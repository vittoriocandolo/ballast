# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Audition: play the exact render buffer, and re-render on every edit.

Every sound is an unchoked one-shot that plays to its end, so on the beat
overlapping tails sum as in a DAW. Hits repeat every rendered length plus
0.5 s, or on the beat, and a new buffer takes over at the next hit. A bed,
such as the track's bass line, can loop under them: each loop starts on the
tempo grid and plays until the next one starts, so it never drifts from the
hits.

When watching, the file is polled every 0.1 s and each new version is
numbered and rendered. A render is used only if the file has not changed
again meanwhile; an invalid edit, or a render over full scale, leaves the
last valid buffer playing.

Ballast never resamples: the stream opens at the patch's sample rate or not
at all. On Windows that takes WASAPI, since MME and DirectSound resample
silently, and on macOS Core Audio is asked to switch the device's rate or
fail. On Linux a sound server such as PipeWire may still resample to its own
rate.
"""

from __future__ import annotations

import importlib
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.io import wavfile

from .patch import Patch, PatchError, loads, read_text
from .render import db, render

if TYPE_CHECKING:
    import os

    import sounddevice
    from numpy.typing import NDArray

GAP_S = 0.5
BAR_TOLERANCE_S = 0.001  # how far a bed may be from a whole number of bars
LAYOUTS = {1: "mono", 2: "stereo"}  # the channel counts a bed may have


class AuditionError(Exception):
    """An audition that cannot start, and why."""


def beat(n: int, rate: int, bpm: float) -> int:
    """Return the sample where beat n starts. Hits and bed loops all start on beats, so they never drift apart."""
    return round(n * rate * 60 / bpm)


@dataclass(frozen=True)
class Bed:
    """A loop of whole bars, as frames × channels."""

    frames: NDArray[np.float32]
    beats: int


def load_bed(path: str | os.PathLike[str], rate: int, bpm: float) -> Bed:
    """Load a mono or stereo WAV at the patch's rate, lasting a whole number of bars at `bpm`."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", wavfile.WavFileWarning)  # chunks it does not know
            file_rate, data = wavfile.read(path)
    except (OSError, ValueError) as e:
        msg = f"{path}: {e}"
        raise AuditionError(msg) from None
    if file_rate != rate:
        msg = f"{path}: {file_rate} Hz, but the patch renders at {rate} Hz; audition never resamples"
        raise AuditionError(msg)
    if data.dtype == np.uint8:
        frames = (data.astype(float) - 128.0) / 128.0
    elif data.dtype.kind == "i":  # 16-bit, and 24- or 32-bit left-justified in int32
        frames = data.astype(float) / float(2 ** (8 * data.dtype.itemsize - 1))
    else:
        frames = data.astype(float)
    frames = frames[:, None] if frames.ndim == 1 else frames
    channels = frames.shape[1]
    if channels not in LAYOUTS:
        msg = f"{path}: {channels} channels; a bed must be mono or stereo"
        raise AuditionError(msg)
    bar = rate * 4 * 60 / bpm
    bars = round(len(frames) / bar)
    if bars < 1 or abs(len(frames) - bars * bar) > BAR_TOLERANCE_S * rate:
        msg = f"{path}: {len(frames) / bar:.3f} bars at {bpm:g} bpm; a bed must be a whole number of bars"
        raise AuditionError(msg)
    sys.stdout.write(f"{'bed':<9}{path}: {bars} bar{'s' if bars > 1 else ''} at {bpm:g} bpm, {LAYOUTS[channels]}\n")
    return Bed(frames.astype(np.float32), 4 * bars)


class Hits:
    """The output: the hits from sample 0 on, and the bed looping under them, mixed block by block."""

    def __init__(
        self,
        buffer: NDArray[np.float32],
        rate: int,
        bpm: float | None = None,
        bed: Bed | None = None,
    ) -> None:
        """Start at sample 0, with a hit after each kick and a gap, or on every beat of `bpm`, over any `bed`."""
        if bed is not None and bpm is None:
            msg = "a bed needs a tempo"
            raise ValueError(msg)
        self.buffer = buffer  # what the next hit plays; the main thread replaces it while the stream runs
        self.rate, self.bpm, self.bed = rate, bpm, bed
        self.channels = 1 if bed is None else bed.frames.shape[1]
        self.sounding: list[tuple[int, NDArray[np.float32]]] = []  # (start sample, frames × channels) still playing
        self.clock = 0  # the next output sample
        self.next_hit = self.hits = self.loops = 0
        self.underflows = 0  # how often the device ran out of samples

    def loop_start(self, k: int) -> int:
        """Return the sample where loop k of the bed starts."""
        if self.bed is None or self.bpm is None:
            msg = "there is no bed to loop"
            raise ValueError(msg)
        return beat(k * self.bed.beats, self.rate, self.bpm)

    def wait(self) -> float:
        """Return the seconds until the next hit."""
        return max(0, self.next_hit - self.clock) / self.rate

    def fill(self, frames: int) -> NDArray[np.float32]:
        """Mix the next `frames` samples, as frames × channels."""
        end = self.clock + frames
        while self.next_hit < end:
            buffer = self.buffer
            self.sounding.append((self.next_hit, buffer[:, None]))
            self.hits += 1
            if self.bpm:
                self.next_hit = beat(self.hits, self.rate, self.bpm)
            else:
                self.next_hit += buffer.size + round(GAP_S * self.rate)
        while self.bed is not None and (start := self.loop_start(self.loops)) < end:
            self.loops += 1
            self.sounding.append((start, self.bed.frames[: self.loop_start(self.loops) - start]))
        self.sounding.sort(key=lambda sound: sound[0])  # summed in order of start, so block sizes never change the mix
        out = np.zeros((frames, self.channels), dtype=np.float32)
        for start, sound in self.sounding:
            a, b = max(self.clock, start), min(end, start + len(sound))
            out[a - self.clock : b - self.clock] += sound[a - start : b - start]
        self.sounding = [(start, sound) for start, sound in self.sounding if start + len(sound) > end]
        self.clock = end
        return out


class Watcher:
    """A file's text, read again whenever its size or modification time changes."""

    def __init__(self, path: str | os.PathLike[str], text: str) -> None:
        """Watch the file at `path`, which held `text` when the session started."""
        self.path, self.text = Path(path), text
        self.stamp: tuple[int, int] | None = None

    def poll(self) -> str | None:
        """Return the file's new text, or None while its content stays the same."""
        try:
            st = self.path.stat()
            stamp = (st.st_mtime_ns, st.st_size)
            if stamp == self.stamp:
                return None
            text = read_text(self.path)
        except (OSError, UnicodeDecodeError):  # unreadable mid-save: the next poll retries
            return None
        self.stamp = stamp
        if text == self.text:
            return None
        self.text = text
        return text


def audition(
    p: Patch,
    *,
    watch: Watcher | None = None,
    bpm: float | None = None,
    bed: str | os.PathLike[str] | None = None,
) -> None:
    """Play the patch until interrupted; with a watcher, render again on every edit of its file."""
    audio, report = render(p)
    sys.stdout.write(f"{report}\n")
    if report.over:
        msg = f"over full scale by {report.over_db:.2f} dB: lower output.db to audition it"
        raise AuditionError(msg)
    rate = report.sample_rate
    loop = None
    if bed:
        if bpm is None:
            msg = "a bed needs a tempo, so the hits land on its beats"
            raise AuditionError(msg)
        loop = load_bed(bed, rate, bpm)
    hits = Hits(audio, rate, bpm, loop)
    _play(hits, audio)
    generation = reported = 0
    with _stream(rate, hits):
        while True:
            time.sleep(0.1)
            if hits.underflows > reported:
                reported = hits.underflows
                sys.stdout.write(
                    f"warning: playback dropped out {reported} time(s) so far; the device ran out of samples\n",
                )
            if watch is not None:
                generation = _follow(watch, hits, generation)


def _play(hits: Hits, audio: NDArray[np.float32]) -> None:
    """Make `audio` the next hit, with a warning if it would clip over the bed."""
    hits.buffer = audio
    if hits.bed is not None:
        # One loop, plus a kick's length, holds every overlap of the steady state.
        mix = Hits(audio, hits.rate, hits.bpm, hits.bed).fill(hits.loop_start(1) + audio.size)
        peak = float(np.max(np.abs(mix)))
        if peak > 1.0:
            sys.stdout.write(
                f"warning: the hits over the bed peak at {db(peak):+.1f} dBFS, which the device will clip\n",
            )


def _follow(watch: Watcher, hits: Hits, generation: int) -> int:
    """Render each edit of the watched file until it holds still, and return the last edit's number."""
    edit = watch.poll()
    while edit is not None:
        generation += 1
        message, update = rerender(edit, hits)
        edit = watch.poll()
        if edit is not None:
            message, update = "dropped: a newer edit is rendering", None
        sys.stdout.write(f"[{generation}] {message}\n")
        if update is not None:
            _play(hits, update)
    return generation


def rerender(text: str, hits: Hits) -> tuple[str, NDArray[np.float32] | None]:
    """Render an edit: the message to show, and the buffer to play if it can replace the last one."""
    start = time.perf_counter()
    try:
        audio, report = render(loads(text))
    except PatchError as e:
        return "invalid patch, still playing the last valid one:\n" + "\n".join(f"  {m}" for m in e.problems), None
    except MemoryError:
        return "render failed: out of memory; still playing the last valid one", None
    elapsed = time.perf_counter() - start
    if report.over:
        return f"over full scale by {report.over_db:.2f} dB; still playing the last valid one", None
    if report.sample_rate != hits.rate:
        return f"now at {report.sample_rate} Hz; restart audition to hear it. Still playing the last valid one", None
    message = f"rendered in {1000 * elapsed:.0f} ms; plays at the next hit, in {1000 * hits.wait():.0f} ms\n{report}"
    return message, audio


def _stream(rate: int, hits: Hits) -> sounddevice.OutputStream:
    """Open an output stream for `hits` at exactly `rate`, at the device's high latency so renders do not starve it."""
    try:
        # Imported here: importing sounddevice starts PortAudio, which only audition needs.
        sd = importlib.import_module("sounddevice")
    except (ImportError, OSError) as e:
        msg = f"audition needs the sounddevice package and PortAudio ({e})"
        raise AuditionError(msg) from None
    choices: list[tuple[dict[str, Any], str]] = [({}, "")]  # stream settings, and what to say about them
    if sys.platform == "win32":
        # WASAPI refuses a rate it cannot play: shared mode plays the mixer's rate, exclusive mode
        # any rate the device itself supports, while it silences other applications.
        wasapi = [api for api in sd.query_hostapis() if api["name"] == "Windows WASAPI"]
        if wasapi:
            shared = {"device": wasapi[0]["default_output_device"]}
            exclusive = {**shared, "extra_settings": sd.WasapiSettings(exclusive=True)}
            choices = [(shared, ""), (exclusive, ", in exclusive mode")]
    elif sys.platform == "darwin":
        switch = sd.CoreAudioSettings(change_device_parameters=True, fail_if_conversion_required=True)
        choices = [({"extra_settings": switch}, "")]

    def callback(outdata: NDArray[np.float32], frames: int, _time: object, status: sounddevice.CallbackFlags) -> None:
        # On the audio thread: count, never print.
        outdata[:] = hits.fill(frames)
        if status.output_underflow:
            hits.underflows += 1

    for settings, note in choices:
        try:
            stream = sd.OutputStream(
                samplerate=rate,
                channels=hits.channels,
                dtype="float32",
                latency="high",
                callback=callback,
                **settings,
            )
        except (sd.PortAudioError, ValueError):
            continue
        sys.stdout.write(f"{'device':<9}{sd.query_devices(stream.device)['name']}{note}\n")
        return stream
    msg = f"the output device cannot play {rate} Hz; audition never resamples"
    raise AuditionError(msg)
