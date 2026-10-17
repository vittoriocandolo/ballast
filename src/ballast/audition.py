"""Audition: play the exact render buffer, and re-render on every edit.

Hits are unchoked one-shots: each plays its buffer to the end, so on the beat
overlapping tails sum as in a DAW. Hits repeat every rendered length plus
0.5 s, or on the beat, and a new buffer takes over at the next hit. A bed,
such as the track's bass line, can loop under them; it restarts on the tempo
grid, so it never drifts from the hits.

When watching, the file is polled every 0.1 s and each new version is
numbered and rendered. A render is used only if the file has not changed
again meanwhile; an invalid edit, or a render over full scale, leaves the
last valid buffer playing.

Nothing is ever resampled: the device opens at the patch's sample rate or
not at all. On Windows that takes WASAPI, since MME and DirectSound resample
silently.
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
from scipy.io import wavfile

from .patch import Patch, PatchError, loads, read_text
from .render import db, render

GAP_S = 0.5
BAR_TOLERANCE_S = 0.001  # how far a bed may be from a whole number of bars


class AuditionError(Exception):
    pass


def beat(n: int, rate: int, bpm: float) -> int:
    """The sample where beat n starts. Hits and bed loops both start on it, so they never drift apart."""
    return round(n * rate * 60 / bpm)


class Bed:
    """A loop restarted every `beats` beats, playing as much of itself as fits before the next start."""

    def __init__(self, frames: np.ndarray, rate: int, bpm: float, beats: int):
        self.frames, self.rate, self.bpm, self.beats = frames, rate, bpm, beats

    def start(self, k: int) -> int:
        return beat(k * self.beats, self.rate, self.bpm)

    def add(self, out: np.ndarray, clock: int) -> None:
        """Add the bed to `out`, the block that starts at sample `clock`."""
        end = clock + len(out)
        k = max(0, int(clock // (self.beats * self.rate * 60 / self.bpm)) - 1)
        while self.start(k) < end:
            s = self.start(k)
            a, b = max(clock, s), min(end, self.start(k + 1), s + len(self.frames))
            if a < b:
                out[a - clock : b - clock] += self.frames[a - s : b - s]
            k += 1


def load_bed(path, rate: int, bpm: float) -> Bed:
    """A mono or stereo WAV at the patch's rate, lasting a whole number of bars at `bpm`."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", wavfile.WavFileWarning)  # chunks it does not know
            file_rate, data = wavfile.read(path)
    except (OSError, ValueError) as e:
        raise AuditionError(f"{path}: {e}") from None
    if file_rate != rate:
        raise AuditionError(f"{path}: {file_rate} Hz, but the patch renders at {rate} Hz; audition never resamples")
    if data.dtype == np.uint8:
        frames = (data.astype(float) - 128.0) / 128.0
    elif data.dtype.kind == "i":  # 16-bit, and 24- or 32-bit left-justified in int32
        frames = data.astype(float) / float(2 ** (8 * data.dtype.itemsize - 1))
    else:
        frames = data.astype(float)
    frames = frames if frames.ndim == 2 else frames[:, None]
    if frames.shape[1] > 2:
        raise AuditionError(f"{path}: {frames.shape[1]} channels; a bed must be mono or stereo")
    bar = rate * 4 * 60 / bpm
    bars = round(len(frames) / bar)
    if bars < 1 or abs(len(frames) - bars * bar) > BAR_TOLERANCE_S * rate:
        raise AuditionError(
            f"{path}: {len(frames) / bar:.3f} bars at {bpm:g} bpm; a bed must be a whole number of bars"
        )
    layout = "stereo" if frames.shape[1] == 2 else "mono"
    print(f"{'bed':<9}{path}: {bars} bar{'s' if bars > 1 else ''} at {bpm:g} bpm, {layout}")
    return Bed(frames.astype(np.float32), rate, bpm, 4 * bars)


class Hits:
    """The output: the hits, from sample 0 on, and the bed under them, mixed block by block."""

    def __init__(self, buffer: np.ndarray, rate: int, bpm: float | None = None, bed: Bed | None = None):
        self.buffer = buffer  # what the next hit plays; the main thread replaces it while the stream runs
        self.rate, self.bpm, self.bed = rate, bpm, bed
        self.channels = 1 if bed is None else bed.frames.shape[1]
        self.voices: list[tuple[int, np.ndarray]] = []  # (start sample, buffer) of the hits still sounding
        self.clock = 0  # the next output sample
        self.next_hit = self.count = 0

    def wait(self) -> float:
        """Seconds until the next hit."""
        return max(0, self.next_hit - self.clock) / self.rate

    def fill(self, frames: int) -> np.ndarray:
        """The next `frames` samples, as frames × channels."""
        end = self.clock + frames
        while self.next_hit < end:
            buffer = self.buffer
            self.voices.append((self.next_hit, buffer))
            self.count += 1
            if self.bpm:
                self.next_hit = beat(self.count, self.rate, self.bpm)
            else:
                self.next_hit += buffer.size + round(GAP_S * self.rate)
        out = np.zeros((frames, self.channels), dtype=np.float32)
        for start, buf in self.voices:
            a, b = max(self.clock, start), min(end, start + buf.size)
            out[a - self.clock : b - self.clock] += buf[a - start : b - start, None]
        self.voices = [(start, buf) for start, buf in self.voices if start + buf.size > end]
        if self.bed is not None:
            self.bed.add(out, self.clock)
        self.clock = end
        return out


class Watcher:
    """A file's text, read again whenever its size or modification time changes."""

    def __init__(self, path, text: str):
        self.path, self.text, self.stamp = path, text, None

    def poll(self) -> str | None:
        """The file's new text, or None while its content stays the same."""
        try:
            st = os.stat(self.path)
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


def audition(path, text: str, p: Patch, *, watch: bool = False, bpm: float | None = None, bed=None) -> None:
    """Play the patch until interrupted; when watching, re-render on every edit of the file."""
    audio, report = render(p)
    print(report)
    if report.over:
        raise AuditionError(f"over full scale by {report.over_db:.2f} dB: lower output.db to audition it")
    rate = report.sample_rate
    hits = Hits(audio, rate, bpm, load_bed(bed, rate, bpm) if bed else None)

    def play(audio: np.ndarray) -> None:
        hits.buffer = audio
        if hits.bed is not None:
            # One loop, plus a kick's length, holds every overlap of the steady state.
            peak = float(np.max(np.abs(Hits(audio, rate, bpm, hits.bed).fill(hits.bed.start(1) + audio.size))))
            if peak > 1.0:
                print(f"warning: the hits over the bed peak at {db(peak):+.1f} dBFS, which the device will clip")

    play(audio)
    watcher = Watcher(path, text)
    generation = 0
    with _stream(rate, hits.channels, hits.fill):
        while True:
            time.sleep(0.1)
            edit = watcher.poll() if watch else None
            while edit is not None:  # render edits until the file holds still
                generation += 1
                message, audio = _rerender(edit, rate, hits)
                edit = watcher.poll()
                if edit is not None:
                    message, audio = "dropped: a newer edit is rendering", None
                print(f"[{generation}] {message}")
                if audio is not None:
                    play(audio)


def _rerender(text: str, rate: int, hits: Hits) -> tuple[str, np.ndarray | None]:
    """Render an edit: the message to show, and the buffer to play if it can replace the last one."""
    start = time.perf_counter()
    try:
        audio, report = render(loads(text))
    except PatchError as e:
        return "invalid patch, still playing the last valid one:\n" + "\n".join(f"  {m}" for m in e.problems), None
    except Exception as e:  # whatever goes wrong, the last valid buffer keeps playing
        return f"render failed, still playing the last valid one: {e}", None
    elapsed = time.perf_counter() - start
    if report.over:
        return f"over full scale by {report.over_db:.2f} dB; still playing the last valid one", None
    if report.sample_rate != rate:
        return f"now at {report.sample_rate} Hz; restart audition to hear it. Still playing the last valid one", None
    message = f"rendered in {1000 * elapsed:.0f} ms; plays at the next hit, in {1000 * hits.wait():.0f} ms\n{report}"
    return message, audio


def _stream(rate: int, channels: int, fill):
    """An output stream at exactly `rate`, at the device's high latency so that renders do not starve it."""
    try:
        import sounddevice as sd
    except (ImportError, OSError) as e:
        raise AuditionError(f"audition needs the sounddevice package and PortAudio ({e})") from None
    choices = [{}]
    if sys.platform == "win32":
        # WASAPI refuses a rate it cannot play: shared mode plays the mixer's rate, exclusive mode
        # any rate the device itself supports, while it silences other applications.
        wasapi = [api for api in sd.query_hostapis() if api["name"] == "Windows WASAPI"]
        if wasapi:
            device = wasapi[0]["default_output_device"]
            choices = [{"device": device}, {"device": device, "extra_settings": sd.WasapiSettings(exclusive=True)}]
    for choice in choices:
        try:
            sd.check_output_settings(samplerate=rate, channels=channels, dtype="float32", **choice)
        except (sd.PortAudioError, ValueError):
            continue
        device = sd.query_devices(choice["device"]) if "device" in choice else sd.query_devices(kind="output")
        print(f"{'device':<9}{device['name']}" + (", in exclusive mode" if "extra_settings" in choice else ""))

        def callback(outdata, frames, _time, _status):
            outdata[:] = fill(frames)

        return sd.OutputStream(
            samplerate=rate, channels=channels, dtype="float32", latency="high", callback=callback, **choice
        )
    raise AuditionError(f"the output device cannot play {rate} Hz; audition never resamples")
