"""Audition: play the exact render buffer, and re-render on every edit.

Hits are unchoked one-shots: each plays its buffer to the end, so on the beat
overlapping tails sum as in a DAW. Hits repeat every rendered length plus
0.5 s, or on the beat, and a new buffer takes over at the next hit. A bed,
such as the track's bass line, can loop under them; it restarts on the tempo
grid, so it never drifts from the hits.

When watching, each change to the file gets an increasing generation and is
rendered in the background. A finished render is used only if its generation
is still the newest, so stale results are dropped; an invalid edit, or a
render over full scale, leaves the last valid buffer playing.

Nothing is ever resampled: the device opens at the patch's sample rate or
not at all. On Windows that takes WASAPI, since MME and DirectSound resample
silently.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import warnings

import numpy as np

from .patch import Patch, PatchError, loads, read_text
from .render import db, render

GAP_S = 0.5
BAR_TOLERANCE_S = 0.001  # how far a bed may be from a whole number of bars


class AuditionError(Exception):
    pass


class Bed:
    """A loop restarted every `bars` bars on the tempo grid.

    Loop k starts at round(k · bars · 4 · 60 / bpm · rate), the same rounding
    as the hits, and plays as much of the bed as fits before the next start.
    """

    def __init__(self, frames: np.ndarray, rate: int, bpm: float, bars: int):
        self.frames, self.rate, self.bpm, self.bars = frames, rate, bpm, bars

    def start(self, k: int) -> int:
        return round(k * self.bars * 4 * self.rate * 60 / self.bpm)

    def add(self, out: np.ndarray, clock: int) -> None:
        """Add the bed to `out`, the block that starts at sample `clock`."""
        end = clock + len(out)
        k = max(0, int(clock // (self.bars * 4 * self.rate * 60 / self.bpm)) - 1)
        while self.start(k) < end:
            s = self.start(k)
            a, b = max(clock, s), min(end, self.start(k + 1), s + len(self.frames))
            if a < b:
                out[a - clock : b - clock] += self.frames[a - s : b - s]
            k += 1


def load_bed(path, rate: int, bpm: float) -> Bed:
    """A mono or stereo WAV at the patch's rate, lasting a whole number of bars at `bpm`."""
    from scipy.io import wavfile

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
    return Bed(frames.astype(np.float32), rate, bpm, bars)


class Hits:
    """The output: the hits, and the bed under them, mixed block by block. `offer` may be called from any thread."""

    def __init__(self, rate: int, bpm: float | None = None, bed: Bed | None = None):
        self.rate, self.bpm, self.bed = rate, bpm, bed
        self.channels = 1 if bed is None else bed.frames.shape[1]
        self._offer: tuple[int, np.ndarray] | None = None
        self.generation, self.buffer = -1, None  # what new hits play
        self.voices: list[tuple[int, np.ndarray]] = []  # (start sample, buffer) of the hits still sounding
        self.clock = 0  # the next output sample
        self.next_hit: int | None = None
        self.origin, self.count = 0, 0

    def offer(self, generation: int, buffer: np.ndarray) -> None:
        """Queue a buffer for the next hit; an older generation never replaces a newer one."""
        offered = self._offer
        if offered is None or generation > offered[0]:
            self._offer = (generation, buffer)

    def wait(self) -> float:
        """Seconds until the next hit."""
        return 0.0 if self.next_hit is None else max(0, self.next_hit - self.clock) / self.rate

    def fill(self, frames: int) -> np.ndarray:
        """The next `frames` samples, as frames × channels."""
        end = self.clock + frames
        if self.next_hit is None and self._offer is not None:
            self.next_hit = self.origin = self.clock
        while self.next_hit is not None and self.next_hit < end:
            offered = self._offer
            if offered[0] > self.generation:
                self.generation, self.buffer = offered
            self.voices.append((self.next_hit, self.buffer))
            self.count += 1
            if self.bpm:
                self.next_hit = self.origin + round(self.count * self.rate * 60 / self.bpm)
            else:
                self.next_hit += self.buffer.size + round(GAP_S * self.rate)
        out = np.zeros((frames, self.channels), dtype=np.float32)
        for start, buf in self.voices:
            a, b = max(self.clock, start), min(end, start + buf.size)
            if a < b:
                out[a - self.clock : b - self.clock] += buf[a - start : b - start, None]
        self.voices = [(start, buf) for start, buf in self.voices if start + buf.size > end]
        if self.bed is not None:
            self.bed.add(out, self.clock)
        self.clock = end
        return out


class Watcher:
    """Renders the newest version of a file in the background, and hands each result to `done`."""

    def __init__(self, path, text: str, work, done):
        self.path, self.work, self.done = path, work, done
        self.generation = 0
        self._stamp, self._text = None, text  # the version already rendered
        self._pending: tuple[int, str] | None = None
        self._cv = threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def poll(self) -> None:
        """Start a new generation if the file's content changed."""
        try:
            st = os.stat(self.path)
            stamp = (st.st_mtime_ns, st.st_size)
            if stamp == self._stamp:
                return
            text = read_text(self.path)
        except (OSError, UnicodeDecodeError):  # unreadable mid-save: the next poll retries
            return
        self._stamp = stamp
        if text == self._text:
            return
        self._text = text
        with self._cv:
            self.generation += 1
            self._pending = (self.generation, text)  # an older request not yet started is superseded
            self._cv.notify()

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._pending is None:
                    self._cv.wait()
                generation, text = self._pending
                self._pending = None
            start = time.perf_counter()
            try:
                result = self.work(text)
            except Exception as e:  # handed to `done`; the last valid buffer keeps playing
                result = e
            elapsed = time.perf_counter() - start
            with self._cv:
                newest = generation == self.generation
            self.done(generation, result, elapsed, newest)


def audition(path, text: str, p: Patch, *, watch: bool = False, bpm: float | None = None, bed=None) -> None:
    """Play the patch until interrupted; when watching, re-render on every edit of the file."""
    audio, report = render(p)
    print(report)
    if report.over:
        raise AuditionError(f"over full scale by {report.over_db:.2f} dB: lower output.db to audition it")
    rate = report.sample_rate
    hits = Hits(rate, bpm, load_bed(bed, rate, bpm) if bed else None)

    def play(generation: int, audio: np.ndarray) -> None:
        hits.offer(generation, audio)
        if hits.bed is not None:
            # One loop, plus a kick's length, holds every overlap of the steady state.
            probe = Hits(rate, bpm, hits.bed)
            probe.offer(0, audio)
            peak = float(np.max(np.abs(probe.fill(hits.bed.start(1) + audio.size))))
            if peak > 1.0:
                print(f"warning: the hits over the bed peak at {db(peak):+.1f} dBFS, which the device will clip")

    def done(generation: int, result, elapsed: float, newest: bool) -> None:
        def say(message):
            print(f"[{generation}] {message}")

        audio, report = result if isinstance(result, tuple) else (None, None)
        if not newest:
            say("dropped: a newer edit is rendering")
        elif isinstance(result, PatchError):
            say("invalid patch, still playing the last valid one:\n" + "\n".join(f"  {m}" for m in result.problems))
        elif isinstance(result, Exception):
            say(f"render failed, still playing the last valid one: {result}")
        elif report.over:
            say(f"over full scale by {report.over_db:.2f} dB; still playing the last valid one")
        elif report.sample_rate != rate:
            say(f"now at {report.sample_rate} Hz; restart audition to hear it. Still playing the last valid one")
        else:
            say(f"rendered in {1000 * elapsed:.0f} ms; plays at the next hit, in {1000 * hits.wait():.0f} ms\n{report}")
            play(generation, audio)

    play(0, audio)
    watcher = Watcher(path, text, lambda t: render(loads(t)), done) if watch else None
    with _stream(rate, hits.channels, hits.fill):
        while True:
            time.sleep(0.1)
            if watcher is not None:
                watcher.poll()


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
