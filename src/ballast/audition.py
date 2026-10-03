"""Audition: play the exact render buffer, and re-render on every edit (plan §7).

Hits are unchoked one-shots: each plays its buffer to the end, so on the beat
overlapping tails sum as in a DAW. A new buffer takes over at the next hit
boundary. Hits repeat every rendered length plus 0.5 s, or on the beat.

The watcher tags each file change with an increasing generation and renders
it in the background. A finished render is used only if its generation is
still the newest, so stale results are dropped; an invalid edit, or a
gain-mode render over full scale, leaves the last valid buffer playing.

A bed (`--with`), such as the track's bass line or loop, plays under the
hits; it restarts on the tempo grid, so it never drifts from them. It is
never resampled.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import warnings
from typing import Callable

import numpy as np

from .patch import PatchError, loads, read_text
from .render import db, render

GAP_S = 0.5
BAR_TOLERANCE_S = 0.001  # how far a bed may be from a whole number of bars


class AuditionError(Exception):
    pass


def load_audio(path) -> tuple[int, np.ndarray]:
    """Any PCM or float WAV as float64 frames × channels, never resampled."""
    from scipy.io import wavfile

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", wavfile.WavFileWarning)  # chunks it does not know
        rate, data = wavfile.read(path)
    if data.dtype == np.uint8:
        x = (data.astype(float) - 128.0) / 128.0
    elif data.dtype.kind == "i":  # 16-bit, and 24- or 32-bit left-justified in int32
        x = data.astype(float) / float(2 ** (8 * data.dtype.itemsize - 1))
    else:
        x = data.astype(float)
    return rate, x if x.ndim == 2 else x[:, None]


def _load(path) -> tuple[int, np.ndarray]:
    try:
        return load_audio(path)
    except (OSError, ValueError) as e:
        raise AuditionError(f"{path}: {e}") from None


class Bed:
    """A loop under the hits, restarted every `bars` bars on the tempo grid.

    Loop k starts at round(k · bars · 4 · 60 / bpm · rate), the same rounding
    as the hits, and plays as much of the bed as fits before the next start.
    """

    def __init__(self, frames: np.ndarray, rate: int, bpm: float, bars: int):
        self.frames = np.ascontiguousarray(frames, dtype=np.float32)
        self.channels = self.frames.shape[1]
        self.rate, self.bpm, self.bars = rate, bpm, bars

    def start(self, k: int) -> int:
        return round(k * self.bars * 4 * self.rate * 60 / self.bpm)

    def fill(self, clock: int, frames: int) -> np.ndarray:
        out = np.zeros((frames, self.channels), dtype=np.float32)
        k = max(0, int(clock // (self.bars * 4 * self.rate * 60 / self.bpm)) - 1)
        while self.start(k) < clock + frames:
            s, e = self.start(k), min(self.start(k + 1), self.start(k) + len(self.frames))
            a, b = max(clock, s), min(clock + frames, e)
            if a < b:
                out[a - clock : b - clock] += self.frames[a - s : b - s]
            k += 1
        return out


class Hits:
    """Mixes hits into output blocks. `offer` may be called from any thread."""

    def __init__(self, rate: int, bpm: float | None = None, *, bed: Bed | None = None):
        self.rate, self.bpm, self.bed = rate, bpm, bed
        self.channels = bed.channels if bed is not None else 1
        self._offer: tuple[int, np.ndarray] | None = None
        self.generation, self.buffer = -1, None  # what new hits play
        self.voices: list[tuple[int, np.ndarray]] = []  # (start sample, buffer)
        self.clock = 0  # index of the next output sample
        self.next_hit: int | None = None
        self.origin, self.count = 0, 0

    def offer(self, generation: int, buffer: np.ndarray) -> None:
        """Queue a buffer for the next hit boundary. An older generation never replaces a newer one."""
        current = self._offer
        if current is None or generation > current[0]:
            self._offer = (generation, buffer)

    def wait(self) -> float:
        """Seconds until the next hit."""
        return 0.0 if self.next_hit is None else max(0, self.next_hit - self.clock) / self.rate

    def fill(self, frames: int) -> np.ndarray:
        out = np.zeros(frames, dtype=np.float32)
        end = self.clock + frames
        if self.next_hit is None and self._offer is not None:
            self.next_hit = self.origin = self.clock
        while self.next_hit is not None and self.next_hit < end:
            offered = self._offer
            if offered is not None and offered[0] > self.generation:
                self.generation, self.buffer = offered
            self.voices.append((self.next_hit, self.buffer))
            self.count += 1
            if self.bpm:
                self.next_hit = self.origin + round(self.count * self.rate * 60 / self.bpm)
            else:
                self.next_hit += self.buffer.size + round(GAP_S * self.rate)
        playing = []
        for start, buf in self.voices:
            a, b = max(self.clock, start), min(end, start + buf.size)
            if a < b:
                out[a - self.clock : b - self.clock] += buf[a - start : b - start]
            if start + buf.size > end:
                playing.append((start, buf))
        self.voices = playing
        if self.bed is not None:
            mix = self.bed.fill(self.clock, frames)
            mix += out[:, None]
            out = mix[:, 0] if self.channels == 1 else mix
        self.clock = end
        return out


class Watcher:
    """Renders the newest version of a file in the background."""

    def __init__(
        self,
        path,
        work: Callable[[str], object],
        done: Callable[[int, object, float, bool], None],
        text: str | None = None,
    ):
        self.path, self.work, self.done = path, work, done
        self.generation = 0
        self._stamp, self._text = None, text  # `text`: the version already rendered, if any
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
            except Exception as e:  # reported by `done`; the last valid buffer keeps playing
                result = e
            elapsed = time.perf_counter() - start
            with self._cv:
                newest = generation == self.generation
            self.done(generation, result, elapsed, newest)


class Session:
    """One audition: the hits, and the watcher when watching. `poll` is called about every 0.1 s."""

    def __init__(self, path, *, watch: bool = False, bpm: float | None = None, bed=None, out=sys.stdout):
        self.out = out
        text = read_text(path)
        audio, report = render(loads(text))
        print(report, file=out)
        if report.over:
            raise OverError(report)
        self.rate, self.bpm = report.sample_rate, bpm
        self.bed = self._bed(bed, bpm) if bed is not None else None
        self.hits = Hits(self.rate, bpm, bed=self.bed)
        self._offer(0, audio)
        self.watcher = Watcher(path, lambda t: render(loads(t)), self._done, text) if watch else None

    def _bed(self, path, bpm) -> Bed:
        if bpm is None:
            raise AuditionError("--with needs --bpm, so the hits land on the bed's beats")
        rate, frames = _load(path)
        if rate != self.rate:
            raise AuditionError(f"{path}: {rate} Hz, but the patch renders at {self.rate} Hz; audition never resamples")
        if frames.shape[1] > 2:
            raise AuditionError(f"{path}: {frames.shape[1]} channels; a bed must be mono or stereo")
        bar = self.rate * 4 * 60 / bpm
        bars = round(frames.shape[0] / bar)
        if bars < 1 or abs(frames.shape[0] - bars * bar) > BAR_TOLERANCE_S * self.rate:
            raise AuditionError(
                f"{path}: {frames.shape[0] / bar:.3f} bars at {bpm:g} bpm; a bed must be a whole number of bars"
            )
        print(
            f"{'bed':<9}{path}: {bars} bar{'s' if bars > 1 else ''} at {bpm:g} bpm, "
            f"{'stereo' if frames.shape[1] == 2 else 'mono'}",
            file=self.out,
        )
        return Bed(frames, self.rate, bpm, bars)

    def _offer(self, generation: int, audio: np.ndarray) -> None:
        """Queue the render, and warn if the hits over the bed would clip."""
        self.hits.offer(generation, audio)
        if self.bed is not None:
            # One loop after a kick's length has passed holds every overlap of the steady state.
            probe = Hits(self.rate, self.bpm, bed=self.bed)
            probe.offer(0, audio)
            peak = float(np.max(np.abs(probe.fill(self.bed.start(1) + audio.size))))
            if peak > 1.0:
                print(
                    f"warning: the hits over the bed peak at {db(peak):+.1f} dBFS, which the device "
                    "will clip; lower the bed or output.db",
                    file=self.out,
                )

    def poll(self) -> None:
        if self.watcher is not None:
            self.watcher.poll()

    def _done(self, generation, result, elapsed, newest) -> None:
        say = lambda msg: print(f"[{generation}] {msg}", file=self.out)  # noqa: E731
        if not newest:
            say("dropped: a newer edit is rendering")
        elif isinstance(result, PatchError):
            say(f"invalid patch, still playing the last valid one:\n{result}")
        elif isinstance(result, Exception):
            say(f"render failed, still playing the last valid one: {result}")
        else:
            audio, report = result
            if report.over:
                say(
                    f"over full scale by {db(report.peak):.2f} dB; lower output.db. Still playing the last valid buffer."
                )
            elif report.sample_rate != self.rate:
                say(
                    f"sample rate changed to {report.sample_rate} Hz; restart audition to hear it. "
                    "Still playing the last valid buffer."
                )
            else:
                say(
                    f"rendered in {1000 * elapsed:.0f} ms; plays at the next hit, in "
                    f"{1000 * self.hits.wait():.0f} ms\n{report}"
                )
                self._offer(generation, audio)


class OverError(Exception):
    def __init__(self, report):
        super().__init__(f"over full scale by {db(report.peak):.2f} dB: lower output.db to audition it")


def audition(
    path,
    *,
    watch: bool = False,
    bpm: float | None = None,
    bed=None,
    out=sys.stdout,
    stream_factory=None,
) -> int:
    """Play the patch until interrupted; with `watch`, re-render on every edit."""
    try:
        session = Session(path, watch=watch, bpm=bpm, bed=bed, out=out)
    except (OSError, PatchError) as e:
        print(f"{path}: {e}", file=sys.stderr)
        return 2
    except AuditionError as e:
        print(e, file=sys.stderr)
        return 2
    except OverError as e:
        print(e, file=sys.stderr)
        return 1
    try:
        stream = (stream_factory or _sounddevice_stream)(session.rate, session.hits.channels, session.hits.fill)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1
    try:
        with stream:
            while True:
                time.sleep(0.1)
                session.poll()
    except KeyboardInterrupt:
        return 0


def _sounddevice_stream(rate: int, channels: int, fill):
    try:
        import sounddevice as sd
    except (ImportError, OSError) as e:
        raise RuntimeError(f"audition needs the sounddevice package and PortAudio ({e})") from None
    try:
        sd.check_output_settings(samplerate=rate, channels=channels, dtype="float32")
    except Exception as e:
        raise RuntimeError(f"the output device does not support {rate} Hz ({e}); audition never resamples") from None

    def callback(outdata, frames, _time, _status):
        outdata[:] = fill(frames).reshape(frames, channels)

    return sd.OutputStream(samplerate=rate, channels=channels, dtype="float32", callback=callback)
