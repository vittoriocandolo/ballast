"""Rendering: rates, the frozen decimators, the output stage and the exported file.

    out[m] = p · g_out · Σ_{k<L} h[k] y[mK − k],    y[n] = y(n / F) for 0 ≤ n < N, else 0

The internal rate is F = K × base, with base 44.1 or 48 kHz. The 88.2 and
96 kHz exports reuse their family's F and filter with factor K/2, so a 96 kHz
file is exactly the 48 kHz file with the in-between samples filled in.
Nothing trims, fades, high-passes or removes DC, and the whole filter tail
is kept.

The internal signal is computed in blocks aligned to the trigger. Every
sample is a pure function of its time, so the blocks never change a value,
and a longer render never changes earlier samples.
"""

from __future__ import annotations

import functools
import hashlib
import math
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib import resources

import numpy as np
import scipy
from scipy.signal import resample_poly, upfirdn

from . import __version__, noise, wav
from .patch import Output, Patch, describe_tuning
from .voice import sample_count, saturate, sources, voice

K = 16
BLOCK = 2**16
FAMILY = {44100: 44100, 88200: 44100, 48000: 48000, 96000: 48000}
# One frozen minimum-phase FIR per family, shipped as data so the sound cannot drift with SciPy:
# flat to 0.001 dB up to 20 kHz, at least 110 dB down from base/2, taps summing to exactly 1.
DECIMATORS = {
    44100: ("decimator-44100x16.f64", "0932afce82f960c811e1a3bf74ccc4636a5aefd2161f5d69466b0a522955c7d0"),
    48000: ("decimator-48000x16.f64", "753e10eadb9b6550d7dfc81847d7a3057b578c8d3bfe07278d0b474d7e29b676"),
}
FORMATS = {"float32": None, "pcm24": 24, "pcm16": 16}  # the PCM bit depth of each export format


class RenderError(RuntimeError):
    pass


def rates(sample_rate: int) -> tuple[int, int, int]:
    """(family base rate, internal rate F, decimation factor) for an export rate."""
    base = FAMILY[sample_rate]
    return base, base * K, K * base // sample_rate


@functools.cache
def decimator(base: int) -> np.ndarray:
    """The family's frozen taps, refused unless they match their checksum."""
    name, sha256 = DECIMATORS[base]
    raw = resources.files(__package__).joinpath("data", name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise RenderError(f"{name} does not match its checksum")
    taps = np.frombuffer(raw, dtype="<f8").astype(float)
    taps.flags.writeable = False
    return taps


def decimate(taps: np.ndarray, y: np.ndarray, factor: int) -> np.ndarray:
    """upfirdn(taps, y, down=factor), computed in parallel chunks that match it bit for bit.

    Each output of upfirdn is one sequential sum over its own inputs. A chunk is
    given every input its outputs need, starting on the factor's grid, so each
    output is computed by exactly the same operations as in a single call.
    """
    L = taps.size
    n_out = (y.size + L - 2) // factor + 1
    workers = min(os.cpu_count() or 1, max(1, n_out // 4096))
    if workers == 1:
        return upfirdn(taps, y, down=factor)
    edges = [round(i * n_out / workers) for i in range(workers + 1)]

    def chunk(m0: int, m1: int) -> np.ndarray:
        start = max(0, (m0 * factor - (L - 1)) // factor * factor)
        stop = y.size if m1 == n_out else (m1 - 1) * factor + 1
        return upfirdn(taps, y[start:stop], down=factor)[m0 - start // factor : m1 - start // factor]

    with ThreadPoolExecutor(workers) as pool:
        return np.concatenate(list(pool.map(chunk, edges[:-1], edges[1:])))


def synthesize(p: Patch) -> np.ndarray:
    """The decimated signal at unity gain: everything before the output stage."""
    v = voice(p)
    base, rate, factor = rates(p.output.sample_rate)
    count = sample_count(v.t_src, rate)
    table = noise.table(v.seed, *v.band, rate) if v.uses_noise else None
    y = np.empty(count)
    for n0 in range(0, count, BLOCK):
        n1 = min(n0 + BLOCK, count)
        y[n0:n1] = saturate(v, sources(v, rate, n0, n1, table))
    return decimate(decimator(base), y, factor)


def output_stage(out: np.ndarray, o: Output, top: float = 1.0) -> np.ndarray:
    """Output gain or peak normalization, then polarity, then one conversion to float32.

    `top` is the format's ceiling: a peak target never goes above it.
    """
    if o.mode == "gain":
        scaled = out * 10.0 ** (o.db / 20.0)
    else:
        peak = float(np.max(np.abs(out), initial=0.0))
        scaled = out * (min(10.0 ** (o.db / 20.0), top) / peak) if peak > 0 else out.copy()
    if o.polarity < 0:
        scaled = -scaled
    return scaled.astype(np.float32)


def ceiling(bits: int | None) -> float:
    """The largest sample a format takes: full scale for float, 2 LSB below it for PCM, so dither never clips."""
    return 1.0 if bits is None else 1.0 - 2.0 ** (2 - bits)


def true_peak(audio: np.ndarray) -> float:
    """A 4× oversampled estimate of the peak between samples."""
    x = np.concatenate((audio.astype(float), np.zeros(32)))
    return max(float(np.max(np.abs(resample_poly(x, 4, 1)))), float(np.max(np.abs(audio), initial=0.0)))


def db(x: float) -> float:
    return 20 * math.log10(x) if x > 0 else -math.inf


@dataclass(frozen=True)
class Report:
    tune_hz: float
    source_ms: float
    samples: int
    sample_rate: int
    bits: int | None  # PCM bit depth, None for float32
    peak: float
    true_peak: float
    ceiling: float
    over: bool  # a gain-mode peak above the format's ceiling
    render_s: float

    @property
    def over_db(self) -> float:
        """How far the peak is above the format's ceiling."""
        return db(self.peak / self.ceiling)

    def __str__(self) -> str:
        if self.peak == 0:
            peak = "silent"
        elif self.bits is None and self.over:
            peak = f"{db(self.peak):+.2f} dBFS sample, over full scale by {self.over_db:.2f} dB"
        elif self.over:  # the PCM ceiling is within a thousandth of a dB of full scale
            peak = (
                f"{db(self.peak):+.4f} dBFS sample, over the {self.bits}-bit ceiling of {db(self.ceiling):.4f} dBFS, "
                f"which keeps 2 LSB for dither, by {self.over_db:.4f} dB"
            )
        else:
            peak = f"{db(self.peak):.2f} dBFS sample, {db(self.true_peak):.2f} dBTP true"
        pcm = "" if self.bits is None else f", {self.bits}-bit PCM"
        length = (
            f"{self.source_ms:.1f} ms source, {1000 * self.samples / self.sample_rate:.1f} ms exported "
            f"({self.samples} samples at {self.sample_rate} Hz{pcm})"
        )
        rows = (
            ("tuning", f"{self.tune_hz:.3f} Hz, {describe_tuning(self.tune_hz)}"),
            ("length", length),
            ("peak", peak),
            ("render", f"{1000 * self.render_s:.0f} ms"),
        )
        return "\n".join(f"{label:<9}{text}" for label, text in rows)


def render(p: Patch, bits: int | None = None) -> tuple[np.ndarray, Report]:
    """The float32 audio and its report; `bits` gives the ceiling of the PCM format it will be exported to."""
    start = time.perf_counter()
    top = ceiling(bits)
    audio = output_stage(synthesize(p), p.output, top)
    peak = float(np.max(np.abs(audio), initial=0.0))
    return audio, Report(
        tune_hz=p.body.tune_hz,
        source_ms=p.length_ms,
        samples=audio.size,
        sample_rate=p.output.sample_rate,
        bits=bits,
        peak=peak,
        true_peak=true_peak(audio),
        ceiling=top,
        over=p.output.mode == "gain" and peak > top,
        render_s=time.perf_counter() - start,
    )


def quantize(audio: np.ndarray, bits: int, seed: int) -> np.ndarray:
    """Round to `bits`-bit PCM with the seed's TPDF dither of ±1 LSB.

    Exact zeros stay zero, so the silent first sample and any digital silence
    stay silent. A render within the format's ceiling, 2 LSB below full
    scale, still fits after dither.
    """
    x = audio.astype(float) * 2 ** (bits - 1)
    return np.round(np.where(x != 0.0, x + noise.dither(seed, x.size), 0.0)).astype(np.int32)


def export(path, p: Patch, text: str, audio: np.ndarray, bits: int | None = None) -> None:
    """Write the WAV, as float32 or dithered PCM, with its root note, the patch text and provenance."""
    base, rate, factor = rates(p.output.sample_rate)
    meta = {
        "patch": text,
        "renderer": f"ballast {__version__}",
        "decimator": {"sha256": DECIMATORS[base][1], "internal_rate": rate, "factor": factor},
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    if bits is not None:
        meta["pcm"] = {
            "bits": bits,
            "dither": "TPDF, ±1 LSB; exact zeros kept",
            "dither_key": noise.DITHER_KEY.decode(),
            "seed": p.seed,
        }
        audio = quantize(audio, bits, p.seed)
    wav.write(path, audio, p.output.sample_rate, bits, wav.root_note(p.body.tune_hz), meta)
