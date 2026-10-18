# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Rendering: rates, the frozen decimators, the output stage and the exported file.

    out[m] = p · g_out · Σ_{k<L} h[k] y[mK − k],    y = L(y(n / F)) for 0 ≤ n < N, else 0

The internal rate is F = K × base, with base 44.1 or 48 kHz. The 88.2 and
96 kHz exports reuse their family's F and filter with factor K/2, and 176.4
and 192 kHz with K/4, so a 96 kHz file is exactly the 48 kHz file with the
in-between samples filled in. Nothing trims or fades, only the optional low
cut L high-passes, and the whole filter tail is kept.

The internal signal is computed in blocks aligned to the trigger and
decimated in parallel chunks. Every sample is a pure function of its time,
the low cut one causal pass, and every output one fixed sum, so neither the
blocks nor the chunks change a value, and a longer render never changes
earlier samples.
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
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy
from scipy.signal import resample_poly, sosfilt, upfirdn

from . import __version__, noise, wav
from .patch import Output, Patch, describe_tuning
from .voice import Voice

if TYPE_CHECKING:
    from numpy.typing import NDArray

K = 16
CHUNK = 2048  # decimator outputs per parallel chunk
FAMILY = {44100: 44100, 88200: 44100, 176400: 44100, 48000: 48000, 96000: 48000, 192000: 48000}
# One frozen minimum-phase FIR per family, shipped as data so the sound cannot drift with SciPy:
# flat to 0.001 dB up to 20 kHz, at least 110 dB down from base/2, taps summing to exactly 1.
DECIMATORS = {
    44100: ("decimator-44100x16.f64", "0932afce82f960c811e1a3bf74ccc4636a5aefd2161f5d69466b0a522955c7d0"),
    48000: ("decimator-48000x16.f64", "753e10eadb9b6550d7dfc81847d7a3057b578c8d3bfe07278d0b474d7e29b676"),
}
FORMATS = {"float32": None, "pcm24": 24, "pcm16": 16}  # the PCM bit depth of each export format
POOL = ThreadPoolExecutor(os.cpu_count())  # upfirdn releases the GIL


def rates(sample_rate: int) -> tuple[int, int, int]:
    """Return the family's base rate, the internal rate F and the decimation factor for an export rate."""
    base = FAMILY[sample_rate]
    return base, base * K, K * base // sample_rate


@functools.cache
def decimator(base: int) -> NDArray[np.float64]:
    """Load the family's frozen taps, refused unless they match their checksum."""
    name, sha256 = DECIMATORS[base]
    raw = resources.files("ballast_synth").joinpath("data", name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        msg = f"{name} does not match its checksum"
        raise RuntimeError(msg)
    return np.frombuffer(raw, dtype="<f8")  # read-only, like the bytes it views


def decimate(taps: NDArray[np.float64], y: NDArray[np.float64], factor: int) -> NDArray[np.float64]:
    """Compute upfirdn(taps, y, down=factor) in parallel chunks that match it bit for bit.

    Each output of upfirdn is one sequential sum over its own inputs. A chunk is
    given every input its outputs need, starting on the factor's grid, so each
    output is computed by exactly the same operations as in a single call.
    """
    n_out = (y.size + taps.size - 2) // factor + 1

    def chunk(m0: int) -> NDArray[np.float64]:
        m1 = min(m0 + CHUNK, n_out)
        start = max(0, m0 * factor - (taps.size - 1)) // factor * factor
        # Since the filter is longer than the factor, the last chunk's slice reaches the end of y.
        out: NDArray[np.float64] = upfirdn(taps, y[start : (m1 - 1) * factor + 1], down=factor)
        return out[m0 - start // factor : m1 - start // factor]

    return np.concatenate(list(POOL.map(chunk, range(0, n_out, CHUNK))))


def low_cut(hz: float, rate: int) -> NDArray[np.float64]:
    """Design a second-order Butterworth high-pass at `hz` by the bilinear transform, as one section for sosfilt.

    The coefficients are computed here in closed form, so they cannot drift with SciPy's filter design.
    """
    k = math.tan(math.pi * hz / rate)
    norm = 1 + math.sqrt(2) * k + k * k
    return np.array(
        [[1 / norm, -2 / norm, 1 / norm, 1.0, 2 * (k * k - 1) / norm, (1 - math.sqrt(2) * k + k * k) / norm]],
    )


def synthesize(p: Patch) -> NDArray[np.float64]:
    """Compute the decimated signal at unity gain: everything before the output stage.

    The low cut filters the internal signal, so every rate of a family still shares it.
    """
    base, rate, factor = rates(p.output.sample_rate)
    y = Voice(p, rate).samples()
    if p.output.low_cut_hz:
        y = sosfilt(low_cut(p.output.low_cut_hz, rate), y)
    return decimate(decimator(base), y, factor)


def output_stage(out: NDArray[np.float64], o: Output, top: float) -> NDArray[np.float32]:
    """Scale by the output gain, or to a sample or true peak target, apply the polarity, and convert once to float32.

    `top` is the format's ceiling: in either peak mode, the highest sample never goes above it.
    """
    peak = float(np.max(np.abs(out), initial=0.0))
    target: float = 10.0 ** (o.db / 20.0)
    if o.mode == "gain":
        scaled = out * target
    elif peak == 0:
        scaled = out.copy()
    elif o.mode == "peak":
        scaled = out * (min(target, top) / peak)
    else:  # true-peak: the estimate is linear, so scaling the signal scales it alike
        scaled = out * min(target / true_peak(out), top / peak)
    if o.polarity < 0:
        scaled = -scaled
    return scaled.astype(np.float32)


def true_peak(audio: NDArray[np.floating[Any]]) -> float:
    """Estimate the peak between samples, by 4× oversampling."""
    x = np.concatenate((audio.astype(float), np.zeros(32)))
    return max(float(np.max(np.abs(resample_poly(x, 4, 1)))), float(np.max(np.abs(audio), initial=0.0)))


def db(x: float) -> float:
    """Return an amplitude in decibels; silence is -inf."""
    return 20 * math.log10(x) if x > 0 else -math.inf


@dataclass(frozen=True)
class Report:
    """What a render made, as `ballast render` reports it."""

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
        """Format the report as `ballast render` prints it."""
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


def render(p: Patch, bits: int | None = None) -> tuple[NDArray[np.float32], Report]:
    """Render the float32 audio and its report, for export as float32 or, with `bits`, as PCM of that depth."""
    start = time.perf_counter()
    top = 1.0 if bits is None else 1.0 - 2.0 ** (2 - bits)  # PCM keeps 2 LSB of headroom, so dither never clips
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


def quantize(audio: NDArray[np.float32], bits: int, seed: int) -> NDArray[np.int32]:
    """Round to `bits`-bit PCM with the seed's TPDF dither of ±1 LSB.

    Exact zeros stay zero, so the silent first sample and any digital silence
    stay silent. A render within the format's ceiling, 2 LSB below full
    scale, still fits after dither.
    """
    x = audio.astype(float) * 2 ** (bits - 1)
    return np.round(np.where(x != 0.0, x + noise.dither(seed, x.size), 0.0)).astype(np.int32)


def export(
    path: str | os.PathLike[str],
    p: Patch,
    text: str,
    audio: NDArray[np.float32],
    bits: int | None = None,
) -> None:
    """Write the WAV, as float32 or dithered PCM, with its root note, the patch text and provenance."""
    base, rate, factor = rates(p.output.sample_rate)
    meta: dict[str, object] = {
        "patch": text,
        "renderer": f"ballast {__version__}",
        "decimator": {"sha256": DECIMATORS[base][1], "internal_rate": rate, "factor": factor},
        "platform": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    samples: NDArray[np.float32] | NDArray[np.int32] = audio
    if bits is not None:
        meta["pcm"] = {
            "bits": bits,
            "dither": "TPDF, ±1 LSB; exact zeros kept",
            "dither_key": noise.DITHER_KEY.decode(),
            "seed": p.seed,
        }
        samples = quantize(audio, bits, p.seed)
    wav.write(path, samples, wav.Format(p.output.sample_rate, bits), wav.root_note(p.body.tune_hz), meta)
