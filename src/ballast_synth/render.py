# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Rendering: the low cut, the clipper, the decimation, the output stage and the exported file.

    out[m] = p · g_out · Σ_{k<L} h[k] z[mK − k],    z[n] = Q(C(y))[n] for 0 ≤ n < N, else 0

Here y[n] is the voice at the internal rate F = 32 × base, where base is 44.1
or 48 kHz, the family's (see family.py): the sources through the drive's
curve, averaged around n / F so that it does not alias (see shaper.py). C is
the optional low cut, Q the clipper. The decimation factor is K = F / rate:
32 at the base rates, 16 at 88.2 and 96 kHz, 8 at 176.4 and 192 kHz. A higher
rate reuses its family's F and filter, so a 96 kHz file is exactly the 48 kHz
file with the in-between samples filled in. Nothing trims or fades, only the
low cut high-passes, and the whole filter tail is kept.

The internal signal is computed in blocks aligned to the trigger and
decimated in parallel chunks. Every sample is a pure function of its time,
the low cut one causal pass, and every output one fixed sum, so neither the
blocks nor the chunks change a value, and a longer render never changes
earlier samples.
"""

from __future__ import annotations

import math
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, override

import numpy as np
import scipy
from scipy.optimize import brentq
from scipy.signal import resample_poly, sosfilt, upfirdn

from . import __version__, grit, noise, room, shaper, wav
from .family import DECIMATORS, decimator, rates
from .patch import EQ, Output, Patch, describe_tuning
from .voice import Voice

if TYPE_CHECKING:
    from numpy.typing import NDArray

CHUNK = 2048  # decimator outputs per parallel chunk
FORMATS = {"float32": None, "pcm24": 24, "pcm16": 16}  # the PCM bit depth of each export format
POOL = ThreadPoolExecutor(os.cpu_count())  # upfirdn releases the GIL


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


def eq(e: EQ, rate: int) -> NDArray[np.float64]:
    """Design the EQ's bands that are on, by the bilinear transform, as sections for sosfilt (none when all are off).

    The coefficients are the cookbook's, computed here in closed form: shelves of slope 1, a bell of bandwidth set
    by its Q, and a resonant low-pass. At 32 times the base rate they match their analog prototypes up to 20 kHz:
    the shelves and the bell within 0.004 dB, the low-pass within 0.012 dB, where its slope shows the warping.
    """
    sections = []
    if e.low_db != 0.0:
        sections.append(_shelf(e.low_hz, e.low_db, rate, high=False))
    if e.bell_db != 0.0:
        a = 10.0 ** (e.bell_db / 40.0)
        w = 2 * math.pi * e.bell_hz / rate
        alpha, c = math.sin(w) / (2 * e.bell_q), math.cos(w)
        sections.append([1 + alpha * a, -2 * c, 1 - alpha * a, 1 + alpha / a, -2 * c, 1 - alpha / a])
    if e.high_db != 0.0:
        sections.append(_shelf(e.high_hz, e.high_db, rate, high=True))
    if e.cutoff_hz != 0.0:
        w = 2 * math.pi * e.cutoff_hz / rate
        alpha, low = math.sin(w) / (2 * e.resonance), 2 * math.sin(w / 2) ** 2  # 1 - cos ω, without cancellation
        sections.append([low / 2, low, low / 2, 1 + alpha, -2 * math.cos(w), 1 - alpha])
    return np.array([[*(x / s[3] for x in s[:3]), 1.0, s[4] / s[3], s[5] / s[3]] for s in sections]).reshape(-1, 6)


def _shelf(hz: float, db: float, rate: int, *, high: bool) -> list[float]:
    """Return a shelf's b0, b1, b2, a0, a1, a2, of slope 1; a high shelf is the low one with cos ω negated."""
    a = 10.0 ** (db / 40.0)
    w = 2 * math.pi * hz / rate
    c = -math.cos(w) if high else math.cos(w)
    r = 2 * math.sqrt(a) * math.sin(w) / math.sqrt(2)
    sign = -1 if high else 1
    return [
        a * ((a + 1) - (a - 1) * c + r),
        sign * 2 * a * ((a - 1) - (a + 1) * c),
        a * ((a + 1) - (a - 1) * c - r),
        (a + 1) + (a - 1) * c + r,
        sign * -2 * ((a - 1) + (a + 1) * c),
        (a + 1) + (a - 1) * c - r,
    ]


class Synthesis(NamedTuple):
    """A kick at unity gain, before the output stage: its signal, its clipper's threshold, and when it began."""

    signal: NDArray[np.float64]
    threshold: float | None
    start: float  # time.perf_counter() as its render began


def synthesize(p: Patch) -> NDArray[np.float64]:
    """Compute the decimated signal at unity gain: everything before the output stage."""
    return synthesized(p).signal


def synthesized(p: Patch, threshold: float | None = None) -> Synthesis:
    """Compute the decimated signal at unity gain, and the clipper's threshold, solved for or as given.

    The low cut and the clipper work on the internal signal, so every rate of a family still shares it. A given
    threshold plays this kick through a clipper set for another, as the hits of one kit share one clipper.
    """
    start = time.perf_counter()
    base, rate, factor = rates(p.output.sample_rate)
    y = Voice(p, rate).samples()
    if p.room is not None and p.room.level_db != -math.inf:
        y = room.add(y, p.room, p.seed, base, rate)
    if p.grit is not None:
        y = grit.apply(y, p.grit, base, rate)
    bands = eq(p.eq, rate)
    if bands.size:
        y = sosfilt(bands, y)
    if p.output.low_cut_hz != 0.0:
        y = sosfilt(low_cut(p.output.low_cut_hz, rate), y)
    if p.clip.db != 0.0:
        if threshold is None:
            threshold = clip_threshold(y, p.clip.db, p.clip.knee_db, base, factor)
        if threshold is not None:
            y = clipped(y, threshold, p.clip.knee_db)
    return Synthesis(decimate(decimator(base), y, factor), threshold, start)


def clipped(y: NDArray[np.float64], threshold: float, knee_db: float) -> NDArray[np.float64]:
    """Clip the internal signal at `threshold`, anti-aliased, with silence beyond either end."""
    out: NDArray[np.float64] = threshold * shaper.apply(shaper.Clip(knee_db), np.pad(y, 2) / threshold)
    return out


def clip(y: NDArray[np.float64], db: float, knee_db: float, base: int, factor: int) -> NDArray[np.float64]:
    """Clip the internal signal so that the file's true peak comes down by exactly db, and nothing below the knee."""
    threshold = clip_threshold(y, db, knee_db, base, factor)
    return y if threshold is None else clipped(y, threshold, knee_db)


def clip_threshold(y: NDArray[np.float64], db: float, knee_db: float, base: int, factor: int) -> float | None:
    """Solve for the threshold that brings the file's true peak down by exactly db; None for silence.

    The threshold follows the kick's own true peak, as the output stage's peak modes do, so the same setting clips
    every kick and every key by the same amount. The decimator's phase rebuilds part of a clipped peak, so the
    threshold is solved for rather than set: each trial clips and decimates only the outputs it can change, and the
    unclipped kick stands for the rest.
    """
    peak = float(np.max(np.abs(y), initial=0.0))
    if peak == 0.0:
        return None
    taps, curve, padded = decimator(base), shaper.Clip(knee_db), np.pad(y, 2)  # silent beyond either end
    plain = decimate(taps, y, factor)
    between = np.abs(oversample(plain))  # the unclipped kick between its samples, 4 to each
    before, after = np.maximum.accumulate(between), np.maximum.accumulate(between[::-1])[::-1]
    reach, edge = taps.size // factor + 2, 32  # outputs one input reaches; outputs the oversampler's edge spoils
    goal = math.log(true_peak(plain)) - db * math.log(10) / 20

    def miss(log_t: float) -> float:
        """Return how far above the goal the true peak lands, in nepers, when the threshold is e^log_t."""
        t = math.exp(log_t)
        hot = np.flatnonzero(np.abs(y) > curve.start * t)
        if hot.size == 0:
            return math.log(float(before[-1])) - goal
        m0 = max(0, (hot[0] - 2) // factor - edge)
        m1 = min(plain.size - 1, (hot[-1] + 2) // factor + reach + edge)
        s0 = max(0, (m0 * factor - taps.size + 1) // factor * factor)
        s1 = min(y.size, m1 * factor + 1)
        part = decimate(taps, t * shaper.apply(curve, padded[s0 : s1 + 4] / t), factor)[m0 - s0 // factor :]
        first = 0 if m0 == 0 else edge  # at either end of the kick, the oversampler's edge is the kick's own silence
        last = m1 - m0 + 1 - (0 if m1 == plain.size - 1 else edge)
        inside = np.abs(oversample(part[: m1 - m0 + 1]))[4 * first : 4 * last if last < m1 - m0 + 1 else None]
        outside = max(
            float(before[4 * (m0 + first) - 1]) if m0 + first > 0 else 0.0,
            float(after[4 * (m0 + last)]) if 4 * (m0 + last) < after.size else 0.0,
        )
        return math.log(max(float(np.max(inside, initial=0.0)), outside)) - goal

    top = math.log(peak / curve.start) + 1e-9  # nothing clips here
    low = math.log(peak / curve.ratio(db))
    while miss(low) > 0:
        low -= math.log(2)
    return math.exp(brentq(miss, low, top, xtol=1e-7))  # to within 1e-6 dB


def output_gain(out: NDArray[np.float64], o: Output, ceiling: float) -> float:
    """Return the output gain: `db` itself, or what brings the sample or true peak to it, within the `ceiling`."""
    peak = float(np.max(np.abs(out), initial=0.0))
    target: float = 10.0 ** (o.db / 20.0)
    if o.mode == "gain":
        return target
    if peak == 0:
        return 1.0
    if o.mode == "peak":
        return min(target, ceiling) / peak
    return min(target / true_peak(out), ceiling / peak)  # true-peak: the estimate is linear, so it scales alike


def output_stage(out: NDArray[np.float64], gain: float, polarity: int) -> NDArray[np.float32]:
    """Scale by the output gain, apply the polarity, and convert once to float32."""
    scaled = out * gain
    if polarity < 0:
        scaled = -scaled
    return scaled.astype(np.float32)


def oversample(audio: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    """Interpolate the audio 4×, to show the peaks between its samples.

    The result runs on past the end of the audio, for as long as the interpolation rings.
    """
    fine: NDArray[np.float64] = resample_poly(np.concatenate((audio.astype(float), np.zeros(32))), 4, 1)
    return fine


def true_peak(audio: NDArray[np.floating[Any]]) -> float:
    """Estimate the peak between samples, by 4× oversampling."""
    return max(float(np.max(np.abs(oversample(audio)))), float(np.max(np.abs(audio), initial=0.0)))


def k_weighting(rate: int) -> NDArray[np.float64]:
    """Design the K-weighting of ITU-R BS.1770 at `rate`, as two sections for sosfilt: a high shelf, then a high-pass.

    BS.1770 gives the coefficients at 48 kHz only. These come from the analog parameters behind them, prewarped at
    each rate, so they match the standard's table at 48 kHz and keep its response at every other rate.
    """
    k = math.tan(math.pi * 1681.974450955533 / rate)
    q = 0.7071752369554196
    high = 10.0 ** (3.999843853973347 / 20.0)
    mid = high**0.4996667741545416
    a0 = 1 + k / q + k * k
    shelf = [(high + mid * k / q + k * k) / a0, 2 * (k * k - high) / a0, (high - mid * k / q + k * k) / a0]
    shelf += [1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]
    k = math.tan(math.pi * 38.13547087602444 / rate)
    q = 0.5003270373238773
    a0 = 1 + k / q + k * k
    return np.array([shelf, [1.0, -2.0, 1.0, 1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]])


def loudness(audio: NDArray[np.floating[Any]], rate: int) -> float:
    """Measure the highest momentary loudness of a kick played once, in LUFS: EBU R128's 400 ms window, every sample.

    The window slides one sample at a time over the K-weighted kick, with silence before and after it, so the
    measure is exact and does not depend on where a meter's 100 ms steps would fall.
    """
    window = round(0.4 * rate)
    y = sosfilt(k_weighting(rate), np.concatenate((audio.astype(float), np.zeros(window))))
    energy = np.concatenate(([0.0], np.cumsum(y * y)))
    mean = np.max(energy[window:] - energy[:-window], initial=0.0) / window
    return -0.691 + 10 * math.log10(mean) if mean > 0 else -math.inf


def db(x: float) -> float:
    """Return an amplitude in decibels; silence is -inf."""
    return 20 * math.log10(x) if x > 0 else -math.inf


def row(label: str, text: str) -> str:
    """Format a line as the report prints it: the label, then the text, in the column where every line's text starts."""
    return f"{label:<9}{text}"


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
    loudness: float  # the highest momentary loudness, LUFS
    ceiling: float
    over: bool  # a peak above the format's ceiling, at a gain set rather than solved for
    gain: float  # the output gain it was scaled by
    threshold: float | None  # the clipper's threshold, when it clips
    render_s: float

    @property
    def over_db(self) -> float:
        """How far the peak is above the format's ceiling."""
        return db(self.peak / self.ceiling)

    @override
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
            ("loudness", "silent" if self.peak == 0 else f"{self.loudness:.1f} LUFS momentary, at its loudest"),
            ("render", f"{1000 * self.render_s:.0f} ms"),
        )
        return "\n".join(row(label, text) for label, text in rows)


def ceiling_of(bits: int | None) -> float:
    """Return the highest sample a format allows: full scale for float32, 2 LSB under it for PCM, for dither."""
    return 1.0 if bits is None else 1.0 - 2.0 ** (2 - bits)


def render(p: Patch, bits: int | None = None) -> tuple[NDArray[np.float32], Report]:
    """Render the float32 audio and its report, for export as float32 or, with `bits`, as PCM of that depth."""
    s = synthesized(p)
    return finish(p, s, bits, output_gain(s.signal, p.output, ceiling_of(bits)), solved=True)


def finish(
    p: Patch,
    s: Synthesis,
    bits: int | None,
    gain: float,
    *,
    solved: bool,
) -> tuple[NDArray[np.float32], Report]:
    """Scale a synthesized kick by `gain`, and report it; `solved` when the gain is its own output stage's.

    A gain set for another kick, as for the hits of a played kit, reports an over in any output mode.
    """
    ceiling = ceiling_of(bits)
    audio = output_stage(s.signal, gain, p.output.polarity)
    peak = float(np.max(np.abs(audio), initial=0.0))
    return audio, Report(
        tune_hz=p.body.tune_hz,
        source_ms=p.length_ms,
        samples=audio.size,
        sample_rate=p.output.sample_rate,
        bits=bits,
        peak=peak,
        true_peak=true_peak(audio),
        loudness=loudness(audio, p.output.sample_rate),
        ceiling=ceiling,
        over=(p.output.mode == "gain" or not solved) and peak > ceiling,
        gain=gain,
        threshold=s.threshold,
        render_s=time.perf_counter() - s.start,
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
