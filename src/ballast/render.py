"""Rendering policy and output stage (plan §2.6, §3, §7).

    out[m] = p · g_out · Σ_{k<L} h[k] y[mK − k],    y[n] = y(n / F) for 0 ≤ n < N, else 0

The internal rate is F = K × base, with base 44.1 or 48 kHz. The 88.2 and
96 kHz exports reuse their family's F and filter with factor K/2, so a 96 kHz
file is exactly the 48 kHz file with the in-between samples filled in.
Nothing trims, fades, high-passes or removes DC, and the whole filter tail
is kept.
"""

from __future__ import annotations

import functools
import hashlib
import json
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
from .voice import internal, sample_count, voice

K = 16
FAMILY = {44100: 44100, 88200: 44100, 48000: 48000, 96000: 48000}

# Export formats: float32, or PCM with seeded TPDF dither (the value is the bit depth).
FORMATS = {"float32": None, "pcm24": 24, "pcm16": 16}


def ceiling(fmt: str) -> float:
    """The largest sample a format takes: full scale for float, and 2 LSB below it for PCM, so dither never clips."""
    bits = FORMATS[fmt]
    return 1.0 if bits is None else 1.0 - 2.0 ** (2 - bits)


class RenderError(RuntimeError):
    pass


@dataclass(frozen=True)
class Decimator:
    base: int
    k: int
    taps: np.ndarray
    sha256: str
    stopband_db: float


def manifest() -> dict:
    return json.loads(resources.files(__package__).joinpath("data", "decimators.json").read_text())


def load_taps(raw: bytes, entry: dict) -> np.ndarray:
    """Coefficients from their shipped bytes, refused unless they match the checksum."""
    if hashlib.sha256(raw).hexdigest() != entry["sha256"] or len(raw) != 8 * entry["taps"]:
        raise RenderError(f"decimator {entry['base']}x{entry['K']} does not match its checksum")
    taps = np.frombuffer(raw, dtype="<f8").astype(float)
    taps.flags.writeable = False
    return taps


@functools.cache
def decimator(base: int, k: int = K) -> Decimator:
    """A frozen decimator from kick/data, verified against its checksum."""
    entry = manifest()["filters"][f"{base}x{k}"]
    raw = resources.files(__package__).joinpath("data", entry["file"]).read_bytes()
    return Decimator(base, k, load_taps(raw, entry), entry["sha256"], entry["stopband_db"])


def decimate(taps: np.ndarray, y: np.ndarray, factor: int, workers: int | None = None) -> np.ndarray:
    """upfirdn(taps, y, down=factor), computed in parallel chunks that match it bit for bit.

    Each output of upfirdn is one sequential sum over its own inputs. A chunk is
    given every input its outputs need, starting on the factor's grid, so each
    output is computed by exactly the same operations as in a single call.
    """
    L = taps.size
    n_out = (y.size + L - 2) // factor + 1
    workers = min(workers or os.cpu_count() or 1, max(1, n_out // 4096))
    if workers == 1:
        return upfirdn(taps, y, down=factor)
    edges = [round(i * n_out / workers) for i in range(workers + 1)]

    def chunk(m0: int, m1: int) -> np.ndarray:
        start = max(0, (m0 * factor - (L - 1)) // factor * factor)
        stop = y.size if m1 == n_out else (m1 - 1) * factor + 1
        out = upfirdn(taps, y[start:stop], down=factor)
        return out[m0 - start // factor : m1 - start // factor]

    with ThreadPoolExecutor(workers) as pool:
        return np.concatenate(list(pool.map(chunk, edges[:-1], edges[1:])))


def rates(sample_rate: int, k: int = K) -> tuple[int, int]:
    """(internal rate F, decimation factor) for an export rate."""
    base = FAMILY[sample_rate]
    return base * k, k * base // sample_rate


def synthesize(p: Patch) -> np.ndarray:
    """Everything before the output stage, at unity gain."""
    v = voice(p)
    rate, factor = rates(p.output.sample_rate)
    y = internal(v, rate, sample_count(v.t_src, rate))
    return decimate(decimator(FAMILY[p.output.sample_rate]).taps, y, factor)


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


def true_peak(audio: np.ndarray) -> float:
    """4× oversampled peak estimate (information only)."""
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
    peak: float
    true_peak: float
    over: bool
    render_s: float
    internal_rate: int
    factor: int
    decimator: str
    format: str = "float32"
    seed: int = 0
    ceiling: float = 1.0

    @property
    def exported_ms(self) -> float:
        return 1000 * self.samples / self.sample_rate

    def _over(self) -> str:
        if FORMATS[self.format] is None:
            return (
                f"{db(self.peak):+.2f} dBFS sample: over full scale; lower output.db by at least "
                f"{db(self.peak):.2f} dB, or pass --allow-over"
            )
        return (
            f"{db(self.peak):+.4f} dBFS sample: over the {FORMATS[self.format]}-bit ceiling of "
            f"{db(self.ceiling):.4f} dBFS, which keeps 2 LSB for dither; lower output.db by at least "
            f"{db(self.peak / self.ceiling):.4f} dB"
        )

    def lines(self) -> list[str]:
        if self.peak == 0:
            peak = "silent"
        elif self.over:
            peak = self._over()
        else:
            peak = (
                f"{db(self.peak):.2f} dBFS sample, {db(self.true_peak):.2f} dBTP true (4× estimate, information only)"
            )
        pcm = "" if FORMATS[self.format] is None else f", {FORMATS[self.format]}-bit PCM"
        rows = [
            ("tuning", f"{self.tune_hz:.3f} Hz, {describe_tuning(self.tune_hz)}"),
            (
                "length",
                f"{self.source_ms:.1f} ms source, {self.exported_ms:.1f} ms exported "
                f"({self.samples} samples at {self.sample_rate} Hz{pcm})",
            ),
            ("peak", peak),
            ("render", f"{1000 * self.render_s:.0f} ms"),
        ]
        return [f"{label:<9}{text}" for label, text in rows]

    def __str__(self) -> str:
        return "\n".join(self.lines())


def render(p: Patch, fmt: str = "float32") -> tuple[np.ndarray, Report]:
    """The float32 audio that `kick render` writes (before any PCM quantization), and its report."""
    start = time.perf_counter()
    top = ceiling(fmt)
    audio = output_stage(synthesize(p), p.output, top)
    if not np.all(np.isfinite(audio)):
        raise RenderError("the render produced non-finite samples")
    peak = float(np.max(np.abs(audio), initial=0.0))
    rate, factor = rates(p.output.sample_rate)
    return audio, Report(
        tune_hz=p.body.tune_hz,
        source_ms=p.length_ms,
        samples=audio.size,
        sample_rate=p.output.sample_rate,
        peak=peak,
        true_peak=true_peak(audio),
        over=p.output.mode == "gain" and peak > top,
        render_s=time.perf_counter() - start,
        internal_rate=rate,
        factor=factor,
        decimator=decimator(FAMILY[p.output.sample_rate]).sha256,
        format=fmt,
        seed=p.seed,
        ceiling=top,
    )


def quantize(audio: np.ndarray, bits: int, seed: int) -> np.ndarray:
    """Round to `bits`-bit PCM with the seed's TPDF dither of ±1 LSB.

    Exact zeros stay zero, so the silent first sample and any digital silence
    stay silent. The output stage keeps the peak 2 LSB below full scale, so
    the dithered value always fits.
    """
    top = 2 ** (bits - 1)
    x = audio.astype(float) * top
    q = np.round(np.where(x != 0.0, x + noise.dither(seed, x.size), 0.0))
    if q.size and (q.max() > top - 1 or q.min() < -top):
        raise RenderError(f"the render does not fit {bits}-bit PCM")
    return q.astype(np.int32)


def provenance(report: Report) -> dict:
    meta = {
        "renderer": f"kick {__version__}",
        "decimator": {"sha256": report.decimator, "internal_rate": report.internal_rate, "factor": report.factor},
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    if FORMATS[report.format] is not None:
        meta["pcm"] = {
            "bits": FORMATS[report.format],
            "dither": "TPDF, ±1 LSB; exact zeros kept",
            "dither_key": noise.DITHER_KEY.decode(),
            "seed": report.seed,
        }
    return meta


def export(path, audio: np.ndarray, report: Report, patch_text: str) -> None:
    """Write the WAV in the report's format, with its root note, the patch text and provenance."""
    bits = FORMATS[report.format]
    samples = audio if bits is None else quantize(audio, bits, report.seed)
    wav.write(
        path,
        samples,
        report.sample_rate,
        bits=bits,
        root=wav.root_note(report.tune_hz),
        meta={"patch": patch_text, **provenance(report)},
    )
