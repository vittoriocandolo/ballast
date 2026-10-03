"""Rendering policy and output stage (plan §2.6, §3, §7).

    out[m] = p · g_out · Σ_{k<L} h[k] y[mK − k],    y[n] = y(n / F) for 0 ≤ n < N, else 0

The internal rate is F = K × base, with base 44.1 or 48 kHz. The 88.2 and
96 kHz exports reuse their family's F and filter with factor K/2, so a 96 kHz
file is exactly the 48 kHz file with the in-between samples filled in.
Nothing trims, fades, high-passes or removes DC, and the whole filter tail
is kept.

Besides the validated-domain status, the report says what the predicted
aliasing comes from and how much drive would change that status, both from
the render's own records, and gives the loudness of the export.
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
from dataclasses import dataclass, field
from importlib import resources

import numpy as np
import scipy
from scipy.signal import resample_poly, upfirdn

from . import __version__, noise, wav
from .loudness import momentary_max
from .patch import Output, Patch, describe_tuning
from .voice import (
    Kinks,
    Slew,
    drive_alpha,
    internal,
    kink_bound_at,
    kink_bounds,
    kink_knots,
    kink_terms,
    noise_table,
    sample_count,
    sources,
    voice,
)

K = 16
FAMILY = {44100: 44100, 88200: 44100, 48000: 48000, 96000: 48000}

# The validated domain (plan §5), frozen by the adversarial grid (docs/VALIDATION.md).
# The predicted aliasing error at unity gain is
#     SATURATION · exp(−π² / q) · P  +  the kink bound
# where q is the knee slew and P the output peak. A patch is inside when that
# is at most LIMIT · P, or FLOOR when the patch is near-silent.
SATURATION = 2.0
LIMIT = 1e-5  # −100 dB of the peak
FLOOR = 1e-7  # absolute bound for near-silent patches

DRIVE_MAX = 48.0  # the valid domain's largest drive

# Export formats: float32, or PCM with seeded TPDF dither (the value is the bit depth).
FORMATS = {"float32": None, "pcm24": 24, "pcm16": 16}


def ceiling(fmt: str) -> float:
    """The largest sample a format takes: full scale for float, and 2 LSB below it for PCM, so dither never clips."""
    bits = FORMATS[fmt]
    return 1.0 if bits is None else 1.0 - 2.0 ** (2 - bits)


@dataclass(frozen=True)
class Domain:
    predicted: float  # predicted aliasing error, at unity gain
    allowed: float
    peak: float

    @property
    def inside(self) -> bool:
        return self.predicted <= self.allowed

    @property
    def margin_db(self) -> float:
        return db(self.allowed / self.predicted) if self.predicted else math.inf

    @property
    def predicted_db(self) -> float:
        """Relative to the peak."""
        return db(self.predicted / self.peak) if self.peak else -math.inf


def domain(slew: Slew, kink: float, peak: float) -> Domain:
    saturation = SATURATION * math.exp(-(math.pi**2) / slew.knee) * peak if slew.knee else 0.0
    return Domain(saturation + kink, max(LIMIT * peak, FLOOR), peak)


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
    return json.loads(resources.files("kick").joinpath("data", "decimators.json").read_text())


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
    raw = resources.files("kick").joinpath("data", entry["file"]).read_bytes()
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


@dataclass(frozen=True)
class Synthesis:
    """Everything before the output stage, at unity gain."""

    out: np.ndarray
    rate: int
    factor: int
    source_samples: int
    slew: Slew
    kink: float  # bound on the error from the curves' kinks, at unity gain
    alpha: float = 0.0
    band: float = 0.0  # the decimator's passband edge, for the kink bound
    kinks: Kinks | None = field(default=None, compare=False, repr=False)
    probes: Probes | None = field(default=None, compare=False, repr=False)


PROBES = 4  # output positions kept for every drive, in each of three places


@dataclass(frozen=True)
class Probes:
    """The exact output at a few chosen positions, at any drive.

    Every internal sample is a pure function of time, so each probe keeps the
    saturator's input over its filter window, and the output there follows
    exactly for any α. The largest of them never exceeds that drive's output
    peak, and at the render's own drive it is the peak. They sit at the
    loudest outputs of the render itself; at the loudest undriven outputs just
    after the crest of x, where a weak drive peaks; and at the loudest outputs
    at full drive just after the steepest zero crossing, where the overshoot
    of a saturated edge peaks.
    """

    x: np.ndarray  # the windows' saturator input, concatenated
    h: np.ndarray  # the taps that weight each sample
    starts: np.ndarray  # where each window starts

    def peak(self, alphas) -> np.ndarray:
        a = np.asarray(alphas, dtype=float)
        out = np.empty(a.size)
        for i in range(0, a.size, 32):  # bounded temporaries
            c = a[i : i + 32, None]
            y = np.where(c > 0, np.tanh(c * self.x) / np.tanh(np.where(c > 0, c, 1.0)), self.x)
            out[i : i + 32] = np.max(np.abs(np.add.reduceat(y * self.h, self.starts, axis=1)), axis=1)
        return out


def _loudest(mags: np.ndarray, first: int, n: int, taken: list[int]) -> list[int]:
    """Up to n indices of the largest nonzero values, apart from each other and from `taken`."""
    picked: list[int] = []
    for i in np.argsort(mags)[::-1][:256]:
        m = first + int(i)
        if mags[i] == 0 or len(picked) == n:
            break
        if all(abs(m - q) > 1 for q in taken + picked):
            picked.append(m)
    return picked


def _probes(v, rate: int, factor: int, taps: np.ndarray, count: int, out: np.ndarray, slew: Slew) -> Probes | None:
    table = noise_table(v, rate)
    picked = _loudest(np.abs(out), 0, PROBES, [])
    # The output just after the crest without drive, and just after the steepest crossing at full drive,
    # computed exactly from x (the decimator's delay at low frequencies is a few outputs).
    a_max = drive_alpha(DRIVE_MAX)
    for n, alpha in ((slew.crest, 0.0), (slew.steep, a_max)):
        m0, m1 = max(0, n // factor - 8), min(out.size, n // factor + 24)
        if m1 <= m0:
            continue
        n0 = max(0, m0 * factor - taps.size + 1) // factor * factor
        n1 = min(count, (m1 - 1) * factor + 1)
        x = sources(v, rate, n0, n1, table)
        y = np.tanh(alpha * x) / np.tanh(alpha) if alpha else x
        near = upfirdn(taps, y, down=factor)[m0 - n0 // factor : m1 - n0 // factor]
        picked += _loudest(np.abs(near), m0, PROBES, picked)
    xs, hs, starts, pos = [], [], [], 0
    for m in picked:
        # out[m] = Σ_n h[mK − n] · y[n] over the internal samples 0 ≤ n < count that the filter reaches.
        top = m * factor
        n0, n1 = max(0, top - taps.size + 1), min(top, count - 1) + 1
        if n1 > n0:
            xs.append(sources(v, rate, n0, n1, table))
            hs.append(taps[top - np.arange(n0, n1)])
            starts.append(pos)
            pos += n1 - n0
    return Probes(np.concatenate(xs), np.concatenate(hs), np.array(starts)) if xs else None


def synthesize(p: Patch, *, k: int = K, taps: np.ndarray | None = None) -> Synthesis:
    v = voice(p)
    rate, factor = rates(p.output.sample_rate, k)
    if taps is None:
        taps = decimator(FAMILY[p.output.sample_rate], k).taps
    n = sample_count(v.t_src, rate)
    y, slew = internal(v, rate, n)
    band = FAMILY[p.output.sample_rate] / 2
    kinks = kink_terms(v)
    kink = kink_bound_at(kinks, v.alpha, rate, band)
    out = decimate(taps, y, factor)
    return Synthesis(
        out, rate, factor, n, slew, kink, v.alpha, band, kinks, _probes(v, rate, factor, taps, n, out, slew)
    )


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
class Cause:
    """Where the predicted aliasing comes from, at unity gain."""

    saturation: float  # the saturation term of the rule
    kinks: float  # the kink bound
    knot: str | None = None  # the knot that contributes most to the kink bound, e.g. "attack.amp[1]"
    knot_ms: float | None = None
    share: float = 0.0  # its share of the kink bound

    @property
    def dominant(self) -> str | None:
        if not (self.saturation or self.kinks):
            return None
        return "saturation" if self.saturation >= self.kinks else "kinks"

    @property
    def where(self) -> str:
        return f"{100 * self.share:.0f}% at {self.knot} ({self.knot_ms:g} ms)"


def cause_of(s: Synthesis, d: Domain) -> Cause:
    saturation = d.predicted - s.kink
    knots = kink_knots(s.kinks, s.alpha, s.rate, s.band) if s.kinks is not None else []
    total = math.fsum(row[3] for row in knots)
    if not total:
        return Cause(saturation, s.kink)
    curve, i, t, c = knots[0]
    return Cause(saturation, s.kink, f"{curve}[{i}]", round(1000 * t, 6), c / total)


HEADROOM_STEP_DB = 0.5  # the scan's grid; every jump of the knee slew is a candidate as well


@dataclass(frozen=True)
class Headroom:
    """How much the drive can change before the validated-domain status does, estimated from one render.

    `db` is positive inside and negative outside; +inf when the patch stays
    inside at every drive up to 48 dB, −inf when it is outside even without
    drive. `edge_db` is the drive at the domain's edge.
    """

    db: float
    edge_db: float | None = None


def _alphas(drives) -> np.ndarray:
    """α with α / tanh α = 10^(drive/20), for many drives at once: bisection on [G − 1, G]."""
    g = 10.0 ** (np.asarray(drives, dtype=float) / 20.0)
    lo, hi = np.maximum(g - 1.0, 0.0), g.copy()
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        below = mid / np.tanh(np.where(mid > 0, mid, 1.0)) < g
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return np.where(g > 1.0, 0.5 * (lo + hi), 0.0)


def _drives(alphas) -> np.ndarray:
    a = np.asarray(alphas, dtype=float)
    return np.where(a > 0, 20.0 * np.log10(a / np.tanh(np.where(a > 0, a, 1.0))), 0.0)


def _ratios(s: Synthesis, alphas) -> np.ndarray:
    """The rule's prediction over its limit at many α: above 1 is outside.

    The knee slew and the kink bound are exact at every α. The output peak is
    bounded from below by the probes, which can only raise the ratio, so the
    headroom errs on the safe side in both directions.
    """
    a = np.asarray(alphas, dtype=float)
    p = s.probes.peak(a) if s.probes is not None else np.zeros(a.size)
    q = s.slew.knees.at(a)
    saturation = np.where(q > 0, SATURATION * np.exp(-(math.pi**2) / np.where(q > 0, q, 1.0)) * p, 0.0)
    return (saturation + kink_bounds(s.kinks, a, s.rate, s.band)) / np.maximum(LIMIT * p, FLOOR)


def _edge(s: Synthesis, inside: float, outside: float) -> float:
    """Refine an inside/outside bracket in α, 16 points at a time; returns the inside end.

    Four rounds narrow a 0.5 dB bracket below 10⁻⁵ dB.
    """
    for _ in range(4):
        a = np.linspace(inside, outside, 17)[1:-1]
        out = np.flatnonzero(_ratios(s, a) > 1.0)  # a runs from the inside end to the outside end
        if out.size:
            j = out[0]
            inside, outside = (a[j - 1] if j else inside), a[j]
        else:
            inside = a[-1]
    return float(inside)


def drive_headroom(s: Synthesis, drive_db: float, d: Domain) -> Headroom | None:
    """The drive at which the validated-domain status first changes, searched from the patch's own drive.

    The rule is not monotone in drive: a step near the knee counts only while
    α·reach ≤ 1, so just below the α where it stops counting the domain can
    have a narrow outside island. Candidates are every 0.5 dB of drive and
    both sides of every α at which the knee slew drops, so such an island is
    never skipped; the first change is then refined.
    """
    knees = s.slew.knees
    if knees is None or s.kinks is None or d.peak == 0:
        return None
    a_max = drive_alpha(DRIVE_MAX)
    drops = knees.drops()
    if d.inside:
        grid = np.append(np.arange(drive_db + HEADROOM_STEP_DB, DRIVE_MAX, HEADROOM_STEP_DB), DRIVE_MAX)
        below = drops * (1.0 - 1e-12)  # just before each drop, where the step still counts
        cand = np.concatenate((_alphas(grid), below[(below > s.alpha) & (below <= a_max)]))
        cand = np.sort(cand[cand > s.alpha])
        out = np.flatnonzero(_ratios(s, cand) > 1.0)
        if not out.size:
            return Headroom(math.inf)
        j = out[0]
        edge = _edge(s, cand[j - 1] if j else s.alpha, cand[j])
    else:
        grid = np.append(np.arange(drive_db - HEADROOM_STEP_DB, 0.0, -HEADROOM_STEP_DB), 0.0)
        above = drops * (1.0 + 1e-12)  # just past each drop, where the step no longer counts
        cand = np.concatenate((_alphas(grid), above[above < s.alpha]))
        cand = np.sort(cand[cand < s.alpha])[::-1]
        inside = np.flatnonzero(_ratios(s, cand) <= 1.0)
        if not inside.size:
            return Headroom(-math.inf)
        j = inside[0]
        edge = _edge(s, cand[j], cand[j - 1] if j else s.alpha)
    edge_db = float(_drives(edge))
    return Headroom(edge_db - drive_db, edge_db)


def _floor1(x: float) -> float:
    """Round down to 0.1, so a suggested drive errs on the inside."""
    return math.floor(x * 10 + 1e-9) / 10


@dataclass(frozen=True)
class Report:
    tune_hz: float
    source_ms: float
    samples: int
    sample_rate: int
    peak: float
    true_peak: float
    over: bool
    domain: Domain
    render_s: float
    internal_rate: int
    factor: int
    decimator: str
    loudness: float = -math.inf  # the largest 400 ms momentary loudness, LUFS
    cause: Cause | None = None
    headroom: Headroom | None = None
    drive_db: float = 0.0
    format: str = "float32"
    seed: int = 0
    ceiling: float = 1.0

    @property
    def exported_ms(self) -> float:
        return 1000 * self.samples / self.sample_rate

    @property
    def inside(self) -> bool:
        return self.domain.inside

    @property
    def margin_db(self) -> float:
        return self.domain.margin_db

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

    def _detail(self) -> str | None:
        """What the prediction comes from, and what drive would change the status."""
        c, h, d = self.cause, self.headroom, self.domain
        if c is None or c.dominant is None or not d.predicted:
            return None
        text = "mostly saturation" if c.dominant == "saturation" else f"mostly kinks, {c.where}"
        if h is None:
            return text
        if h.db == math.inf:
            return f"{text}; inside at every drive up to {DRIVE_MAX:g} dB"
        if h.db == -math.inf:
            knot = "" if c.dominant == "kinks" or c.knot is None else f", most at {c.knot} ({c.knot_ms:g} ms)"
            return f"{text}; outside even without drive: the kinks alone exceed the limit{knot}"
        if h.db >= 0:
            return f"{text}; drive headroom {_floor1(h.db):+.1f} dB"
        to = _floor1(h.edge_db)
        return f"{text}; drive headroom {to - self.drive_db:+.1f} dB: lower drive.db to {to:.1f} dB"

    def lines(self) -> list[str]:
        if self.peak == 0:
            peak = "silent"
        elif self.over:
            peak = self._over()
        else:
            peak = (
                f"{db(self.peak):.2f} dBFS sample, {db(self.true_peak):.2f} dBTP true (4× estimate, information only)"
            )
        loudness = (
            f"{self.loudness:.1f} LUFS, the largest 400 ms momentary loudness (BS.1770)"
            if self.loudness > -math.inf
            else "silent"
        )
        d = self.domain
        if not d.predicted:
            domain = "inside the validated domain (no predicted aliasing)"
        elif self.inside:
            domain = (
                f"inside the validated domain: predicted aliasing {d.predicted_db:.0f} dB, margin {d.margin_db:.1f} dB"
            )
        else:
            domain = (
                f"OUTSIDE the validated domain by {-d.margin_db:.1f} dB (predicted aliasing {d.predicted_db:.0f} dB); "
                "check it with `kick verify`"
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
            ("loudness", loudness),
            ("domain", domain),
        ]
        detail = self._detail()
        if detail:
            rows.append(("", detail))
        rows.append(("render", f"{1000 * self.render_s:.0f} ms"))
        return [f"{label:<9}{text}" for label, text in rows]

    def __str__(self) -> str:
        return "\n".join(self.lines())


def render(p: Patch, fmt: str = "float32") -> tuple[np.ndarray, Report]:
    """The float32 audio that `kick render` writes (before any PCM quantization), and its report."""
    start = time.perf_counter()
    top = ceiling(fmt)
    s = synthesize(p)
    audio = output_stage(s.out, p.output, top)
    if not np.all(np.isfinite(audio)):
        raise RenderError("the render produced non-finite samples")
    peak = float(np.max(np.abs(audio), initial=0.0))
    tp = true_peak(audio)
    d = domain(s.slew, s.kink, float(np.max(np.abs(s.out), initial=0.0)))
    cause, headroom = cause_of(s, d), drive_headroom(s, p.drive.db, d)
    loudness = momentary_max(audio, p.output.sample_rate)
    return audio, Report(
        tune_hz=p.body.tune_hz,
        source_ms=p.length_ms,
        samples=audio.size,
        sample_rate=p.output.sample_rate,
        peak=peak,
        true_peak=tp,
        over=p.output.mode == "gain" and peak > top,
        domain=d,
        render_s=time.perf_counter() - start,
        internal_rate=s.rate,
        factor=s.factor,
        decimator=decimator(FAMILY[p.output.sample_rate]).sha256,
        loudness=loudness,
        cause=cause,
        headroom=headroom,
        drive_db=p.drive.db,
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
