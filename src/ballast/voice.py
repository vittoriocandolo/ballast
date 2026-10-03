"""Body, attack and drive as pure functions of time (plan §2.3–§2.5, §3).

    x(t) = b(t) + a(t),    y(t) = S_α(x(t))

Nothing here keeps state: sample n at rate F is computed from t_n = n / F
alone, so a block boundary, a longer render or a different rate never
changes a sample's value.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from . import noise
from .curve import Curve, amplitude_curve, pitch_curve
from .patch import Patch

BLOCK = 2**16


def drive_alpha(db: float) -> float:
    """α with α / tanh α = 10^(db/20): the saturator's small-signal gain. 0 dB is bypass."""
    if db == 0.0:
        return 0.0
    g = 10.0 ** (db / 20.0)
    # x / tanh x lies between x and x + 1, so [g - 1, g] brackets the root.
    return brentq(lambda a: a / math.tanh(a) - g, g - 1.0, g, xtol=1e-300, rtol=4 * np.finfo(float).eps, maxiter=500)


def mix_weights(m: float) -> tuple[float, float]:
    """Constant-power tone and noise weights, exact at both ends."""
    if m == 0.0:
        return 1.0, 0.0
    if m == 1.0:
        return 0.0, 1.0
    return math.cos(math.pi * m / 2), math.sin(math.pi * m / 2)


@dataclass(frozen=True)
class Voice:
    """The patch reduced to what synthesis needs; times in seconds."""

    f0: float
    body_gain: float  # g_b / √(1 + h2²)
    h2: float
    phi0: float
    tail: bool
    pitch: Curve
    body_amp: Curve
    attack_gain: float
    tone_hz: float
    tone_weight: float
    noise_weight: float
    band: tuple[float, float]
    seed: int
    attack_amp: Curve
    alpha: float
    t_src: float

    @property
    def uses_noise(self) -> bool:
        return self.attack_gain != 0.0 and self.noise_weight != 0.0


def voice(p: Patch) -> Voice:
    b, a = p.body, p.attack
    body_amp = amplitude_curve([m / 1000 for m, _ in b.amp], [g for _, g in b.amp])
    attack_amp = amplitude_curve([m / 1000 for m, _ in a.amp], [g for _, g in a.amp])
    tone, noise_w = mix_weights(a.noise_mix)
    return Voice(
        f0=b.tune_hz,
        body_gain=10.0 ** (b.level_db / 20.0) / math.sqrt(1.0 + b.h2 * b.h2),
        h2=b.h2,
        phi0=math.radians(b.phase_deg),
        tail=b.phase_ref == "tail",
        pitch=pitch_curve([m / 1000 for m, _ in b.pitch], [r for _, r in b.pitch]),
        body_amp=body_amp,
        attack_gain=10.0 ** (a.level_db / 20.0),
        tone_hz=a.tone_hz,
        tone_weight=tone,
        noise_weight=noise_w,
        band=a.noise_band_hz,
        seed=p.seed,
        attack_amp=attack_amp,
        alpha=drive_alpha(p.drive.db),
        t_src=max(body_amp.t[-1], attack_amp.t[-1]),
    )


def sample_count(t_src: float, rate: int) -> int:
    """N = ⌊T_src·F⌋ + 1: every internal sample with n / F ≤ T_src, decided in float64."""
    n = math.floor(t_src * rate)
    while (n + 1) / rate <= t_src:
        n += 1
    while n / rate > t_src:
        n -= 1
    return n + 1


def phase_time(v: Voice, t: np.ndarray) -> np.ndarray:
    """τ(t) = R(t) − R(t_ref) + c, so that φ = φ0 + 2π f0 τ."""
    if not v.tail:
        return v.pitch.integral(t)
    # Tail mode: τ = t exactly from the last pitch knot on, whatever the sweep does before it.
    t_n = v.pitch.t[-1]
    return np.where(t >= t_n, t, t_n - v.pitch.remaining(t))


def body(v: Voice, t: np.ndarray) -> np.ndarray:
    phi = v.phi0 + (2 * math.pi * v.f0) * phase_time(v, t)
    wave = np.sin(phi)
    if v.h2:
        # sin φ + h2 sin 2φ = sin φ (1 + 2 h2 cos φ): no extra zero crossings while h2 ≤ ½.
        wave *= 1.0 + 2.0 * v.h2 * np.cos(phi)
    return v.body_gain * v.body_amp(t) * wave


def attack(v: Voice, t: np.ndarray, noise_samples: np.ndarray | None) -> np.ndarray:
    carrier = np.zeros_like(t)
    if v.tone_weight:
        carrier += v.tone_weight * np.sin((2 * math.pi * v.tone_hz) * t)
    if v.noise_weight:
        carrier += v.noise_weight * noise_samples
    return v.attack_gain * v.attack_amp(t) * carrier


def _support(t: np.ndarray, curve: Curve) -> slice:
    return slice(np.searchsorted(t, curve.t[0], "left"), np.searchsorted(t, curve.t[-1], "right"))


def sources(v: Voice, rate: int, n0: int, n1: int, table: np.ndarray | None) -> np.ndarray:
    """x[n] = b(n / rate) + a(n / rate) for n0 ≤ n < n1; each source only over its support."""
    t = np.arange(n0, n1) / rate
    x = np.zeros(n1 - n0)
    if v.body_gain:
        s = _support(t, v.body_amp)
        x[s] += body(v, t[s])
    if v.attack_gain:
        s = _support(t, v.attack_amp)
        n = np.arange(n0 + s.start, n0 + s.stop) % rate if v.noise_weight else None
        x[s] += attack(v, t[s], table[n] if n is not None else None)
    return x


def saturate(v: Voice, x: np.ndarray) -> np.ndarray:
    """S_α(x) = tanh(αx) / tanh α, odd by construction; the identity when α = 0."""
    if not v.alpha:
        return x
    norm = np.tanh(np.array([v.alpha]))[0]  # the same tanh as below, so S(1) = 1 exactly
    with np.errstate(over="ignore"):
        return np.copysign(np.tanh(v.alpha * np.abs(x)), x) / norm


def slope(v: Voice, x: np.ndarray) -> np.ndarray:
    """S'_α(x), the saturator's local gain."""
    return _slope(v.alpha, x)


def _slope(alpha: float, x: np.ndarray) -> np.ndarray:
    if not alpha:
        return np.ones_like(x)
    return alpha * (1.0 - np.tanh(alpha * x) ** 2) / np.tanh(alpha)


ZETA3, ZETA4 = 1.2020569031595942, math.pi**4 / 90


def _state(v: Voice, t: np.ndarray) -> dict:
    """The saturator input and each source's carrier, with first derivatives, at times t."""
    phi = v.phi0 + (2 * math.pi * v.f0) * phase_time(v, t)
    s = {
        "dphi": 2 * math.pi * v.f0 * v.pitch(t),
        "w": np.sin(phi) * (1.0 + 2.0 * v.h2 * np.cos(phi)),  # sin φ + h2 sin 2φ
        "dw": np.cos(phi) + 2.0 * v.h2 * np.cos(2 * phi),
        "eb": v.body_amp(t),
        "c": v.tone_weight * np.sin((2 * math.pi * v.tone_hz) * t),
        "dc": v.tone_weight * (2 * math.pi * v.tone_hz) * np.cos((2 * math.pi * v.tone_hz) * t),
    }
    ea = v.attack_amp(t)
    live = (t >= v.attack_amp.t[0]) & (t <= v.attack_amp.t[-1])  # the noise only matters where the attack sounds
    if v.uses_noise and live.any():
        for key, derivative in (("c", False), ("dc", True)):
            n = np.zeros_like(t)
            n[live] = noise.at(v.seed, v.band[0], v.band[1], t[live], derivative=derivative)
            s[key] = s[key] + v.noise_weight * n
    s["x"] = v.body_gain * s["eb"] * s["w"] + v.attack_gain * ea * s["c"]
    s["dx"] = v.body_gain * (v.body_amp.derivative(t) * s["w"] + s["eb"] * s["dw"] * s["dphi"]) + v.attack_gain * (
        v.attack_amp.derivative(t) * s["c"] + ea * s["dc"]
    )
    return s


@dataclass(frozen=True)
class Kinks:
    """The jumps of x″ and x‴ at every knot, with x and x′ there: all the kink bound needs but α.

    Drive acts only through the saturator, so these are computed once per
    render and the bound follows at any drive for the cost of a few knots.
    """

    groups: tuple  # ((curve, knot times in s, x, x′, Δx″, Δx‴), ...), in the bound's summation order


def kink_terms(v: Voice) -> Kinks:
    groups = []
    if v.body_gain:
        s = _state(v, v.body_amp.t)
        j2, j3 = v.body_amp.jumps()
        groups.append(
            (
                "body.amp",
                v.body_amp.t,
                s["x"],
                s["dx"],
                v.body_gain * j2 * s["w"],
                v.body_gain * (j3 * s["w"] + 3 * j2 * s["dw"] * s["dphi"]),
            )
        )
        s = _state(v, v.pitch.t)
        j2, _ = v.pitch.jumps()  # φ‴ = 2π f0 r″ jumps where r″ does
        groups.append(
            ("body.pitch", v.pitch.t, s["x"], s["dx"], 0.0, v.body_gain * s["eb"] * s["dw"] * 2 * math.pi * v.f0 * j2)
        )
    if v.attack_gain:
        s = _state(v, v.attack_amp.t)
        j2, j3 = v.attack_amp.jumps()
        groups.append(
            (
                "attack.amp",
                v.attack_amp.t,
                s["x"],
                s["dx"],
                v.attack_gain * j2 * s["c"],
                v.attack_gain * (j3 * s["c"] + 3 * j2 * s["dc"]),
            )
        )
    return Kinks(tuple(groups))


def _kink_weights(rate: int, band: float) -> tuple[float, float]:
    return 4 * ZETA3 * band / (2 * math.pi * rate) ** 3, 4 * ZETA4 * band / (2 * math.pi * rate) ** 4


def kink_bound(v: Voice, rate: int, band: float) -> float:
    """Upper bound on the output error from the curves' kinks, at unity gain.

    The shapes are C¹: at each amplitude knot the second and third derivatives
    of y jump, and at each pitch knot the third. A jump J in the n-th derivative
    gives a spectrum falling as J / (2πf)^(n+1); sampling at F folds the parts
    near every multiple of F into the band the decimator passes (below `band`),
    so the error is at most 4 ζ(n+1) · band · |J| / (2πF)^(n+1). Higher orders
    are smaller by about the signal's bandwidth over F.
    """
    return kink_bound_at(kink_terms(v), v.alpha, rate, band)


def kink_bound_at(k: Kinks, alpha: float, rate: int, band: float) -> float:
    """The kink bound at one drive's α."""
    k2, k3 = _kink_weights(rate, band)
    total = 0.0
    for _, _, x, dx, d2, d3 in k.groups:
        # y = S(x): y″ jumps by S′·Δx″, and y‴ by S′·Δx‴ + 3·S″·x′·Δx″.
        g1 = _slope(alpha, x)
        g2 = -2.0 * alpha * np.tanh(alpha * x) * g1 if alpha else 0.0
        total += k2 * np.sum(np.abs(g1 * d2)) + k3 * np.sum(np.abs(g1 * d3 + 3 * g2 * dx * d2))
    return float(total)


def kink_bounds(k: Kinks, alphas: np.ndarray, rate: int, band: float) -> np.ndarray:
    """The kink bound at many drives at once (agrees with kink_bound_at to rounding)."""
    k2, k3 = _kink_weights(rate, band)
    a = np.asarray(alphas, dtype=float)[:, None]
    driven = a > 0
    total = np.zeros(a.shape[0])
    for _, _, x, dx, d2, d3 in k.groups:
        th = np.tanh(a * x)
        g1 = np.where(driven, a * (1.0 - th**2) / np.tanh(np.where(driven, a, 1.0)), 1.0)
        g2 = np.where(driven, -2.0 * a * th * g1, 0.0)
        total += k2 * np.sum(np.abs(g1 * d2), axis=1) + k3 * np.sum(np.abs(g1 * d3 + 3 * g2 * dx * d2), axis=1)
    return total


def kink_knots(k: Kinks, alpha: float, rate: int, band: float) -> list[tuple[str, int, float, float]]:
    """Each knot's share of the kink bound: (curve, index, time in s, contribution), largest first."""
    k2, k3 = _kink_weights(rate, band)
    out = []
    for curve, t, x, dx, d2, d3 in k.groups:
        g1 = _slope(alpha, x)
        g2 = -2.0 * alpha * np.tanh(alpha * x) * g1 if alpha else 0.0
        c = k2 * np.abs(g1 * d2) + k3 * np.abs(g1 * d3 + 3 * g2 * dx * d2)
        out += [(curve, i, float(t[i]), float(c[i])) for i in range(len(t))]
    return sorted(out, key=lambda row: -row[3])


def noise_table(v: Voice, rate: int) -> np.ndarray | None:
    """The attack noise at every internal sample of one period, cached while the seed and band stay the same."""
    return _table(v.seed, v.band[0], v.band[1], rate) if v.uses_noise else None


@functools.lru_cache(maxsize=2)
def _table(seed: int, lo: float, hi: float, rate: int) -> np.ndarray:
    table = noise.table(seed, lo, hi, rate)
    table.flags.writeable = False
    return table


@dataclass(frozen=True)
class Knees:
    """The knee slew at every drive, recorded in one pass.

    Drive leaves the saturator's input x unchanged, so for any α

        q(α) = α · M(1/α),   M(m) = the largest step of x that crosses zero or has an end within |x| ≤ m.

    M is kept as its upper frontier: `crossing`, the largest step that crosses
    zero, which qualifies at every α, and the steps that beat it, each with its
    `reach`, the smaller |x| at its two ends. As α grows a step stops qualifying
    once α·reach > 1, and never starts again, so q grows at most in proportion
    to α and drops where α passes 1/reach.
    """

    crossing: float
    reach: np.ndarray  # ascending
    step: np.ndarray  # ascending: the largest step with at most that reach

    def largest(self, m) -> np.ndarray:
        """M(m) for m ≥ 0."""
        i = np.searchsorted(self.reach, np.asarray(m, dtype=float), side="right")
        best = self.step[np.maximum(i - 1, 0)] if self.step.size else np.zeros(np.shape(i))
        return np.maximum(self.crossing, np.where(i > 0, best, 0.0))

    def at(self, alpha) -> np.ndarray:
        """q(α); 0 without drive."""
        a = np.asarray(alpha, dtype=float)
        driven = a > 0
        return np.where(driven, a * self.largest(np.where(driven, 1.0 / np.where(driven, a, 1.0), 0.0)), 0.0)

    def drops(self) -> np.ndarray:
        """The α at which q drops: the last α at which each frontier step still qualifies."""
        return 1.0 / self.reach[self.reach > 0]


def _frontier(reach: np.ndarray, step: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The points no other beats with a smaller or equal reach and a larger or equal step."""
    if not reach.size:
        return reach, step
    order = np.lexsort((-step, reach))
    r, s = reach[order], step[order]
    keep = np.empty(s.size, dtype=bool)
    keep[0], keep[1:] = True, s[1:] > np.maximum.accumulate(s)[:-1]
    return r[keep], s[keep]


@dataclass(frozen=True)
class Slew:
    """Steps of the saturator's argument u = αx between consecutive internal samples."""

    peak: float  # the largest step anywhere
    knee: float  # the largest step that crosses zero or touches |u| ≤ 1, where tanh bends
    knees: Knees | None = field(default=None, compare=False, repr=False)  # the knee slew at every other drive
    crest: int = field(default=0, compare=False, repr=False)  # the internal sample with the largest |x|
    steep: int = field(default=0, compare=False, repr=False)  # the end of the largest step that crosses zero


def internal(v: Voice, rate: int, count: int) -> tuple[np.ndarray, Slew]:
    """y[n] = y(n / rate) for 0 ≤ n < count, in blocks aligned to n = 0, and its slew."""
    table = noise_table(v, rate)
    y = np.empty(count)
    prev, peak, knee = 0.0, 0.0, 0.0  # silence before the trigger
    unit = v.alpha or 1.0  # steps are taken on u = unit · x; without drive, on x itself
    crossing, reach, steps, crest, crest_x, steep = 0.0, [], [], 0, 0.0, 0
    for n0 in range(0, count, BLOCK):
        n1 = min(n0 + BLOCK, count)
        x = sources(v, rate, n0, n1, table)
        y[n0:n1] = saturate(v, x)
        if not x.size:
            continue
        u = unit * np.concatenate(([prev], x))
        step = np.abs(np.diff(u))
        a, b = u[:-1], u[1:]
        near = np.minimum(np.abs(a), np.abs(b))
        crosses = np.signbit(a) != np.signbit(b)
        if v.alpha:
            bends = (near <= 1.0) | crosses
            peak = max(peak, float(step.max()))
            knee = max(knee, float(np.max(step, where=bends, initial=0.0)))
        # For the knee slew at every other drive: the largest crossing step, and the steps that beat it.
        c = float(np.max(step, where=crosses, initial=0.0))
        if c > crossing:
            crossing, steep = c, n0 + int(np.argmax(np.where(crosses, step, -1.0)))
        beats = ~crosses & (step > crossing)
        if beats.any():
            r, s = _frontier(near[beats], step[beats])
            reach.append(r)
            steps.append(s)
        i = int(np.argmax(np.abs(x)))
        if abs(x[i]) > crest_x:
            crest, crest_x = n0 + i, abs(float(x[i]))
        prev = x[-1]
    r = np.concatenate(reach) if reach else np.zeros(0)
    s = np.concatenate(steps) if steps else np.zeros(0)
    keep = s > crossing
    r, s = _frontier(r[keep], s[keep])
    return y, Slew(peak, knee, Knees(crossing / unit, r / unit, s / unit), crest, steep)
