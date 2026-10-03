"""Body, attack and drive as pure functions of time.

    x(t) = b(t) + a(t),    y(t) = S_α(x(t))

Nothing here keeps state: sample n at rate F is computed from t_n = n / F
alone, so a block boundary, a longer render or a different rate never
changes a sample's value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .curve import Curve, amplitude_curve, pitch_curve
from .patch import Patch


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

