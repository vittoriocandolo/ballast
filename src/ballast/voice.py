"""Body, attack and drive as pure functions of time.

    x(t) = b(t) + a(t),    y(t) = S_α(x(t))

Nothing here keeps state: sample n at rate F is computed from t_n = n / F
alone, so a block boundary, a longer render or a different rate never
changes a sample's value.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from . import noise
from .curve import Curve
from .patch import Patch

# Samples per block. Blocks bound the size of the temporaries; threads would only contend for the GIL.
BLOCK = 2**15


def drive_alpha(db: float) -> float:
    """α with α / tanh α = 10^(db/20): the saturator's small-signal gain. A gain of 1 is bypass."""
    g = 10.0 ** (db / 20.0)
    if g == 1.0:  # 0 dB, or so little drive that the gain rounds to 1
        return 0.0
    # x / tanh x lies between x and x + 1, so [g - 1, g] brackets the root.
    return brentq(lambda a: a / math.tanh(a) - g, g - 1.0, g, xtol=1e-300, rtol=4 * np.finfo(float).eps, maxiter=500)


def mix_weights(m: float) -> tuple[float, float]:
    """Constant-power tone and noise weights, exact at both ends."""
    if m == 0.0:
        return 1.0, 0.0
    if m == 1.0:
        return 0.0, 1.0
    return math.cos(math.pi * m / 2), math.sin(math.pi * m / 2)


def sample_count(t_src: float, rate: int) -> int:
    """N = ⌊T_src·F⌋ + 1: every internal sample with n / F ≤ T_src, decided in float64."""
    n = math.floor(t_src * rate)
    while (n + 1) / rate <= t_src:
        n += 1
    while n / rate > t_src:
        n -= 1
    return n + 1


def _curve(knots, **ends) -> Curve:
    """The curve through a patch's (ms, value) knots, in seconds."""
    return Curve([ms / 1000 for ms, _ in knots], [v for _, v in knots], **ends)


def _support(t: np.ndarray, curve: Curve) -> slice:
    return slice(np.searchsorted(t, curve.t[0], "left"), np.searchsorted(t, curve.t[-1], "right"))


class Voice:
    """The patch at internal rate F: y[n] = S_α(b(n / F) + a(n / F))."""

    def __init__(self, p: Patch, rate: int):
        b, a = p.body, p.attack
        self.rate = rate
        self.f0, self.h2, self.phi0, self.tail = b.tune_hz, b.h2, math.radians(b.phase_deg), b.phase_ref == "tail"
        self.body_gain = 10.0 ** (b.level_db / 20.0) / math.sqrt(1.0 + b.h2 * b.h2)  # g_b / √(1 + h2²)
        self.pitch = _curve(b.pitch, last=0.0)  # lands smoothly on the tuning
        # Amplitude shapes have zero end slopes, so they join silence smoothly.
        self.body_amp = _curve(b.amp, first=0.0, last=0.0)
        self.attack_amp = _curve(a.amp, first=0.0, last=0.0)
        self.attack_gain = 10.0 ** (a.level_db / 20.0)
        self.tone_hz = a.tone_hz
        self.tone_weight, self.noise_weight = mix_weights(a.noise_mix)
        uses_noise = self.attack_gain != 0.0 and self.noise_weight != 0.0
        self.noise = noise.table(p.seed, *a.noise_band_hz, rate) if uses_noise else None
        self.alpha = drive_alpha(p.drive.db)
        self.count = sample_count(p.length_ms / 1000, rate)

    def samples(self) -> np.ndarray:
        """y[n] for 0 ≤ n < N, computed in blocks aligned to the trigger."""
        return np.concatenate([self._block(n, min(n + BLOCK, self.count)) for n in range(0, self.count, BLOCK)])

    def _block(self, n0: int, n1: int) -> np.ndarray:
        """y[n] for n0 ≤ n < n1, each source only over its support."""
        n = np.arange(n0, n1)
        t = n / self.rate
        x = np.zeros(n1 - n0)
        if self.body_gain:
            s = _support(t, self.body_amp)
            x[s] += self.body(t[s])
        if self.attack_gain:
            s = _support(t, self.attack_amp)
            x[s] += self.attack(t[s], n[s])
        return self.saturate(x)

    def body(self, t: np.ndarray) -> np.ndarray:
        # φ = φ0 + 2π f0 τ, where τ(t) = R(t) − R(t_ref) + c integrates the pitch ratio.
        if self.tail:  # τ = t exactly from the last pitch knot on, whatever the sweep does before it
            t_n = self.pitch.t[-1]
            tau = np.where(t >= t_n, t, t_n - self.pitch.remaining(t))
        else:
            tau = self.pitch.integral(t)
        phi = self.phi0 + (2 * math.pi * self.f0) * tau
        wave = np.sin(phi)
        if self.h2:
            # sin φ + h2 sin 2φ = sin φ (1 + 2 h2 cos φ): no extra zero crossings while h2 ≤ ½.
            wave *= 1.0 + 2.0 * self.h2 * np.cos(phi)
        return self.body_gain * self.body_amp(t) * wave

    def attack(self, t: np.ndarray, n: np.ndarray) -> np.ndarray:
        carrier = np.zeros_like(t)
        if self.tone_weight:
            carrier += self.tone_weight * np.sin((2 * math.pi * self.tone_hz) * t)
        if self.noise_weight:
            assert self.noise is not None  # computed whenever the attack has noise
            carrier += self.noise_weight * self.noise[n % self.rate]  # the noise repeats every second
        return self.attack_gain * self.attack_amp(t) * carrier

    def saturate(self, x: np.ndarray) -> np.ndarray:
        """S_α(x) = tanh(αx) / tanh α, odd by construction; the identity when α = 0."""
        if not self.alpha:
            return x
        norm = np.tanh(np.array([self.alpha]))[0]  # the same tanh as below, so S(1) = 1 exactly
        with np.errstate(over="ignore"):
            return np.copysign(np.tanh(self.alpha * np.abs(x)), x) / norm
