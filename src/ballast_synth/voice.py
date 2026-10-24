# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Body, attack, transient and drive as pure functions of time.

    x(t) = b(t) + a(t) + v(t),    y[n] = the drive's curve S_α, averaged around t_n = n / F

Nothing here keeps state: sample n at rate F is computed from x at t_(n-2) … t_(n+2)
alone, so a block boundary, a longer render or another rate of the same family
never changes a sample's value. The averaging keeps the drive from aliasing; see
shaper.py.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import brentq
from scipy.signal import upfirdn

from . import family, noise, shaper
from .curve import Curve
from .patch import decode

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .patch import Knots, Patch, Transient

# Samples per block. Blocks bound the size of the temporaries; threads would only contend for the GIL.
BLOCK = 2**15
NODES = range(-2, 4)  # where the 6-point interpolation of a transient reads it, around each position


def drive_alpha(db: float) -> float:
    """Solve α / tanh α = 10^(db/20), the saturator's small-signal gain, for α. A gain of 1 is a bypass: α = 0."""
    g = 10.0 ** (db / 20.0)
    if g == 1.0:  # 0 dB, or so little drive that the gain rounds to 1
        return 0.0
    # x / tanh x lies between x and x + 1, so [g - 1, g] brackets the root.
    root = brentq(lambda a: a / math.tanh(a) - g, g - 1.0, g, xtol=1e-300, rtol=4 * np.finfo(float).eps, maxiter=500)
    return float(root)


def mix_weights(m: float) -> tuple[float, float]:
    """Return the constant-power tone and noise weights, exact at both ends."""
    if m == 0.0:
        return 1.0, 0.0
    if m == 1.0:
        return 0.0, 1.0
    return math.cos(math.pi * m / 2), math.sin(math.pi * m / 2)


def sample_count(t_src: float, rate: int) -> int:
    """Count every internal sample with n / F ≤ T_src: N = ⌊T_src·F⌋ + 1, decided in float64."""
    n = math.floor(t_src * rate)
    while (n + 1) / rate <= t_src:
        n += 1
    while n / rate > t_src:
        n -= 1
    return n + 1


def _curve(knots: Knots, **ends: float) -> Curve:
    """Make the curve through a patch's (ms, value) knots, in seconds."""
    return Curve([ms / 1000 for ms, _ in knots], [v for _, v in knots], **ends)


def lagrange(u: NDArray[np.float64], q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Read u at positions q by the Lagrange polynomial through u[⌊q⌋ − 2] … u[⌊q⌋ + 3], with silence beyond u.

    Each q must lie within −3 < q < u.size + 2, where the result can differ from 0. On a whole position the weights
    are exactly 1 and 0, so the result is u itself.
    """
    whole = np.floor(q)
    f = q - whole
    padded = np.pad(u, 6)  # silence, as far as any node reaches
    i = whole.astype(np.int64) + 6
    out = np.zeros_like(q)
    for j in NODES:
        weight = np.ones_like(f)
        for m in NODES:
            if m != j:
                weight *= (f - m) / (j - m)
        out += weight * padded[i + j]
    return out


def place(t: Transient, rate: int) -> tuple[int, NDArray[np.float64]]:
    """Place a transient at internal rate F: the first sample n_1 it can reach, and v(n / F) from there on.

    The frozen filter of the samples' own family brings them up to that family's internal rate F_t, as u, and the
    voice reads u at q = ρ (n − n_0) F_t / F, where ρ is the pitch and n_0 = start × F. That is u's own samples
    when the families match, the pitch is 1 and the start falls on a sample, and lagrange() between them otherwise.
    """
    base, internal, factor = family.rates(t.rate)
    u: NDArray[np.float64] = upfirdn(factor * family.decimator(base), decode(t.samples).astype(float), up=factor)
    step = t.pitch * internal / rate
    start = t.start_ms * rate / 1000
    first = math.floor(start - 3 / step) + 1
    n = np.arange(first, math.ceil(start + (u.size + 2) / step), dtype=float)
    return first, t.polarity * 10.0 ** (t.level_db / 20.0) * lagrange(u, (n - start) * step)


def _support(t: NDArray[np.float64], curve: Curve) -> slice:
    """Return the slice of the times t within an amplitude shape's knots, outside which its source is silent."""
    return slice(np.searchsorted(t, curve.t[0], "left"), np.searchsorted(t, curve.t[-1], "right"))


class Voice:
    """The patch at internal rate F: y[n], the drive's curve averaged around n / F over x(t) = b(t) + a(t) + v(t)."""

    def __init__(self, p: Patch, rate: int) -> None:
        """Prepare the patch's curves, gains, noise, transient and drive for synthesis at `rate`."""
        b, a = p.body, p.attack
        self.rate = rate
        self.f0, self.h2, self.phi0, self.tail = b.tune_hz, b.h2, math.radians(b.phase_deg), b.phase_ref == "tail"
        self.bright = b.bright_hz
        self.body_gain: float = 10.0 ** (b.level_db / 20.0) / math.sqrt(1.0 + b.h2 * b.h2)  # g_b / √(1 + h2²)
        self.pitch = _curve(b.pitch, last=0.0)  # lands smoothly on the tuning
        # Amplitude shapes have zero end slopes, so they join silence smoothly.
        self.body_amp = _curve(b.amp, first=0.0, last=0.0)
        self.attack_amp = _curve(a.amp, first=0.0, last=0.0)
        self.attack_gain: float = 10.0 ** (a.level_db / 20.0)
        self.tone_hz = a.tone_hz
        # A flat pitch curve is the fixed tone, whose phase 2π f_t t needs no integral.
        self.tone_pitch = _curve(a.pitch, last=0.0) if any(r != 1.0 for _, r in a.pitch) else None
        self.tone_weight, self.noise_weight = mix_weights(a.noise_mix)
        uses_noise = self.attack_gain != 0.0 and self.noise_weight != 0.0
        self.noise = noise.table(p.seed, *a.noise_band_hz, rate) if uses_noise else None
        self.alpha = drive_alpha(p.drive.db)
        # No curve at all at 0 dB. Asymmetry adds the even term k S²(1 - S²), with k up to the monotonic limit 1/2.
        self.curve = shaper.Tanh(self.alpha, p.drive.asymmetry / 2) if self.alpha != 0.0 else None
        self.mix = p.drive.mix
        self.count = sample_count(p.length_ms / 1000, rate)
        self.transient = None if p.transient is None else place(p.transient, rate)
        if self.transient is not None:  # the samples' interpolation rings on past their end
            first, v = self.transient
            self.count = max(self.count, first + v.size)

    def samples(self) -> NDArray[np.float64]:
        """Compute y[n] for 0 ≤ n < N, in blocks aligned to the trigger."""
        return np.concatenate([self._block(n, min(n + BLOCK, self.count)) for n in range(0, self.count, BLOCK)])

    def _block(self, n0: int, n1: int) -> NDArray[np.float64]:
        """Compute y[n] for n0 ≤ n < n1, from the sources two samples either side, which the drive averages over."""
        x = self.sources(n0 - 2, n1 + 2)
        if self.curve is None:
            return x[2:-2]
        driven = shaper.apply(self.curve, x)
        if self.mix == 1.0:
            return driven
        return (1 - self.mix) * x[2:-2] + self.mix * driven  # dry and driven line up: the drive adds no delay

    def sources(self, n0: int, n1: int) -> NDArray[np.float64]:
        """Compute x[n] = b(n / F) + a(n / F) + v(n / F) for n0 ≤ n < n1, each source only over its support."""
        n = np.arange(n0, n1)
        t = n / self.rate
        x = np.zeros(n1 - n0)
        if self.body_gain != 0.0:
            s = _support(t, self.body_amp)
            x[s] += self.body(t[s])
        if self.attack_gain != 0.0:
            s = _support(t, self.attack_amp)
            x[s] += self.attack(t[s], n[s])
        if self.transient is not None:
            first, v = self.transient
            lo, hi = max(n0, first), min(n1, first + v.size)
            if lo < hi:
                x[lo - n0 : hi - n0] += v[lo - first : hi - first]
        return x

    def body(self, t: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the body at times t: the swept sine and its second harmonic, under their amplitude shape."""
        # φ = φ0 + 2π f0 τ, where τ integrates the pitch ratio R: τ(t) = ∫₀ᵗ R, or with a tail reference
        # τ(t) = t_n − ∫ₜ^{t_n} R, which is exactly t from the last pitch knot t_n on, whatever the sweep does.
        if self.tail:
            t_n = self.pitch.t[-1]
            tau = np.where(t >= t_n, t, t_n - self.pitch.remaining(t))
        else:
            tau = self.pitch.integral(t)
        phi = self.phi0 + (2 * math.pi * self.f0) * tau
        if self.bright == 0.0:
            wave = np.sin(phi)
            if self.h2 != 0.0:
                # sin φ + h2 sin 2φ = sin φ (1 + 2 h2 cos φ): no extra zero crossings while h2 ≤ ½.
                wave *= 1.0 + 2.0 * self.h2 * np.cos(phi)
        else:
            # Σ r^(k-1) sin(kφ) / k = atan(r sin φ / (1 - r cos φ)) / r, with r = e^(-f/bright) at the frequency f
            # of the moment: a saw whose k-th harmonic, at k f, fades by e^(-(k-1) f/bright), a roll-off fixed in
            # Hz whatever the pitch, while the fundamental stays at exactly 1. h2 adds to its second harmonic.
            r = np.exp(-(self.f0 / self.bright) * self.pitch(t))
            wave = np.arctan(r * np.sin(phi) / (1.0 - r * np.cos(phi))) / r
            if self.h2 != 0.0:
                wave += self.h2 * np.sin(2.0 * phi)
        return self.body_gain * self.body_amp(t) * wave

    def attack(self, t: NDArray[np.float64], n: NDArray[np.int_]) -> NDArray[np.float64]:
        """Return the attack at times t, sample indices n: the tone and the noise, under their amplitude shape."""
        carrier = np.zeros_like(t)
        if self.tone_weight != 0.0:
            # φ = 2π f_t τ, where τ(t) = ∫₀ᵗ R integrates the tone's own pitch ratio, as the body's does.
            tau = t if self.tone_pitch is None else self.tone_pitch.integral(t)
            carrier += self.tone_weight * np.sin((2 * math.pi * self.tone_hz) * tau)
        if self.noise is not None:  # computed whenever the attack has noise
            carrier += self.noise_weight * self.noise[n % self.rate]  # the noise repeats every second
        return self.attack_gain * self.attack_amp(t) * carrier
