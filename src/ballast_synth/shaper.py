# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The drive's curve, anti-aliased with its antiderivatives.

    S(x) = tanh(αx) / tanh α

A curve makes harmonics without limit, and those near a multiple of the internal rate fold back into the
audio band. So each output sample is the average of S along the input's path from one sample before to one
after, weighted by a triangle that peaks on the sample itself, with the input taken as linear between samples:

    y[n] = R(x[n], x[n+1]) + R(x[n], x[n-1]),    R(a, b) = ∫₀¹ (1 - u) S(a + u (b - a)) du

This is second-order antiderivative anti-aliasing, centred on the sample. The triangle's spectrum has a double
zero at every multiple of the internal rate, exactly where the aliases come from, and it is symmetric, so the
drive adds no delay. R has a closed form in S's antiderivatives, arranged so that rounding cancels nothing
large; for steps much shorter than the curve's own scale, 1/α, a Taylor series in the step is more accurate.

For a small signal the average is (x[n-1] + 4 x[n] + x[n+1]) / 6, which takes 0.0097 dB off at 20 kHz. The
three-tap correction y[n] + (2 y[n] - y[n-1] - y[n+1]) / 6 undoes it, to within 0.00002 dB up to 20 kHz.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import spence

if TYPE_CHECKING:
    from numpy.typing import NDArray

    type Floats = NDArray[np.float64]

TAYLOR = 0.02  # the step, in units of 1/α, below which the Taylor series is the more accurate form of R
LOG2 = math.log(2)
PI2_12 = math.pi**2 / 12


def _li2_tail(u: Floats) -> Floats:
    """Return Li₂(-e^(-2u)) + π² / 12 for u ≥ 0: 0 at u = 0, rising to π² / 12, with Li₂ the dilogarithm."""
    tail: Floats = spence(1 + np.exp(-2 * u)) + PI2_12  # spence(z) = Li₂(1 - z)
    return tail


class Tanh:
    """The curve S(x) = tanh(αx) / tanh α: odd, with S(±1) = ±1 exactly and a small-signal gain of α / tanh α."""

    def __init__(self, alpha: float) -> None:
        """Make the curve of scale α, which must be positive: α = 0 is no drive at all."""
        self.alpha = alpha
        self.norm: float = np.tanh(np.array([alpha]))[0]  # the same tanh as S's own, so S(1) = 1 exactly

    def __call__(self, x: Floats) -> Floats:
        """Evaluate S at x."""
        return np.copysign(np.tanh(self.alpha * np.abs(x)), x) / self.norm

    def average(self, x: Floats) -> Floats:
        """Return S averaged around each of the samples x[1:-1], over its neighbours' span, with triangle weights."""
        a = x[1:-1]
        series = self._series(a)
        return self._side(a, x[2:], series) + self._side(a, x[:-2], series)

    def half(self, a: Floats, b: Floats) -> Floats:
        """Return R(a, b) = ∫₀¹ (1 - u) S(a + u (b - a)) du: the triangle's side from a sample a to a neighbour b."""
        return self._side(a, b, self._series(a))

    def _series(self, a: Floats) -> tuple[Floats, Floats, Floats, Floats, Floats]:
        """Return R's Taylor coefficients at a, for powers of q = α (b - a): S⁽ᵏ⁾(a) / (αᵏ (k + 1)(k + 2) k!).

        S's derivatives are polynomials in t = tanh(αa) times 1 - t², which may round to 0 where S is flat: there
        the terms it scales are negligible anyway.
        """
        t = np.copysign(np.tanh(self.alpha * np.abs(a)), a)  # as S's own, so that R(a, a) is S(a) / 2 exactly
        s = (1 - t * t) / self.norm
        return t / (2 * self.norm), s / 6, -t * s / 12, (6 * t * t - 2) * s / 120, t * (16 - 24 * t * t) * s / 720

    def _side(self, a: Floats, b: Floats, series: tuple[Floats, Floats, Floats, Floats, Floats]) -> Floats:
        """Return R(a, b) from its Taylor series at a for a short step, or in closed form for a long one."""
        c0, c1, c2, c3, c4 = series
        d = b - a
        q = self.alpha * d
        out: Floats = c0 + q * (c1 + q * (c2 + q * (c3 + q * c4)))
        far = np.abs(q) >= TAYLOR
        if np.any(far):
            out[far] = self._remainder(a[far], b[far]) / (d[far] * d[far])
        return out

    def _remainder(self, a: Floats, b: Floats) -> Floats:
        """Return ∫ₐᵇ (b - s) S(s) ds = F₂(b) - F₂(a) - (b - a) F₁(a) in closed form, from S's antiderivatives.

        With u = αx, F₁ = log cosh u / (α tanh α) and F₂ = G(u) / (α² tanh α), where G is the odd antiderivative
        of log cosh: G(u) = u² / 2 - u log 2 + (Li₂(-e^(-2u)) + π² / 12) / 2 for u ≥ 0. S is odd, so the integral
        from a to b is minus the one from -a to -b, and a is made non-negative. When b is too, the u² and log 2
        terms cancel exactly on paper, so they are left out rather than cancelled in rounding.
        """
        sign = np.where((a < 0) | ((a == 0) & (b < 0)), -1.0, 1.0)
        ua, ub = self.alpha * sign * a, self.alpha * sign * b
        step = ub - ua
        tail_a = np.log1p(np.exp(-2 * ua))  # log cosh ua = ua - log 2 + tail_a
        li_a, li_b = _li2_tail(ua), _li2_tail(np.abs(ub))
        same = step * step / 2 + (li_b - li_a) / 2 - step * tail_a
        g_a = ua * ua / 2 - ua * LOG2 + li_a / 2  # G(ua)
        g_b = ub * ub / 2 + ub * LOG2 + li_b / 2  # G(-ub), which is -G(ub), where b is negative
        crossing = -g_b - g_a - step * (ua - LOG2 + tail_a)
        return sign * np.where(ub >= 0, same, crossing) / (self.alpha * self.alpha * self.norm)


def drive(curve: Tanh, x: Floats) -> Floats:
    """Drive the samples x[2:-2]: average the curve around each, with triangle weights, then flatten the passband."""
    y = curve.average(x)
    return y[1:-1] + (2 * y[1:-1] - y[:-2] - y[2:]) / 6
