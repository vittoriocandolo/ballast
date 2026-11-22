# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The curves of the drive and the clipper, anti-aliased with their antiderivatives.

    drive     D(x) = S(x) + k S(x)² (1 - S(x)²),    S(x) = tanh(αx) / tanh α
    clipper   C(u) = u up to the knee, then a parabola that meets ±1 with zero slope

A curve makes harmonics without limit, and those near a multiple of the internal rate fold back into the
audio band. So each output sample is the average of the curve along the input's path from one sample before to
one after, weighted by a triangle that peaks on the sample itself, with the input taken as linear between
samples:

    y[n] = R(x[n], x[n+1]) + R(x[n], x[n-1]),    R(a, b) = ∫₀¹ (1 - u) f(a + u (b - a)) du

This is second-order antiderivative anti-aliasing, centred on the sample. The triangle's spectrum has a double
zero at every multiple of the internal rate, exactly where the aliases come from, and it is symmetric, so no
curve adds delay. Each curve gives R in closed form, arranged so that rounding cancels nothing large.

For a small signal the average is (x[n-1] + 4 x[n] + x[n+1]) / 6, which takes 0.0097 dB off at 20 kHz. The
three-tap correction y[n] + (2 y[n] - y[n-1] - y[n+1]) / 6 undoes it, to within 0.00002 dB up to 20 kHz.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol

import numpy as np
from numpy.polynomial import Polynomial
from scipy.special import spence

if TYPE_CHECKING:
    from numpy.typing import NDArray

    type Floats = NDArray[np.float64]
    type Series = tuple[Floats, Floats, Floats, Floats, Floats]

TAYLOR = 0.02  # the step, in units of 1/α, below which a Taylor series is the more accurate form of R
LOG2 = math.log(2)
PI2_12 = math.pi**2 / 12


class Curve(Protocol):
    """A curve that can average itself around each sample: what apply needs."""

    def average(self, x: Floats) -> Floats:
        """Return the curve averaged around each of the samples x[1:-1], with triangle weights."""
        ...


def apply(curve: Curve, x: Floats) -> Floats:
    """Put the samples x[2:-2] through the curve: average it around each, then flatten the passband."""
    y = curve.average(x)
    return y[1:-1] + (2 * y[1:-1] - y[:-2] - y[2:]) / 6


def _horner(s: Floats, coefficients: Floats) -> Floats:
    """Evaluate the polynomial with these coefficients, lowest power first, at s."""
    value: Floats = np.full_like(s, coefficients[-1])
    for c in coefficients[-2::-1]:
        value = value * s + c
    return value


def _li2_tail(u: Floats) -> Floats:
    """Return Li₂(-e^(-2u)) + π² / 12 for u ≥ 0: 0 at u = 0, rising to π² / 12, with Li₂ the dilogarithm."""
    tail: Floats = spence(1 + np.exp(-2 * u)) + PI2_12  # spence(z) = Li₂(1 - z)
    return tail


class Tanh:
    """The drive's curve: S(x) = tanh(αx) / tanh α, with k S²(1 - S²) added for even harmonics.

    S is odd, with S(±1) = ±1 exactly and a small-signal gain of α / tanh α. The even term keeps all three: it is 0
    at 0 and at ±1, flat at 0, and the curve stays monotonic for k up to 1/2. With k = 0 it is not computed at all.
    """

    def __init__(self, alpha: float, even: float = 0.0) -> None:
        """Make the curve of scale α, which must be positive, with k = even."""
        self.alpha, self.even = alpha, even
        self.norm: float = np.tanh(np.array([alpha]))[0]  # the same tanh as S's own, so S(1) = 1 exactly
        self._even = Even(alpha, self.norm) if even != 0.0 else None

    def __call__(self, x: Floats) -> Floats:
        """Evaluate the curve at x."""
        s: Floats = np.copysign(np.tanh(self.alpha * np.abs(x)), x) / self.norm
        return s if self._even is None else s + self.even * self._even(x)

    def average(self, x: Floats) -> Floats:
        """Return the curve averaged around each of the samples x[1:-1], over its neighbours' span, by the triangle."""
        a = x[1:-1]
        series = self._series(a)
        y = self._side(a, x[2:], series) + self._side(a, x[:-2], series)
        return y if self._even is None else y + self.even * self._even.average(x)

    def half(self, a: Floats, b: Floats) -> Floats:
        """Return R(a, b) = ∫₀¹ (1 - u) D(a + u (b - a)) du: the triangle's side from a sample a to a neighbour b."""
        y = self._side(a, b, self._series(a))
        return y if self._even is None else y + self.even * self._even.half(a, b)

    def _series(self, a: Floats) -> Series:
        """Return S's part of R's Taylor coefficients at a, for powers of q = α (b - a): S⁽ᵏ⁾(a) / (αᵏ (k+1)(k+2) k!).

        S's derivatives are polynomials in t = tanh(αa) times 1 - t², which may round to 0 where S is flat: there
        the terms it scales are negligible anyway.
        """
        t = np.copysign(np.tanh(self.alpha * np.abs(a)), a)  # as S's own, so that R(a, a) is S(a) / 2 exactly
        s = (1 - t * t) / self.norm
        return t / (2 * self.norm), s / 6, -t * s / 12, (6 * t * t - 2) * s / 120, t * (16 - 24 * t * t) * s / 720

    def _side(self, a: Floats, b: Floats, series: Series) -> Floats:
        """Return S's part of R(a, b): its Taylor series at a for a short step, its closed form for a long one."""
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


class Even:
    """The drive's even term E(x) = s² (1 - s²), with s = tanh(αx) / tanh α: 0 at 0 and at ±1, and flat at 0."""

    def __init__(self, alpha: float, norm: float) -> None:
        """Make the even term of the curve of scale α, whose S is tanh(αx) / norm."""
        self.alpha, self.norm = alpha, norm
        # E as a polynomial in s, and its derivatives in u = αx, by the chain rule ds/du = (1 - norm² s²) / norm.
        chain = Polynomial([1 / norm, 0, -norm])
        terms = [Polynomial([0, 0, 1, 0, -1])]
        for _ in range(4):
            terms.append(terms[-1].deriv() * chain)
        weights = [(k + 1) * (k + 2) * math.factorial(k) for k in range(len(terms))]
        self._coefficients = [np.asarray((term / w).coef, dtype=float) for term, w in zip(terms, weights, strict=True)]

    def __call__(self, x: Floats) -> Floats:
        """Evaluate E at x."""
        s = np.copysign(np.tanh(self.alpha * np.abs(x)), x) / self.norm
        value: Floats = s * s * (1 - s * s)  # exactly 0 where s is exactly ±1
        return value

    def average(self, x: Floats) -> Floats:
        """Return E averaged around each of the samples x[1:-1], over its neighbours' span, by the triangle."""
        a = x[1:-1]
        series = self._series(a)
        return self._side(a, x[2:], series) + self._side(a, x[:-2], series)

    def half(self, a: Floats, b: Floats) -> Floats:
        """Return ∫₀¹ (1 - u) E(a + u (b - a)) du: the triangle's side from a sample a to a neighbour b."""
        return self._side(a, b, self._series(a))

    def _series(self, a: Floats) -> Series:
        """Return R's Taylor coefficients at a, for powers of q = α (b - a), from the polynomials in s."""
        s = np.copysign(np.tanh(self.alpha * np.abs(a)), a) / self.norm
        c1, c2, c3, c4 = (_horner(s, c) for c in self._coefficients[1:])
        return s * s * (1 - s * s) / 2, c1, c2, c3, c4  # c₀ as E's own value, so that R(a, a) is E(a) / 2 exactly

    def _side(self, a: Floats, b: Floats, series: Series) -> Floats:
        """Return R(a, b) for E, from its Taylor series at a for a short step, or in closed form for a long one."""
        c0, c1, c2, c3, c4 = series
        d = b - a
        q = self.alpha * d
        out: Floats = c0 + q * (c1 + q * (c2 + q * (c3 + q * c4)))
        far = np.abs(q) >= TAYLOR
        if np.any(far):
            out[far] = self._remainder(a[far], b[far]) / (d[far] * d[far])
        return out

    def _remainder(self, a: Floats, b: Floats) -> Floats:
        """Return ∫ₐᵇ (b - s) E(s) ds in closed form: E = t² / n² - t⁴ / n⁴ in t = tanh u, u = αx, n = tanh α.

        With L = log cosh u, the antiderivatives in u are ∫ t² = u - t and ∫∫ t² = u² / 2 - L, ∫ t⁴ = u - t - t³ / 3
        and ∫∫ t⁴ = u² / 2 - 4L / 3 + t² / 6. E is even, so the integral from a to b is the one from -a to -b, and a
        is made non-negative. When b is too, the terms in u cancel on paper, and what remains is written in
        ε = 1 - t and log(1 + e^(-2u)), which rounding keeps accurate however large u is.
        """
        flip = (a < 0) | ((a == 0) & (b < 0))
        ua, ub = self.alpha * np.where(flip, -a, a), self.alpha * np.where(flip, -b, b)
        step = ub - ua
        vb = np.abs(ub)
        e_a, e_b = np.exp(-2 * ua), np.exp(-2 * vb)
        eps_a, eps_b = 2 * e_a / (1 + e_a), 2 * e_b / (1 + e_b)  # 1 - tanh u, without the cancellation
        tail_a, tail_b = np.log1p(e_a), np.log1p(e_b)  # log cosh u = u - log 2 + tail
        same2 = step * step / 2 - step * eps_a - (tail_b - tail_a)
        same4 = (
            step * step / 2
            - step * (2 * eps_a - eps_a * eps_a + eps_a**3 / 3)
            - 4 * (tail_b - tail_a) / 3
            + (eps_a - eps_b) * (2 - eps_a - eps_b) / 6
        )
        log_a, log_b = ua - LOG2 + tail_a, vb - LOG2 + tail_b
        t_a, t_b = 1 - eps_a, 1 - eps_b
        cross2 = (vb * vb / 2 - log_b) - (ua * ua / 2 - log_a) - step * (ua - t_a)
        cross4 = (
            (vb * vb / 2 - 4 * log_b / 3 + t_b * t_b / 6)
            - (ua * ua / 2 - 4 * log_a / 3 + t_a * t_a / 6)
            - step * (ua - t_a - t_a**3 / 3)
        )
        n2 = self.norm * self.norm
        rem = np.where(ub >= 0, same2 / n2 - same4 / (n2 * n2), cross2 / n2 - cross4 / (n2 * n2))
        result: Floats = rem / (self.alpha * self.alpha)
        return result


class Clip:
    """The clipper's curve, for a threshold of 1: C(u) = u up to the knee's start u₀, then a parabola to ±1.

    The parabola leaves the line with its slope and meets ±1 with zero slope, at 2 - u₀, so the curve is smooth
    once. A knee of 0 dB makes u₀ = 1: a hard clip. C'' is constant in the knee, so R has an exact form there,
    and a hard clip's corner is a jump in C', which the same form takes as a point.
    """

    def __init__(self, knee_db: float) -> None:
        """Make the curve whose knee starts knee_db below the threshold."""
        self.start = 10 ** (-knee_db / 20)
        self.end = 2 - self.start
        self.width = self.end - self.start

    def __call__(self, u: Floats) -> Floats:
        """Evaluate C at u."""
        v = np.abs(u)
        k = np.clip(v - self.start, 0, self.width)
        bent = self.start + k - k * k / (2 * self.width) if self.width else np.minimum(v, 1.0)
        out: Floats = np.copysign(np.where(v <= self.start, v, bent), u)
        return out

    def ratio(self, db: float) -> float:
        """Return how far above the threshold an input peak must be for C to bring it down by db, in peak units."""
        r = 10 ** (-db / 20)
        if 1 / r >= self.end:  # the peak lands on the flat
            return 1 / r
        m = self.start + self.width * (1 - r)  # C(ρ) = r ρ in the knee is a quadratic in ρ
        return m + math.sqrt(m * m - self.start * self.start)

    def average(self, x: Floats) -> Floats:
        """Return C averaged around each of the samples x[1:-1], over its neighbours' span, by the triangle."""
        a = x[1:-1]
        return self.half(a, x[2:]) + self.half(a, x[:-2])

    def half(self, a: Floats, b: Floats) -> Floats:
        """Return R(a, b) = C(a) / 2 + d C'(a) / 6 + ∫ₐᵇ C''(t) (b - t)³ / 6 dt / d², with d = b - a: exact.

        The integral is Taylor's remainder; C'' is ∓1/width over the knees, or a jump of ∓1 at ±1 for a hard clip.
        """
        d = b - a
        v = np.abs(a)
        if self.width:
            slope = np.clip(1 - (v - self.start) / self.width, 0, 1)
        else:  # a hard clip, whose slope at a corner is the one towards b
            slope = np.where((v < 1) | ((a == 1) & (b < 1)) | ((a == -1) & (b > -1)), 1.0, 0.0)
        out = self(a) / 2 + d * slope / 6
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        for side in (1.0, -1.0):  # the upper knee, where C'' is negative, and the lower one: only steps that reach it
            p, q = (self.start, self.end) if side > 0 else (-self.end, -self.start)
            near = (
                np.flatnonzero((lo < q) & (hi > p) & (lo < hi))
                if self.width
                else np.flatnonzero((lo < side) & (side < hi))
            )
            if near.size == 0:
                continue
            bn, dn = b[near], d[near]
            if self.width:
                left, right = np.maximum(lo[near], p), np.minimum(hi[near], q)
                rest = ((bn - left) ** 4 - (bn - right) ** 4) / 24 * (-side / self.width)
            else:
                rest = (bn - side) ** 3 / 6 * -side
            out[near] += np.sign(dn) * rest / (dn * dn)
        result: Floats = out
        return result
