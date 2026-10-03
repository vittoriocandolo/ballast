"""Monotone cubic Hermite curves with exact integrals (plan §2.2).

A curve passes through its knots with slopes from the PCHIP rule:

* interior knots take the weighted harmonic mean of the adjacent secants
  (Fritsch–Butland), or zero where the secants differ in sign or one is zero;
* end knots take the shape-preserving one-sided three-point rule.

End-slope overrides are applied before the curve is evaluated or integrated.
Every slope the rule or an override can produce keeps both normalized end
slopes of each segment in [0, 3], inside the Fritsch–Carlson monotonicity
region, so each segment is monotone and stays between its two knot values.
Evaluation clips to that interval, which removes only rounding excursions.

Outside its knots a curve holds its end values, and its integral grows
linearly there. Times are in seconds.
"""

from __future__ import annotations

import numpy as np


def pchip_slopes(t, v) -> np.ndarray:
    """Knot slopes from the PCHIP rule, before any override."""
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)
    h = np.diff(t)
    m = np.diff(v) / h
    if t.size == 2:
        return np.array([m[0], m[0]])
    ml, mr, hl, hr = m[:-1], m[1:], h[:-1], h[1:]
    agree = (np.sign(ml) == np.sign(mr)) & (ml != 0) & (mr != 0)
    w1, w2 = 2 * hr + hl, hr + 2 * hl
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = (w1 + w2) / (w1 / ml + w2 / mr)
    d = np.empty(t.size)
    d[1:-1] = np.where(agree, mean, 0.0)
    d[0] = _end_slope(h[0], h[1], m[0], m[1])
    d[-1] = _end_slope(h[-1], h[-2], m[-1], m[-2])
    return d


def _end_slope(h0: float, h1: float, m0: float, m1: float) -> float:
    d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    if np.sign(d) != np.sign(m0):
        return 0.0
    if np.sign(m0) != np.sign(m1) and abs(d) > 3 * abs(m0):
        return 3 * m0
    return float(d)


class Curve:
    """Cubic Hermite curve through knots (t, v) with knot slopes d."""

    def __init__(self, t, v, d):
        self.t = np.array(t, dtype=float)
        self.v = np.array(v, dtype=float)
        self.d = np.array(d, dtype=float)
        if not (self.t.ndim == 1 and self.t.size >= 2 and self.v.shape == self.t.shape == self.d.shape):
            raise ValueError("a curve needs at least two knots with one value and one slope each")
        self.h = np.diff(self.t)
        if not np.all(self.h > 0):
            raise ValueError("knot times must increase")
        v, d, h = self.v, self.d, self.h
        full = h * (v[:-1] + v[1:]) / 2 + h * h * (d[:-1] - d[1:]) / 12
        # Integral from the first knot to knot i, and from knot i to the last knot.
        self._before = np.concatenate(([0.0], np.cumsum(full)))
        self._after = np.concatenate((np.cumsum(full[::-1])[::-1], [0.0]))
        for a in (self.t, self.v, self.d, self.h, self._before, self._after):
            a.flags.writeable = False

    @classmethod
    def pchip(cls, t, v, *, first: float | None = None, last: float | None = None) -> Curve:
        """PCHIP curve; `first` and `last` override the end slopes."""
        d = pchip_slopes(t, v)
        if first is not None:
            d[0] = first
        if last is not None:
            d[-1] = last
        return cls(t, v, d)

    @property
    def total(self) -> float:
        """Integral over the knot span."""
        return float(self._before[-1])

    def _segment(self, x: np.ndarray):
        i = np.clip(np.searchsorted(self.t, x, side="right") - 1, 0, self.t.size - 2)
        return i, self.h[i], self.v[i], self.v[i + 1], self.d[i], self.d[i + 1]

    def __call__(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        i, h, va, vb, da, db = self._segment(x)
        s = (x - self.t[i]) / h
        p = s * s * (3 - 2 * s)
        y = va * (1 - p) + vb * p + h * s * (1 - s) * (da * (1 - s) - db * s)
        y = np.clip(y, np.minimum(va, vb), np.maximum(va, vb))
        y = np.where(x < self.t[0], self.v[0], y)
        return np.where(x > self.t[-1], self.v[-1], y)

    def integral(self, x) -> np.ndarray:
        """Integral from the first knot to x."""
        x = np.asarray(x, dtype=float)
        i, h, va, vb, da, db = self._segment(x)
        s = (x - self.t[i]) / h
        y = self._before[i] + h * _partial(s, va, vb, h * da, h * db)
        y = np.where(x < self.t[0], self.v[0] * (x - self.t[0]), y)
        return np.where(x > self.t[-1], self.total + self.v[-1] * (x - self.t[-1]), y)

    def derivative(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        i, h, va, vb, da, db = self._segment(x)
        s = (x - self.t[i]) / h
        y = 6 * (vb - va) / h * s * (1 - s) + da * (1 - s) * (1 - 3 * s) - db * s * (2 - 3 * s)
        return np.where((x < self.t[0]) | (x > self.t[-1]), 0.0, y)

    def jumps(self) -> tuple[np.ndarray, np.ndarray]:
        """Jumps in the second and third derivatives at each knot; the curve is constant outside.

        The curve is only C¹, so these jumps set how fast its spectrum falls.
        """
        h, d = self.h, self.d
        m = np.diff(self.v) / h
        start = (6 * m - 4 * d[:-1] - 2 * d[1:]) / h  # second derivative just after each segment starts
        end = (-6 * m + 2 * d[:-1] + 4 * d[1:]) / h  # and just before it ends
        third = 6 * (d[:-1] + d[1:] - 2 * m) / h**2  # constant on each segment
        jump = lambda after, before: np.concatenate((after, [0.0])) - np.concatenate(([0.0], before))  # noqa: E731
        return jump(start, end), jump(third, third)

    def remaining(self, x) -> np.ndarray:
        """Integral from x to the last knot."""
        x = np.asarray(x, dtype=float)
        i, h, va, vb, da, db = self._segment(x)
        s = (self.t[i + 1] - x) / h
        # The segment read backwards is the same Hermite form with the knots swapped.
        y = self._after[i + 1] + h * _partial(s, vb, va, -h * db, -h * da)
        y = np.where(x < self.t[0], self.total + self.v[0] * (self.t[0] - x), y)
        return np.where(x > self.t[-1], self.v[-1] * (self.t[-1] - x), y)


def _partial(s, va, vb, ma, mb):
    """Integral over [0, s] of the unit Hermite segment with scaled slopes ma, mb."""
    s2 = s * s
    p = s2 * s * (1 - s / 2)  # antiderivative of s²(3 - 2s)
    a = s2 * (6 - 8 * s + 3 * s2) / 12  # antiderivative of s(1 - s)²
    b = s2 * s * (4 - 3 * s) / 12  # antiderivative of s²(1 - s)
    return va * (s - p) + vb * p + ma * a - mb * b


def amplitude_curve(t, v) -> Curve:
    """Amplitude shape: zero end slopes, so it joins silence smoothly."""
    return Curve.pchip(t, v, first=0.0, last=0.0)


def pitch_curve(t, v) -> Curve:
    """Pitch ratio: zero last slope, so it lands smoothly on the tuning."""
    return Curve.pchip(t, v, last=0.0)
