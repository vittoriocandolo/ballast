# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Monotone cubic Hermite curves with exact integrals.

A curve passes through its knots with slopes from the PCHIP rule:

* interior knots take the weighted harmonic mean of the adjacent secants
  (Fritsch–Butland), or zero where the secants differ in sign or one is zero;
* end knots take the shape-preserving one-sided three-point rule, unless the
  caller fixes them.

Every slope the rule or a fixed end slope of zero can produce keeps both
normalized end slopes of each segment in [0, 3], inside the Fritsch–Carlson
monotonicity region, so each segment is monotone and stays between its two
knot values. Evaluation clips to that interval, which removes only rounding
excursions.

Outside its knots a curve holds its end values, and its integral grows
linearly there. Times are in seconds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

    type Floats = NDArray[np.float64]


MIN_KNOTS = 2  # a curve's fewest knots, for one segment; patches never give fewer


def _slopes(h: Floats, m: Floats) -> Floats:
    """Compute the knot slopes by the PCHIP rule, from the knot spacings h and the secants m."""
    if h.size == 1:
        return np.array([m[0], m[0]])
    ml, mr, hl, hr = m[:-1], m[1:], h[:-1], h[1:]
    agree = (np.sign(ml) == np.sign(mr)) & (ml != 0) & (mr != 0)
    w1, w2 = 2 * hr + hl, hr + 2 * hl
    # Zero secants divide by zero, and tiny ones overflow, where the mean is unused or rounds to 0.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        mean = (w1 + w2) / (w1 / ml + w2 / mr)
    d = np.empty(h.size + 1)
    d[1:-1] = np.where(agree, mean, 0.0)
    d[0] = _end_slope(h[0], h[1], m[0], m[1])
    d[-1] = _end_slope(h[-1], h[-2], m[-1], m[-2])
    return d


def _end_slope(h0: float, h1: float, m0: float, m1: float) -> float:
    """Return an end knot's slope by the shape-preserving three-point rule.

    h0 and m0 are the width and the secant of the knot's segment, h1 and m1 those of the next one.
    """
    d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    if np.sign(d) != np.sign(m0):
        return 0.0
    if np.sign(m0) != np.sign(m1) and abs(d) > 3 * abs(m0):
        return 3 * m0
    return float(d)


class Curve:
    """PCHIP curve through knots (t, v), with spacings h and slopes d; `first` and `last` fix the end slopes."""

    def __init__(self, t: ArrayLike, v: ArrayLike, *, first: float | None = None, last: float | None = None) -> None:
        """Fit the curve to its knots, and integrate it from end to end."""
        self.t = np.array(t, dtype=float)
        if self.t.size < MIN_KNOTS:
            msg = f"a curve needs at least {MIN_KNOTS} knots, found {self.t.size}"
            raise ValueError(msg)
        self.v = np.array(v, dtype=float)
        self.h = h = np.diff(self.t)
        self.d = d = _slopes(h, np.diff(self.v) / h)
        if first is not None:
            d[0] = first
        if last is not None:
            d[-1] = last
        full = h * (self.v[:-1] + self.v[1:]) / 2 + h * h * (d[:-1] - d[1:]) / 12
        # Integral from the first knot to knot i, and from knot i to the last knot.
        self._before = np.concatenate(([0.0], np.cumsum(full)))
        self._after = np.concatenate((np.cumsum(full[::-1])[::-1], [0.0]))
        self.total: float = self._before[-1]  # the integral over the knot span

    @classmethod
    def of(cls, knots: Sequence[tuple[float, float]], **ends: float) -> Curve:
        """Make the curve through a patch's (ms, value) knots, in seconds; `ends` fixes end slopes, as above."""
        return cls([ms / 1000 for ms, _ in knots], [v for _, v in knots], **ends)

    def _segment(self, x: Floats) -> tuple[NDArray[np.intp], Floats, Floats, Floats, Floats, Floats]:
        """Return the segment of each x: its index, width, end values and end slopes."""
        i = np.clip(np.searchsorted(self.t, x, side="right") - 1, 0, self.t.size - 2)
        return i, self.h[i], self.v[i], self.v[i + 1], self.d[i], self.d[i + 1]

    def __call__(self, x: ArrayLike) -> Floats:
        """Evaluate the curve at x."""
        at = np.asarray(x, dtype=float)
        i, h, va, vb, da, db = self._segment(at)
        s = (at - self.t[i]) / h
        p = s * s * (3 - 2 * s)
        y = va * (1 - p) + vb * p + h * s * (1 - s) * (da * (1 - s) - db * s)
        y = np.clip(y, np.minimum(va, vb), np.maximum(va, vb))
        y = np.where(at < self.t[0], float(self.v[0]), y)
        return np.where(at > self.t[-1], float(self.v[-1]), y)

    def integral(self, x: ArrayLike) -> Floats:
        """Integrate from the first knot to x."""
        at = np.asarray(x, dtype=float)
        i, h, va, vb, da, db = self._segment(at)
        s = (at - self.t[i]) / h
        y = self._before[i] + h * _partial(s, va, vb, h * da, h * db)
        y = np.where(at < self.t[0], self.v[0] * (at - self.t[0]), y)
        return np.where(at > self.t[-1], self.total + self.v[-1] * (at - self.t[-1]), y)

    def remaining(self, x: ArrayLike) -> Floats:
        """Integrate from x to the last knot."""
        at = np.asarray(x, dtype=float)
        i, h, va, vb, da, db = self._segment(at)
        s = (self.t[i + 1] - at) / h
        # The segment read backwards is the same Hermite form with the knots swapped.
        y = self._after[i + 1] + h * _partial(s, vb, va, -h * db, -h * da)
        y = np.where(at < self.t[0], self.total + self.v[0] * (self.t[0] - at), y)
        return np.where(at > self.t[-1], self.v[-1] * (self.t[-1] - at), y)


def _partial(s: Floats, va: Floats, vb: Floats, ma: Floats, mb: Floats) -> Floats:
    """Integrate the unit Hermite segment with scaled slopes ma, mb over [0, s]."""
    s2 = s * s
    p = s2 * s * (1 - s / 2)  # antiderivative of s²(3 - 2s)
    a = s2 * (6 - 8 * s + 3 * s2) / 12  # antiderivative of s(1 - s)²
    b = s2 * s * (4 - 3 * s) / 12  # antiderivative of s²(1 - s)
    return va * (s - p) + vb * p + ma * a - mb * b
