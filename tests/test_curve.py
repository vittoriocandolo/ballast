# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Monotone cubic Hermite curves: through their knots, monotone between them, with exact integrals."""

import warnings

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from ballast_synth.curve import Curve

ROUNDING = 1e-15  # how far rounding may step against a monotone segment


def random_curve(seed: int, **ends: float) -> Curve:
    """Make a curve through 2 to 11 random knots, with plateaus when the seed is even."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 12))
    t = np.cumsum(rng.uniform(1e-3, 0.1, n))
    v = rng.uniform(0, 1, n) if seed % 2 else np.round(rng.uniform(0, 1, n), 1)  # rounding makes plateaus
    return Curve(t, v, **ends)


@pytest.mark.parametrize("seed", range(30))
def test_agrees_with_scipy_pchip(seed: int) -> None:
    """Compare a curve with free end slopes to SciPy's PCHIP, which follows the same rule."""
    c = random_curve(seed)
    x = np.linspace(c.t[0], c.t[-1], 2001)
    assert np.allclose(c(x), PchipInterpolator(c.t, c.v)(x), rtol=0, atol=1e-12)


@pytest.mark.parametrize("seed", range(30))
def test_passes_through_its_knots_and_is_monotone_between_them(seed: int) -> None:
    """Evaluate a curve at its knots exactly, and densely between them for any step the wrong way."""
    c = random_curve(seed, first=0.0, last=0.0)
    assert np.array_equal(c(c.t), c.v)
    for i in range(c.t.size - 1):
        y = c(np.linspace(c.t[i], c.t[i + 1], 500))
        steps = np.diff(y) * np.sign(c.v[i + 1] - c.v[i])
        assert np.all(steps >= -ROUNDING)


@pytest.mark.parametrize("seed", range(30))
def test_integrals_are_exact(seed: int) -> None:
    """Check each segment's integral by Simpson's rule, which is exact for cubics, up to rounding."""
    c = random_curve(seed, last=0.0)
    for i in range(c.t.size - 1):
        a, b = c.t[i], c.t[i] + 0.7 * c.h[i]
        simpson = (b - a) / 6 * (c(a) + 4 * c((a + b) / 2) + c(b))
        assert c.integral(b) - c.integral(a) == pytest.approx(simpson, rel=1e-12, abs=1e-15)
    x = np.linspace(c.t[0], c.t[-1], 101)
    assert np.allclose(c.integral(x) + c.remaining(x), c.total, rtol=1e-14, atol=0)


def test_end_slopes_can_be_fixed() -> None:
    """Fix both end slopes at zero, where the rule would slope them."""
    c = Curve([0.0, 1.0, 3.0], [0.0, 1.0, 0.0], first=0.0, last=0.0)
    assert c.d[0] == 0.0
    assert c.d[-1] == 0.0


def test_holds_its_end_values_outside_its_knots() -> None:
    """Evaluate and integrate a curve before its first knot and after its last."""
    first, last = 0.25, 0.5
    c = Curve([1.0, 2.0, 4.0], [first, 1.0, last])
    assert c(0.0) == first
    assert c(9.0) == last
    assert c.integral(0.0) == -first  # the integral grows linearly before the first knot, for 1 s here
    assert c.integral(5.0) == pytest.approx(c.total + last)
    assert c.remaining(5.0) == -last
    assert c.remaining(0.0) == pytest.approx(c.total + first)  # and so does what remains, counted back from it
    x = np.array([-3.0, 0.0, 0.5, 4.5, 9.0])
    assert np.allclose(c.integral(x) + c.remaining(x), c.total, rtol=1e-14, atol=0)  # either side, as within


def test_two_knots_make_a_straight_line() -> None:
    """Draw a curve through two knots, and integrate it as a trapezoid."""
    c = Curve([0.0, 2.0], [1.0, 3.0])
    assert np.allclose(c(np.linspace(0, 2, 9)), np.linspace(1, 3, 9))
    assert c.total == 2.0 * (1.0 + 3.0) / 2  # its width times its mean height


def test_tiny_values_raise_no_warnings() -> None:
    """Fit a curve through values so small that its harmonic means overflow, with warnings as errors."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c = Curve([0.0, 0.001, 0.002, 0.003, 0.004], [0.0, 1e-310, 1.0, 1e-310, 0.0], first=0.0, last=0.0)
        assert np.all(np.isfinite(c.d))
        assert np.all(np.isfinite(c(np.linspace(0, 0.004, 101))))


def test_a_curve_needs_two_knots() -> None:
    """Refuse a curve of fewer than two knots, which has no segment to draw, with the reason."""
    with pytest.raises(ValueError, match="at least 2 knots, found 1"):
        Curve([0.0], [1.0])
