"""Monotone cubic Hermite curves: through their knots, monotone between them, with exact integrals."""

import warnings

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from ballast_synth.curve import Curve


def random_curve(seed: int, **ends) -> Curve:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 12))
    t = np.cumsum(rng.uniform(1e-3, 0.1, n))
    v = rng.uniform(0, 1, n) if seed % 2 else np.round(rng.uniform(0, 1, n), 1)  # rounding makes plateaus
    return Curve(t, v, **ends)


@pytest.mark.parametrize("seed", range(30))
def test_agrees_with_scipy_pchip(seed):
    c = random_curve(seed)
    x = np.linspace(c.t[0], c.t[-1], 2001)
    assert np.allclose(c(x), PchipInterpolator(c.t, c.v)(x), rtol=0, atol=1e-12)


@pytest.mark.parametrize("seed", range(30))
def test_passes_through_its_knots_and_is_monotone_between_them(seed):
    c = random_curve(seed, first=0.0, last=0.0)
    assert np.array_equal(c(c.t), c.v)
    for i in range(c.t.size - 1):
        y = c(np.linspace(c.t[i], c.t[i + 1], 500))
        steps = np.diff(y) * np.sign(c.v[i + 1] - c.v[i])
        assert np.all(steps >= -1e-15)


@pytest.mark.parametrize("seed", range(30))
def test_integrals_are_exact(seed):
    """Simpson's rule is exact for cubics, so it checks each segment's integral up to rounding."""
    c = random_curve(seed, last=0.0)
    for i in range(c.t.size - 1):
        a, b = c.t[i], c.t[i] + 0.7 * c.h[i]
        simpson = (b - a) / 6 * (c(a) + 4 * c((a + b) / 2) + c(b))
        assert c.integral(b) - c.integral(a) == pytest.approx(simpson, rel=1e-12, abs=1e-15)
    x = np.linspace(c.t[0], c.t[-1], 101)
    assert np.allclose(c.integral(x) + c.remaining(x), c.total, rtol=1e-14, atol=0)


def test_end_slopes_can_be_fixed():
    c = Curve([0.0, 1.0, 3.0], [0.0, 1.0, 0.0], first=0.0, last=0.0)
    assert c.d[0] == 0.0 and c.d[-1] == 0.0


def test_holds_its_end_values_outside_its_knots():
    c = Curve([1.0, 2.0, 4.0], [0.25, 1.0, 0.5])
    assert c(0.0) == 0.25 and c(9.0) == 0.5
    assert c.integral(0.0) == -0.25  # the integral grows linearly before the first knot
    assert c.integral(5.0) == pytest.approx(c.total + 0.5)
    assert c.remaining(5.0) == -0.5


def test_two_knots_make_a_straight_line():
    c = Curve([0.0, 2.0], [1.0, 3.0])
    assert np.allclose(c(np.linspace(0, 2, 9)), np.linspace(1, 3, 9))
    assert c.total == 4.0


def test_tiny_values_raise_no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        c = Curve([0.0, 0.001, 0.002, 0.003, 0.004], [0.0, 1e-310, 1.0, 1e-310, 0.0], first=0.0, last=0.0)
        assert np.all(np.isfinite(c.d)) and np.all(np.isfinite(c(np.linspace(0, 0.004, 101))))
