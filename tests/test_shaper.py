# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The drive's curve, anti-aliased: exact where the curve is, accurate in its computation, flat, and clean."""

import math

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.integrate import quad

from ballast_synth import family, render, shaper
from ballast_synth.voice import drive_alpha

DRIVES = [0.5, 6.0, 21.0, 48.0]
BAND_HZ = (20, 20000)  # the audio band, where an alias would be heard
WORST_ALIAS_DB = -104  # the most any setting aliases, as the README states it


def weighted(u: float, curve: shaper.Tanh, a: float, b: float) -> float:
    """Return the integrand of R(a, b) at u: the curve along the step from a to b, weighted by 1 - u."""
    return (1 - u) * float(curve(np.array([a + u * (b - a)]))[0])


def tone(hz: float, rate: int, seconds: float) -> NDArray[np.float64]:
    """Return a full-scale sine at `rate`, with the two samples either side that the drive averages over."""
    t = (np.arange(round(seconds * rate) + 4) - 2) / rate
    return np.sin(2 * np.pi * hz * t)


@pytest.mark.parametrize("db", DRIVES)
def test_the_curve_is_odd_and_keeps_full_scale_exactly(db: float) -> None:
    """Evaluate the curve at full scale, its negative and silence exactly, and odd everywhere."""
    curve = shaper.Tanh(drive_alpha(db))
    x = np.linspace(-3.0, 3.0, 6001)
    assert curve(np.array([1.0, -1.0, 0.0])).tolist() == [1.0, -1.0, 0.0]
    assert np.array_equal(curve(-x), -curve(x))


@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")  # the reference's own rounding notice
@pytest.mark.parametrize("db", DRIVES)
def test_each_side_of_the_triangle_matches_quadrature(db: float) -> None:
    """Compare R(a, b) with adaptive quadrature over steps of every length, near zero, at full scale and far beyond."""
    curve = shaper.Tanh(drive_alpha(db))
    rng = np.random.default_rng(round(10 * db))
    for _ in range(300):
        scale = rng.choice([0.1, 0.66, curve.alpha, 16 * curve.alpha]) / curve.alpha  # where the curve bends, or not
        a = rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.5) * scale
        b = a + rng.choice([-1.0, 1.0]) * 10.0 ** rng.uniform(-9, 0.5) / curve.alpha  # either form of R
        exact, _ = quad(weighted, 0, 1, args=(curve, a, b), epsabs=1e-15, epsrel=1e-13)
        assert curve.half(np.array([a]), np.array([b]))[0] == pytest.approx(exact, rel=0, abs=1e-10)


@pytest.mark.parametrize("db", DRIVES)
def test_the_drive_is_odd_and_exact_on_steady_inputs(db: float) -> None:
    """Drive a signal and its negative into exact negatives, and steady full scale and silence into themselves."""
    curve = shaper.Tanh(drive_alpha(db))
    x = np.random.default_rng(0).standard_normal(10_000)
    assert np.array_equal(shaper.apply(curve, -x), -shaper.apply(curve, x))
    for level in (1.0, -1.0, 0.0):
        assert np.all(shaper.apply(curve, np.full(100, level)) == level)


@pytest.mark.parametrize("hz", [1000.0, 10000.0, 20000.0])
def test_the_drive_is_flat_for_small_signals(hz: float) -> None:
    """Drive a tone 120 dB down, and find the curve's own small-signal gain within 0.00002 dB up to 20 kHz."""
    _, rate, _ = family.rates(48000)
    curve = shaper.Tanh(drive_alpha(21.0))
    level = 1e-6
    y = shaper.apply(curve, level * tone(hz, rate, 0.01))  # a whole number of periods
    t = np.arange(y.size) / rate
    gain = abs(2 * np.mean(y * np.exp(-2j * np.pi * hz * t))) / level
    assert 20 * math.log10(gain / 10 ** (21.0 / 20)) == pytest.approx(0.0, abs=0.00002)


@pytest.mark.parametrize("base", [44100, 48000])
def test_the_worst_setting_aliases_at_least_104_db_down(base: int) -> None:
    """Drive a full-scale 20 kHz tone with the maximum 48 dB, and find everything that is no harmonic 104 dB down.

    The tone is 4 × 4999 Hz, so over a quarter of a second every alias lands on a 4 Hz bin of its own.
    """
    _, rate, factor = family.rates(base)
    hz = 19996
    driven = shaper.apply(shaper.Tanh(drive_alpha(48.0)), tone(hz, rate, 0.27))
    out = family.decimate(family.decimator(base), driven, factor)
    spectrum = np.abs(np.fft.rfft(out[base // 50 : base // 50 + base // 4]))  # a steady quarter second
    f = 4 * np.arange(spectrum.size)
    band = (f >= BAND_HZ[0]) & (f <= BAND_HZ[1]) & (f % hz != 0)
    assert 20 * math.log10(np.max(spectrum[band]) / spectrum[hz // 4]) < WORST_ALIAS_DB


@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
@pytest.mark.parametrize("asymmetry", [0.5, 1.0])
def test_the_asymmetric_curve_matches_quadrature_and_keeps_full_scale(asymmetry: float) -> None:
    """Compare R with quadrature for the curve with its even term, which keeps full scale and stays monotonic."""
    curve = shaper.Tanh(drive_alpha(21.0), asymmetry / 2)
    assert curve(np.array([1.0, -1.0, 0.0])).tolist() == [1.0, -1.0, 0.0]
    assert np.all(np.diff(curve(np.linspace(-1.0, 1.0, 100_001))) >= 0)
    rng = np.random.default_rng(3)
    for _ in range(300):
        a = rng.uniform(-2.0, 2.0) * rng.choice([1.0, 1 / curve.alpha])
        b = a + rng.choice([-1.0, 1.0]) * 10.0 ** rng.uniform(-9, 0.5) / curve.alpha
        exact, _ = quad(weighted, 0, 1, args=(curve, a, b), epsabs=1e-15, epsrel=1e-13)
        assert curve.half(np.array([a]), np.array([b]))[0] == pytest.approx(exact, rel=0, abs=1e-10)


@pytest.mark.filterwarnings("ignore::scipy.integrate.IntegrationWarning")
@pytest.mark.parametrize("knee_db", [0.0, 3.0, 6.0])
def test_the_clipper_matches_quadrature_and_takes_exactly_its_db_off(knee_db: float) -> None:
    """Compare R with quadrature for the clipper, from its corners too, and bring a peak down by exactly db."""
    curve = shaper.Clip(knee_db)
    rng = np.random.default_rng(5)
    corners = [curve.start, curve.end, -curve.start, -curve.end]
    for _ in range(500):
        a = rng.choice([rng.uniform(-3.0, 3.0), *corners])
        b = a + rng.choice([-1.0, 1.0]) * 10.0 ** rng.uniform(-9, 0.6)
        points = [(c - a) / (b - a) for c in corners if 0 < (c - a) / (b - a) < 1]
        exact, _ = quad(weighted, 0, 1, args=(curve, a, b), points=points or None, epsabs=1e-15, epsrel=1e-13)
        assert curve.half(np.array([a]), np.array([b]))[0] == pytest.approx(exact, rel=0, abs=1e-13)
    for db in (0.5, 3.0, 12.0):
        ratio = curve.ratio(db)
        assert 20 * math.log10(curve(np.array([ratio]))[0] / ratio) == pytest.approx(-db, abs=1e-12)


def worst_alias(hz: int, factor: int, base: int, signal: NDArray[np.float64]) -> float:
    """Return the strongest component of the band that is no harmonic of hz, in dB re hz, over a steady 1/4 s."""
    out = family.decimate(family.decimator(base), signal, factor)
    spectrum = np.abs(np.fft.rfft(out[base // 50 : base // 50 + base // 4]))
    f = 4 * np.arange(spectrum.size)
    band = (f >= BAND_HZ[0]) & (f <= BAND_HZ[1]) & (f % hz != 0)
    return 20 * math.log10(np.max(spectrum[band]) / spectrum[hz // 4])


@pytest.mark.parametrize("base", [44100, 48000])
def test_asymmetry_and_hard_clipping_alias_no_more_than_the_drive(base: int) -> None:
    """Drive a full-scale 20 kHz tone to the limit with full asymmetry, and hard-clip one by 12 dB: 104 dB down."""
    _, rate, factor = family.rates(base)
    hz = 19996
    x = tone(hz, rate, 0.27)
    driven = shaper.apply(shaper.Tanh(drive_alpha(48.0), 0.5), x)
    assert worst_alias(hz, factor, base, driven) < WORST_ALIAS_DB
    clipped = render.clip(x[2:-2], 12.0, 0.0, base)
    assert worst_alias(hz, factor, base, clipped) < WORST_ALIAS_DB


@pytest.mark.parametrize("base", [44100, 48000])
def test_the_brightest_body_aliases_no_more_than_the_drive(base: int) -> None:
    """Drive a full-scale saw at 7996 Hz that fades only above 20 kHz with 48 dB, and find no alias 104 dB up."""
    _, rate, factor = family.rates(base)
    hz = 7996  # no divisor of either internal rate, so its aliases miss its harmonics
    phi = 2 * math.pi * hz * np.arange(-2, round(0.27 * rate) + 2) / rate
    r = math.exp(-hz / 20000)
    saw = np.arctan(r * np.sin(phi) / (1 - r * np.cos(phi))) / r
    driven = shaper.apply(shaper.Tanh(drive_alpha(48.0)), saw / np.max(np.abs(saw)))
    assert worst_alias(hz, factor, base, driven) < WORST_ALIAS_DB
