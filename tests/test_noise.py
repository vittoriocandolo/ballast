# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Seeded continuous-time noise and the PCM dither: reproducible, rate-independent, band-editable."""

import numpy as np
import pytest

from ballast_synth import noise


def test_the_noise_is_the_same_at_every_internal_rate() -> None:
    """Sample the noise at both internal rates, and compare it at the times both grids share."""
    a = noise.table(1, 700.0, 9000.0, 768000)
    b = noise.table(1, 700.0, 9000.0, 705600)
    k = np.arange(2400)  # times k / 2400 s fall on both grids
    assert np.allclose(a[k * 320], b[k * 294], rtol=0, atol=1e-12)


def test_editing_the_band_reweights_the_same_components() -> None:
    """Change both band edges, and find every component with the same phase as before."""
    a = np.fft.rfft(noise.table(1, 700.0, 9000.0, 768000))
    b = np.fft.rfft(noise.table(1, 300.0, 4000.0, 768000))
    k = np.arange(1, 20001)
    assert np.allclose(a[k] / np.abs(a[k]), b[k] / np.abs(b[k]), rtol=0, atol=1e-9)  # same phases


def test_each_seed_draws_its_own_noise() -> None:
    """Draw the same noise again from the same seed, uncached, and other noise from another seed."""
    a = noise.table(1, 700.0, 9000.0, 768000)
    assert np.array_equal(a, noise.table.__wrapped__(1, 700.0, 9000.0, 768000))
    assert not np.allclose(a, noise.table(2, 700.0, 9000.0, 768000))


@pytest.mark.parametrize("seed", [0, 1, 2**63 - 1])
def test_the_noise_has_the_power_of_a_unit_sine(seed: int) -> None:
    """Measure the mean square of a second of noise, which a unit sine would have."""
    assert np.mean(noise.table(seed, 300.0, 12000.0, 768000) ** 2) == pytest.approx(0.5, rel=0.1)


def test_the_noise_is_band_limited() -> None:
    """Find nothing in the noise's spectrum above 20 kHz but rounding."""
    spectrum = np.abs(np.fft.rfft(noise.table(1, 700.0, 9000.0, 768000)))
    assert np.allclose(spectrum[20001:], 0, rtol=0, atol=1e-6)


def test_dither_is_triangular_and_prefix_stable() -> None:
    """Measure the dither's range and first two moments, and draw a shorter run that starts the same."""
    d = noise.dither(7, 200_000)
    assert np.all(np.abs(d) < 1)
    assert np.mean(d) == pytest.approx(0, abs=0.01)
    assert np.var(d) == pytest.approx(1 / 6, rel=0.02)  # the variance of a triangle on (-1, 1)
    assert np.array_equal(noise.dither(7, 1000), d[:1000])
