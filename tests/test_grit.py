# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Grit: a sampler's aliasing and staircase, exactly as modeled, and nothing of Ballast's own."""

import math

import numpy as np
import pytest

from ballast_synth import family, grit, patch

BASE = 48000
_, INTERNAL, FACTOR = family.rates(BASE)
SP = patch.Grit(26040.0, 16, 0.0)  # a 26.04 kHz, 16-bit sampler
MATCH_DB = 0.01  # how close each tone comes to the zero-order hold's prediction
FLOOR_DB = -110.0  # below which lie only the converter's own rounding spurs
PASSBAND_HZ = 20000
HARMONIC_DB = -20.0  # how strong a clipped tone's third harmonic is, at least, against the tone


def test_the_staircase_is_averaged_exactly() -> None:
    """Average a staircase under each output's triangle, against the midpoint rule on a grid 2000 times finer."""
    v = np.random.default_rng(3).uniform(-1, 1, 40)
    spacing, n, fine = 7.37, 280, 2000
    u = (np.arange(-2 * fine, (n + 2) * fine) + 0.5) / fine  # from two samples before the first output to after
    k = np.floor(u / spacing).astype(int)
    stair = np.where((u >= 0) & (k < v.size), v[np.clip(k, 0, v.size - 1)], 0.0)
    triangle = 1 - np.abs(np.arange(2 * fine) + 0.5 - fine) / fine
    avg = np.array([np.sum(triangle * stair[(j + 1) * fine : (j + 3) * fine]) / fine for j in range(-1, n + 1)])
    expected = avg[1:-1] + (2 * avg[1:-1] - avg[:-2] - avg[2:]) / 6
    assert np.allclose(grit.hold(v, spacing, n), expected, rtol=0, atol=1e-12)


def tones(y: np.ndarray, g: patch.Grit) -> tuple[np.ndarray, np.ndarray]:
    """Pass y through the sampler and the decimator, and return a steady quarter second's levels, 4 Hz apart."""
    out = family.decimate(family.decimator(BASE), grit.apply(y, g, BASE, INTERNAL), FACTOR)
    steady = out[BASE // 10 : BASE // 10 + BASE // 4]
    return 4.0 * np.arange(steady.size // 2 + 1), np.abs(np.fft.rfft(steady)) / (steady.size / 2)


def test_a_tone_above_half_the_rate_aliases_and_images_as_the_hold_predicts() -> None:
    """Sample 15 kHz at 26.04 kHz: its alias at 11.04 kHz and its image at 15 kHz, at sinc(f / rate), and no more."""
    t = np.arange(round(0.6 * INTERNAL)) / INTERNAL
    f, level = tones(np.sin(2 * math.pi * 15000 * t), SP)
    for hz in (11040.0, 15000.0):
        assert 20 * math.log10(level[round(hz / 4)]) == pytest.approx(
            20 * math.log10(np.sinc(hz / 26040)),
            abs=MATCH_DB,
        )
        level[round(hz / 4) - 2 : round(hz / 4) + 3] = 0.0
    assert 20 * math.log10(np.max(level[f <= PASSBAND_HZ])) < FLOOR_DB


def test_the_input_gain_is_undone_and_drives_the_converter_into_clipping() -> None:
    """Drive a quiet tone 12 dB harder, which keeps its level, and a loud one, which clips at full scale."""
    t = np.arange(round(0.6 * INTERNAL)) / INTERNAL
    quiet = 0.1 * np.sin(2 * math.pi * 1000 * t)
    _, plain = tones(quiet, SP)
    _, hot = tones(quiet, patch.Grit(26040.0, 16, 12.0))
    assert 20 * math.log10(hot[250] / plain[250]) == pytest.approx(0.0, abs=0.001)
    _, clipped = tones(10 * quiet, patch.Grit(26040.0, 16, 12.0))
    assert 20 * math.log10(clipped[750] / clipped[250]) > HARMONIC_DB  # 3 kHz, from the clipped converter


def test_bits_are_whole_and_grit_reads_back(tight: patch.Patch) -> None:
    """Refuse 12.0 bits and 3 bits, and read a patch with grit back the same."""
    p = patch.edit(tight, "grit", SP)
    assert patch.loads(patch.dumps(p)) == p
    data = patch.to_dict(p)
    data["grit"]["bits"] = 12.0
    with pytest.raises(patch.PatchError, match=r"grit\.bits: expected an integer, found a float"):
        patch.from_dict(data)
    data["grit"]["bits"] = 3
    with pytest.raises(patch.PatchError, match=r"grit\.bits: 3 is outside 4 … 16"):
        patch.from_dict(data)
