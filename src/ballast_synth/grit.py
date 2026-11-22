# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Grit: a low-rate sampler, modelled on purpose, aliasing and all.

The sampler takes the kick as an analog signal, band-limited to 20 kHz by the
family's frozen filter, down to the base rate and back up. It samples it at its
own rate, with no filter against aliasing; rounds each sample to its bit depth,
without dither, clipping at its full scale; and plays the samples back through
a zero-order hold, the staircase a converter makes. Its aliasing and staircase
are computed exactly: Ballast's own sampling of the staircase does not add to
them.

At internal rate F, each output is the staircase's exact average under the
triangle from one sample before to one after, centred on its own, as the
drive averages its curve, with the same 3-tap correction to keep the
passband flat. The steps lie at least two internal samples apart, so a
triangle meets at most one of them. `input_db` sets the level into the
converter, and the output is brought back by as much.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from . import family
from .voice import lagrange

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .patch import Grit


def heard(y: NDArray[np.float64], base: int) -> NDArray[np.float64]:
    """Band-limit the internal signal to 20 kHz, as the sampler hears it: to the base rate and back up."""
    taps = family.decimator(base)
    _, _, factor = family.rates(base)
    return family.interpolate(taps, family.decimate(taps, y, factor), factor)


def convert(x: NDArray[np.float64], bits: int, gain: float) -> NDArray[np.float64]:
    """Round to `bits`, without dither, clipping at the converter's full scale, after `gain`."""
    scale = 2.0 ** (bits - 1)
    codes = np.clip(np.round(gain * x * scale), -scale, scale - 1)
    out: NDArray[np.float64] = codes / scale
    return out


def hold(v: NDArray[np.float64], spacing: float, n: int) -> NDArray[np.float64]:
    """Average the staircase of the samples v, `spacing` internal samples apart, under each output's triangle.

    Output j covers [j - 1, j + 1] with weights 1 - |u - j|. The step after it lies at p = (k + 1) · spacing,
    where k = ⌊j / spacing⌋ holds at j, and gives the triangle's part beyond it, (j + 1 - p)² / 2, to the next
    sample; the step before, at k · spacing, gives the part before it, (k · spacing - j + 1)² / 2, to the last.
    """
    j = np.arange(-1, n + 1, dtype=float)  # one more on each side, for the correction
    k = np.floor(j / spacing)
    after = np.clip(j + 1 - (k + 1) * spacing, 0.0, None) ** 2 / 2
    before = np.clip(k * spacing - j + 1, 0.0, None) ** 2 / 2
    padded = np.concatenate(([0.0, 0.0], v, [0.0, 0.0]))  # silence before the first sample and after the last
    i = k.astype(np.int64) + 2  # k runs from -1 to ⌈n / spacing⌉, all within the padding
    y = (1 - after - before) * padded[i] + after * padded[i + 1] + before * padded[i - 1]
    out: NDArray[np.float64] = y[1:-1] + (2 * y[1:-1] - y[:-2] - y[2:]) / 6
    return out


def apply(y: NDArray[np.float64], g: Grit, base: int, rate: int) -> NDArray[np.float64]:
    """Pass the internal signal through the sampler, which runs on for as long as its input filter rings."""
    x = heard(y, base)
    spacing = rate / g.rate_hz
    count = math.ceil(x.size / spacing)
    gain = 10.0 ** (g.input_db / 20.0)
    v = convert(lagrange(x, np.arange(count, dtype=float) * spacing), g.bits, gain)
    out: NDArray[np.float64] = hold(v, spacing, x.size) / gain
    return out
