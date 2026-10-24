# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The room: a seeded, decaying tail of Gaussian noise that the kick is heard in.

    h(t) = Σ_k ρ_k cos(2π f_k t + θ_k) · e^(−t ln 1000 / T(f_k)),    T(f) = decay / (1 + f / damping)

Each frequency fades 60 dB in its own time T(f), so the tail darkens as it
fades, as air and walls darken a real room's. The amplitudes and phases come
from SHAKE-256 under `ballast/room/v1` and the patch's seed, as the attack's
noise does, spaced 1/S Hz apart over S whole seconds, so the tail never
repeats within itself. Rather than fade every component on its own, h fades
octave bands, each with T at its center, and weights every component across
its two nearest bands, linearly in octaves: the weights sum to exactly 1, so
h starts as plain white noise.

The tail is built at the family's base rate and brought up to the internal
rate by the family's frozen filter, as a transient is, so every rate of a
family shares it, and normalized to unit energy, so that 0 dB puts its energy
level with the kick's. It lasts ROOM_TAIL × decay, to -90 dB, and fades out
over its last tenth.
"""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import fftconvolve, upfirdn

from . import family, noise
from .patch import ROOM_TAIL

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .patch import Room

ROOM_KEY = b"ballast/room/v1"
CENTERS = 1000.0 * 2.0 ** np.arange(-4, 5)  # octave bands from 62.5 Hz to 16 kHz
TOP_HZ = 20000.0  # the tail's components reach up to here
FADE = 0.1  # the share of the tail over which it fades out


@functools.lru_cache(maxsize=4)
def impulse(seed: int, decay_ms: float, damping_hz: float, base: int) -> NDArray[np.float64]:
    """Compute the tail at its family's internal rate, with unit energy (read-only)."""
    length_s = ROOM_TAIL * decay_ms / 1000
    seconds = math.ceil(length_s)
    count = round(TOP_HZ * seconds)
    rho, theta = noise.coefficients(seed, ROOM_KEY, count)
    f = np.arange(1, count + 1) / seconds  # 1/S Hz apart
    octave = np.clip(np.log2(f / CENTERS[0]), 0.0, CENTERS.size - 1.0)  # each component's place among the bands
    n = round(length_s * base)
    t = np.arange(n) / base
    h = np.zeros(n)
    for b, center in enumerate(CENTERS):
        weight = np.clip(1.0 - np.abs(octave - b), 0.0, None)  # linear in octaves, summing to 1 over the bands
        spectrum = np.zeros(base * seconds // 2 + 1, dtype=complex)
        spectrum[1 : count + 1] = weight * rho * np.exp(1j * theta)
        band = np.fft.irfft(spectrum, n=base * seconds)[:n]
        h += band * np.exp(-t * math.log(1000.0) / (decay_ms / 1000 / (1 + center / damping_hz)))
    fade = round(FADE * n)
    h[n - fade :] *= np.cos(np.linspace(0.0, math.pi / 2, fade)) ** 2  # to exactly 0 at its last sample
    _, _, factor = family.rates(base)
    up: NDArray[np.float64] = upfirdn(factor * family.decimator(base), h, up=factor)
    up /= math.sqrt(math.fsum(up * up))
    up.flags.writeable = False
    return up


def add(y: NDArray[np.float64], r: Room, seed: int, base: int, rate: int) -> NDArray[np.float64]:
    """Add the room to the internal signal y, which runs on long enough to hold its tail."""
    h = impulse(seed, r.decay_ms, r.damping_hz, base)
    delay = round(r.predelay_ms * rate / 1000)
    wet = fftconvolve(y, h)[: max(0, y.size - delay)]
    out = y.copy()
    out[delay : delay + wet.size] += 10.0 ** (r.level_db / 20.0) * wet
    return out
