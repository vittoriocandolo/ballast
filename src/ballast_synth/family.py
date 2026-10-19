# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The rate families: the base and internal rates of every sample rate, and each family's frozen filter.

Every sample rate belongs to the family of 44.1 or 48 kHz. A family computes
at the internal rate F = 32 × its base rate, and one frozen minimum-phase FIR
per family takes F down to each of its rates and brings a transient up from
them. The filters ship as data, so the sound cannot drift with SciPy, and are
refused unless they match their checksums.
"""

from __future__ import annotations

import functools
import hashlib
from importlib import resources
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

OVERSAMPLING = 32  # the internal rate F is 32 × the base rate
FAMILY = {44100: 44100, 88200: 44100, 176400: 44100, 48000: 48000, 96000: 48000, 192000: 48000}
# Designed by tools/decimators.py: flat to 0.001 dB up to 20 kHz, at least 110 dB down from base/2, taps summing
# to exactly 1.
DECIMATORS = {
    44100: ("decimator-44100x32.f64", "f74810b0e955e0aada486f9f064b82435ea11a64e155afd2905c29d99e662efd"),
    48000: ("decimator-48000x32.f64", "229115e8e64a113a77c5c008c7cd2272c87431cd9b682007de8b21b47d24b41d"),
}


def rates(sample_rate: int) -> tuple[int, int, int]:
    """Return the family's base rate, the internal rate F and the factor K = F / rate for a sample rate."""
    base = FAMILY[sample_rate]
    internal = base * OVERSAMPLING
    return base, internal, internal // sample_rate


@functools.cache
def decimator(base: int) -> NDArray[np.float64]:
    """Load the family's frozen taps, refused unless they match their checksum."""
    name, sha256 = DECIMATORS[base]
    raw = resources.files("ballast_synth").joinpath("data", name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        msg = f"{name} does not match its checksum"
        raise RuntimeError(msg)
    return np.frombuffer(raw, dtype="<f8")  # read-only, like the bytes it views
