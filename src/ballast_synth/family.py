# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The rate families: the base and internal rates of every sample rate, each family's frozen filter, and its use.

Every sample rate belongs to the family of 44.1 or 48 kHz. A family computes
at the internal rate F = 32 × its base rate, and one frozen minimum-phase FIR
per family takes F down to each of its rates and brings a transient up from
them. The filters ship as data, so the sound cannot drift with SciPy, and are
refused unless they match their checksums.
"""

from __future__ import annotations

import functools
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from importlib import resources
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import upfirdn

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


CHUNK = 2048  # outputs per parallel chunk, at the lower rate when decimating
_POOLS: dict[int, ThreadPoolExecutor] = {}  # by process ID


def pool() -> ThreadPoolExecutor:
    """Return this process's threads for the chunks, which upfirdn computes with the GIL released.

    A process forked after a render, as multiprocessing's default start method on Linux before Python 3.14 does,
    inherits its parent's pool but none of its threads, and would wait on it forever; so each process makes its own.
    """
    pid = os.getpid()
    if pid not in _POOLS:
        _POOLS[pid] = ThreadPoolExecutor(os.cpu_count())
    return _POOLS[pid]


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


def decimate(taps: NDArray[np.float64], y: NDArray[np.float64], factor: int) -> NDArray[np.float64]:
    """Compute upfirdn(taps, y, down=factor) in parallel chunks that match it bit for bit.

    Each output of upfirdn is one sequential sum over its own inputs. A chunk is
    given every input its outputs need, starting on the factor's grid, so each
    output is computed by exactly the same operations as in a single call.
    """
    n_out = (y.size + taps.size - 2) // factor + 1

    def chunk(m0: int) -> NDArray[np.float64]:
        m1 = min(m0 + CHUNK, n_out)
        start = max(0, m0 * factor - (taps.size - 1)) // factor * factor
        # Since the filter is longer than the factor, the last chunk's slice reaches the end of y.
        out: NDArray[np.float64] = upfirdn(taps, y[start : (m1 - 1) * factor + 1], down=factor)
        return out[m0 - start // factor : m1 - start // factor]

    return np.concatenate(list(pool().map(chunk, range(0, n_out, CHUNK))))


def interpolate(taps: NDArray[np.float64], x: NDArray[np.float64], factor: int) -> NDArray[np.float64]:
    """Compute upfirdn(factor * taps, x, up=factor) in parallel chunks that match it bit for bit.

    Output n sums taps[n - j K] x[j] over the inputs j it reaches. A chunk is given every input its outputs reach,
    from the first on the factor's grid, so each output is computed by exactly the same operations as in a single
    call.
    """
    scaled = factor * taps
    n_out = (x.size - 1) * factor + taps.size
    step = CHUNK * factor

    def chunk(n0: int) -> NDArray[np.float64]:
        n1 = min(n0 + step, n_out)
        j0 = max(0, -(-(n0 - taps.size + 1) // factor))  # the first input that reaches output n0
        out: NDArray[np.float64] = upfirdn(scaled, x[j0 : (n1 - 1) // factor + 1], up=factor)
        return out[n0 - j0 * factor : n1 - j0 * factor]

    return np.concatenate(list(pool().map(chunk, range(0, n_out, step))))
