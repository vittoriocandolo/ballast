# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Design the frozen decimators, check them against their specification, and write them as data.

    uv run python tools/decimators.py

Ballast never designs a filter while it renders: each family's decimator ships as data, checked against its
SHA-256, so the sound cannot drift with SciPy. This script records how they were made. Each one is a
Kaiser-window linear-phase low-pass, made minimum phase by the homomorphic method at full length, with its taps
scaled to sum to exactly 1. Another SciPy may round the design differently in the last bits: the shipped data,
not this script, defines the sound.
"""

from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import firwin, kaiserord, minimum_phase

from ballast_synth import family

if TYPE_CHECKING:
    from numpy.typing import NDArray

DATA = Path(__file__).resolve().parent.parent / "src" / "ballast_synth" / "data"
PASSBAND_HZ = 20000.0
PROTOTYPE_DB = 117.0  # the linear-phase prototype's attenuation; the minimum-phase filter keeps more than 116 dB
N_FFT = 1 << 21  # the homomorphic method's FFT length: a shorter one leaves more ripple in the passband
RIPPLE_DB = 0.001  # the specification: the passband flat within ±0.001 dB,
STOPBAND_DB = 110.0  # and everything from half the base rate up at least 110 dB down


def design(base: int) -> NDArray[np.float64]:
    """Design the minimum-phase decimator of a family, with taps that sum to exactly 1."""
    rate = base * family.OVERSAMPLING
    numtaps, beta = kaiserord(PROTOTYPE_DB, (base / 2 - PASSBAND_HZ) / (rate / 2))
    prototype = firwin(numtaps | 1, (PASSBAND_HZ + base / 2) / 2, window=("kaiser", beta), fs=rate)
    taps: NDArray[np.float64] = minimum_phase(prototype, method="homomorphic", n_fft=N_FFT, half=False)
    taps = taps / math.fsum(taps)
    taps[np.argmax(taps)] += 1.0 - math.fsum(taps)  # whatever rounding left over, on the largest tap
    if math.fsum(taps) != 1.0:
        msg = f"the {base} Hz family's taps do not sum to exactly 1"
        raise RuntimeError(msg)
    return taps


def measure(taps: NDArray[np.float64], base: int) -> tuple[float, float, float]:
    """Measure the passband ripple and the stopband attenuation in dB, and the ms that hold 99% of the energy."""
    n = 1 << 22
    f = np.fft.rfftfreq(n, 1 / (base * family.OVERSAMPLING))
    db = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(taps, n)), 1e-300))
    energy = np.cumsum(taps * taps) / np.sum(taps * taps)
    energy99_ms = (np.searchsorted(energy, 0.99) + 1) * 1000 / (base * family.OVERSAMPLING)
    return float(np.max(np.abs(db[f <= PASSBAND_HZ]))), float(-np.max(db[f >= base / 2])), float(energy99_ms)


def main() -> None:
    """Design both families' decimators and compare them with the shipped data, which --force replaces."""
    for base in sorted(set(family.FAMILY.values())):
        taps = design(base)
        ripple, stopband, energy99_ms = measure(taps, base)
        if ripple > RIPPLE_DB or stopband < STOPBAND_DB:
            msg = f"the {base} Hz family misses its specification: ±{ripple} dB, {stopband} dB"
            raise SystemExit(msg)
        raw = taps.astype("<f8").tobytes()
        path = DATA / f"decimator-{base}x{family.OVERSAMPLING}.f64"
        if not path.exists() or "--force" in sys.argv:
            path.write_bytes(raw)
            verdict = "written"
        else:
            verdict = "matches the shipped data" if path.read_bytes() == raw else "differs from the shipped data"
        sys.stdout.write(
            f"{path.name}: {taps.size} taps, ±{ripple:.7f} dB to 20 kHz, {stopband:.2f} dB down from {base / 2:g} Hz,"
            f" 99% of the energy in {energy99_ms:.3f} ms, sha256 {hashlib.sha256(raw).hexdigest()}; {verdict}\n",
        )


if __name__ == "__main__":
    main()
