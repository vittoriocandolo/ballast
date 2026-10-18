"""Seeded continuous-time Gaussian noise for the attack, and the dither of PCM exports.

    n(t) = γ Σ_{k=1}^{20000} |H(f_k)| ρ_k cos(2π f_k t + θ_k),    f_k = k Hz

The coefficients are a pure function of the seed. SHAKE-256 over a
domain-separated key yields 16 bytes per component, which become two 53-bit
uniforms and then, by the polar form of Box–Muller, a Rayleigh amplitude ρ_k
and a uniform phase θ_k. Each component is a circular Gaussian, so n(t) is
exactly Gaussian at every t.

|H| is the magnitude of second-order Butterworth high- and low-pass skirts,
each with its half-power point at a band edge, and γ sets the expected mean
square to ½, that of a unit sine. The shaping is zero-phase, so editing the
band reweights the same components instead of rerolling them.

The noise repeats every second, and its samples at an even rate above 40 kHz
come exactly from one inverse real FFT of length rate × 1 s. The same
construction, under its own key, gives the TPDF dither of PCM exports, so a
PCM file is as reproducible as a float one.
"""

from __future__ import annotations

import functools
import hashlib
import math

import numpy as np

COMPONENTS = 20000
KEY = b"ballast/noise/v1"
DITHER_KEY = b"ballast/dither/v1"


def _shake(key: bytes, seed: int, size: int) -> bytes:
    return hashlib.shake_256(key + seed.to_bytes(8, "little")).digest(size)


def coefficients(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Amplitudes ρ_k and phases θ_k for f_k = 1 … 20000 Hz."""
    words = np.frombuffer(_shake(KEY, seed, 16 * COMPONENTS), dtype="<u8").reshape(COMPONENTS, 2) >> np.uint64(11)
    u = (words[:, 0] + np.uint64(1)).astype(float) * 2.0**-53  # (0, 1]
    v = words[:, 1].astype(float) * 2.0**-53  # [0, 1)
    return np.sqrt(-2.0 * np.log(u)), 2.0 * np.pi * v


@functools.lru_cache(maxsize=2)
def table(seed: int, lo: float, hi: float, rate: int) -> np.ndarray:
    """n(j / rate) for j = 0 … rate − 1, in the band with half-power points lo and hi (read-only)."""
    rho, theta = coefficients(seed)
    f = np.arange(1, COMPONENTS + 1, dtype=float)
    power = 1.0 / ((1.0 + (lo / f) ** 4) * (1.0 + (f / hi) ** 4))  # |H(f_k)|²
    amp = math.sqrt(0.5 / math.fsum(power)) * np.sqrt(power) * rho
    spectrum = np.zeros(rate // 2 + 1, dtype=complex)
    spectrum[1 : COMPONENTS + 1] = (rate / 2) * amp * np.exp(1j * theta)
    samples = np.fft.irfft(spectrum, n=rate)
    samples.flags.writeable = False
    return samples


def dither(seed: int, count: int) -> np.ndarray:
    """Triangular (TPDF) dither in LSB for `count` samples.

    Each value is the difference of two 32-bit uniforms on [0, 1), so it lies in
    (−1, 1) with a triangular density: the dither that makes the first two
    moments of the quantization error independent of the signal. The sequence
    is prefix-stable, like the noise.
    """
    u = np.frombuffer(_shake(DITHER_KEY, seed, 8 * count), dtype="<u4").reshape(count, 2).astype(float) * 2.0**-32
    return u[:, 0] - u[:, 1]
