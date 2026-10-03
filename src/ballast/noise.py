"""Seeded continuous-time Gaussian noise for the attack (plan §2.4, §6).

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
come exactly from one inverse real FFT of length rate × 1 s.

The same construction, under its own key, gives the TPDF dither of PCM
exports, so a PCM file is as reproducible as a float one.
"""

from __future__ import annotations

import functools
import hashlib
import math

import numpy as np

COMPONENTS = 20000
KEY = b"kick/noise/v1"
DITHER_KEY = b"kick/dither/v1"


def uniforms(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The two 53-bit integers per component, from SHAKE-256 of the key and seed."""
    if not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer from 0 to 2**63 - 1")
    stream = hashlib.shake_256(KEY + seed.to_bytes(8, "little")).digest(16 * COMPONENTS)
    words = np.frombuffer(stream, dtype="<u8").reshape(COMPONENTS, 2) >> np.uint64(11)
    return words[:, 0], words[:, 1]


def coefficients(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Amplitudes ρ_k and phases θ_k for f_k = 1 … 20000 Hz."""
    a, b = uniforms(seed)
    u = (a + np.uint64(1)).astype(float) * 2.0**-53  # (0, 1]
    v = b.astype(float) * 2.0**-53  # [0, 1)
    return np.sqrt(-2.0 * np.log(u)), 2.0 * np.pi * v


def band_power(lo: float, hi: float) -> np.ndarray:
    """|H(f_k)|² for the band with half-power points lo and hi."""
    f = np.arange(1, COMPONENTS + 1, dtype=float)
    return 1.0 / ((1.0 + (lo / f) ** 4) * (1.0 + (f / hi) ** 4))


@functools.lru_cache(maxsize=8)
def amplitudes(seed: int, lo: float, hi: float) -> tuple[np.ndarray, np.ndarray]:
    """Calibrated component amplitudes γ|H(f_k)|ρ_k and phases θ_k (read-only)."""
    rho, theta = coefficients(seed)
    power = band_power(lo, hi)
    gamma = math.sqrt(0.5 / math.fsum(power))
    amp = gamma * np.sqrt(power) * rho
    amp.flags.writeable = theta.flags.writeable = False
    return amp, theta


def at(seed: int, lo: float, hi: float, t, derivative: bool = False) -> np.ndarray:
    """n(t), or n′(t), at arbitrary times, by summing the components (for a few points)."""
    amp, theta = amplitudes(seed, lo, hi)
    k = np.arange(1, COMPONENTS + 1, dtype=float)
    phase = 2 * np.pi * np.mod(np.outer(np.atleast_1d(t), k), 1.0) + theta
    if derivative:
        return -(np.sin(phase) @ (2 * np.pi * k * amp))
    return np.cos(phase) @ amp


def dither(seed: int, count: int) -> np.ndarray:
    """Triangular (TPDF) dither in LSB for `count` samples, from SHAKE-256 of its own key and the seed.

    Each value is the difference of two 32-bit uniforms on [0, 1), so it lies in
    (−1, 1) with a triangular density: the dither that makes the first two
    moments of the quantization error independent of the signal. The sequence
    is prefix-stable, like the noise.
    """
    if not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer from 0 to 2**63 - 1")
    stream = hashlib.shake_256(DITHER_KEY + seed.to_bytes(8, "little")).digest(8 * count)
    u = np.frombuffer(stream, dtype="<u4").reshape(count, 2).astype(float) * 2.0**-32
    return u[:, 0] - u[:, 1]


def table(seed: int, lo: float, hi: float, rate: int) -> np.ndarray:
    """n(j / rate) for j = 0 … rate − 1."""
    if rate % 2 or rate <= 2 * COMPONENTS:
        raise ValueError("the noise table needs an even rate above 40 kHz")
    amp, theta = amplitudes(seed, lo, hi)
    spectrum = np.zeros(rate // 2 + 1, dtype=complex)
    spectrum[1 : COMPONENTS + 1] = (rate / 2) * amp * np.exp(1j * theta)
    return np.fft.irfft(spectrum, n=rate)
