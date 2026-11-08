# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""WAV files: a small RIFF writer for mono IEEE float32 or PCM at 24 or 16 bits, and a reader for PCM and float WAVs.

Files carry fmt, fact (float only), smpl (so samplers map the root key), a
private chunk with the patch text and provenance, and data. PCM uses the
plain PCM format tag, which hardware samplers read most widely. RIFF readers
skip chunks they do not know.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .patch import midi_pitch

if TYPE_CHECKING:
    from numpy.typing import NDArray

FLOAT = 3  # WAVE_FORMAT_IEEE_FLOAT
PCM = 1  # WAVE_FORMAT_PCM
EXTENSIBLE = 0xFFFE  # WAVE_FORMAT_EXTENSIBLE: the real tag opens the subformat GUID
FMT, EXTENSIBLE_FMT, EXTENSIBLE_TAG = 16, 40, 24  # the plain and extensible fmt chunks' sizes, and where the tag lies
PCM8, PCM24 = 8, 24  # unsigned bytes, and three bytes a sample
SAMPLES = {(PCM, 8): "u1", (PCM, 16): "<i2", (PCM, 24): "u1", (PCM, 32): "<i4", (FLOAT, 32): "<f4", (FLOAT, 64): "<f8"}
PRIVATE = b"blst"  # the chunk that holds the patch text and provenance
LAYOUTS = {1: "mono", 2: "stereo"}  # the channel counts of the WAVs Ballast takes in


@dataclass(frozen=True)
class Format:
    """How a file stores its samples: the sample rate, and the PCM bit depth, or None for float32."""

    rate: int
    bits: int | None = None


def root_note(hz: float) -> tuple[int, int]:
    """Return the smpl unity note and pitch fraction (units of 2⁻³² semitone, always upward).

    The unity note is the note at or below the pitch, except that a pitch less
    than half a cent flat of a note counts as that note, so samplers that
    ignore the fraction are never a semitone off.
    """
    pitch = midi_pitch(hz)
    note = math.floor(pitch + 0.005)
    return note, round(max(0.0, pitch - note) * 2**32)


def save(path: str | os.PathLike[str], data: bytes) -> None:
    """Write `data` to `path` whole or not at all: to a temporary file beside it, then renamed over it.

    A failed or interrupted write leaves the file as it was.
    """
    target = os.path.realpath(path)  # through a symbolic link, to the file it names
    temporary = Path(f"{target}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        temporary.replace(target)
    except BaseException:
        with contextlib.suppress(OSError):
            temporary.unlink()
        raise


def _chunk(cid: bytes, data: bytes) -> bytes:
    return cid + struct.pack("<I", len(data)) + data + b"\0" * (len(data) % 2)


def write(
    path: str | os.PathLike[str],
    samples: NDArray[np.float32] | NDArray[np.int32],
    fmt: Format,
    root: tuple[int, int],
    meta: dict[str, object],
) -> None:
    """Write a WAV file of float32 samples, or of integer samples already quantized to `fmt.bits`, 16 or 24, for PCM."""
    if fmt.bits is None:
        chunks = _chunk(b"fmt ", struct.pack("<HHIIHHH", FLOAT, 1, fmt.rate, 4 * fmt.rate, 4, 32, 0))
        chunks += _chunk(b"fact", struct.pack("<I", samples.size))
        data = samples.astype("<f4").tobytes()
    else:
        full_scale = 2 ** (fmt.bits - 1)
        if samples.size and not (-full_scale <= np.min(samples) and np.max(samples) < full_scale):
            msg = f"samples out of range for {fmt.bits}-bit PCM"
            raise ValueError(msg)
        width = fmt.bits // 8
        chunks = _chunk(b"fmt ", struct.pack("<HHIIHH", PCM, 1, fmt.rate, width * fmt.rate, width, fmt.bits))
        # The low `width` bytes of each little-endian 32-bit integer.
        data = samples.astype("<i4").view(np.uint8).reshape(-1, 4)[:, :width].tobytes()
    chunks += _chunk(b"smpl", struct.pack("<9I", 0, 0, round(1e9 / fmt.rate), *root, 0, 0, 0, 0))
    chunks += _chunk(PRIVATE, json.dumps(meta, ensure_ascii=False, indent=1).encode("utf-8"))
    chunks += _chunk(b"data", data)
    save(path, b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks)


def read(path: str | os.PathLike[str]) -> tuple[int, NDArray[np.float64]]:
    """Read a WAV file's sample rate and its frames, one row per frame, as floats with full scale at ±1.

    It may hold PCM of 8, 16, 24 or 32 bits or floats of 32 or 64, plain or
    extensible. Every size is checked against what the file holds, so a
    damaged or crafted file is refused with a ValueError, never misread.
    """
    found = chunks(path)
    fmt, data = found.get(b"fmt "), found.get(b"data")
    if fmt is None or data is None:
        msg = "missing fmt or data chunk"
        raise ValueError(msg)
    if len(fmt) < FMT:
        msg = "the fmt chunk is cut short"
        raise ValueError(msg)
    tag, channels, rate, _, _, bits = struct.unpack_from("<HHIIHH", fmt)
    if tag == EXTENSIBLE and len(fmt) >= EXTENSIBLE_FMT:
        tag = struct.unpack_from("<H", fmt, EXTENSIBLE_TAG)[0]
    if (tag, bits) not in SAMPLES or not channels or not rate:
        msg = f"unsupported format: tag {tag}, {bits}-bit samples, {channels} channels at {rate} Hz"
        raise ValueError(msg)
    width = bits // 8
    raw = data[: len(data) // (width * channels) * (width * channels)]  # whole frames; a cut tail is dropped
    if bits == PCM24:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = ((b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)) << 8 >> 8).astype(float)  # sign-extend from bit 23
    else:
        values = np.frombuffer(raw, dtype=SAMPLES[tag, bits]).astype(float)
    if tag == PCM:
        values = (values - 128.0) / 128.0 if bits == PCM8 else values / float(2 ** (bits - 1))
    return rate, values.reshape(-1, channels)


def chunks(path: str | os.PathLike[str]) -> dict[bytes, bytes]:
    """Read every top-level chunk of a RIFF WAVE file, by ID (first occurrence wins)."""
    data = Path(path).read_bytes()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        msg = "not a RIFF WAVE file"
        raise ValueError(msg)
    end = min(8 + struct.unpack_from("<I", data, 4)[0], len(data))
    found: dict[bytes, bytes] = {}
    pos = 12
    while pos + 8 <= end:
        cid, size = data[pos : pos + 4], struct.unpack_from("<I", data, pos + 4)[0]
        found.setdefault(cid, data[pos + 8 : pos + 8 + size])
        pos += 8 + size + size % 2
    return found
