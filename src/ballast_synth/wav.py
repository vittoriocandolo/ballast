"""A small RIFF writer for mono WAV: IEEE float32, or PCM at 24 or 16 bits.

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

import numpy as np

from .patch import midi_pitch

FLOAT = 3  # WAVE_FORMAT_IEEE_FLOAT
PCM = 1  # WAVE_FORMAT_PCM
PRIVATE = b"blst"  # the chunk that holds the patch text and provenance


def root_note(hz: float) -> tuple[int, int]:
    """smpl unity note and pitch fraction (units of 2⁻³² semitone, always upward).

    The unity note is the note at or below the pitch, except that a pitch less
    than half a cent flat of a note counts as that note, so samplers that
    ignore the fraction are never a semitone off.
    """
    p = midi_pitch(hz)
    note = math.floor(p + 0.005)
    return note, round(max(0.0, p - note) * 2**32)


def save(path, data: bytes) -> None:
    """Write `data` to `path` whole or not at all: to a temporary file beside it, then renamed over it.

    A failed or interrupted write leaves the file as it was.
    """
    target = os.path.realpath(path)  # through a symbolic link, to the file it names
    temporary = f"{target}.{os.getpid()}.tmp"
    try:
        with open(temporary, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(temporary)
        raise


def _chunk(cid: bytes, data: bytes) -> bytes:
    return cid + struct.pack("<I", len(data)) + data + b"\0" * (len(data) % 2)


def write(path, samples: np.ndarray, rate: int, bits: int | None, root: tuple[int, int], meta: dict) -> None:
    """Write float32 samples, or with `bits` (16 or 24) integer samples already quantized."""
    if bits is None:
        chunks = _chunk(b"fmt ", struct.pack("<HHIIHHH", FLOAT, 1, rate, 4 * rate, 4, 32, 0))
        chunks += _chunk(b"fact", struct.pack("<I", samples.size))
        data = samples.astype("<f4").tobytes()
    else:
        top = 2 ** (bits - 1)
        if samples.size and not (-top <= samples.min() and samples.max() < top):
            raise ValueError(f"samples out of range for {bits}-bit PCM")
        width = bits // 8
        chunks = _chunk(b"fmt ", struct.pack("<HHIIHH", PCM, 1, rate, width * rate, width, bits))
        if bits == 16:
            data = samples.astype("<i2").tobytes()
        else:  # 24 bits: the low three bytes of each little-endian 32-bit integer
            data = samples.astype("<i4").view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
    chunks += _chunk(b"smpl", struct.pack("<9I", 0, 0, round(1e9 / rate), *root, 0, 0, 0, 0))
    chunks += _chunk(PRIVATE, json.dumps(meta, ensure_ascii=False, indent=1).encode("utf-8"))
    chunks += _chunk(b"data", data)
    save(path, b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks)


def chunks(path) -> dict[bytes, bytes]:
    """Every top-level chunk of a RIFF WAVE file, by ID (first occurrence wins)."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF WAVE file")
    end = min(8 + struct.unpack_from("<I", data, 4)[0], len(data))
    found: dict[bytes, bytes] = {}
    pos = 12
    while pos + 8 <= end:
        cid, size = data[pos : pos + 4], struct.unpack_from("<I", data, pos + 4)[0]
        found.setdefault(cid, data[pos + 8 : pos + 8 + size])
        pos += 8 + size + size % 2
    return found
