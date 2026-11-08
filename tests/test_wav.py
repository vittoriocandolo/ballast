# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The WAV files: float32 or PCM that other readers agree on, the root note, and the embedded patch."""

import json
import os
import re
import struct
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.io import wavfile

from ballast_synth import family, patch, render, wav

RATE = 48000
CD_RATE = 44100
ROOT = (36, 0)  # C2, exactly

# SciPy skips the smpl and blst chunks, as RIFF readers should, and says so.
pytestmark = pytest.mark.filterwarnings("ignore::scipy.io.wavfile.WavFileWarning")


def test_float_samples_read_back_exactly(tmp_path: Path) -> None:
    """Write float32 samples, and read the same rate and samples back with SciPy."""
    samples = np.linspace(-1, 1, 1001, dtype=np.float32)
    wav.write(tmp_path / "f.wav", samples, wav.Format(RATE), ROOT, {})
    rate, data = wavfile.read(tmp_path / "f.wav")
    assert rate == RATE
    assert data.dtype == np.float32
    assert np.array_equal(data, samples)


@pytest.mark.parametrize("bits", [16, 24])
def test_pcm_samples_read_back_exactly(tmp_path: Path, bits: int) -> None:
    """Write PCM samples up to both ends of the range, and read them back with SciPy."""
    full_scale = 2 ** (bits - 1)
    q = np.array([-full_scale, -full_scale + 1, -1, 0, 1, 1234, full_scale - 1], dtype=np.int32)
    wav.write(tmp_path / "p.wav", q, wav.Format(CD_RATE, bits), ROOT, {})
    rate, data = wavfile.read(tmp_path / "p.wav")
    assert rate == CD_RATE
    # SciPy left-justifies samples in its integer type: 24-bit ones in int32.
    assert np.array_equal(data.astype(np.int64), q << (8 * data.dtype.itemsize - bits))


@pytest.mark.parametrize("bits", [None, 24, 16])
def test_read_gives_floats_with_full_scale_at_one(tmp_path: Path, bits: int | None) -> None:
    """Write samples from full scale to full scale in each format Ballast writes, and read them back as floats."""
    full_scale = 1.0 if bits is None else 2.0 ** (bits - 1)
    steps = np.array([-full_scale, -full_scale / 2, 0.0, full_scale / 4, full_scale - 1])
    samples = steps.astype(np.float32) if bits is None else steps.astype(np.int32)
    wav.write(tmp_path / "x.wav", samples, wav.Format(CD_RATE, bits), ROOT, {})
    rate, frames = wav.read(tmp_path / "x.wav")
    assert (rate, frames.shape, frames.dtype) == (CD_RATE, (5, 1), np.dtype(np.float64))
    assert np.array_equal(frames[:, 0], steps / full_scale)


def test_read_keeps_the_channels_of_each_frame(tmp_path: Path) -> None:
    """Read a stereo file of 8-bit PCM, one row per frame, with the channels in order."""
    wavfile.write(tmp_path / "x.wav", RATE, np.array([[0, 255], [128, 192]], dtype=np.uint8))
    assert np.array_equal(wav.read(tmp_path / "x.wav")[1], [[-1.0, 127 / 128], [0.0, 0.5]])


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (np.array([[0, 255], [128, 64]], np.uint8), [[-1.0, 127 / 128], [0.0, -0.5]]),
        (np.array([[-(2**15), 2**15 - 1], [0, -(2**14)]], np.int16), [[-1.0, 1 - 2.0**-15], [0.0, -0.5]]),
        (np.array([[-(2**31), 2**31 - 1], [0, -(2**30)]], np.int32), [[-1.0, 1 - 2.0**-31], [0.0, -0.5]]),
        (np.array([[-1.0, 0.25], [0.5, -0.125]], np.float32), [[-1.0, 0.25], [0.5, -0.125]]),
        (np.array([[-1.0, 0.25], [0.5, 1e-300]], np.float64), [[-1.0, 0.25], [0.5, 1e-300]]),
    ],
)
def test_read_decodes_what_scipy_writes(tmp_path: Path, data: NDArray[Any], expected: list[list[float]]) -> None:
    """Read stereo PCM of 8, 16 and 32 bits and floats of 32 and 64, as SciPy writes them, to full scale at one."""
    wavfile.write(tmp_path / "x.wav", RATE, data)
    rate, frames = wav.read(tmp_path / "x.wav")
    assert rate == RATE
    assert np.array_equal(frames, expected)


def test_read_takes_the_tag_of_an_extensible_file(tmp_path: Path) -> None:
    """Read an extensible fmt chunk by the tag that opens its subformat GUID."""
    fmt = struct.pack("<HHIIHHHHI", 0xFFFE, 1, RATE, 4 * RATE, 4, 32, 22, 32, 4)
    fmt += struct.pack("<H", wav.FLOAT) + bytes(14)
    data = np.array([0.5, -0.25], dtype="<f4").tobytes()
    body = b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
    (tmp_path / "x.wav").write_bytes(b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body)
    assert np.array_equal(wav.read(tmp_path / "x.wav")[1][:, 0], [0.5, -0.25])


@pytest.mark.parametrize(
    ("fmt", "data", "message"),
    [
        (None, b"\0\0", "missing fmt or data chunk"),
        (struct.pack("<HHIIHH", 1, 1, 48000, 96000, 2, 16), None, "missing fmt or data chunk"),
        (b"\1\0", b"\0\0", "the fmt chunk is cut short"),
        (struct.pack("<HHIIHH", 1, 0, 48000, 0, 0, 16), b"\0\0", "0 channels"),
        (struct.pack("<HHIIHH", 1, 1, 0, 0, 2, 16), b"\0\0", "at 0 Hz"),
        (struct.pack("<HHIIHH", 3, 1, 48000, 0, 2, 16), b"\0\0", "tag 3, 16-bit samples"),
        (struct.pack("<HHIIHH", 1, 1, 48000, 0, 2, 12), b"\0\0", "tag 1, 12-bit samples"),
    ],
)
def test_read_refuses_undecodable_files(tmp_path: Path, fmt: bytes | None, data: bytes | None, message: str) -> None:
    """Refuse a file missing its format or samples, a format cut short, and formats no WAV reader agrees on."""
    body = b"".join(
        cid + struct.pack("<I", len(c)) + c + b"\0" * (len(c) % 2)
        for cid, c in ((b"fmt ", fmt), (b"data", data))
        if c is not None
    )
    (tmp_path / "x.wav").write_bytes(b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body)
    with pytest.raises(ValueError, match=re.escape(message)):
        wav.read(tmp_path / "x.wav")


def test_pcm_out_of_range_is_refused(tmp_path: Path) -> None:
    """Refuse a sample one step beyond the 16-bit range."""
    with pytest.raises(ValueError, match="out of range"):
        wav.write(tmp_path / "p.wav", np.array([2**15], dtype=np.int32), wav.Format(RATE, 16), ROOT, {})


def test_an_exported_render_carries_its_patch_and_root_note(tmp_path: Path, tight_text: str) -> None:
    """Export a render as 24-bit PCM, and read back its patch text, provenance, root note and samples."""
    bits = 24
    text = tight_text.replace("seed = 1", "seed = 1  # my favourite kick")  # the text is kept as written
    p = patch.loads(text)
    audio, report = render.render(p, bits)
    render.export(tmp_path / "kick.wav", p, text, audio, report)
    found = wav.chunks(tmp_path / "kick.wav")
    meta = json.loads(found[wav.PRIVATE])
    assert meta["patch"] == text
    assert meta["pcm"]["bits"] == bits
    assert meta["decimator"]["sha256"] == family.DECIMATORS[RATE][1]
    assert struct.unpack("<9I", found[b"smpl"])[3:5] == wav.root_note(55.0)
    rate, data = wavfile.read(tmp_path / "kick.wav")
    assert rate == RATE
    assert np.array_equal(data >> 8, render.quantize(audio, bits, p.seed))


@pytest.mark.parametrize(
    ("hz", "note"),
    [
        (440.0, (69, 0)),
        (440.0 * 2 ** (-0.4 / 1200), (69, 0)),  # less than half a cent flat counts as the note itself
        (440.0 * 2 ** (-0.6 / 1200), (68, round(0.994 * 2**32))),
        (440.0 * 2 ** (50 / 1200), (69, 2**31)),
    ],
)
def test_root_note(hz: float, note: tuple[int, int]) -> None:
    """Find the unity note and the upward fraction that the smpl chunk holds for a frequency."""
    unity, fraction = wav.root_note(hz)
    assert unity == note[0]
    assert fraction == pytest.approx(note[1], abs=2**12)


def test_chunks_refuses_what_is_not_riff_wave(tmp_path: Path) -> None:
    """Refuse to read the chunks of a file that is not a RIFF WAVE file."""
    (tmp_path / "x.wav").write_bytes(b"not a wave file")
    with pytest.raises(ValueError, match="not a RIFF WAVE file"):
        wav.chunks(tmp_path / "x.wav")


def test_save_replaces_a_file_whole_or_not_at_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace a file, then interrupt the next replacement, which leaves the file and no temporary behind."""
    path = tmp_path / "kick.toml"
    path.write_bytes(b"the old patch")
    wav.save(path, b"the new patch")
    assert path.read_bytes() == b"the new patch"

    def interrupted(_src: object, _dst: object) -> NoReturn:
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        wav.save(path, b"a patch that never lands")
    assert path.read_bytes() == b"the new patch"
    assert [p.name for p in tmp_path.iterdir()] == ["kick.toml"]


def riff_chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    """Split a RIFF WAVE file into its chunks, by the spec, independently of the reader under test."""
    assert data[:4] == b"RIFF"
    assert data[8:12] == b"WAVE"
    assert struct.unpack_from("<I", data, 4)[0] == len(data) - 8  # the RIFF size counts every byte after it
    found, pos = [], 12
    while pos < len(data):
        cid, size = data[pos : pos + 4], struct.unpack_from("<I", data, pos + 4)[0]
        found.append((cid, data[pos + 8 : pos + 8 + size]))
        pos += 8 + size + size % 2
    assert pos == len(data)
    return found


@pytest.mark.parametrize("bits", [None, 24, 16])
def test_every_header_field_is_what_the_spec_asks(tmp_path: Path, bits: int | None) -> None:
    """Write each format and find every field of fmt, fact and smpl as WAVE defines it, and the data whole."""
    samples = np.array([0.5, -0.25, 0.125], np.float32) if bits is None else np.array([0x123456 >> (24 - bits), -7, 0])
    meta: dict[str, object] = {"patch": "# café → kick\n"}
    wav.write(tmp_path / "x.wav", samples, wav.Format(CD_RATE, bits), (37, 123), meta)
    chunks = dict(riff_chunks((tmp_path / "x.wav").read_bytes()))
    width = 4 if bits is None else bits // 8
    if bits is None:
        assert chunks[b"fmt "] == struct.pack("<HHIIHHH", 3, 1, CD_RATE, 4 * CD_RATE, 4, 32, 0)  # with cbSize 0
        assert chunks[b"fact"] == struct.pack("<I", samples.size)
        assert chunks[b"data"] == samples.astype("<f4").tobytes()
    else:
        assert chunks[b"fmt "] == struct.pack("<HHIIHH", 1, 1, CD_RATE, width * CD_RATE, width, bits)
        assert b"fact" not in chunks
        assert chunks[b"data"] == b"".join(int(v).to_bytes(width, "little", signed=True) for v in samples)
    # smpl: no manufacturer or product, the sample period in ns, the root note and its fraction, no SMPTE, no loops.
    assert chunks[b"smpl"] == struct.pack("<9I", 0, 0, round(1e9 / CD_RATE), 37, 123, 0, 0, 0, 0)
    assert chunks[wav.PRIVATE].decode("utf-8") == json.dumps(meta, ensure_ascii=False, indent=1)  # UTF-8, as written


def test_read_takes_each_byte_of_24_bit_samples_in_its_place(tmp_path: Path) -> None:
    """Read 24-bit samples whose three bytes all differ, positive and negative, little-endian."""
    values = np.array([0x123456, -0x123456, 0x7FFFFE, -0x800000, 0x0000FF, 0x00FF00])
    wav.write(tmp_path / "x.wav", values, wav.Format(RATE, 24), ROOT, {})
    assert np.array_equal(wav.read(tmp_path / "x.wav")[1][:, 0], values / 2**23)


@pytest.mark.parametrize("head", [b"RIFX\4\0\0\0WAVE", b"RIFF\4\0\0\0AVI ", b"FORM\4\0\0\0WAVE"])
def test_chunks_refuse_a_file_that_is_riff_or_wave_but_not_both(tmp_path: Path, head: bytes) -> None:
    """Refuse a big-endian RIFX, a RIFF that is not WAVE, and a WAVE id under another container."""
    (tmp_path / "x.wav").write_bytes(head)
    with pytest.raises(ValueError, match="not a RIFF WAVE file"):
        wav.chunks(tmp_path / "x.wav")


def test_chunks_end_where_the_riff_size_says(tmp_path: Path) -> None:
    """Read every chunk up to the declared end, an empty last one included, and nothing appended after it."""
    body = b"WAVE" + b"abcd" + struct.pack("<I", 3) + b"xyz\0" + b"none" + struct.pack("<I", 0)
    junk = b"tail" + struct.pack("<I", 2) + b"!!"
    (tmp_path / "x.wav").write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body + junk)
    assert wav.chunks(tmp_path / "x.wav") == {b"abcd": b"xyz", b"none": b""}
