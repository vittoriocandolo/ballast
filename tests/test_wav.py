# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The WAV files: float32 or PCM that other readers agree on, the root note, and the embedded patch."""

import json
import os
import struct
from pathlib import Path
from typing import NoReturn

import numpy as np
import pytest
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


def test_pcm_out_of_range_is_refused(tmp_path: Path) -> None:
    """Refuse a sample one step beyond the 16-bit range."""
    with pytest.raises(ValueError, match="out of range"):
        wav.write(tmp_path / "p.wav", np.array([2**15], dtype=np.int32), wav.Format(RATE, 16), ROOT, {})


def test_an_exported_render_carries_its_patch_and_root_note(tmp_path: Path, tight_text: str) -> None:
    """Export a render as 24-bit PCM, and read back its patch text, provenance, root note and samples."""
    bits = 24
    text = tight_text.replace("seed = 1", "seed = 1  # my favourite kick")  # the text is kept as written
    p = patch.loads(text)
    audio, _ = render.render(p, bits)
    render.export(tmp_path / "kick.wav", p, text, audio, bits)
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
