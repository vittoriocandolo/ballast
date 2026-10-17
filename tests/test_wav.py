"""The WAV files: float32 or PCM that other readers agree on, the root note, and the embedded patch."""

import json
import struct

import numpy as np
import pytest
from scipy.io import wavfile

from ballast import patch, render, wav

# SciPy skips the smpl and blst chunks, as RIFF readers should, and says so.
pytestmark = pytest.mark.filterwarnings("ignore::scipy.io.wavfile.WavFileWarning")


def test_float_samples_read_back_exactly(tmp_path):
    samples = np.linspace(-1, 1, 1001, dtype=np.float32)
    wav.write(tmp_path / "f.wav", samples, 48000, None, (36, 0), {})
    rate, data = wavfile.read(tmp_path / "f.wav")
    assert rate == 48000 and data.dtype == np.float32 and np.array_equal(data, samples)


@pytest.mark.parametrize("bits", [16, 24])
def test_pcm_samples_read_back_exactly(tmp_path, bits):
    top = 2 ** (bits - 1)
    q = np.array([-top, -top + 1, -1, 0, 1, 1234, top - 1], dtype=np.int32)
    wav.write(tmp_path / "p.wav", q, 44100, bits, (36, 0), {})
    rate, data = wavfile.read(tmp_path / "p.wav")
    expected = q if bits == 16 else q << 8  # SciPy left-justifies 24-bit samples in int32
    assert rate == 44100 and np.array_equal(data.astype(np.int64), expected)


def test_pcm_out_of_range_is_refused(tmp_path):
    with pytest.raises(ValueError, match="out of range"):
        wav.write(tmp_path / "p.wav", np.array([2**15], dtype=np.int32), 48000, 16, (36, 0), {})


def test_an_exported_render_carries_its_patch_and_root_note(tmp_path, tight_text):
    text = tight_text.replace("seed = 1", "seed = 1  # my favourite kick")  # the text is kept as written
    p = patch.loads(text)
    audio, _ = render.render(p, 24)
    render.export(tmp_path / "kick.wav", p, text, audio, 24)
    found = wav.chunks(tmp_path / "kick.wav")
    meta = json.loads(found[wav.PRIVATE])
    assert meta["patch"] == text
    assert meta["pcm"]["bits"] == 24 and meta["decimator"]["sha256"] == render.DECIMATORS[48000][1]
    assert struct.unpack("<9I", found[b"smpl"])[3:5] == wav.root_note(55.0)
    rate, data = wavfile.read(tmp_path / "kick.wav")
    assert rate == 48000 and np.array_equal(data >> 8, render.quantize(audio, 24, p.seed))


@pytest.mark.parametrize(
    ("hz", "note"),
    [
        (440.0, (69, 0)),
        (440.0 * 2 ** (-0.4 / 1200), (69, 0)),  # less than half a cent flat counts as the note itself
        (440.0 * 2 ** (-0.6 / 1200), (68, round(0.994 * 2**32))),
        (440.0 * 2 ** (50 / 1200), (69, 2**31)),
    ],
)
def test_root_note(hz, note):
    unity, fraction = wav.root_note(hz)
    assert unity == note[0] and fraction == pytest.approx(note[1], abs=2**12)


def test_chunks_refuses_what_is_not_riff_wave(tmp_path):
    (tmp_path / "x.wav").write_bytes(b"not a wave file")
    with pytest.raises(ValueError, match="not a RIFF WAVE file"):
        wav.chunks(tmp_path / "x.wav")
