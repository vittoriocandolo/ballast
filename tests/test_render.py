"""Rendering: the frozen decimators, exact parallel chunks, rates, the output stage, PCM and the report."""

import math

import numpy as np
import pytest
from scipy.signal import upfirdn

from ballast import patch, render


def with_output(p, **values):
    for key, value in values.items():
        p = patch.edit(p, f"output.{key}", value)
    return p


@pytest.mark.parametrize("base", [44100, 48000])
def test_decimators_meet_their_specification(base):
    taps = render.decimator(base)
    assert math.fsum(taps) == 1.0  # unity gain at DC, exactly
    n = 1 << 21
    db = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(taps, n)), 1e-300))
    f = np.fft.rfftfreq(n, 1 / (base * render.K))
    assert np.max(np.abs(db[f <= 20000])) < 0.001  # flat within 0.001 dB up to 20 kHz
    assert np.max(db[f >= base / 2]) < -110  # at least 110 dB down from base/2


def test_a_decimator_that_fails_its_checksum_is_refused(monkeypatch):
    monkeypatch.setitem(render.DECIMATORS, 1, ("decimator-48000x16.f64", "0" * 64))
    with pytest.raises(RuntimeError, match="does not match its checksum"):
        render.decimator(1)


@pytest.mark.parametrize("size", [1, 10, 1447, 30_000])
@pytest.mark.parametrize("chunk", [1, 7, 2048])
def test_parallel_decimation_matches_one_upfirdn_call_bit_for_bit(size, chunk, monkeypatch):
    y = np.random.default_rng(size).standard_normal(size)
    taps = render.decimator(48000)
    monkeypatch.setattr(render, "CHUNK", chunk)
    for factor in (8, 16):
        assert np.array_equal(render.decimate(taps, y, factor), upfirdn(taps, y, down=factor))


def test_decimation_is_causal():
    y = np.random.default_rng(0).standard_normal(50_000)
    taps = render.decimator(44100)
    whole, part = render.decimate(taps, y, 16), render.decimate(taps, y[:20_000], 16)
    n = 20_000 // 16  # outputs whose inputs all lie in the first 20 000 samples
    assert np.array_equal(part[:n], whole[:n])


@pytest.mark.parametrize(("base", "high"), [(44100, 88200), (48000, 96000)])
def test_a_high_rate_render_fills_in_between_the_base_rate_samples(tight, base, high):
    low, _ = render.render(with_output(tight, sample_rate=base))
    double, _ = render.render(with_output(tight, sample_rate=high))
    assert np.array_equal(double[::2][: low.size], low)


@pytest.mark.parametrize("db", [0.0, -0.3, -12.0])
def test_peak_mode_reaches_its_target(tight, db):
    audio, report = render.render(with_output(tight, mode="peak", db=db))
    assert np.max(np.abs(audio)) == pytest.approx(10 ** (db / 20), rel=2**-24)
    assert not report.over


@pytest.mark.parametrize("bits", [16, 24])
def test_peak_mode_stays_two_lsb_below_full_scale_for_pcm(tight, bits):
    audio, report = render.render(with_output(tight, mode="peak", db=0.0), bits)
    assert np.max(np.abs(audio)) == pytest.approx(1 - 2.0 ** (2 - bits), rel=2**-24)
    assert report.ceiling == 1 - 2.0 ** (2 - bits)


def test_polarity_inverts_exactly(tight):
    up, _ = render.render(tight)
    down, _ = render.render(with_output(tight, polarity=-1))
    assert np.array_equal(down, -up)


def test_gain_mode_reports_overs_without_clipping(tight):
    audio, report = render.render(with_output(tight, db=12.0))
    assert report.over and report.over_db == pytest.approx(20 * math.log10(np.max(np.abs(audio))))
    assert np.max(np.abs(audio)) > 1  # nothing is ever clipped
    assert "over full scale by" in str(report)


@pytest.mark.parametrize("bits", [16, 24])
def test_dithered_pcm_is_seeded_keeps_zeros_and_never_clips(tight, bits):
    audio, _ = render.render(with_output(tight, mode="peak", db=0.0), bits)
    q = render.quantize(audio, bits, tight.seed)
    assert np.array_equal(q, render.quantize(audio, bits, tight.seed))
    assert not np.array_equal(q, render.quantize(audio, bits, tight.seed + 1))
    assert np.all(q[audio == 0] == 0)
    assert -(2 ** (bits - 1)) <= q.min() and q.max() < 2 ** (bits - 1)
    assert np.max(np.abs(q - audio.astype(float) * 2 ** (bits - 1))) < 1.5  # ±1 LSB of dither, then rounding


def test_the_report(tight):
    _, report = render.render(tight)
    lines = str(report).splitlines()
    assert [line.split()[0] for line in lines] == ["tuning", "length", "peak", "render"]
    assert lines[0] == "tuning   55.000 Hz, A1 +0.0 cents"
    assert lines[1] == "length   240.0 ms source, 241.9 ms exported (11611 samples at 48000 Hz)"
    assert report.true_peak >= report.peak


def test_silence_renders_and_says_so(tight):
    silent = patch.edit(patch.edit(tight, "body.level_db", -math.inf), "attack.level_db", -math.inf)
    audio, report = render.render(silent)
    assert not np.any(audio) and "silent" in str(report)
