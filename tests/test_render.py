# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Rendering: the frozen decimators, exact parallel chunks, rates, the output stage, PCM and the report."""

import math

import numpy as np
import pytest
from scipy.signal import sosfreqz, upfirdn

from ballast_synth import patch, render

PASSBAND_HZ = 20000
RIPPLE_DB = 0.001  # the specified flatness of the passband
STOPBAND_DB = -110  # the specified attenuation from half the base rate up


def with_output(p: patch.Patch, **values: object) -> patch.Patch:
    """Return the patch with output fields replaced, one edit at a time."""
    for key, value in values.items():
        p = patch.edit(p, f"output.{key}", value)
    return p


@pytest.mark.parametrize("base", [44100, 48000])
def test_decimators_meet_their_specification(base: int) -> None:
    """Measure each frozen filter's gain at DC, its passband ripple and its stopband attenuation."""
    taps = render.decimator(base)
    assert math.fsum(taps) == 1.0  # unity gain at DC, exactly
    n = 1 << 21
    db = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(taps, n)), 1e-300))
    f = np.fft.rfftfreq(n, 1 / (base * render.K))
    assert np.max(np.abs(db[f <= PASSBAND_HZ])) < RIPPLE_DB
    assert np.max(db[f >= base / 2]) < STOPBAND_DB


def test_a_decimator_that_fails_its_checksum_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse taps whose checksum is not the one the code expects."""
    monkeypatch.setitem(render.DECIMATORS, 1, ("decimator-48000x16.f64", "0" * 64))
    with pytest.raises(RuntimeError, match="does not match its checksum"):
        render.decimator(1)


@pytest.mark.parametrize("size", [1, 10, 1447, 30_000])
@pytest.mark.parametrize("chunk", [1, 7, 2048])
def test_parallel_decimation_matches_one_upfirdn_call_bit_for_bit(
    size: int,
    chunk: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decimate in chunks of three sizes, and compare every output with one call to upfirdn."""
    y = np.random.default_rng(size).standard_normal(size)
    taps = render.decimator(48000)
    monkeypatch.setattr(render, "CHUNK", chunk)
    for factor in (8, 16):
        assert np.array_equal(render.decimate(taps, y, factor), upfirdn(taps, y, down=factor))


def test_decimation_is_causal() -> None:
    """Decimate the start of a signal alone, and get the outputs that depend only on it unchanged."""
    y = np.random.default_rng(0).standard_normal(50_000)
    taps = render.decimator(44100)
    whole, part = render.decimate(taps, y, 16), render.decimate(taps, y[:20_000], 16)
    n = 20_000 // 16  # outputs whose inputs all lie in the first 20 000 samples
    assert np.array_equal(part[:n], whole[:n])


@pytest.mark.parametrize(("base", "high"), [(44100, 88200), (44100, 176400), (48000, 96000), (48000, 192000)])
@pytest.mark.parametrize("low_cut", [0.0, 30.0])
def test_a_high_rate_render_fills_in_between_the_base_rate_samples(
    tight: patch.Patch,
    base: int,
    high: int,
    low_cut: float,
) -> None:
    """Render at a high rate and at its base rate, and find every base-rate sample in the high-rate file."""
    low, _ = render.render(with_output(tight, sample_rate=base, low_cut_hz=low_cut))
    more, _ = render.render(with_output(tight, sample_rate=high, low_cut_hz=low_cut))
    assert np.array_equal(more[:: high // base][: low.size], low)


@pytest.mark.parametrize("db", [0.0, -0.3, -12.0])
def test_peak_mode_reaches_its_target(tight: patch.Patch, db: float) -> None:
    """Normalize to a sample peak, and measure it at the target."""
    audio, report = render.render(with_output(tight, mode="peak", db=db))
    assert np.max(np.abs(audio)) == pytest.approx(10 ** (db / 20), rel=2**-24)
    assert not report.over


@pytest.mark.parametrize("bits", [16, 24])
def test_peak_mode_stays_two_lsb_below_full_scale_for_pcm(tight: patch.Patch, bits: int) -> None:
    """Normalize to full scale for PCM, and stop at the ceiling that leaves room for dither."""
    audio, report = render.render(with_output(tight, mode="peak", db=0.0), bits)
    assert np.max(np.abs(audio)) == pytest.approx(1 - 2.0 ** (2 - bits), rel=2**-24)
    assert report.ceiling == 1 - 2.0 ** (2 - bits)


def test_polarity_inverts_exactly(tight: patch.Patch) -> None:
    """Render with each polarity, and get exact negatives."""
    up, _ = render.render(tight)
    down, _ = render.render(with_output(tight, polarity=-1))
    assert np.array_equal(down, -up)


def test_gain_mode_reports_overs_without_clipping(tight: patch.Patch) -> None:
    """Render 12 dB too loud, and report the over instead of clipping it."""
    audio, report = render.render(with_output(tight, db=12.0))
    assert report.over
    assert report.over_db == pytest.approx(20 * math.log10(np.max(np.abs(audio))))
    assert np.max(np.abs(audio)) > 1  # nothing is ever clipped
    assert "over full scale by" in str(report)


@pytest.mark.parametrize("bits", [16, 24])
def test_dithered_pcm_is_seeded_keeps_zeros_and_never_clips(tight: patch.Patch, bits: int) -> None:
    """Quantize a full-scale render twice with its seed and once with another, and check the range and error."""
    audio, _ = render.render(with_output(tight, mode="peak", db=0.0), bits)
    q = render.quantize(audio, bits, tight.seed)
    assert np.array_equal(q, render.quantize(audio, bits, tight.seed))
    assert not np.array_equal(q, render.quantize(audio, bits, tight.seed + 1))
    assert np.all(q[audio == 0] == 0)
    assert -(2 ** (bits - 1)) <= np.min(q)
    assert np.max(q) < 2 ** (bits - 1)
    assert np.max(np.abs(q - audio.astype(float) * 2 ** (bits - 1))) < 1 + 1 / 2  # 1 LSB of dither, ½ of rounding


def test_the_report(tight: patch.Patch) -> None:
    """Print the report of a render, line by line."""
    _, report = render.render(tight)
    lines = str(report).splitlines()
    assert [line.split()[0] for line in lines] == ["tuning", "length", "peak", "render"]
    assert lines[0] == "tuning   55.000 Hz, A1 +0.0 cents"
    assert lines[1] == "length   240.0 ms source, 241.9 ms exported (11611 samples at 48000 Hz)"
    assert report.true_peak >= report.peak


@pytest.mark.parametrize("mode", ["gain", "peak", "true-peak"])
def test_silence_renders_and_says_so(tight: patch.Patch, mode: str) -> None:
    """Render silence in every output mode, and report it as silent."""
    silent = patch.edit(patch.edit(tight, "body.level_db", -math.inf), "attack.level_db", -math.inf)
    audio, report = render.render(with_output(silent, mode=mode, db=-1.0))
    assert not np.any(audio)
    assert "silent" in str(report)


@pytest.mark.parametrize(("db", "bits"), [(-1.0, None), (-0.1, None), (0.0, 16), (0.0, 24)])
def test_true_peak_mode_reaches_its_target_within_the_pcm_ceiling(
    tight: patch.Patch,
    db: float,
    bits: int | None,
) -> None:
    """Normalize to a true peak, and reach either the target or the format's ceiling, never beyond."""
    _, report = render.render(with_output(tight, mode="true-peak", db=db), bits)
    assert report.true_peak <= 10 ** (db / 20) * (1 + 1e-6)
    assert report.peak <= report.ceiling
    hit_target = math.isclose(20 * math.log10(report.true_peak), db, rel_tol=0, abs_tol=1e-4)
    hit_ceiling = report.peak == pytest.approx(report.ceiling, rel=2**-24)
    assert hit_target or hit_ceiling  # one of the two limits binds


@pytest.mark.parametrize("hz", [10.0, 30.0, 1000.0])
def test_the_low_cut_is_a_12_db_per_octave_butterworth_high_pass(hz: float) -> None:
    """Measure the low cut at its corner, a decade below it, and a decade above it and at 20 kHz."""
    rate = 768000
    _, h = sosfreqz(render.low_cut(hz, rate), worN=[hz / 10, hz, hz * 10, 20000.0], fs=rate)
    gain = 20 * np.log10(np.abs(h))
    assert gain[1] == pytest.approx(-3.0103, abs=1e-3)  # half power at the corner
    assert gain[0] == pytest.approx(-40.0, abs=0.1)  # a decade below: 40 dB down
    assert np.allclose(gain[2:], 0, rtol=0, atol=0.001)  # flat from a decade above


def test_the_low_cut_removes_dc_and_off_changes_nothing(tight: patch.Patch) -> None:
    """Render with the low cut on, which removes the DC, and off, which changes no sample."""
    plain, _ = render.render(tight)
    cut, _ = render.render(with_output(tight, low_cut_hz=40.0))
    assert abs(np.mean(cut)) < abs(np.mean(plain)) / 20
    zero = patch.edit(tight, "output.low_cut_hz", 0.0)
    assert np.array_equal(render.render(zero)[0], plain)
