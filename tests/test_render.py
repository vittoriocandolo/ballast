# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Rendering: the frozen decimators, exact parallel chunks, rates, the output stage, PCM and the report."""

import dataclasses
import math
import multiprocessing
import sys
import warnings

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.signal import hilbert, sosfilt, sosfreqz, upfirdn

from ballast_synth import family, patch, render
from ballast_synth.curve import Curve

PASSBAND_HZ = 20000
PEAK_DB = 0.00002  # how close the true peak comes to a sine's amplitude: the decimator's ripple
CLIP_DB = 1e-7  # how close the clipper comes to taking exactly its db off the true peak
EQ_MATCH_DB = 0.004  # how close the EQ comes to its analog prototypes, up to 20 kHz
LOW_PASS_MATCH_DB = 0.012  # and the low-pass, whose -12 dB/oct slope shows the warping, 40 log10(tan x / x)
SWEEP_ROUNDING = 1e-10  # of the peak, -200 dB: poles this near 1 amplify the rounding of different algebra
FOLLOW_DB = 0.05  # how close a slow sweep stays to the analog low-pass at each moment's cutoff: 0.021 dB measured
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
    taps = family.decimator(base)
    assert math.fsum(taps) == 1.0  # unity gain at DC, exactly
    n = 1 << 21
    db = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(taps, n)), 1e-300))
    f = np.fft.rfftfreq(n, 1 / (base * family.OVERSAMPLING))
    assert np.max(np.abs(db[f <= PASSBAND_HZ])) < RIPPLE_DB
    assert np.max(db[f >= base / 2]) < STOPBAND_DB


def test_a_decimator_that_fails_its_checksum_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse taps whose checksum is not the one the code expects."""
    monkeypatch.setitem(family.DECIMATORS, 1, (family.DECIMATORS[48000][0], "0" * 64))
    with pytest.raises(RuntimeError, match="does not match its checksum"):
        family.decimator(1)


@pytest.mark.parametrize("size", [1, 10, family.decimator(48000).size, 30_000])
@pytest.mark.parametrize("chunk", [1, 7, 2048])
def test_parallel_decimation_matches_one_upfirdn_call_bit_for_bit(
    size: int,
    chunk: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decimate in chunks of three sizes, and compare every output with one call to upfirdn."""
    y = np.random.default_rng(size).standard_normal(size)
    taps = family.decimator(48000)
    monkeypatch.setattr(family, "CHUNK", chunk)
    for factor in (8, 16, 32):  # the factors of the three rates of a family
        assert np.array_equal(family.decimate(taps, y, factor), upfirdn(taps, y, down=factor))


@pytest.mark.parametrize("size", [1, 10, 200, 3000])
@pytest.mark.parametrize("chunk", [1, 7, 2048])
def test_parallel_interpolation_matches_one_upfirdn_call_bit_for_bit(
    size: int,
    chunk: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interpolate in chunks of three sizes, and compare every output with one call to upfirdn."""
    x = np.random.default_rng(size).standard_normal(size)
    taps = family.decimator(44100)
    monkeypatch.setattr(family, "CHUNK", chunk)
    for factor in (8, 16, 32):
        assert np.array_equal(family.interpolate(taps, x, factor), upfirdn(factor * taps, x, up=factor))


def test_decimation_is_causal() -> None:
    """Decimate the start of a signal alone, and get the outputs that depend only on it unchanged."""
    y = np.random.default_rng(0).standard_normal(50_000)
    taps = family.decimator(44100)
    whole, part = family.decimate(taps, y, 32), family.decimate(taps, y[:20_000], 32)
    n = 20_000 // 32  # outputs whose inputs all lie in the first 20 000 samples
    assert np.array_equal(part[:n], whole[:n])


@pytest.mark.parametrize(("base", "high"), [(44100, 88200), (44100, 176400), (48000, 96000), (48000, 192000)])
@pytest.mark.parametrize("low_cut", [0.0, 30.0])
def test_a_high_rate_render_fills_in_between_the_base_rate_samples(
    gain: patch.Patch,
    base: int,
    high: int,
    low_cut: float,
) -> None:
    """Render at a high rate and at its base rate, and find every base-rate sample in the high-rate file."""
    low, _ = render.render(with_output(gain, sample_rate=base, low_cut_hz=low_cut))
    more, _ = render.render(with_output(gain, sample_rate=high, low_cut_hz=low_cut))
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


def test_gain_mode_reports_overs_without_clipping(gain: patch.Patch) -> None:
    """Render 12 dB too loud, and report the over instead of clipping it."""
    audio, report = render.render(with_output(gain, db=12.0))
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
    assert [line.split()[0] for line in lines] == ["tuning", "length", "peak", "loudness", "render"]
    assert lines[0] == "tuning   55.000 Hz, A1 +0.0 cents"
    assert lines[1] == "length   240.0 ms source, 241.9 ms exported (11612 samples at 48000 Hz)"
    assert lines[3] == f"loudness {report.loudness:.1f} LUFS momentary, at its loudest"
    assert report.true_peak >= report.peak


@pytest.mark.parametrize("mode", ["gain", "peak", "true-peak"])
def test_silence_renders_and_says_so(tight: patch.Patch, mode: str) -> None:
    """Render silence in every output mode, through the clipper and a room, and report it as silent."""
    silent = patch.edit(patch.edit(tight, "body.level_db", -math.inf), "attack.level_db", -math.inf)
    silent = patch.edit(patch.edit(silent, "clip.db", 4.0), "room", patch.Room(-10.0, 600.0, 4000.0, 8.0))
    assert render.internal(silent)[1] is None  # no threshold clips silence
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
    _, rate, _ = family.rates(48000)
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


@pytest.mark.parametrize("rate", patch.SAMPLE_RATES)
def test_a_transient_passes_flat_to_20_khz(tight: patch.Patch, rate: int) -> None:
    """Render an impulse as the only source, at its own rate, through the interpolation and the decimation."""
    alone = patch.edit(patch.edit(tight, "body.level_db", -math.inf), "attack.level_db", -math.inf)
    impulse = patch.Transient.of(np.array([1.0]), rate)
    p = patch.edit(patch.edit(with_output(alone, sample_rate=rate), "drive.db", 0.0), "transient", impulse)
    n = 1 << 18
    passband = np.abs(np.fft.rfft(render.synthesize(p), n))[np.fft.rfftfreq(n, 1 / rate) <= PASSBAND_HZ]
    assert np.max(np.abs(20 * np.log10(passband))) < 2 * 0.0000135  # the filter's ripple, once up and once down


def analog(kind: str, hz: float, db: float, q: float, f: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Return the response of a band's analog prototype at frequencies f."""
    a, s, slope = 10 ** (db / 40), 1j * f / hz, math.sqrt(2)  # a shelf of slope 1 has Q = 1/√2
    if kind == "bell":
        out: NDArray[np.complex128] = (s * s + s * a / q + 1) / (s * s + s / (a * q) + 1)
    elif kind == "low":
        out = a * (s * s + math.sqrt(a) * slope * s + a) / (a * s * s + math.sqrt(a) * slope * s + 1)
    else:
        out = a * (a * s * s + math.sqrt(a) * slope * s + 1) / (s * s + math.sqrt(a) * slope * s + a)
    return out


@pytest.mark.parametrize("base", [44100, 48000])
@pytest.mark.parametrize(
    "band",
    [
        ("low", 20.0, 12.0, 1.0),
        ("low", 1000.0, -12.0, 1.0),
        ("bell", 60.0, 12.0, 10.0),
        ("bell", 10000.0, -12.0, 1.0),
        ("bell", 20000.0, 6.0, 0.3),
        ("high", 1000.0, -12.0, 1.0),
        ("high", 20000.0, 12.0, 1.0),
    ],
)
def test_the_eq_matches_its_analog_prototypes(
    tight: patch.Patch,
    base: int,
    band: tuple[str, float, float, float],
) -> None:
    """Design one band at the internal rate, and find its response within 0.004 dB of the analog one to 20 kHz."""
    kind, hz, db, q = band
    e = dataclasses.replace(patch.edit(patch.edit(tight, f"eq.{kind}_hz", hz), f"eq.{kind}_db", db).eq, bell_q=q)
    _, rate, _ = family.rates(base)
    bands = render.eq(e, rate)
    assert bands.shape == (1, 6)
    f = np.geomspace(10, PASSBAND_HZ, 2000)
    _, h = sosfreqz(bands, worN=f, fs=rate)
    assert np.max(np.abs(20 * np.log10(np.abs(h / analog(kind, hz, db, q, f))))) < EQ_MATCH_DB


def test_a_bell_reaches_its_gain_at_its_frequency_and_off_designs_nothing(tight: patch.Patch) -> None:
    """Boost a bell by 9 dB at 200 Hz, find exactly 9 dB there, and design no band at all with every gain at 0."""
    _, rate, _ = family.rates(48000)
    assert render.eq(tight.eq, rate).shape == (0, 6)
    _, h = sosfreqz(render.eq(dataclasses.replace(tight.eq, bell_db=9.0), rate), worN=[200.0], fs=rate)
    assert 20 * math.log10(abs(h[0])) == pytest.approx(9.0, abs=1e-6)


def test_k_weighting_matches_the_table_of_bs_1770_at_48_khz() -> None:
    """Design the K-weighting at 48 kHz, and find the coefficients the standard gives."""
    table = [
        [1.53512485958697, -2.69169618940638, 1.19839281085285, 1.0, -1.69065929318241, 0.73248077421585],
        [1.0, -2.0, 1.0, 1.0, -1.99004745483398, 0.99007225036621],
    ]
    assert np.allclose(render.k_weighting(48000), table, rtol=0, atol=1e-10)


@pytest.mark.parametrize("rate", [44100, 88200, 96000, 176400, 192000])
def test_k_weighting_keeps_the_48_khz_response_at_every_rate(rate: int) -> None:
    """Design the K-weighting at another rate and find the 48 kHz response within 0.01 dB from 20 Hz to 20 kHz."""
    f = np.geomspace(20, 20000, 400)
    ref = np.abs(sosfreqz(render.k_weighting(48000), worN=f, fs=48000)[1])
    got = np.abs(sosfreqz(render.k_weighting(rate), worN=f, fs=rate)[1])
    assert np.max(np.abs(20 * np.log10(got / ref))) < K_WEIGHTING_DB


K_WEIGHTING_DB = 0.01  # every rate's K-weighting against the 48 kHz one


@pytest.mark.parametrize("rate", [44100, 48000, 96000, 192000])
def test_a_full_scale_sine_at_997_hz_is_minus_3_lufs(rate: int) -> None:
    """Measure a full-scale 997 Hz sine of 2 s: -3.01 LUFS, within the 0.1 LU EBU Tech 3341 allows, and silence."""
    sine = np.sin(2 * math.pi * 997 * np.arange(2 * rate) / rate)
    assert render.loudness(sine, rate) == pytest.approx(-3.01, abs=0.05)
    assert render.loudness(np.zeros(100), rate) == -math.inf


def test_the_loudest_moment_of_a_kick_counts_its_first_400_ms(gain: patch.Patch) -> None:
    """Find a kick 6 dB louder at 6 dB more gain, and as loud however much silence follows it."""
    audio, report = render.render(gain)
    louder, _ = render.render(with_output(gain, db=0.0))
    assert render.loudness(louder, 48000) == pytest.approx(report.loudness + 6.0, abs=1e-6)
    assert render.loudness(np.concatenate((audio, np.zeros(48000))), 48000) == report.loudness


@pytest.mark.parametrize("base", [44100, 48000])
@pytest.mark.parametrize(("hz", "q"), [(40.0, 0.707), (800.0, 4.0), (20000.0, 10.0)])
def test_the_low_pass_matches_its_analog_prototype(tight: patch.Patch, base: int, hz: float, q: float) -> None:
    """Design the resonant low-pass at the internal rate, and find it within 0.012 dB of 1 / (s² + s/Q + 1)."""
    _, rate, _ = family.rates(base)
    bands = render.eq(dataclasses.replace(tight.eq, cutoff_hz=hz, resonance=q), rate)
    assert bands.shape == (1, 6)
    f = np.geomspace(10, PASSBAND_HZ, 2000)
    _, h = sosfreqz(bands, worN=f, fs=rate)
    s = 1j * f / hz
    assert np.max(np.abs(20 * np.log10(np.abs(h * (s * s + s / q + 1))))) < LOW_PASS_MATCH_DB


def swept(tight: patch.Patch, hz: float, q: float, sweep: patch.Knots) -> patch.EQ:
    """Return the tight preset's EQ with its low-pass at hz and q, swept."""
    return dataclasses.replace(tight.eq, cutoff_hz=hz, resonance=q, sweep=sweep)


def state_variable(x: NDArray[np.float64], e: patch.EQ, rate: int) -> NDArray[np.float64]:
    """Low-pass x through the state-variable filter at every sample's cutoff, from the first sample to the last."""
    curve = Curve.of(e.sweep, last=0.0)
    g = np.tan(math.pi * e.cutoff_hz * curve(np.arange(x.size) / rate) / rate)
    out, s1, s2 = np.empty_like(x), 0.0, 0.0
    for n, (v, gn) in enumerate(zip(x.tolist(), g.tolist(), strict=True)):
        h = (v - s2 - (gn + 1 / e.resonance) * s1) / (1 + gn * (gn + 1 / e.resonance))  # the high-pass
        band = gn * h + s1
        low = gn * band + s2
        s1, s2 = gn * h + band, gn * band + low
        out[n] = low
    return out


@pytest.mark.parametrize(
    "sweep",
    [
        ((0.0, 1.0), (5.0, 1.0)),
        ((0.0, 12.0), (3.0, 0.5), (20.0, 1.0)),
        ((0.0, 12.0), (900.0, 1.0)),  # still sweeping when the kick ends, so there is nothing to hand over
    ],
)
def test_a_sweep_hands_over_to_the_static_low_pass_exactly(tight: patch.Patch, sweep: patch.Knots) -> None:
    """Sweep the tight kick's low-pass, and find it as the state-variable filter run throughout, to rounding.

    The kick lasts 240 ms, so the last sweep outlasts it.

    A still sweep is the static biquad itself, its transfer function the same and its handover exact.
    """
    _, rate, _ = family.rates(48000)
    x, _ = render.internal(tight)
    e = swept(tight, 1500.0, 4.0, sweep)
    low = render.eq(e, rate)[-1]
    y = render.sweep(x, e, low, rate)
    scale = float(np.max(np.abs(y)))
    assert np.max(np.abs(y - state_variable(x, e, rate))) < SWEEP_ROUNDING * scale
    if all(r == 1.0 for _, r in sweep):
        assert np.max(np.abs(y - sosfilt(low[None], x))) < SWEEP_ROUNDING * scale


def test_a_slow_sweep_follows_the_analog_low_pass(tight: patch.Patch) -> None:
    """Sweep a 1 kHz sine's low-pass from 4 kHz to 500 Hz in a second: its level is the analog filter's at each cutoff.

    The level is read by the Hilbert transform, from 0.1 s to 1.1 s, away from its edges.
    """
    _, rate, _ = family.rates(48000)
    e = swept(tight, 500.0, 0.707, ((0.0, 8.0), (1000.0, 1.0)))
    t = np.arange(round(1.2 * rate)) / rate
    y = render.sweep(np.sin(2 * math.pi * 1000.0 * t), e, render.eq(e, rate)[-1], rate)
    s = 1j * 1000.0 / (500.0 * Curve([0.0, 1.0], [8.0, 1.0], last=0.0)(t))
    level = np.abs(hilbert(y)) * np.abs(s * s + s / 0.707 + 1)
    inside = slice(round(0.1 * rate), round(1.1 * rate))
    assert np.max(np.abs(20 * np.log10(level[inside]))) < FOLLOW_DB


def test_a_sweep_changes_only_the_low_pass(tight: patch.Patch) -> None:
    """Render a still sweep exactly as the static low-pass, and a moving one differently, through the whole chain."""
    dark = patch.edit(tight, "eq.cutoff_hz", 1500.0)
    still, _ = render.internal(dark)
    assert np.array_equal(render.internal(patch.edit(dark, "eq.sweep", ((0.0, 1.0), (5.0, 1.0))))[0], still)
    plucked, _ = render.internal(patch.edit(dark, "eq.sweep", ((0.0, 8.0), (50.0, 1.0))))
    assert plucked.size == still.size
    assert not np.allclose(plucked, still)


@pytest.mark.parametrize("base", [44100, 48000])
def test_the_true_peak_is_read_between_the_samples_exactly(base: int) -> None:
    """Measure a 19 kHz sine, whose peaks fall between the file's samples, and find its amplitude to 0.00002 dB."""
    _, rate, _ = family.rates(base)
    n = np.arange(round(0.05 * rate))
    amplitude = 0.7
    sine = (
        amplitude * np.sin(2 * math.pi * 19000 * n / rate + 0.3) * np.sin(math.pi * n / n.size) ** 2
    )  # faded in and out
    assert 20 * math.log10(render.true_peak(sine, base) / amplitude) == pytest.approx(0.0, abs=PEAK_DB)


@pytest.mark.parametrize(("db", "knee_db"), [(0.01, 0.0), (0.01, 6.0), (0.5, 0.0), (4.0, 0.0), (4.0, 6.0), (12.0, 3.0)])
def test_the_clipper_takes_exactly_its_db_off_the_true_peak(tight: patch.Patch, db: float, knee_db: float) -> None:
    """Clip the tight preset, and find its true peak, read on the reconstruction, lower by db within 1e-7 dB."""
    plain, _ = render.internal(tight)
    clipped, threshold = render.internal(patch.edit(patch.edit(tight, "clip.db", db), "clip.knee_db", knee_db))
    assert threshold is not None
    drop = 20 * math.log10(render.true_peak(plain, 48000) / render.true_peak(clipped, 48000))
    assert drop == pytest.approx(db, abs=CLIP_DB)


def _renders_as(text: str, expected: bytes) -> None:
    """Render a patch in this process and fail unless it is exactly `expected`."""
    if render.render(patch.loads(text))[0].tobytes() != expected:
        raise AssertionError


def test_a_process_forked_after_a_render_renders_too(tight_text: str) -> None:
    """Render, fork as multiprocessing does by default on Linux before Python 3.14, and render the same in the child.

    The child inherits the parent's pool of threads but none of the threads; it must make its own, not wait forever.
    """
    if sys.platform != "win32":  # a check the type checker reads too, unlike a skip mark
        expected = render.render(patch.loads(tight_text))[0].tobytes()
        with warnings.catch_warnings():  # Python warns that forking a process with threads may deadlock: the point here
            warnings.simplefilter("ignore", DeprecationWarning)
            child = multiprocessing.get_context("fork").Process(target=_renders_as, args=(tight_text, expected))
            child.start()
        child.join(timeout=60)
        alive = child.is_alive()
        child.kill()
        child.join()
        assert not alive, "the forked child hung"
        assert child.exitcode == 0
        return
    pytest.skip("Windows has no fork")
