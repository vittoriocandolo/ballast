# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Synthesis: pure functions of time, exact wherever the design promises exactness."""

import math

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.integrate import cumulative_simpson
from scipy.signal import upfirdn

from ballast_synth import family, patch, voice
from ballast_synth.voice import Voice

from .conftest import SUB

_, INTERNAL, _ = family.rates(48000)  # the internal rate of the 48 kHz family
EXACT_DB = -140  # how close a transient read between its samples comes to exact interpolation, re its peak


@pytest.mark.parametrize("db", [6.0, 48.0])
def test_blocks_never_change_a_sample(tight: patch.Patch, db: float, monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthesize in three block sizes, one longer than the kick, gently and hard driven, and compare every sample."""
    driven = patch.edit(tight, "drive.db", db)
    whole = Voice(driven, INTERNAL).samples()
    for block in (999, 4096, 2**18):
        monkeypatch.setattr(voice, "BLOCK", block)
        assert np.array_equal(Voice(driven, INTERNAL).samples(), whole)


def test_a_longer_render_starts_with_the_same_samples(tight: patch.Patch) -> None:
    """Synthesize a shorter run of the same voice, and compare it with the start of the whole."""
    v = Voice(tight, INTERNAL)
    whole = v.samples()
    v.count = 10_000
    assert np.array_equal(v.samples(), whole[:10_000])


def test_tail_phase_meets_the_free_running_sine_exactly(tight: patch.Patch) -> None:
    """Compare the body from the last pitch knot on with the sine at tune_hz itself, with phase_ref = "tail"."""
    v = Voice(patch.edit(patch.edit(tight, "body.phase_ref", "tail"), "body.h2", 0.0), INTERNAL)
    t = np.arange(round(v.pitch.t[-1] * INTERNAL), round(v.body_amp.t[-1] * INTERNAL)) / INTERNAL
    t = t[t >= v.pitch.t[-1]]
    sine = np.sin(v.phi0 + (2 * math.pi * v.f0) * t)
    assert np.array_equal(v.body(t), v.body_gain * v.body_amp(t) * sine)


def test_onset_phase_starts_at_phase_deg(tight: patch.Patch) -> None:
    """Start the body at a phase of 90 degrees, under an amplitude shape that starts at 0."""
    v = Voice(patch.edit(tight, "body.phase_deg", 90.0), INTERNAL)
    assert v.body(np.array([0.0]))[0] == 0.0
    assert v.phi0 == math.pi / 2


def test_drive_at_0_db_is_an_exact_bypass(tight: patch.Patch) -> None:
    """Synthesize with no drive, and get the sources themselves, sample for sample."""
    v = Voice(patch.edit(tight, "drive.db", 0.0), INTERNAL)
    assert v.curve is None
    assert np.array_equal(v.samples(), v.sources(0, v.count))


@pytest.mark.parametrize("db", [0.5, 6.0, 21.0, 48.0])
def test_drive_alpha_solves_for_the_small_signal_gain(db: float) -> None:
    """Solve for α, and recover the gain α / tanh α to within rounding."""
    a = voice.drive_alpha(db)
    assert a / math.tanh(a) == pytest.approx(10 ** (db / 20), rel=1e-15)


def test_the_attack_mix_keeps_constant_power_and_is_exact_at_both_ends() -> None:
    """Weigh tone and noise at both ends of the mix, and at constant power between them."""
    assert voice.mix_weights(0.0) == (1.0, 0.0)
    assert voice.mix_weights(1.0) == (0.0, 1.0)
    for m in np.linspace(0.05, 0.95, 19):
        tone, noise = voice.mix_weights(m)
        assert tone**2 + noise**2 == pytest.approx(1.0, abs=1e-15)


# Lengths where t_src × rate rounds across an integer in float64: below it at either family's rate (real
# patch lengths: 21.875 and 35.625 ms at 1411.2 kHz, 1.125 and 2.25 ms at 1536 kHz), and above it just below a
# sample (3.3 µs, shorter than a patch).
EDGES = [
    0.021875,
    0.035625,
    0.001125,
    0.00225,
    float(np.nextafter(5 / 1536000, 0)),
    float(np.nextafter(5 / 1411200, 0)),
]


@pytest.mark.parametrize("t_src", [0.0, 1e-6, 0.24, 0.1 + 0.2, 1.45, 8.0, *EDGES])
@pytest.mark.parametrize("rate", [1411200, 1536000])
def test_sample_count_covers_every_sample_up_to_the_source_length(t_src: float, rate: int) -> None:
    """Count the samples of a source, so that the last one is at or before its end and the next one after."""
    n = voice.sample_count(t_src, rate)
    assert (n - 1) / rate <= t_src < n / rate


def test_silent_sources_render_nothing(tight: patch.Patch) -> None:
    """Silence both sources, and synthesize only zeros, without drawing any noise."""
    silent = patch.edit(patch.edit(tight, "body.level_db", -math.inf), "attack.level_db", -math.inf)
    v = Voice(silent, INTERNAL)
    assert v.noise is None
    assert not np.any(v.samples())
    assert not np.any(v.sub(np.linspace(0.0, 0.1, 100)))  # the patch has no sub


@pytest.mark.parametrize(("mix", "tone", "noise"), [(0.0, True, False), (1.0, False, True)])
def test_the_attack_is_pure_tone_or_pure_noise_at_the_ends_of_the_mix(
    tight: patch.Patch,
    *,
    mix: float,
    tone: bool,
    noise: bool,
) -> None:
    """Synthesize the attack at each end of the mix, as exactly the tone or exactly the noise."""
    v = Voice(patch.edit(tight, "attack.noise_mix", mix), INTERNAL)
    t = np.arange(0, round(0.006 * INTERNAL)) / INTERNAL
    attack = v.attack(t, np.arange(t.size))
    assert (v.noise is not None) == noise
    if tone:
        assert np.array_equal(attack, v.attack_gain * v.attack_amp(t) * np.sin((2 * math.pi * v.tone_hz) * t))
    else:
        assert v.noise is not None
        assert np.array_equal(attack, v.attack_gain * v.attack_amp(t) * v.noise[: t.size])


def test_a_flat_tone_pitch_is_the_fixed_tone(tight: patch.Patch) -> None:
    """Hold the tone's pitch at ratio 1 over two knots, and synthesize exactly the tone of one knot."""
    pure = patch.edit(tight, "attack.noise_mix", 0.0)
    flat = Voice(patch.edit(pure, "attack.pitch", ((0.0, 1.0), (3.0, 1.0))), INTERNAL)
    assert flat.tone_pitch is None
    assert np.array_equal(flat.samples(), Voice(pure, INTERNAL).samples())


def test_the_tone_sweeps_along_its_pitch_curve(tight: patch.Patch) -> None:
    """Sweep the tone from 4 times tone_hz down to it, and find the phase of 2π f_t ∫ R, integrated numerically."""
    zap = patch.edit(patch.edit(tight, "attack.noise_mix", 0.0), "attack.pitch", ((0.0, 4.0), (2.0, 1.0)))
    v = Voice(zap, INTERNAL)
    assert v.tone_pitch is not None
    t = np.arange(round(0.006 * INTERNAL)) / INTERNAL
    phase = 2 * math.pi * v.tone_hz * cumulative_simpson(v.tone_pitch(t), x=t, initial=0.0)
    assert v.tone_pitch(np.array([0.0, 0.002, 0.004])).tolist() == [4.0, 1.0, 1.0]
    tone = v.attack(t, np.arange(t.size)) / (v.attack_gain * v.tone_weight)
    assert np.allclose(tone, v.attack_amp(t) * np.sin(phase), rtol=0, atol=1e-9)


@pytest.mark.parametrize("bright", [50.0, 1000.0, 20000.0])
def test_a_bright_body_is_a_saw_that_fades_above_bright_hz(tight: patch.Patch, bright: float) -> None:
    """Sum the series Σ r^(k-1) sin(kφ) / k itself, with r = e^(-f/bright), and find the body's closed form."""
    v = Voice(patch.edit(patch.edit(tight, "body.bright_hz", bright), "body.h2", 0.0), INTERNAL)
    t = np.linspace(0.0, 0.2, 4001)
    phi = v.phi0 + 2 * math.pi * v.f0 * v.pitch.integral(t)
    r = np.exp(-v.f0 * v.pitch(t) / bright)
    k = np.arange(1, 20001)[:, None]
    series = np.sum(r ** (k - 1) * np.sin(k * phi) / k, axis=0)
    assert np.allclose(v.body(t), v.body_gain * v.body_amp(t) * series, rtol=0, atol=1e-12)


def test_a_bright_body_fades_by_the_same_hz_at_every_pitch(tight: patch.Patch) -> None:
    """Find the harmonic nearest 2 kHz e^(-f/bright) below a saw's, on the note and an octave up, over one period."""
    for tune in (55.0, 110.0):
        bright = patch.edit(patch.edit(tight, "body.tune_hz", tune), "body.bright_hz", 1000.0)
        v = Voice(patch.edit(bright, "body.h2", 0.0), INTERNAL)
        t = 0.1 + np.arange(4096) / 4096 / tune  # one period, past the sweep, where φ grows at 2π tune_hz
        wave = v.body(t) / (v.body_gain * v.body_amp(t))
        phi = v.phi0 + 2 * math.pi * v.f0 * v.pitch.integral(t)
        k = round(2000 / tune)
        harmonic = abs(2 * np.mean(wave * np.exp(-1j * k * phi)))
        assert harmonic == pytest.approx(math.exp(-(k - 1) * tune / 1000.0) / k, rel=1e-9)


@pytest.mark.parametrize("reference", ["onset", "tail"])
def test_the_sub_is_the_sine_the_body_tail_meets(tight: patch.Patch, reference: str) -> None:
    """Find the sub in phase with the body, sample for sample to rounding, from the last pitch knot on."""
    p = patch.edit(patch.edit(patch.edit(tight, "body.h2", 0.0), "body.phase_ref", reference), "sub", SUB)
    v = Voice(p, INTERNAL)
    t = np.linspace(v.pitch.t[-1], v.body_amp.t[-1] - 0.001, 5001)
    assert v.sub_amp is not None
    sine = v.sub(t) / (v.sub_gain * v.sub_amp(t))
    assert np.allclose(sine, v.body(t) / (v.body_gain * v.body_amp(t)), rtol=0, atol=1e-11)


def test_the_sub_stays_clean_however_hard_the_body_is_driven(tight: patch.Patch) -> None:
    """Drive the body by 48 dB, and find the sub added after the drive, untouched."""
    hard = patch.edit(tight, "drive.db", 48.0)
    v = Voice(patch.edit(hard, "sub", SUB), INTERNAL)
    without = Voice(hard, INTERNAL).samples()
    t = np.arange(v.count) / INTERNAL
    added = v.samples()[: without.size] - without
    assert np.allclose(added, v.sub(t[: without.size]), rtol=0, atol=1e-15)
    assert v.count > without.size  # the sub outlasts the body


@pytest.mark.parametrize("reference", ["onset", "tail"])
def test_modes_ring_from_the_strike_along_the_glide(tight: patch.Patch, reference: str) -> None:
    """Add two modes, and find each the sine at its ratio of the body's phase from the trigger, under its own decay."""
    modes = ((1.59, -9.0, 60.0), (2.65, -20.0, 20.0))
    plain = patch.edit(patch.edit(tight, "body.h2", 0.0), "body.phase_ref", reference)
    v, sine = Voice(patch.edit(plain, "body.modes", modes), INTERNAL), Voice(plain, INTERNAL)
    t = np.linspace(0.0, 0.2, 4001)
    struck = 2 * math.pi * v.f0 * v.pitch.integral(t)  # from the trigger, whatever the phase reference
    modal = np.zeros_like(t)
    for ratio, db, ms in modes:
        modal += 10 ** (db / 20) * np.exp(-math.log(1000) * t / (ms / 1000)) * np.sin(ratio * struck)
    assert np.allclose(v.body(t), sine.body(t) + v.body_gain * v.body_amp(t) * modal, rtol=0, atol=1e-12)


def test_a_mode_falls_60_db_in_its_decay_time(tight: patch.Patch) -> None:
    """Find a mode's envelope 60 dB down after decay_ms, whatever the body's shape does."""
    v = Voice(patch.edit(tight, "body.modes", ((1.59, 0.0, 50.0),)), INTERNAL)
    ratio, gain, rate = v.modes[0]
    assert (ratio, gain) == (1.59, 1.0)
    assert 20 * math.log10(math.exp(-rate * 0.05)) == pytest.approx(-60.0, abs=1e-9)


def reconstruction(t: patch.Transient) -> NDArray[np.float64]:
    """Bring a transient's samples up to its family's internal rate, with the family's frozen filter."""
    base, _, factor = family.rates(t.rate)
    u: NDArray[np.float64] = upfirdn(factor * family.decimator(base), patch.decode(t.samples).astype(float), up=factor)
    return u


def read_exactly(u: NDArray[np.float64], up: int, down: int) -> NDArray[np.float64]:
    """Read u at the positions n · down / up by its trigonometric interpolation, padded far enough not to wrap."""
    n = down * -(-4 * u.size // down)
    m = n * up // down
    out: NDArray[np.float64] = np.fft.irfft(np.fft.rfft(u, n)[: m // 2 + 1], m) * up / down
    return out


def test_a_transient_in_its_own_family_is_its_reconstruction_exactly(layered: patch.Patch) -> None:
    """Place the click at pitch 1 from the trigger, where the voice reads its reconstruction sample for sample."""
    t = layered.transient
    assert t is not None
    first, v = voice.place(t, INTERNAL)
    assert first == voice.NODES.start  # the interpolation could reach back that far, where it is silent
    assert np.array_equal(v, 10 ** (t.level_db / 20) * np.pad(reconstruction(t), 2))


@pytest.mark.parametrize(("base", "pitch", "up", "down"), [(44100, 1.0, 147, 160), (48000, 1.5, 2, 3)])
def test_a_transient_read_between_its_samples_is_exact_to_140_db(
    layered: patch.Patch,
    *,
    base: int,
    pitch: float,
    up: int,
    down: int,
) -> None:
    """Read the 48 kHz click in the 44.1 kHz family, and 1.5 times as fast in its own, against exact interpolation."""
    t = layered.transient
    assert t is not None
    _, rate, _ = family.rates(base)
    first, v = voice.place(patch.edit(layered, "transient.pitch", pitch).transient or t, rate)
    exact = 10 ** (t.level_db / 20) * read_exactly(reconstruction(t), up, down)
    n = np.arange(max(first, 0), first + v.size)
    error = np.max(np.abs(v[n - first] - exact[n])) / np.max(np.abs(exact))
    assert 20 * math.log10(error) < EXACT_DB


def test_blocks_never_change_a_sample_of_a_transient(layered: patch.Patch, monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthesize a transient read between its samples in three block sizes, and compare every sample."""
    moved = patch.edit(patch.edit(layered, "transient.pitch", 1.5), "transient.start_ms", 0.3)
    whole = Voice(moved, INTERNAL).samples()
    for block in (999, 4096, 2**18):
        monkeypatch.setattr(voice, "BLOCK", block)
        assert np.array_equal(Voice(moved, INTERNAL).samples(), whole)


def test_the_voice_runs_on_until_a_late_transient_has_rung_out(layered: patch.Patch) -> None:
    """End the transient after the body, and synthesize until its interpolation's ring has died away."""
    late = patch.edit(layered, "transient.start_ms", 250.0)
    v = Voice(late, INTERNAL)
    assert v.transient is not None
    first, values = v.transient
    assert v.count == first + values.size > voice.sample_count(late.length_ms / 1000, INTERNAL)


@pytest.mark.parametrize("db", [5e-324, 1e-300, 1e-16])
def test_a_drive_too_small_to_change_the_gain_is_a_bypass(tight: patch.Patch, db: float) -> None:
    """Drive by less than the gain can show, and synthesize exactly what no drive would."""
    assert voice.drive_alpha(db) == 0.0
    bypass = Voice(patch.edit(tight, "drive.db", 0.0), INTERNAL).samples()
    assert np.array_equal(Voice(patch.edit(tight, "drive.db", db), INTERNAL).samples(), bypass)


def test_mix_blends_the_dry_and_the_driven_signal_exactly(tight: patch.Patch) -> None:
    """Mix nothing of the drive into the very sources, all of it into the driven signal, and half into their mean."""
    hard = patch.edit(tight, "drive.db", 21.0)
    dry = Voice(hard, INTERNAL).sources(0, Voice(hard, INTERNAL).count)
    driven = Voice(hard, INTERNAL).samples()
    assert np.array_equal(Voice(patch.edit(hard, "drive.mix", 0.0), INTERNAL).samples(), dry)
    half = Voice(patch.edit(hard, "drive.mix", 0.5), INTERNAL).samples()
    assert np.allclose(half, (dry + driven) / 2, rtol=0, atol=1e-15)
