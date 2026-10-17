"""Synthesis: pure functions of time, exact wherever the design promises exactness."""

import math

import numpy as np
import pytest

from ballast import patch, voice
from ballast.voice import Voice

RATE = 768000


def test_blocks_never_change_a_sample(tight, monkeypatch):
    whole = Voice(tight, RATE).samples()
    for block in (999, 4096, 2**17):
        monkeypatch.setattr(voice, "BLOCK", block)
        assert np.array_equal(Voice(tight, RATE).samples(), whole)


def test_a_longer_render_starts_with_the_same_samples(tight):
    v = Voice(tight, RATE)
    whole = v.samples()
    v.count = 10_000
    assert np.array_equal(v.samples(), whole[:10_000])


def test_tail_phase_meets_the_free_running_sine_exactly(tight):
    """With phase_ref = "tail", the body is the sine at tune_hz itself from the last pitch knot on."""
    v = Voice(patch.edit(patch.edit(tight, "body.phase_ref", "tail"), "body.h2", 0.0), RATE)
    t = np.arange(round(v.pitch.t[-1] * RATE), round(v.body_amp.t[-1] * RATE)) / RATE
    t = t[t >= v.pitch.t[-1]]
    sine = np.sin(v.phi0 + (2 * math.pi * v.f0) * t)
    assert np.array_equal(v.body(t), v.body_gain * v.body_amp(t) * sine)


def test_onset_phase_starts_at_phase_deg(tight):
    v = Voice(patch.edit(tight, "body.phase_deg", 90.0), RATE)
    assert v.body(np.array([0.0]))[0] == 0.0  # the amplitude shape starts at 0
    assert v.phi0 == math.pi / 2


@pytest.mark.parametrize("db", [1e-6, 3.0, 21.0, 48.0])
def test_the_saturator_is_odd_and_keeps_full_scale_exactly(tight, db):
    v = Voice(patch.edit(tight, "drive.db", db), RATE)
    x = np.linspace(-2.0, 2.0, 4001)
    assert v.saturate(np.array([1.0, -1.0, 0.0])).tolist() == [1.0, -1.0, 0.0]
    assert np.array_equal(v.saturate(-x), -v.saturate(x))
    small = 1e-9  # the small-signal gain is 10^(db/20)
    assert v.saturate(np.array([small]))[0] / small == pytest.approx(10 ** (db / 20), rel=1e-9)


def test_drive_at_0_db_is_an_exact_bypass(tight):
    v = Voice(patch.edit(tight, "drive.db", 0.0), RATE)
    x = np.random.default_rng(0).standard_normal(1000)
    assert v.saturate(x) is x


@pytest.mark.parametrize("db", [0.5, 6.0, 21.0, 48.0])
def test_drive_alpha_solves_for_the_small_signal_gain(db):
    a = voice.drive_alpha(db)
    assert a / math.tanh(a) == pytest.approx(10 ** (db / 20), rel=1e-15)


def test_the_attack_mix_keeps_constant_power_and_is_exact_at_both_ends():
    assert voice.mix_weights(0.0) == (1.0, 0.0)
    assert voice.mix_weights(1.0) == (0.0, 1.0)
    for m in np.linspace(0.05, 0.95, 19):
        tone, noise = voice.mix_weights(m)
        assert tone**2 + noise**2 == pytest.approx(1.0, abs=1e-15)


# Lengths where t_src × rate rounds across an integer in float64: below it at the 44.1 kHz family's rate
# (real patch lengths, 21.875 and 35.625 ms), and above it just below a sample (6.5 µs, shorter than a patch).
EDGES = [0.021875, 0.035625, float(np.nextafter(5 / 768000, 0)), float(np.nextafter(5 / 705600, 0))]


@pytest.mark.parametrize("t_src", [0.0, 1e-6, 0.24, 0.1 + 0.2, 1.45, 8.0, *EDGES])
@pytest.mark.parametrize("rate", [705600, 768000])
def test_sample_count_covers_every_sample_up_to_the_source_length(t_src, rate):
    n = voice.sample_count(t_src, rate)
    assert (n - 1) / rate <= t_src < n / rate


def test_silent_sources_render_nothing(tight):
    silent = patch.edit(patch.edit(tight, "body.level_db", -math.inf), "attack.level_db", -math.inf)
    v = Voice(silent, RATE)
    assert v.noise is None
    assert not np.any(v.samples())


@pytest.mark.parametrize(("mix", "tone", "noise"), [(0.0, True, False), (1.0, False, True)])
def test_the_attack_is_pure_tone_or_pure_noise_at_the_ends_of_the_mix(tight, mix, tone, noise):
    v = Voice(patch.edit(tight, "attack.noise_mix", mix), RATE)
    t = np.arange(0, round(0.006 * RATE)) / RATE
    attack = v.attack(t, np.arange(t.size))
    assert (v.noise is not None) == noise
    if tone:
        assert np.array_equal(attack, v.attack_gain * v.attack_amp(t) * np.sin((2 * math.pi * v.tone_hz) * t))
    else:
        assert np.array_equal(attack, v.attack_gain * v.attack_amp(t) * v.noise[: t.size])
