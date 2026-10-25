# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Playing: softer hits as [play] describes them, repeats drawn the same way in every version, and played kits."""

import math

import pytest

from ballast_synth import pack, patch, play

PLAY = patch.Play(-18.0, -6.0, 0.6, 0.5)
DRAWS = (856659851727918938, [-0.6616851628848233, 0.5125035092305257, -0.9384975636726016, 0.17258370779338672])
LAYER_RANGE_DB = 6.0  # how much quieter the softest of three layers is, at least, than the hardest


@pytest.fixture
def played(tight: patch.Patch) -> patch.Patch:
    """Return the tight preset with a [play] section."""
    return patch.edit(tight, "play", PLAY)


def test_the_hardest_hit_is_the_patch_and_the_softest_loses_what_play_says(played: patch.Patch) -> None:
    """Hit the kick at full strength, which is the patch itself, then at its softest and halfway."""
    assert play.softer(played, 1.0) is played
    soft = play.softer(played, 0.0)
    assert soft.body.level_db == played.body.level_db + PLAY.soft_db
    assert soft.attack.level_db == played.attack.level_db + PLAY.soft_db + PLAY.soft_click_db
    first = played.body.pitch[0][1]
    assert soft.body.pitch[0][1] == pytest.approx(1 + (first - 1) * PLAY.soft_punch)
    assert soft.body.pitch[-1][1] == 1.0
    half = play.softer(played, 0.5)
    assert half.body.level_db == played.body.level_db + PLAY.soft_db / 2


def test_a_soft_hit_moves_the_sub_with_the_body_and_the_transient_with_the_click(
    layered: patch.Patch,
) -> None:
    """Hit a kick with a sub and a transient softly, at full punch, and find the pitch curve as it was.

    The sub falls with the body, and the transient with the click.
    """
    full = patch.edit(layered, "sub", patch.Sub(-6.0, ((0.0, 0.0), (5.0, 1.0), (200.0, 0.0))))
    full = patch.edit(full, "play", patch.Play(-18.0, -6.0, 1.0, 0.0))
    soft = play.softer(full, 0.0)
    assert full.sub is not None
    assert full.transient is not None
    assert soft.sub is not None
    assert soft.transient is not None
    assert soft.sub.level_db == full.sub.level_db - 18.0
    assert soft.transient.level_db == round(full.transient.level_db - 24.0, 3)
    assert soft.body.pitch == full.body.pitch


def test_a_silent_source_stays_silent_however_softly_it_is_hit(played: patch.Patch) -> None:
    """Hit a kick whose attack is off, which stays off."""
    silent = patch.edit(played, "attack.level_db", -math.inf)
    assert play.softer(silent, 0.0).attack.level_db == -math.inf


def test_the_repeats_never_change(played: patch.Patch) -> None:
    """Draw repeat 1 of seed 1, as every version must, and play it: another seed, small moves within spread."""
    assert play.draws(1, 1) == DRAWS
    assert play.again(played, 0) is played
    again = play.again(played, 1)
    assert again.seed == DRAWS[0]
    assert abs(again.body.level_db - played.body.level_db) <= play.LEVEL_DB * PLAY.spread
    assert play.again(played, 1) == again
    assert play.again(played, 2) != again


def test_a_kit_needs_play_and_its_hits_share_one_chain(played: patch.Patch, tight: patch.Patch) -> None:
    """Refuse layers for a patch without [play]; render a kit whose hits share a clipper and whose loudest is at -1."""
    with pytest.raises(ValueError, match=r"tight: velocity layers and round robins need a \[play\] section"):
        pack.kicks([("tight", "", tight)], [], 24, 2, 1)
    clipped = patch.edit(played, "clip.db", 3.0)
    kit = pack.kicks([("kick", "", clipped)], [], 24, 3, 2)
    assert [k.path for k in kit] == [f"kick-A1-v{j}-rr{k}.wav" for j in (1, 2, 3) for k in (1, 2)]
    assert [k.velocity for k in kit[::2]] == [(1, 42), (43, 84), (85, 127)]
    assert len({k.report.gain for k in kit}) == 1
    assert len({k.report.threshold for k in kit}) == 1
    assert max(k.report.true_peak for k in kit) == pytest.approx(10 ** (-1 / 20), rel=1e-6)
    assert kit[0].report.loudness < kit[-1].report.loudness - LAYER_RANGE_DB


def test_a_kit_maps_its_layers_and_repeats_for_samplers(played: patch.Patch) -> None:
    """Map a kit of two layers and two repeats: a group per layer on the drum key, its repeats in turn."""
    kit = pack.kicks([("kick", "", played)], [], 24, 2, 2)
    assert pack.sfz("kick-A1", kit).splitlines() == [
        "// kick-A1: 2 velocity layers, 2 round robins, on key 36",
        "<group> key=36 lovel=1 hivel=63 amp_veltrack=0 seq_length=2",
        "<region> sample=kick-A1-v1-rr1.wav seq_position=1",
        "<region> sample=kick-A1-v1-rr2.wav seq_position=2",
        "<group> key=36 lovel=64 hivel=127 amp_veltrack=0 seq_length=2",
        "<region> sample=kick-A1-v2-rr1.wav seq_position=1",
        "<region> sample=kick-A1-v2-rr2.wav seq_position=2",
    ]


def test_a_spread_of_zero_repeats_the_hit_exactly(played: patch.Patch) -> None:
    """Play a kick with no spread, whose every repeat is the hit itself, noise and all."""
    still = patch.edit(played, "play.spread", 0.0)
    assert play.again(still, 1) is still


def test_a_kit_refuses_hits_that_no_patch_can_hold(played: patch.Patch) -> None:
    """Play the softest hit so soft that its click falls below -60 dB, and get that problem, named by its file."""
    soft = patch.edit(patch.edit(played, "play.soft_db", -40.0), "play.soft_click_db", -24.0)
    with pytest.raises(patch.PatchError) as caught:
        pack.kicks([("kick", "", soft)], [], 24, 3, 1)
    assert caught.value.problems == ["kick-A1-v1.wav: attack.level_db: -80 is outside -60 … 24 (or -inf)"]
