# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Playing: softer hits as [play] describes them, repeats drawn the same way in every version, and played kits."""

import dataclasses
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

from ballast_synth import pack, patch, play, render, wav

PLAY = patch.Play(-18.0, -6.0, 0.6, 0.5)
DRAWS = (856659851727918938, [-0.6616851628848233, 0.5125035092305257, -0.9384975636726016, 0.17258370779338672])
LAYER_RANGE_DB = 6.0  # how much quieter the softest of three layers is, at least, than the hardest


@pytest.fixture
def played(tight: patch.Patch) -> patch.Patch:
    """Return the tight preset with a [play] section."""
    return patch.edit(tight, "play", PLAY)


MAX_VELOCITY = 127
CURVE_DB = 1e-4  # the curve is written to 9 decimals


def test_the_hardest_hit_is_the_patch_and_the_softest_loses_what_play_says(played: patch.Patch) -> None:
    """Hit the kick at full strength, which is the patch itself as one hit, then at its softest and halfway."""
    assert play.hit(played, 1.0, 0) == dataclasses.replace(played, play=None)
    soft = play.hit(played, 0.0, 0)
    assert soft.play is None
    assert soft.body.level_db == played.body.level_db + PLAY.soft_db
    assert soft.attack.level_db == played.attack.level_db + PLAY.soft_db + PLAY.soft_click_db
    first = played.body.pitch[0][1]
    assert soft.body.pitch[0][1] == pytest.approx(1 + (first - 1) * PLAY.soft_drop)
    assert soft.body.pitch[-1][1] == 1.0
    half = play.hit(played, 0.5, 0)
    assert half.body.level_db == played.body.level_db + PLAY.soft_db / 2


def test_a_soft_hit_moves_the_sub_with_the_body_and_the_transient_with_the_click(
    layered: patch.Patch,
) -> None:
    """Hit a kick with a sub and a transient softly, keeping its whole pitch drop, and find the pitch curve as it was.

    The sub falls with the body, and the transient with the click.
    """
    full = patch.edit(layered, "sub", patch.Sub(-6.0, ((0.0, 0.0), (5.0, 1.0), (200.0, 0.0))))
    full = patch.edit(full, "play", patch.Play(-18.0, -6.0, 1.0, 0.0))
    soft = play.hit(full, 0.0, 0)
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
    assert play.hit(silent, 0.0, 0).attack.level_db == -math.inf


def test_the_repeats_never_change(played: patch.Patch) -> None:
    """Draw repeat 1 of seed 1, as every version must, and play it: another seed, small moves within spread."""
    assert play.draws(1, 1) == DRAWS
    again = play.hit(played, 1.0, 1)
    assert again.seed == DRAWS[0]
    assert abs(again.body.level_db - played.body.level_db) <= patch.LEVEL_DB * PLAY.spread
    assert play.hit(played, 1.0, 1) == again
    assert play.hit(played, 1.0, 2) != again


def test_a_kit_needs_play_and_its_hits_share_one_chain(played: patch.Patch, tight: patch.Patch) -> None:
    """Refuse layers for a patch without [play]; render a kit whose hits share a clipper and whose loudest is at -1."""
    with pytest.raises(ValueError, match=r"tight: velocity layers and round robins need a \[play\] section"):
        pack.kicks([("tight", "", tight)], [], 24, 2, 1)
    clipped = patch.edit(played, "clip.db", 3.0)
    kit = pack.kicks([("kick", "", clipped)], [], 24, 3, 2)
    assert [k.path for k in kit] == [f"kick-A1-v{j}-rr{k}.wav" for j in (1, 2, 3) for k in (1, 2)]
    ranges = [k.velocity for k in kit[::2]]
    assert ranges[0][0] == 1
    assert ranges[-1][1] == MAX_VELOCITY
    assert all(b[0] == a[1] + 1 for a, b in itertools.pairwise(ranges))  # softest first, every velocity once
    assert len({k.report.gain for k in kit}) == 1
    assert len({k.report.threshold for k in kit}) == 1
    assert kit[0].report.threshold == render.synthesized(clipped).threshold  # the hardest hit's, not none
    assert {(k.report.bits, k.report.ceiling) for k in kit} == {(24, render.ceiling_of(24))}  # the format asked for
    assert max(k.report.true_peak for k in kit) == pytest.approx(10 ** (-1 / 20), rel=1e-6)
    assert kit[0].report.loudness < kit[-1].report.loudness - LAYER_RANGE_DB


def test_a_kit_maps_its_layers_and_repeats_for_samplers(played: patch.Patch) -> None:
    """Map a kit of two layers and two repeats: a group per layer on the drum key, one-shot, its repeats in turn."""
    kit = pack.kicks([("kick", "", played)], [], 24, 2, 2)
    lines = pack.sfz("kick-A1", kit).splitlines()
    assert lines[0] == "// kick-A1: 2 velocity layers, 2 round robins, on key 36"
    groups = [line for line in lines if line.startswith("<group>")]
    assert len(groups) == len({k.velocity for k in kit})
    assert all("key=36 " in g and "amp_veltrack=100" in g and "loop_mode=one_shot seq_length=2" in g for g in groups)
    regions = [line for line in lines if line.startswith("<region>")]
    assert regions == [f"<region> sample={k.path} seq_position={k.robin}" for k in kit]


def test_the_kit_follows_the_square_law_at_every_velocity(played: patch.Patch) -> None:
    """Find each velocity's level, its layer's loudness plus its curve, on 40·log10(v/127) dB, and never above it."""
    kit = pack.kicks([("kick", "", played)], [], 24, 3, 2)
    loudness = pack.layer_loudness(kit)
    levels: dict[int, float] = {}
    gains: list[float] = []
    layer: tuple[int, int] = (0, 0)
    for line in pack.sfz("kick-A1", kit).splitlines():
        if line.startswith("<group>"):
            low, high = (int(w.split("=")[1]) for w in line.split() if w.startswith(("lovel=", "hivel=")))
            layer = (low, high)
            assert "volume=" not in line  # a group never raises its hits
        elif line.startswith("amp_velcurve_"):
            name, value = line.split("=")
            gains.append(float(value))
            levels[int(name.removeprefix("amp_velcurve_"))] = loudness[layer] + 20 * math.log10(float(value))
    assert sorted(levels) == list(range(1, MAX_VELOCITY + 1))
    assert max(gains) <= 1.0  # no velocity plays a hit louder than it was played
    for v, level in levels.items():
        assert level == pytest.approx(40 * math.log10(v / MAX_VELOCITY), abs=CURVE_DB)


def test_layers_sit_on_the_square_law_or_are_refused() -> None:
    """Run each layer's velocities up to its own on the law; refuse layers too close, or a softest out of reach."""
    assert pack.velocities([-12.0, -6.0, 0.0]) == [(1, 63), (64, 89), (90, 127)]  # their own: 63.6, 89.9 and 127
    assert pack.velocities([0.0]) == [(1, 127)]
    with pytest.raises(ValueError, match="layers v1 and v2 are within a velocity step"):
        pack.velocities([-3.0, -3.0, 0.0])
    with pytest.raises(ValueError, match=r"layer v1 is 90\.0 dB under the loudest"):
        pack.velocities([-90.0, 0.0])
    gains = pack.curve(-6.0, 64, 89)
    assert max(gains) <= 1.0
    assert 20 * math.log10(gains[0]) == pytest.approx(pack.law_db(64) + 6.0)


def test_a_spread_of_zero_repeats_the_hit_exactly(played: patch.Patch) -> None:
    """Play a kick with no spread, whose every repeat is the hit itself, noise and all."""
    still = patch.edit(played, "play.spread", 0.0)
    assert play.hit(still, 1.0, 1) == play.hit(still, 1.0, 0)


def test_only_a_played_patch_is_hit(tight: patch.Patch) -> None:
    """Refuse to hit a patch without [play] softer or again."""
    with pytest.raises(ValueError, match=r"only a patch with a \[play\] section"):
        play.hit(tight, 0.5, 1)


SOFTEST = "play: its softest hit, as far as spread moves it, would be invalid: "
HARDEST = "play: its hardest hit, as far as spread moves it, would be invalid: "


def test_a_patch_refuses_play_that_would_take_a_hit_outside_a_patch(played: patch.Patch) -> None:
    """Refuse [play] whose softest or loudest hit no patch could hold, naming every problem of both, when it is read.

    The softest hit's click falls below -60 dB; the loudest hit's attack rises above +24 dB, its pitch drop takes
    the body above 20 kHz, and its longer shape ends after 8 s.
    """
    soft = dataclasses.replace(played, play=dataclasses.replace(PLAY, soft_db=-30.0, soft_click_db=-24.0))
    with pytest.raises(patch.PatchError) as caught:
        patch.from_dict(patch.to_dict(soft))
    assert caught.value.problems == [f"{SOFTEST}attack.level_db: -71.5 is outside -60 … 24 (or -inf)"]
    loud = dataclasses.replace(
        played,
        attack=dataclasses.replace(played.attack, level_db=24.0),
        body=dataclasses.replace(
            played.body,
            pitch=((0.0, 20000.0 / played.body.tune_hz), (6.0, 1.0)),
            amp=((0.0, 0.0), (0.25, 1.0), (7900.0, 0.0)),
        ),
    )
    with pytest.raises(patch.PatchError) as caught:
        patch.from_dict(patch.to_dict(loud))
    found = caught.value.problems
    assert all(p.startswith(HARDEST) for p in found)
    assert {p.removeprefix(HARDEST).split(":")[0] for p in found} >= {
        "attack.level_db",
        "body.pitch[1].ratio",
        "body.amp[3].ms",
    }


def test_a_shorter_repeat_keeps_its_knots_apart(played: patch.Patch) -> None:
    """Refuse [play] whose shortest repeat would bring two knots of the body's shape closer than 10 µs."""
    amp = ((0.0, 0.0), (0.2505, 1.0), (0.2605, 0.5), (30.0, 0.0))  # 10 µs apart, off the µs grid
    close = dataclasses.replace(played, body=dataclasses.replace(played.body, amp=amp))
    with pytest.raises(patch.PatchError) as caught:
        patch.from_dict(patch.to_dict(close))
    assert caught.value.problems == [
        f"{SOFTEST}body.amp[3].ms: is 0.0095 ms after the previous knot; knots need 0.01 ms",
    ]


def test_every_hit_of_a_kit_renders_again_from_what_its_file_records(played: patch.Patch, tmp_path: Path) -> None:
    """Pack a clipped kit, and render each hit again from its WAV's patch and chain: the very samples written.

    The patch alone renders a kick of its own, and a kick packed on its own records no chain.
    """
    clipped = patch.edit(played, "clip.db", 3.0)
    kicks = pack.kicks([("kick", patch.dumps(clipped), clipped)], [], None, 3, 2)
    pack.write(tmp_path, kicks, None)
    for k in kicks:
        meta = json.loads(wav.chunks(tmp_path / k.path)[wav.PRIVATE])
        hit, chain = patch.loads(meta["patch"]), render.Chain(meta["kit"]["clip_threshold"], meta["kit"]["gain"])
        written = wav.read(tmp_path / k.path)[1][:, 0]
        assert np.array_equal(render.render(hit, None, chain)[0], written)
    soft = kicks[0]
    assert not np.array_equal(render.render(soft.patch)[0], soft.audio)
    single = pack.kicks([("kick", patch.dumps(clipped), clipped)], [], None)
    pack.write(tmp_path / "single", single, None)
    assert "kit" not in json.loads(wav.chunks(tmp_path / "single" / single[0].path)[wav.PRIVATE])


def test_each_layer_ends_where_the_law_reaches_the_mean_of_its_hits(played: patch.Patch) -> None:
    """Find each layer's top velocity where 40·log10(v/127) dB reaches its hits' mean loudness against the loudest."""
    kit = pack.kicks([("kick", "", patch.edit(played, "play.spread", 1.0))], [], None, 3, 3)
    layers = dict.fromkeys(k.velocity for k in kit)
    means = {v: float(np.mean([k.report.loudness for k in kit if k.velocity == v])) for v in layers}
    top = max(means.values())
    for (_, high), mean in means.items():
        assert high == math.floor(MAX_VELOCITY * 10 ** ((mean - top) / 40) + 1e-9)


def test_repeats_alone_make_a_kit_of_one_layer(played: patch.Patch) -> None:
    """Pack two repeats and no layers: one layer over every velocity, its files named by repeat only."""
    kit = pack.kicks([("kick", "", played)], [], None, 1, 2)
    assert [(k.path, k.velocity, k.robin) for k in kit] == [
        ("kick-A1-rr1.wav", (1, MAX_VELOCITY), 1),
        ("kick-A1-rr2.wav", (1, MAX_VELOCITY), 2),
    ]


def test_a_pack_keeps_every_kit_it_plays(played: patch.Patch) -> None:
    """Pack two played patches, each into its own kit, every hit of both kept."""
    kit = pack.kicks([("a", "", played), ("b", "", played)], [], None, 2, 1)
    assert [k.path for k in kit] == ["a-A1-v1.wav", "a-A1-v2.wav", "b-A1-v1.wav", "b-A1-v2.wav"]
    assert {k.kit for k in kit} == {"a-A1", "b-A1"}
