# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Variations: drawn from a seed, the same in every version, always valid, near the patch they vary."""

import math

import pytest

from ballast_synth import patch, vary
from ballast_synth.cli import PRESETS

from .conftest import preset_text


def test_the_draws_never_change() -> None:
    """Draw the moves of seed 1, variation 1, as every version of Ballast must."""
    assert vary.draws(1, 1) == {
        "punch": -0.3565903717224823,
        "snap": -0.923924873105054,
        "length": 0.5429777761580306,
        "click": 0.8699582567616528,
        "tone": -0.2944284380396489,
        "drive": 0.3964504150628425,
        "bright": 0.31754644019652867,
    }


def test_each_seed_and_number_draws_its_own_moves() -> None:
    """Draw for two seeds and two numbers: four different sets, every draw in [-1, 1)."""
    sets = [vary.draws(seed, number) for seed in (1, 2) for number in (1, 2)]
    assert len({tuple(d.values()) for d in sets}) == len(sets)
    assert all(-1.0 <= u < 1.0 for d in sets for u in d.values())


@pytest.mark.parametrize("name", PRESETS)
def test_variations_are_valid_and_keep_the_tuning_and_levels(name: str) -> None:
    """Vary every preset 20 times: valid patches, each the same for the same seed, within their moves' reach."""
    p = patch.loads(preset_text(name))
    for number in range(1, 21):
        q, moves = vary.vary(p, 5, number)
        assert patch.loads(patch.dumps(q)) == q
        assert vary.vary(p, 5, number)[0] == q
        assert (q.body.tune_hz, q.body.level_db, q.sub, q.transient, q.output) == (
            p.body.tune_hz,
            p.body.level_db,
            p.sub,
            p.transient,
            p.output,
        )
        reach = 2.0**vary.PUNCH + 0.001  # and rounding to 3 decimals
        assert 1 / reach <= q.body.pitch[0][1] / p.body.pitch[0][1] <= reach
        assert abs(q.drive.db - p.drive.db) <= vary.DRIVE_DB
        assert len(moves) == 6 + (p.body.bright_hz != 0.0)


def test_a_silent_click_stays_silent(tight: patch.Patch) -> None:
    """Vary a patch whose attack is off, which keeps it off and says nothing of the click."""
    q, moves = vary.vary(patch.edit(tight, "attack.level_db", -math.inf), 1, 1)
    assert q.attack.level_db == -math.inf
    assert not any(move.startswith("click") for move in moves)
