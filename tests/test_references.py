# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The presets sound the same on every platform: samples of each, within a millionth of full scale of the stored."""

import numpy as np
import pytest

from ballast_synth import patch, render
from ballast_synth.cli import PRESET_DIR, PRESETS
from tools import references

STORED = references.stored()


def test_every_preset_has_its_references_at_both_base_rates() -> None:
    """Find a stored reference for every preset at 44.1 and 48 kHz, and none for a preset that is gone."""
    assert sorted(STORED) == sorted(f"{name}@{rate}" for name in PRESETS for rate in references.RATES)


@pytest.mark.parametrize("key", sorted(STORED))
def test_every_preset_renders_its_references(key: str) -> None:
    """Render a preset at a base rate, and find its kept samples within a millionth of full scale of the stored."""
    name, rate = key.split("@")
    preset = patch.loads(PRESET_DIR.joinpath(f"{name}.toml").read_text(encoding="utf-8"))
    audio, _ = render.render(patch.edit(preset, "output.sample_rate", int(rate)))
    kept = audio[references.picks(audio.size)].astype(float)
    assert np.max(np.abs(kept - STORED[key])) <= references.TOLERANCE
