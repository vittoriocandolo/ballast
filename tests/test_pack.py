# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Packs: names by key or frequency, and the gates every kick must pass."""

import dataclasses

import numpy as np
import pytest

from ballast_synth import pack, patch, render


@pytest.mark.parametrize(("hz", "name"), [(55.0, "A1"), (43.654, "F1"), (50.0, "50Hz"), (55.016, "55.016Hz")])
def test_a_kick_bears_its_key_only_within_half_a_cent(hz: float, name: str) -> None:
    """Name kicks on a note by the note, and any other by its frequency."""
    assert pack.label(hz) == name


def test_the_presets_pass_every_gate_and_a_click_at_either_edge_does_not(tight: patch.Patch) -> None:
    """Pass the tight preset, then fail it with a loud first or last sample."""
    audio, report = render.render(tight, 24)
    kick = pack.Kick("tight-A1.wav", tight, patch.dumps(tight), audio, report)
    assert pack.gates(kick) == []
    for edge in (0, -1):
        clicked = audio.copy()
        clicked[edge] = 0.01
        problems = pack.gates(dataclasses.replace(kick, audio=clicked))
        assert problems == [f"{'first' if edge == 0 else 'last'} sample at -40.0 dBFS, above -90 dBFS: it would click"]
    assert np.array_equal(kick.audio, audio)
