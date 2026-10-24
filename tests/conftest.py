# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared fixtures: the shipped presets, as text, as patches and as plain TOML data, and a click to layer."""

import math
from collections.abc import Callable
from importlib import resources
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from ballast_synth import patch
from ballast_synth.cli import PRESETS

CLICK_RATE = 48000


def preset_text(name: str) -> str:
    """Read the text of a shipped preset."""
    return resources.files("ballast_synth").joinpath("data", "presets", f"{name}.toml").read_text(encoding="utf-8")


@pytest.fixture
def read_preset() -> Callable[[str], str]:
    """Return the reader of a shipped preset's text, by name."""
    return preset_text


@pytest.fixture(params=PRESETS)
def preset(request: pytest.FixtureRequest) -> str:
    """Return the text of each shipped preset in turn."""
    return preset_text(request.param)


@pytest.fixture
def tight_text() -> str:
    """Return the text of the tight preset."""
    return preset_text("tight")


@pytest.fixture
def tight(tight_text: str) -> patch.Patch:
    """Return the tight preset."""
    return patch.loads(tight_text)


@pytest.fixture
def gain_text(tight_text: str) -> str:
    """Return the tight preset's text at a plain gain of -6 dB, for the behavior of gain mode."""
    return tight_text.replace('mode        = "true-peak"', 'mode        = "gain"     ').replace(
        "db          = -1.0",
        "db          = -6.0",
    )


@pytest.fixture
def gain(gain_text: str) -> patch.Patch:
    """Return the tight preset at a plain gain of -6 dB."""
    return patch.loads(gain_text)


@pytest.fixture
def raw(tight: patch.Patch) -> dict[str, Any]:
    """Return the tight preset as the plain data TOML parses to, ready to be broken on purpose."""
    return patch.to_dict(tight)


SUB = patch.Sub(-6.0, ((0.0, 0.0), (5.0, 1.0), (150.0, 0.6), (400.0, 0.0)))  # a sub that outlasts the body


@pytest.fixture
def click() -> NDArray[np.float64]:
    """Return a click to layer, at 48 kHz: three decaying modes of a struck block, 30 ms long, peaking at -12 dBFS."""
    t = np.arange(round(0.03 * CLICK_RATE)) / CLICK_RATE
    x = np.zeros_like(t)
    for hz, ms, amplitude in ((1730.0, 3.0, 1.0), (4210.0, 1.2, 0.6), (7650.0, 0.6, 0.35)):
        x += amplitude * np.exp(-1000 * t / ms) * np.sin(2 * math.pi * hz * t)
    x *= 10 ** (-12 / 20) / float(np.max(np.abs(x)))
    return x


@pytest.fixture
def layered(tight: patch.Patch, click: NDArray[np.float64]) -> patch.Patch:
    """Return the tight preset with the click layered as its transient."""
    return patch.edit(tight, "transient", patch.Transient.of(click, CLICK_RATE))
