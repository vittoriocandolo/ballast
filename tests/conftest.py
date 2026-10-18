# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared fixtures: the shipped presets, as text, as patches and as plain TOML data."""

from collections.abc import Callable
from importlib import resources
from typing import Any

import pytest

from ballast_synth import patch

PRESETS = ("round", "tight", "long", "driven")


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
def raw(tight: patch.Patch) -> dict[str, Any]:
    """Return the tight preset as the plain data TOML parses to, ready to be broken on purpose."""
    return patch.to_dict(tight)
