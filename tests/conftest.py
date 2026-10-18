"""Shared fixtures: the shipped presets, as text, as patches and as plain TOML data."""

from importlib import resources

import pytest

from ballast_synth import patch

PRESETS = ("round", "tight", "long", "driven")


def preset_text(name: str) -> str:
    return resources.files("ballast_synth").joinpath("data", "presets", f"{name}.toml").read_text(encoding="utf-8")


@pytest.fixture
def read_preset():
    """The text of a shipped preset, by name."""
    return preset_text


@pytest.fixture(params=PRESETS)
def preset(request) -> str:
    """The text of each shipped preset in turn."""
    return preset_text(request.param)


@pytest.fixture
def tight_text() -> str:
    return preset_text("tight")


@pytest.fixture
def tight(tight_text) -> patch.Patch:
    return patch.loads(tight_text)


@pytest.fixture
def raw(tight) -> dict:
    """The tight preset as the plain data TOML parses to, ready to be broken on purpose."""
    return patch.to_dict(tight)
