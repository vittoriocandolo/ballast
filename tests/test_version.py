# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The version the code reports is the version the package was built with."""

from importlib.metadata import version

import ballast_synth


def test_the_version_is_declared_once() -> None:
    """Compare the version in the code with the one in the installed package's metadata."""
    assert ballast_synth.__version__ == version("ballast-synth")
