"""The version the code reports is the version the package was built with."""

from importlib.metadata import version

import ballast


def test_the_version_is_declared_once():
    assert ballast.__version__ == version("ballast")
