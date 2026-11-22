# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The documentation's links lead somewhere, every figure and sound in docs/ is shown, and its example is real."""

import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

from tools import readme

ROOT = Path(__file__).resolve().parent.parent
PAGES = sorted([*ROOT.glob("*.md"), *(ROOT / "docs").glob("*.md")])
LINK = re.compile(r"\]\(([^)\s]+)\)")  # the target of a Markdown link or image


@pytest.fixture
def read_pyproject() -> dict[str, Any]:
    """Return the project's pyproject.toml, parsed."""
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def prose(page: Path) -> str:
    """Read a page without its code blocks, where brackets and hashes are neither links nor headings."""
    return re.sub(r"^```.*?^```", "", page.read_text(encoding="utf-8"), flags=re.MULTILINE | re.DOTALL)


def anchors(page: Path) -> set[str]:
    """Return the anchors that GitHub gives a page's headings."""
    headings = re.findall(r"^#+ (.+)$", prose(page), flags=re.MULTILINE)
    return {re.sub(r"[^\w\- ]", "", heading.strip()).lower().replace(" ", "-") for heading in headings}


def targets(page: Path) -> list[str]:
    """List the targets of a page's links and images, other than web addresses."""
    return [target for target in LINK.findall(prose(page)) if not re.match(r"[a-z]+:", target)]


@pytest.mark.parametrize("page", PAGES, ids=lambda page: page.name)
def test_every_link_leads_somewhere(page: Path) -> None:
    """Find the file of every local link, and the heading of every anchor."""
    for target in targets(page):
        path, _, anchor = target.partition("#")
        linked = (page.parent / path).resolve() if path else page
        assert linked.is_file(), f"{target}: no such file"
        if anchor and linked.suffix == ".md":
            assert anchor in anchors(linked), f"{target}: no such heading"


def test_every_figure_and_sound_is_shown() -> None:
    """Find a link to every file in docs/figures and docs/audio, so that none is left behind."""
    linked = {(page.parent / target.partition("#")[0]).resolve() for page in PAGES for target in targets(page)}
    files = {file.resolve() for folder in ("figures", "audio") for file in (ROOT / "docs" / folder).iterdir()}
    assert files <= linked, sorted(file.name for file in files - linked)


def test_the_patch_reference_shows_the_tight_preset_as_it_ships(tight_text: str) -> None:
    """Find the patch reference's first example the very text of the tight preset, which it says it is."""
    page = (ROOT / "docs" / "patch.md").read_text(encoding="utf-8")
    start = page.index("```toml\n") + len("```toml\n")
    assert page[start : page.index("```", start)] == tight_text


def test_the_readme_on_pypi_links_into_the_repository_at_the_release(read_pyproject: dict[str, Any]) -> None:
    """Make the README's links absolute as the build does: pages in the repository, media raw, all at the tag."""
    repository = read_pyproject["project"]["urls"]["Repository"]
    text = readme.absolute((ROOT / "README.md").read_text(encoding="utf-8"), repository, "v1.2.3")
    found = LINK.findall(text)
    assert found
    assert all(t.startswith(("https://", "#")) for t in found)
    raw = repository.replace("https://github.com/", "https://raw.githubusercontent.com/")
    assert f"]({raw}/v1.2.3/docs/figures/presets.svg)" in text
    assert f"]({repository}/blob/v1.2.3/docs/guide.md)" in text
    assert all(url.startswith(repository) for url in read_pyproject["project"]["urls"].values())
