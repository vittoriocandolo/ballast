# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Make the README's relative links absolute, for PyPI, which shows the README without the repository around it.

Pages and files point at the repository's page for them, figures and sounds at their raw files, all at one ref,
the tag of the version being built, so the README on PyPI shows the documentation of that very release.
"""

from __future__ import annotations

import re

LINK = re.compile(r"(\]\()([^)\s]+)\)")  # the target of a Markdown link or image
MEDIA = (".svg", ".png", ".wav")  # shown or played from the raw file, not from the repository's page


def absolute(text: str, repository: str, ref: str) -> str:
    """Point every relative link and image of README text at a GitHub repository, at `ref`."""
    raw = repository.replace("https://github.com/", "https://raw.githubusercontent.com/", 1)

    def target(match: re.Match[str]) -> str:
        path = match[2]
        if re.match(r"[a-z]+:|#", path):  # a web address, or an anchor on the same page
            return match[0]
        base = f"{raw}/{ref}" if path.endswith(MEDIA) else f"{repository}/blob/{ref}"
        return f"{match[1]}{base}/{path})"

    return LINK.sub(target, text)
