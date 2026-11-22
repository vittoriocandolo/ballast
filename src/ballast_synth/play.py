# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Playing: the kick hit softer, and repeated.

The patch is the hardest hit. A softer one, at x from 1 (the hardest) down to 0
(the softest), loses (1 - x) times what [play] says the softest loses: its
level, before the drive, so the drive adds fewer harmonics to softer hits; a
further drop of its click; and part of its pitch drop, every pitch ratio r
becoming 1 + (r - 1) · (1 - (1 - soft_drop)(1 - x)).

Each repeat after the first is the same hit again, unless the spread is 0: its
attack's noise is drawn from a seed of its own, and its level, pitch drop,
length and click move by small amounts, drawn from SHAKE-256 under
`ballast/play/v1`, the patch's seed and the repeat's number:

    level    ± 1 dB × spread
    drop     every pitch ratio's distance from 1, × (1 ± 4% × spread)
    length   the body's shape after its peak, × (1 ± 4% × spread)
    click    ± 2 dB × spread

Each hit is a patch of its own, without [play]: one hit, as its file records
it. The parser refuses a patch whose [play] would take any hit outside what a
patch may hold, and the moves themselves are in patch.py, which checks them.
"""

from __future__ import annotations

import hashlib

from .patch import MAX_SEED, Patch, struck

PLAY_KEY = b"ballast/play/v1"


def draws(seed: int, number: int) -> tuple[int, list[float]]:
    """Draw a repeat's own seed, and its four moves as uniforms in [-1, 1), from SHAKE-256."""
    raw = hashlib.shake_256(PLAY_KEY + seed.to_bytes(8, "little") + number.to_bytes(8, "little")).digest(40)
    words = [int.from_bytes(raw[8 * i : 8 * i + 8], "little") for i in range(5)]
    return words[0] & MAX_SEED, [2.0 * (w >> 11) * 2.0**-53 - 1.0 for w in words[1:]]


def hit(p: Patch, x: float, number: int) -> Patch:
    """Return the kick hit at strength x, from 1 (the patch itself) down to 0 (its softest), as repeat `number`.

    Repeat 0 is the hit itself, and so is every repeat at a spread of 0. The hit is a patch of one hit: it has no
    [play].
    """
    play = p.play
    if play is None:
        msg = "only a patch with a [play] section can be hit softer or again"
        raise ValueError(msg)
    if number == 0 or play.spread == 0.0:
        return struck(p, x, p.seed, (0.0, 0.0, 0.0, 0.0))
    seed, (level, drop, length, click) = draws(p.seed, number)
    return struck(p, x, seed, (level, drop, length, click))
