# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Playing: the kick hit softer, and hit again, as a drummer would.

The patch is the hardest hit. A softer one, at x from 1 (the hardest) down to
0 (the softest), loses (1 - x) times what [play] says the softest loses: its
level, before the drive, so that softer hits come out cleaner, a further
drop of its click, since a soft beater barely clicks, and part of its pitch
drop, since a soft hit bends the head less, every pitch ratio r becoming
1 + (r - 1) · (1 - (1 - soft_punch)(1 - x)).

Each repeat after the first is the same hit played again: its attack's noise
is drawn from a seed of its own, the greatest difference between two real
hits, and its level, pitch drop, length and click move by small amounts,
drawn from SHAKE-256 under `ballast/play/v1`, the patch's seed and the
repeat's number:

    level    ± 1 dB × spread
    punch    every pitch ratio's distance from 1, × (1 ± 4% × spread)
    length   the body's shape after its peak, × (1 ± 4% × spread)
    click    ± 2 dB × spread
"""

from __future__ import annotations

import dataclasses
import hashlib
import math

from .patch import MAX_SEED, Patch, edit, stretch

PLAY_KEY = b"ballast/play/v1"
LEVEL_DB, PUNCH, LENGTH, CLICK_DB = 1.0, 0.04, 0.04, 2.0  # how far a repeat moves, at a spread of 1


def _levels(p: Patch, db: float, click_db: float) -> Patch:
    """Return the patch with every source's level moved by db, and the click's and transient's by click_db more."""
    moves = [("body.level_db", db), ("attack.level_db", db + click_db)]
    if p.sub is not None:
        moves.append(("sub.level_db", db))
    if p.transient is not None:
        moves.append(("transient.level_db", db + click_db))
    for path, by in moves:
        section, name = path.split(".")
        level = getattr(getattr(p, section), name)
        if math.isfinite(level) and by != 0.0:
            p = edit(p, path, round(level + by, 3))
    return p


def _punch(p: Patch, keep: float) -> Patch:
    """Return the patch with every pitch ratio's distance from 1 scaled by `keep`."""
    if keep == 1.0:
        return p
    return edit(p, "body.pitch", tuple((ms, round(1 + (r - 1) * keep, 6)) for ms, r in p.body.pitch))


def softer(p: Patch, x: float) -> Patch:
    """Return the patch hit softer: at x = 1 the patch itself, at x = 0 its softest hit, as [play] describes it."""
    play = p.play
    if play is None or x == 1.0:
        return p
    soft = 1.0 - x
    return _punch(_levels(p, play.soft_db * soft, play.soft_click_db * soft), 1 - (1 - play.soft_punch) * soft)


def draws(seed: int, number: int) -> tuple[int, list[float]]:
    """Draw a repeat's own seed, and its four moves as uniforms in [-1, 1), from SHAKE-256."""
    raw = hashlib.shake_256(PLAY_KEY + seed.to_bytes(8, "little") + number.to_bytes(8, "little")).digest(40)
    words = [int.from_bytes(raw[8 * i : 8 * i + 8], "little") for i in range(5)]
    return words[0] & MAX_SEED, [2.0 * (w >> 11) * 2.0**-53 - 1.0 for w in words[1:]]


def again(p: Patch, number: int) -> Patch:
    """Return repeat `number` of the hit: 0 is the hit itself, every other one the same hit played again."""
    if p.play is None or number == 0:
        return p
    spread = p.play.spread
    seed, (level, punch, length, click) = draws(p.seed, number)
    p = _levels(dataclasses.replace(p, seed=seed), LEVEL_DB * spread * level, CLICK_DB * spread * click)
    p = _punch(p, 1 + PUNCH * spread * punch)
    return stretch(p, "body.amp", round(1 + LENGTH * spread * length, 6))
