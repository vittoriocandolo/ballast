# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Variations: patches near a patch, drawn from a seed, the same ones every time.

Each variation draws one uniform in [-1, 1) per move from SHAKE-256, keyed by
`ballast/vary/v1`, the seed and the variation's number, and makes these
moves, each bounded to what the patch allows and rounded to read cleanly:

    punch    the first pitch ratio, × 2^(±0.5)
    snap     the pitch curve's times, × 2^(±0.5)
    length   the body's shape after its peak, × 2^(±0.4)
    click    the attack's level, ± 4 dB
    tone     the attack's tone, × 2^(±0.5)
    drive    the drive, ± 4 dB
    bright   the body's brightness, × 2^(±1), when it has any

The tuning, the levels of the body, sub and transient, and everything else
stay as they were. The draws never depend on NumPy, so the same seed gives
the same variations in every version.
"""

from __future__ import annotations

import hashlib
import math

from .patch import BODY_HZ, MAX_BODY_MS, TONE_HZ, Patch, edit, stretch

VARY_KEY = b"ballast/vary/v1"
MOVES = ("punch", "snap", "length", "click", "tone", "drive", "bright")
PUNCH = SNAP = TONE = 0.5  # octaves either way
LENGTH = 0.4  # octaves either way
BRIGHT = 1.0  # octaves either way
CLICK_DB = DRIVE_DB = 4.0  # dB either way


def draws(seed: int, number: int) -> dict[str, float]:
    """Draw one uniform in [-1, 1) per move for variation `number` of `seed`: 53 bits of SHAKE-256 each."""
    raw = hashlib.shake_256(VARY_KEY + seed.to_bytes(8, "little") + number.to_bytes(8, "little")).digest(8 * len(MOVES))
    words = (int.from_bytes(raw[8 * i : 8 * i + 8], "little") >> 11 for i in range(len(MOVES)))
    return {move: 2.0 * word * 2.0**-53 - 1.0 for move, word in zip(MOVES, words, strict=True)}


def _factor(u: float, octaves: float, top: float) -> float:
    """Return 2^(u · octaves), at most `top`, rounded to 3 decimals."""
    factor: float = 2.0 ** (u * octaves)
    return round(min(factor, top), 3)


def vary(p: Patch, seed: int, number: int) -> tuple[Patch, list[str]]:
    """Return variation `number` of the patch for `seed`, and what each move did, as "punch ×1.12"."""
    u = draws(seed, number)
    b, a, done = p.body, p.attack, []
    ratio = b.pitch[0][1]
    first = round(min(max(ratio * 2.0 ** (PUNCH * u["punch"]), BODY_HZ[0] / b.tune_hz), BODY_HZ[1] / b.tune_hz), 3)
    p = edit(p, "body.pitch", ((0.0, first), *b.pitch[1:]))
    done.append(f"punch ×{first / ratio:.3g}")
    snap = _factor(u["snap"], SNAP, MAX_BODY_MS / b.pitch[-1][0])
    p = stretch(p, "body.pitch", snap)
    done.append(f"snap ×{snap:g}")
    peak = next(ms for ms, g in b.amp if g == 1.0)
    length = _factor(u["length"], LENGTH, (MAX_BODY_MS - peak) / (b.amp[-1][0] - peak))
    p = stretch(p, "body.amp", length)
    done.append(f"length ×{length:g}")
    if math.isfinite(a.level_db):
        level = round(min(max(a.level_db + CLICK_DB * u["click"], -60.0), 24.0), 1)
        p = edit(p, "attack.level_db", level)
        done.append(f"click {level - a.level_db:+.1f} dB")
    reach = max(r for _, r in a.pitch), min(r for _, r in a.pitch)  # the tone's highest and lowest ratios
    tone = a.tone_hz * 2.0 ** (TONE * u["tone"])
    tone = round(min(max(tone, TONE_HZ[0] / reach[1]), TONE_HZ[1] / reach[0]))
    p = edit(p, "attack.tone_hz", float(tone))
    done.append(f"tone ×{tone / a.tone_hz:.3g}")
    before = p.drive.db
    drive = round(min(max(before + DRIVE_DB * u["drive"], 0.0), 48.0), 1)
    p = edit(p, "drive.db", drive)
    done.append(f"drive {drive - before:+.1f} dB")
    if b.bright_hz != 0.0:
        bright = float(round(min(max(b.bright_hz * 2.0 ** (BRIGHT * u["bright"]), 50.0), 20000.0)))
        p = edit(p, "body.bright_hz", bright)
        done.append(f"bright ×{bright / b.bright_hz:.3g}")
    return p, done
