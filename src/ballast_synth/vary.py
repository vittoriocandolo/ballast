# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Variations: patches near a patch, drawn from a seed the same way every time, or set along a grid of axes.

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

A grid instead sets fields along axes, such as drive.db at 6, 12 and 18, and
makes every combination. A step is a number, which sets the field; xF, which
stretches a curve by F, as `ballast stretch` does, or multiplies a number;
or any other text, which sets a choice, such as phase_ref's "tail".
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import math
from typing import TYPE_CHECKING

from .patch import BODY_HZ, CURVES, MAX_BODY_MS, SECTIONS, TONE_HZ, Patch, PatchError, edit, stretch

if TYPE_CHECKING:
    from collections.abc import Callable

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


def _inside(x: float, lo: float, hi: float, digits: int, fits: Callable[[float], bool]) -> float:
    """Return x within lo … hi, rounded to `digits` decimals, and stepped back inside if the rounding left `fits`.

    The bounds come from dividing a limit, so their rounding can land a hair outside it; `fits` checks the value
    as the patch itself will.
    """
    step = 10.0**-digits
    x = round(min(max(x, lo), hi), digits)
    while not fits(x):
        x = round(x - step if 2 * x > lo + hi else x + step, digits)
    return x


def _factor(u: float, octaves: float, top: float, fits: Callable[[float], bool]) -> float:
    """Return 2^(u · octaves), at most `top`, rounded to 3 decimals, within what `fits`."""
    factor: float = 2.0 ** (u * octaves)
    return _inside(factor, 0.0, top, 3, fits)


def vary(p: Patch, seed: int, number: int) -> tuple[Patch, list[str]]:
    """Return variation `number` of the patch for `seed`, and what each move did, as "punch ×1.12"."""
    u = draws(seed, number)
    b, a, done = p.body, p.attack, []
    ratio = b.pitch[0][1]
    lo, hi = BODY_HZ[0] / b.tune_hz, BODY_HZ[1] / b.tune_hz
    first = _inside(ratio * 2.0 ** (PUNCH * u["punch"]), lo, hi, 3, lambda r: BODY_HZ[0] <= b.tune_hz * r <= BODY_HZ[1])
    p = edit(p, "body.pitch", ((0.0, first), *b.pitch[1:]))
    done.append(f"punch ×{first / ratio:.3g}")
    end = b.pitch[-1][0]
    snap = _factor(u["snap"], SNAP, MAX_BODY_MS / end, lambda f: round(f * end, 3) <= MAX_BODY_MS)
    p = stretch(p, "body.pitch", snap)
    done.append(f"snap ×{snap:g}")
    peak = next(ms for ms, g in b.amp if g == 1.0)
    span = b.amp[-1][0] - peak
    length = _factor(
        u["length"],
        LENGTH,
        (MAX_BODY_MS - peak) / span,
        lambda f: round(peak + f * span, 3) <= MAX_BODY_MS,
    )
    p = stretch(p, "body.amp", length)
    done.append(f"length ×{length:g}")
    if math.isfinite(a.level_db):
        level = round(min(max(a.level_db + CLICK_DB * u["click"], -60.0), 24.0), 1)
        p = edit(p, "attack.level_db", level)
        done.append(f"click {level - a.level_db:+.1f} dB")
    reach = max(r for _, r in a.pitch), min(r for _, r in a.pitch)  # the tone's highest and lowest ratios
    tone = _inside(
        a.tone_hz * 2.0 ** (TONE * u["tone"]),
        TONE_HZ[0] / reach[1],
        TONE_HZ[1] / reach[0],
        0,
        lambda f: TONE_HZ[0] <= f * reach[1] and f * reach[0] <= TONE_HZ[1],
    )
    p = edit(p, "attack.tone_hz", tone)
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


def step(p: Patch, path: str, text: str) -> Patch:
    """Return the patch with one step of an axis taken: a number set, xF stretched or multiplied, or a choice set."""
    section, _, name = path.partition(".")
    kinds = {f.name: f.type for f in dataclasses.fields(SECTIONS[section])} if section in SECTIONS else {}
    if name not in kinds:
        msg = f"{path}: no such field"
        raise PatchError([msg])
    if text.startswith("x"):
        try:
            factor = float(text[1:])
        except ValueError:
            msg = f"{path}: {text!r} is not a factor such as x1.5"
            raise PatchError([msg]) from None
        if path in CURVES:
            return stretch(p, path, factor)
        part = getattr(p, section)
        current = None if part is None else getattr(part, name)
        if not isinstance(current, float):
            msg = f"{path}: only a number or a curve can be multiplied"
            raise PatchError([msg])
        return edit(p, path, round(current * factor, 6))
    try:
        value: object = int(text) if text.lstrip("-").isdigit() else float(text)
    except ValueError:
        value = text
    if isinstance(value, int) and kinds[name] != "int":
        value = float(value)  # 6 means 6.0 for a number; integer fields, such as polarity, keep it whole
    return edit(p, path, value)


def grid(p: Patch, axes: list[tuple[str, list[str]]]) -> list[tuple[Patch, list[str]]]:
    """Make every combination of the axes' steps, the first axis outermost, or report every one that fails."""
    known = {f"{name}.{f.name}" for name, cls in SECTIONS.items() for f in dataclasses.fields(cls)}
    if unknown := [f"{path}: no such field" for path, _ in axes if path not in known]:
        raise PatchError(unknown)
    out, problems = [], []
    for combination in itertools.product(*([(path, text) for text in steps] for path, steps in axes)):
        q, done = p, []
        try:
            for path, text in combination:
                q = step(q, path, text)
                done.append(f"{path} {text}")
        except PatchError as e:
            where = ", ".join(f"{path} {text}" for path, text in combination)
            problems += [f"{where}: {problem}" for problem in e.problems]
            continue
        out.append((q, done))
    if problems:
        raise PatchError(problems)
    return out
