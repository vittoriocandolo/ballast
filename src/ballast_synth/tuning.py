# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tuning: MIDI note numbers and names, equal-tempered frequencies, and the tunings a patch is given.

Shared by `ballast init` and `ballast tune`, the render report, packs and the WAV root note.
"""

from __future__ import annotations

import math
import re

_NOTES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def midi_pitch(hz: float) -> float:
    """Return the fractional MIDI note number of a frequency (A4 = 69 = 440 Hz)."""
    return 69.0 + 12.0 * math.log2(hz / 440.0)


def note_name(n: int) -> str:
    """Return the name of MIDI note n, such as 'F#1'."""
    return f"{_NOTES[n % 12]}{n // 12 - 1}"


def note_number(text: str) -> int | None:
    """Return the MIDI number of a note name such as F1, C#2 or Db2, or None."""
    m = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d+)", text.strip())
    if m is None:
        return None
    return _NOTES.index(m[1].upper()) + {"#": 1, "b": -1, "": 0}[m[2]] + 12 * (int(m[3]) + 1)


def note_hz(n: int) -> float:
    """Return the equal-tempered frequency of MIDI note n, A4 = 440 Hz, rounded to 3 decimals as a patch writes it."""
    ratio: float = 2.0 ** ((n - 69) / 12)
    return round(440.0 * ratio, 3)


def describe_tuning(hz: float) -> str:
    """Describe a frequency by its nearest note and its offset in cents, such as 'G1 +35.3 cents'."""
    pitch = midi_pitch(hz)
    n = round(pitch)
    cents = round(100 * (pitch - n), 1) + 0.0  # + 0.0 turns -0.0 into 0.0
    return f"{note_name(n)} {cents:+.1f} cents"


def parse_tune(text: str) -> float:
    """Parse a note name such as F1, C#2 or Db2 (Hz rounded to 3 decimals), or a frequency in Hz."""
    n = note_number(text)
    if n is not None:
        return note_hz(n)
    try:
        return float(text)
    except ValueError:
        msg = f"not a note name or frequency: {text!r}"
        raise ValueError(msg) from None
