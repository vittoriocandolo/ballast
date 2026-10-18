# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The patch: a complete, readable description of one kick.

Every field is required and the renderer holds no defaults, so a future
default can never change an old patch. Each field declares its rule and its
comment in the canonical TOML once, on its dataclass, and parsing, dumping
and editing all work from those declarations. The parser rejects anything
outside the valid domain, reporting every problem with its field path, and
never clamps or repairs.
"""

from __future__ import annotations

import codecs
import dataclasses
import datetime as dt
import math
import re
import tomllib
from dataclasses import dataclass, field, fields
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast

if TYPE_CHECKING:
    import os
    from collections.abc import Callable, Collection, Iterable

type Knots = tuple[tuple[float, float], ...]  # a curve's (ms, value) knots

SCHEMA = 2  # schema 1 had no output.low_cut_hz; its patches still render as they did
SAMPLE_RATES = (44100, 48000, 88200, 96000, 176400, 192000)
MAX_BODY_MS = 8000.0
MAX_ATTACK_MS = 1000.0
MIN_SPACING_MS = Decimal("0.01")  # 10 µs, compared on the decimal values as written
MAX_SEED = 2**63 - 1
PEAK_MODES = ("peak", "true-peak")  # the modes whose `db` is a target at or below full scale
BODY_HZ = (10.0, 20000.0)  # where the body's frequency must lie at every pitch knot


class PatchError(ValueError):
    """An invalid patch. `problems` lists every reason, each with its field path."""

    def __init__(self, problems: Iterable[str]) -> None:
        """Keep every problem, and make the message one problem per line."""
        self.problems = list(problems)
        super().__init__("\n".join(self.problems))


_KINDS = {  # every type TOML parses to
    bool: "a boolean",
    int: "an integer",
    float: "a float",
    str: "a string",
    list: "an array",
    dict: "a table",
    dt.datetime: "a date-time",
    dt.date: "a date",
    dt.time: "a time",
}


def _kind(v: object) -> str:
    return _KINDS.get(type(v), type(v).__name__)


def _keys(problems: list[str], prefix: str, table: dict[str, Any], expected: Collection[str]) -> bool:
    """Report unknown and missing keys; True when none is missing."""
    problems += [f"{prefix}{key}: unknown field" for key in table if key not in expected]
    missing = [key for key in expected if key not in table]
    problems += [f"{prefix}{key}: missing" for key in missing]
    return not missing


class _Bounds(NamedTuple):
    """The range a number must lie in: lo … hi, or below hi when `below` is set."""

    lo: float
    hi: float
    below: bool = False


def _number(
    path: str,
    v: object,
    problems: list[str],
    bounds: _Bounds | None = None,
    *,
    off: float | None = None,
) -> float | None:
    """Return v as a finite float within `bounds`, or as `off`, a value outside them that turns the field off.

    Report the problem and return None if it is neither.
    """
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        problems.append(f"{path}: expected a number, found {_kind(v)}")
        return None
    try:
        x = float(v)
    except OverflowError:
        problems.append(f"{path}: is too large")
        return None
    if x == off:
        return x
    or_off = "" if off is None else f" (or {off:g})"
    if not math.isfinite(x):
        problems.append(f"{path}: must be finite{or_off}")
        return None
    if bounds is not None:
        lo, hi, below = bounds
        if x < lo or (x >= hi if below else x > hi):
            problems.append(f"{path}: {x:g} is outside {lo:g} … {hi:g}" + (" (exclusive)" if below else "") + or_off)
            return None
    return x


def _toml(v: object) -> str:
    """Format a scalar or a pair in TOML, floats by repr so that they read back exactly."""
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, tuple):
        return f"[{', '.join(map(repr, v))}]"
    return repr(v)


# Field declarations. Each carries `parse(path, raw, problems, section)`, which
# appends any problems and returns the normalized value or None; `section` holds
# the fields of the same section parsed before it. `key` names the value of each
# knot in a curve. A declaration is typed as the value its field holds, as
# dataclasses.field is.

if TYPE_CHECKING:
    type Parse[T] = Callable[[str, Any, list[str], dict[str, Any]], T | None]
    type Check = Callable[[str, Knots, list[str], dict[str, Any]], None]


def _rule[T](parse: Parse[T], note: str = "", key: str | None = None) -> T:
    return cast("T", field(metadata={"parse": parse, "note": note, "key": key}))


def number(
    lo: float,
    hi: float | Callable[[dict[str, Any]], float],
    note: str = "",
    *,
    below: bool = False,
    off: float | None = None,
) -> float:
    """Declare a number in lo … hi, or `off`; `hi` may be a function of the fields before it."""

    def parse(path: str, v: object, problems: list[str], section: dict[str, Any]) -> float | None:
        top = hi(section) if callable(hi) else hi
        return _number(path, v, problems, _Bounds(lo, top, below=below), off=off)

    return _rule(parse, note)


def choice[T](allowed: tuple[T, ...], note: str = "") -> T:
    """Declare one of `allowed`, and of its type: neither true nor 48000.0 passes for an integer."""

    def parse(path: str, v: object, problems: list[str], _section: dict[str, Any]) -> T | None:
        if type(v) is type(allowed[0]) and v in allowed:
            return v
        problems.append(f"{path}: expected one of {', '.join(map(repr, allowed))}, found {v!r}")
        return None

    return _rule(parse, note)


def band(lo: float, hi: float, note: str = "") -> tuple[float, float]:
    """Declare a [low, high] pair of frequencies."""

    def parse(path: str, v: object, problems: list[str], _section: dict[str, Any]) -> tuple[float, float] | None:
        match v:
            case [first, second]:
                low = _number(f"{path}[0]", first, problems, _Bounds(lo, hi))
                high = _number(f"{path}[1]", second, problems, _Bounds(lo, hi))
            case _:
                problems.append(f"{path}: expected [low, high] in Hz")
                return None
        if low is None or high is None:
            return None
        if low >= high:
            problems.append(f"{path}: the low edge must be below the high edge")
        return low, high

    return _rule(parse, note)


def knots(key: str, count: tuple[int, int], max_ms: float, check: Check, note: str = "") -> Knots:
    """Declare a curve: knots ((ms, value), ...), written as an array of { ms, key } tables, then `check`ed whole."""

    def parse(path: str, raw: object, problems: list[str], section: dict[str, Any]) -> Knots | None:
        if not isinstance(raw, list):
            problems.append(f"{path}: expected an array of {{ ms, {key} }} tables, found {_kind(raw)}")
            return None
        malformed = False
        for i, knot in enumerate(raw):
            if not isinstance(knot, dict):
                problems.append(f"{path}[{i}]: expected a {{ ms, {key} }} table, found {_kind(knot)}")
                malformed = True
            elif not _keys(problems, f"{path}[{i}].", knot, ("ms", key)):
                malformed = True
        if malformed:
            return None
        counted = count[0] <= len(raw) <= count[1]
        if not counted:
            problems.append(f"{path}: needs {count[0]} to {count[1]} knots, found {len(raw)}")
        pairs = tuple(
            (
                _number(f"{path}[{i}].ms", k["ms"], problems, _Bounds(0.0, max_ms)),
                _number(f"{path}[{i}].{key}", k[key], problems),
            )
            for i, k in enumerate(raw)
        )
        if not counted or any(None in pair for pair in pairs):
            return None
        curve = cast("Knots", pairs)  # every value is a float: checked just above
        _spacing(path, curve, problems)
        check(path, curve, problems, section)
        return curve

    return _rule(parse, note, key)


def _spacing(path: str, curve: Knots, problems: list[str]) -> None:
    """Check that the knot times strictly increase, at least MIN_SPACING_MS apart as written."""
    for i in range(1, len(curve)):
        a, b = Decimal(repr(curve[i - 1][0])), Decimal(repr(curve[i][0]))
        if b == a:
            problems.append(f"{path}[{i}].ms: repeats the previous knot time")
        elif b < a:
            problems.append(f"{path}[{i}].ms: is earlier than the previous knot")
        elif b - a < MIN_SPACING_MS:
            problems.append(f"{path}[{i}].ms: is {b - a} ms after the previous knot; knots need 0.01 ms (10 µs)")


def _pitch(path: str, curve: Knots, problems: list[str], section: dict[str, Any]) -> None:
    """Check that the pitch curve starts at 0 ms, ends at ratio 1, and keeps the body within 10 Hz … 20 kHz."""
    if curve[0][0] != 0.0:
        problems.append(f"{path}[0].ms: the pitch curve must start at 0 ms")
    if curve[-1][1] != 1.0:
        problems.append(f"{path}[{len(curve) - 1}].ratio: the pitch curve must end at ratio 1")
    tune = section["tune_hz"]
    lo, hi = BODY_HZ
    for i, (_, r) in enumerate(curve):
        if tune is not None and not lo <= tune * r <= hi:
            problems.append(f"{path}[{i}].ratio: body frequency {tune * r:g} Hz is outside 10 Hz … 20 kHz")
        elif tune is None and r <= 0:
            problems.append(f"{path}[{i}].ratio: must be positive")


def _shape(path: str, curve: Knots, problems: list[str], _section: dict[str, Any]) -> None:
    """Check an amplitude shape: gains in 0 … 1, 0 at both ends, highest point exactly 1."""
    gains = [g for _, g in curve]
    for i, g in enumerate(gains):
        if not 0.0 <= g <= 1.0:
            problems.append(f"{path}[{i}].gain: {g:g} is outside 0 … 1")
    if gains[0] != 0.0:
        problems.append(f"{path}[0].gain: an amplitude shape must start at 0")
    if gains[-1] != 0.0:
        problems.append(f"{path}[{len(gains) - 1}].gain: an amplitude shape must end at 0")
    if max(gains) != 1.0:
        problems.append(f"{path}: the highest gain must be exactly 1, found {max(gains)!r}")


@dataclass(frozen=True)
class Body:
    """The tonal body: a sine and its second harmonic, swept by the pitch curve."""

    tune_hz: float = number(20.0, 1000.0)
    level_db: float = number(-60.0, 24.0, off=-math.inf)
    h2: float = number(0.0, 0.5, "second harmonic, 0 … 0.5")
    phase_deg: float = number(0.0, 360.0, below=True)
    phase_ref: str = choice(("onset", "tail"), '"onset" | "tail"')
    pitch: tuple[tuple[float, float], ...] = knots(
        "ratio",
        (2, 32),
        MAX_BODY_MS,
        _pitch,
        "ratio to tune_hz; starts at 0 ms, ends at 1",
    )
    amp: tuple[tuple[float, float], ...] = knots(
        "gain",
        (3, 32),
        MAX_BODY_MS,
        _shape,
        "shape: 0 at both ends, highest point exactly 1",
    )


@dataclass(frozen=True)
class Attack:
    """The attack: a tone and seeded band-limited noise, under a short amplitude shape."""

    level_db: float = number(-60.0, 24.0, off=-math.inf)
    tone_hz: float = number(20.0, 20000.0)
    noise_mix: float = number(0.0, 1.0, "0 = tone … 1 = noise")
    noise_band_hz: tuple[float, float] = band(20.0, 20000.0)
    amp: tuple[tuple[float, float], ...] = knots("gain", (3, 32), MAX_ATTACK_MS, _shape)


@dataclass(frozen=True)
class Drive:
    """The saturator: its small-signal gain."""

    db: float = number(0.0, 48.0, "small-signal gain; 0 = bypass")


@dataclass(frozen=True)
class Output:
    """The exported file: its sample rate, low cut, level and polarity."""

    sample_rate: int = choice(SAMPLE_RATES)
    low_cut_hz: float = number(10.0, 1000.0, "high-pass, 12 dB/oct; 0 = off", off=0.0)
    mode: str = choice(("gain", "peak", "true-peak"), '"gain" | "peak" | "true-peak"')
    db: float = number(-60.0, lambda o: 0.0 if o["mode"] in PEAK_MODES else 24.0, "gain in dB, or peak target")
    polarity: int = choice((1, -1))


@dataclass(frozen=True)
class Patch:
    """One kick: the seed of its noise and dither, and its four sections."""

    seed: int
    body: Body
    attack: Attack
    drive: Drive
    output: Output

    @property
    def length_ms(self) -> float:
        """The source length: the later of the two amplitude shapes' last knots."""
        return max(self.body.amp[-1][0], self.attack.amp[-1][0])


SECTIONS: dict[str, type[Body | Attack | Drive | Output]] = {
    "body": Body,
    "attack": Attack,
    "drive": Drive,
    "output": Output,
}
CURVES = tuple(f"{name}.{f.name}" for name, cls in SECTIONS.items() for f in fields(cls) if f.metadata["key"])


def _schema(data: dict[str, Any]) -> int:
    """Check the schema before anything else: a newer schema may share nothing else with this one."""
    if "schema" not in data:
        raise PatchError(["schema: missing"])
    schema = data["schema"]
    if isinstance(schema, bool) or not isinstance(schema, int):
        raise PatchError([f"schema: expected an integer, found {_kind(schema)}"])
    if schema not in (1, SCHEMA):
        raise PatchError([f"schema: unsupported version {schema}; this renderer reads schemas 1 and {SCHEMA}"])
    return schema


def _section(
    name: str,
    cls: type[Body | Attack | Drive | Output],
    table: dict[str, Any],
    problems: list[str],
    new: dict[str, Any],
) -> dict[str, Any]:
    """Parse the fields of one section's table; `new` holds those its schema did not have yet."""
    rules = {f.name: f.metadata["parse"] for f in fields(cls) if f.name not in new}
    _keys(problems, f"{name}.", table, rules)
    values: dict[str, Any] = {}
    for key, parse in rules.items():  # a missing field does not hide the problems of the others
        values[key] = parse(f"{name}.{key}", table[key], problems, values) if key in table else None
    return values | new


def from_dict(data: object) -> Patch:
    """Build a patch from parsed TOML, rejecting anything outside the schema."""
    if not isinstance(data, dict):
        raise PatchError(["expected a table"])
    added = {"output": {"low_cut_hz": 0.0}} if _schema(data) == 1 else {}  # schema 1 had no low cut
    problems: list[str] = []
    _keys(problems, "", data, ("schema", "seed", *SECTIONS))
    seed = data.get("seed")
    if "seed" in data:
        if isinstance(seed, bool) or not isinstance(seed, int):
            problems.append(f"seed: expected an integer, found {_kind(seed)}")
        elif not 0 <= seed <= MAX_SEED:
            problems.append(f"seed: {seed} is outside 0 … 2**63 - 1")
    sections: dict[str, dict[str, Any]] = {}
    for name, cls in SECTIONS.items():
        if name not in data:  # reported as missing above
            continue
        if not isinstance(data[name], dict):
            problems.append(f"{name}: expected a table, found {_kind(data[name])}")
            continue
        sections[name] = _section(name, cls, data[name], problems, added.get(name, {}))
    if problems:
        raise PatchError(problems)
    return Patch(
        data["seed"],
        Body(**sections["body"]),
        Attack(**sections["attack"]),
        Drive(**sections["drive"]),
        Output(**sections["output"]),
    )


def to_dict(p: Patch) -> dict[str, Any]:
    """Return the plain data that TOML parses to, so that from_dict(to_dict(p)) == p."""
    data: dict[str, Any] = {"schema": SCHEMA, "seed": p.seed}
    for name, cls in SECTIONS.items():
        section = getattr(p, name)
        data[name] = {f.name: _plain(getattr(section, f.name), f.metadata["key"]) for f in fields(cls)}
    return data


def _plain(v: object, key: str | None) -> object:
    """Return a field's value as TOML parses it: a curve as an array of { ms, key } tables, a pair as an array."""
    if not isinstance(v, tuple):
        return v
    if key:
        return [{"ms": ms, key: x} for ms, x in v]
    return list(v)


def loads(text: str) -> Patch:
    """Parse a patch from its TOML text, a syntax error included among its problems."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise PatchError([f"TOML: {e}"]) from None
    return from_dict(data)


def read_text(path: str | os.PathLike[str]) -> str:
    """Read a patch file's text: UTF-8 with or without a BOM, or UTF-16 with one, as Windows PowerShell's `>` writes."""
    raw = Path(path).read_bytes()
    return raw.decode("utf-16" if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)) else "utf-8-sig")


def dumps(p: Patch) -> str:
    """Format the complete patch as canonical TOML; loads(dumps(p)) == p exactly."""
    lines = [f"schema = {SCHEMA}", f"seed = {p.seed}"]
    for name, cls in SECTIONS.items():
        section = getattr(p, name)
        rows: list[tuple[str, str, str]] = []
        curves: list[str] = []
        for f in fields(cls):
            m, v = f.metadata, getattr(section, f.name)
            if m["key"] is None:
                rows.append((f.name, _toml(v), m["note"]))
                continue
            ms = [f"{t!r}," for t, _ in v]
            width = max(map(len, ms))
            curves += [_note(f"{f.name} = [", m["note"])]
            curves += [f"  {{ ms = {t.ljust(width)} {m['key']} = {x!r} }}," for t, (_, x) in zip(ms, v, strict=True)]
            curves += ["]"]
        width = max(len(key) for key, _, _ in rows)
        lines += ["", f"[{name}]", *(_note(f"{key.ljust(width)} = {text}", note) for key, text, note in rows), *curves]
    return "\n".join(lines) + "\n"


def _note(line: str, note: str) -> str:
    return f"{(line + ' ').ljust(28)}# {note}" if note else line


def edit(p: Patch, path: str, value: object) -> Patch:
    """Return the patch with one field, such as "body.tune_hz", replaced, and validated like any patch."""
    section, name = path.split(".")
    part = dataclasses.replace(getattr(p, section), **{name: value})
    return from_dict(to_dict(dataclasses.replace(p, **{section: part})))


def stretch(p: Patch, curve: str, factor: float, anchor_ms: float | None = None) -> Patch:
    """Rescale a curve's knot times after `anchor_ms` by `factor`, rounded to 1 µs.

    The default anchor is the shape's highest point for an amplitude curve (so
    the rise is kept) and 0 ms for the pitch curve. This rewrites the points
    themselves: the patch gains no parameter.
    """
    if curve not in CURVES:
        msg = f"unknown curve {curve!r}; expected one of {', '.join(CURVES)}"
        raise ValueError(msg)
    if not (math.isfinite(factor) and factor > 0):
        msg = "the factor must be a positive number"
        raise ValueError(msg)
    section, name = curve.split(".")
    pairs = getattr(getattr(p, section), name)
    if anchor_ms is None:
        anchor_ms = 0.0 if name == "pitch" else next(ms for ms, g in pairs if g == 1.0)
    moved = tuple((ms if ms <= anchor_ms else round(anchor_ms + factor * (ms - anchor_ms), 3), v) for ms, v in pairs)
    return edit(p, curve, moved)


# Tuning vocabulary shared by `ballast init`, the render report and the WAV root note.

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
    p = midi_pitch(hz)
    n = round(p)
    cents = round(100 * (p - n), 1) + 0.0  # + 0.0 turns -0.0 into 0.0
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
