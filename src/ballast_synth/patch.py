# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The patch: a complete, readable description of one kick.

Every field is required and the renderer holds no defaults, so a future
default can never change an old patch. The sections a patch may leave out are
[sub], [room], [grit], [play] and [transient]: without one, the patch has
none. Each field declares its rule and its comment in the canonical TOML
once, on its dataclass, and parsing, dumping and editing all work from those
declarations. The parser rejects anything outside the valid domain, reporting
every problem with its field path, and never clamps or repairs.
"""

from __future__ import annotations

import base64
import binascii
import codecs
import dataclasses
import datetime as dt
import math
import tomllib
from dataclasses import dataclass, field, fields
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import numpy as np

if TYPE_CHECKING:
    import os
    from collections.abc import Callable, Collection, Iterable, Mapping

    from numpy.typing import NDArray

type Knots = tuple[tuple[float, float], ...]  # a curve's (ms, value) knots

SCHEMA = 3  # each older schema lacks fields, which ADDED fills so that its patches render as they always did
SAMPLE_RATES = (44100, 48000, 88200, 96000, 176400, 192000)
MAX_BODY_MS = 8000.0
MAX_ATTACK_MS = 1000.0
MAX_SWEEP_MS = 1000.0  # how long the low-pass may sweep, which it computes a sample at a time
MIN_SPACING_MS = Decimal("0.01")  # 10 µs, compared on the decimal values as written
MAX_SEED = 2**63 - 1
PEAK_MODES = ("peak", "true-peak")  # the modes whose `db` is a target at or below full scale
BODY_HZ = (10.0, 20000.0)  # where the body's frequency must lie at every pitch knot
ROOM_TAIL = 1.5  # a room rings for this many times its decay: until it has fallen 90 dB
CUTOFF_HZ = (20.0, 20000.0)  # where the low-pass cutoff must lie, at every sweep knot too
TONE_HZ = (20.0, 20000.0)  # where the attack tone's frequency must lie, at every pitch knot too


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


def _keys(
    problems: list[str],
    prefix: str,
    table: dict[str, Any],
    expected: Collection[str],
    optional: Collection[str] = (),
) -> bool:
    """Report unknown keys, and missing ones of those expected; True when none is missing."""
    problems += [f"{prefix}{key}: unknown field" for key in table if key not in expected and key not in optional]
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
# knot in a curve, `columns` the keys of each row of a table, and `dump`, if any,
# writes the value in the canonical TOML. A declaration is typed as the value its
# field holds, as dataclasses.field is.

if TYPE_CHECKING:
    type Parse[T] = Callable[[str, Any, list[str], dict[str, Any]], T | None]
    type Check = Callable[[str, Knots, list[str], dict[str, Any]], None]


def _rule[T](
    parse: Parse[T],
    note: str = "",
    key: str | None = None,
    dump: Callable[[T], str] | None = None,
    columns: tuple[str, ...] = (),
) -> T:
    meta = {"parse": parse, "note": note, "key": key, "dump": dump, "columns": columns}
    return cast("T", field(metadata=meta))


def number(
    lo: float,
    hi: float | Callable[[dict[str, Any]], float],
    note: str = "",
    *,
    below: bool = False,
    off: float | None = None,
) -> float:
    """Declare a number in lo … hi, or `off`; `hi` may be a function of the fields before it."""

    def parse(path: str, raw: object, problems: list[str], section: dict[str, Any]) -> float | None:
        top = hi(section) if callable(hi) else hi
        return _number(path, raw, problems, _Bounds(lo, top, below=below), off=off)

    return _rule(parse, note)


def integer(lo: int, hi: int, note: str = "") -> int:
    """Declare a whole number in lo … hi: an integer, not a float such as 12.0."""

    def parse(path: str, raw: object, problems: list[str], _section: dict[str, Any]) -> int | None:
        if isinstance(raw, bool) or not isinstance(raw, int):
            problems.append(f"{path}: expected an integer, found {_kind(raw)}")
            return None
        if not lo <= raw <= hi:
            problems.append(f"{path}: {raw} is outside {lo} … {hi}")
            return None
        return raw

    return _rule(parse, note)


def choice[T](allowed: tuple[T, ...], note: str = "") -> T:
    """Declare one of `allowed`, and of its type: neither true nor 48000.0 passes for an integer."""

    def parse(path: str, raw: object, problems: list[str], _section: dict[str, Any]) -> T | None:
        if type(raw) is type(allowed[0]) and raw in allowed:
            return raw
        problems.append(f"{path}: expected one of {', '.join(map(repr, allowed))}, found {raw!r}")
        return None

    return _rule(parse, note)


def band(lo: float, hi: float, note: str = "") -> tuple[float, float]:
    """Declare a [low, high] pair of frequencies."""

    def parse(path: str, raw: object, problems: list[str], _section: dict[str, Any]) -> tuple[float, float] | None:
        match raw:
            case [first, second]:
                low = _number(f"{path}[1]", first, problems, _Bounds(lo, hi))
                high = _number(f"{path}[2]", second, problems, _Bounds(lo, hi))
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
        for i, knot in enumerate(raw, 1):
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
            for i, k in enumerate(raw, 1)
        )
        if not counted or any(None in pair for pair in pairs):
            return None
        curve = cast("Knots", pairs)  # every value is a float: checked just above
        _spacing(path, curve, problems)
        check(path, curve, problems, section)
        return curve

    return _rule(parse, note, key)


type Rows = tuple[tuple[float, ...], ...]  # a table's rows, each a value per column
type RowCheck = Callable[[str, Rows, list[str], dict[str, Any]], None]


def table(columns: dict[str, _Bounds], count: tuple[int, int], check: RowCheck, note: str = "") -> Rows:
    """Declare a table: rows of numbers, written as an array of tables of `columns`, each in its bounds."""

    def parse(path: str, raw: object, problems: list[str], section: dict[str, Any]) -> Rows | None:
        keys = ", ".join(columns)
        if not isinstance(raw, list):
            problems.append(f"{path}: expected an array of {{ {keys} }} tables, found {_kind(raw)}")
            return None
        if not count[0] <= len(raw) <= count[1]:
            problems.append(f"{path}: needs {count[0]} to {count[1]} rows, found {len(raw)}")
            return None
        # Every row, every problem; rows count from 1, as every path does.
        rows = [_row(f"{path}[{i}]", row, columns, problems) for i, row in enumerate(raw, 1)]
        if None in rows:
            return None
        checked = cast("Rows", tuple(rows))
        check(path, checked, problems, section)
        return checked

    return _rule(parse, note, columns=tuple(columns))


def _row(path: str, row: object, columns: dict[str, _Bounds], problems: list[str]) -> tuple[float, ...] | None:
    """Read one row of a table: a table of exactly these columns, each a number in its bounds; None if not."""
    if not isinstance(row, dict):
        problems.append(f"{path}: expected a {{ {', '.join(columns)} }} table, found {_kind(row)}")
        return None
    if not _keys(problems, f"{path}.", row, columns):
        return None
    values = [_number(f"{path}.{k}", row[k], problems, bounds) for k, bounds in columns.items()]
    return None if None in values else tuple(cast("list[float]", values))


def _modes(path: str, rows: Rows, problems: list[str], section: dict[str, Any]) -> None:
    """Check that every mode stays within 20 kHz, at the top of the body's sweep too."""
    tune, pitch = section["tune_hz"], section["pitch"]
    if tune is None or pitch is None:
        return
    top = tune * max(r for _, r in pitch)
    for i, (ratio, _, _) in enumerate(rows, 1):
        if top * ratio > BODY_HZ[1]:
            reach = f"the mode reaches {top * ratio:g} Hz at the top of the sweep, above 20 kHz"
            problems.append(f"{path}[{i}].ratio: {reach}")


def encode(samples: NDArray[np.floating[Any]]) -> str:
    """Write samples as a transient stores them: little-endian float32, in base64."""
    return base64.b64encode(samples.astype("<f4").tobytes()).decode("ascii")


def decode(text: str) -> NDArray[np.float32]:
    """Read the samples of a transient's base64 text (read-only)."""
    return np.frombuffer(base64.b64decode(text), dtype="<f4")


def _wrapped(text: str) -> str:
    """Write base64 as a TOML multi-line string of 64 characters a line, each line break escaped away."""
    return '"""\n' + "".join(f"{text[i : i + 64]}\\\n" for i in range(0, len(text), 64)) + '"""'


def sampled(note: str = "") -> str:
    """Declare a transient's samples: base64 of little-endian float32, peaking at exactly 1, ending by 1 s."""

    def parse(path: str, raw: object, problems: list[str], section: dict[str, Any]) -> str | None:
        if not isinstance(raw, str):
            problems.append(f"{path}: expected base64 text, found {_kind(raw)}")
            return None
        try:
            data = base64.b64decode(raw, validate=True)
        except binascii.Error:
            problems.append(f"{path}: is not base64")
            return None
        if not data or len(data) % 4:
            problems.append(f"{path}: needs whole float32 samples, at least one, found {len(data)} bytes")
            return None
        samples = np.frombuffer(data, dtype="<f4")
        if not np.all(np.isfinite(samples)):
            problems.append(f"{path}: every sample must be finite")
            return None
        peak = float(np.max(np.abs(samples)))
        if peak != 1.0:
            problems.append(f"{path}: the samples must peak at exactly 1, found {peak!r}; level_db sets the level")
        rate, start, pitch = section["rate"], section["start_ms"], section["pitch"]
        if rate is not None and start is not None and pitch is not None:
            end = start + 1000 * samples.size / (rate * pitch)
            if end > MAX_ATTACK_MS:
                problems.append(f"{path}: the transient ends at {end:g} ms, after {MAX_ATTACK_MS:g} ms")
        return base64.b64encode(data).decode("ascii")  # canonical, so that equal samples make equal patches

    return _rule(parse, note, dump=_wrapped)


def _spacing(path: str, curve: Knots, problems: list[str]) -> None:
    """Check that the knot times strictly increase, at least MIN_SPACING_MS apart as written."""
    for i in range(2, len(curve) + 1):  # paths count knots from 1
        a, b = Decimal(repr(curve[i - 2][0])), Decimal(repr(curve[i - 1][0]))
        if b == a:
            problems.append(f"{path}[{i}].ms: repeats the previous knot time")
        elif b < a:
            problems.append(f"{path}[{i}].ms: is earlier than the previous knot")
        elif b - a < MIN_SPACING_MS:
            problems.append(f"{path}[{i}].ms: is {b - a} ms after the previous knot; knots need {MIN_SPACING_MS} ms")


def _pitch(reference: str, name: str, hz: tuple[float, float]) -> Check:
    """Make the check of a pitch curve: it starts at 0 ms, ends at ratio 1, and keeps `reference` × ratio within hz."""

    def check(path: str, curve: Knots, problems: list[str], section: dict[str, Any]) -> None:
        if curve[0][0] != 0.0:
            problems.append(f"{path}[1].ms: the curve must start at 0 ms")
        if curve[-1][1] != 1.0:
            problems.append(f"{path}[{len(curve)}].ratio: the curve must end at ratio 1")
        f0 = section[reference]
        lo, hi = hz
        for i, (_, r) in enumerate(curve, 1):
            if f0 is not None and not lo <= f0 * r <= hi:
                problems.append(f"{path}[{i}].ratio: {name} frequency {f0 * r:g} Hz is outside {lo:g} … {hi:g} Hz")
            elif f0 is None and r <= 0:
                problems.append(f"{path}[{i}].ratio: must be positive")

    return check


def _sweep(path: str, curve: Knots, problems: list[str], section: dict[str, Any]) -> None:
    """Check the sweep as a pitch curve of cutoff_hz; with the low-pass off, its ratios need only be positive."""
    off = section["cutoff_hz"] == 0.0
    _pitch("cutoff_hz", "cutoff", CUTOFF_HZ)(path, curve, problems, section | {"cutoff_hz": None} if off else section)


def _shape(path: str, curve: Knots, problems: list[str], _section: dict[str, Any]) -> None:
    """Check an amplitude shape: gains in 0 … 1, 0 at both ends, highest point exactly 1."""
    gains = [g for _, g in curve]
    for i, g in enumerate(gains, 1):
        if not 0.0 <= g <= 1.0:
            problems.append(f"{path}[{i}].gain: {g:g} is outside 0 … 1")
    if gains[0] != 0.0:
        problems.append(f"{path}[1].gain: an amplitude shape must start at 0")
    if gains[-1] != 0.0:
        problems.append(f"{path}[{len(gains)}].gain: an amplitude shape must end at 0")
    if max(gains) != 1.0:
        problems.append(f"{path}: the highest gain must be exactly 1, found {max(gains)!r}")


@dataclass(frozen=True)
class Body:
    """The tonal body: a sine, or a saw that fades above bright_hz, a second harmonic and modes, swept by the pitch."""

    tune_hz: float = number(20.0, 1000.0)
    level_db: float = number(-60.0, 24.0, off=-math.inf)
    h2: float = number(0.0, 0.5, "second harmonic, 0 … 0.5")
    bright_hz: float = number(50.0, 20000.0, "harmonics fade above; 0 = sine", off=0.0)
    phase_deg: float = number(0.0, 360.0, below=True)
    phase_ref: str = choice(("onset", "tail"), '"onset" | "tail"')
    pitch: tuple[tuple[float, float], ...] = knots(
        "ratio",
        (2, 32),
        MAX_BODY_MS,
        _pitch("tune_hz", "body", BODY_HZ),
        "ratio to tune_hz; starts at 0 ms, ends at 1",
    )
    amp: tuple[tuple[float, float], ...] = knots(
        "gain",
        (3, 32),
        MAX_BODY_MS,
        _shape,
        "shape: 0 at both ends, highest point exactly 1",
    )
    modes: tuple[tuple[float, ...], ...] = table(
        {"ratio": _Bounds(0.5, 40.0), "db": _Bounds(-60.0, 12.0), "decay_ms": _Bounds(1.0, MAX_BODY_MS)},
        (0, 16),
        _modes,
        "ratio to the body, dB re its sine, ms to fall 60 dB",
    )


@dataclass(frozen=True)
class Attack:
    """The attack: a tone with its own pitch curve, and seeded band-limited noise, under a short amplitude shape."""

    level_db: float = number(-60.0, 24.0, off=-math.inf)
    tone_hz: float = number(*TONE_HZ)
    pitch: tuple[tuple[float, float], ...] = knots(
        "ratio",
        (1, 32),
        MAX_ATTACK_MS,
        _pitch("tone_hz", "tone", TONE_HZ),
        "ratio to tone_hz; starts at 0 ms, ends at 1",
    )
    noise_mix: float = number(0.0, 1.0, "0 = tone … 1 = noise")
    noise_band_hz: tuple[float, float] = band(20.0, 20000.0)
    amp: tuple[tuple[float, float], ...] = knots("gain", (3, 32), MAX_ATTACK_MS, _shape)


@dataclass(frozen=True)
class Drive:
    """The saturator: its small-signal gain, its even harmonics, and its blend with the undriven signal."""

    db: float = number(0.0, 48.0, "small-signal gain; 0 = bypass")
    asymmetry: float = number(0.0, 1.0, "even harmonics, 0 … 1")
    mix: float = number(0.0, 1.0, "0 = dry … 1 = driven")


@dataclass(frozen=True)
class EQ:
    """After the drive, the sub and the room: a low shelf, a bell and a high shelf, off at 0 dB, and a low-pass."""

    low_hz: float = number(20.0, 1000.0, "low shelf")
    low_db: float = number(-12.0, 12.0, "0 = off")
    bell_hz: float = number(20.0, 20000.0, "bell")
    bell_db: float = number(-12.0, 12.0, "0 = off")
    bell_q: float = number(0.3, 10.0, "0.3 = wide … 10 = narrow")
    high_hz: float = number(1000.0, 20000.0, "high shelf")
    high_db: float = number(-12.0, 12.0, "0 = off")
    cutoff_hz: float = number(*CUTOFF_HZ, "low-pass, 12 dB/oct; 0 = off", off=0.0)
    resonance: float = number(0.5, 10.0, "Q: 0.707 = flat … 10")
    sweep: tuple[tuple[float, float], ...] = knots(
        "ratio",
        (1, 32),
        MAX_SWEEP_MS,
        _sweep,
        "ratio to cutoff_hz; starts at 0 ms, ends at 1",
    )


@dataclass(frozen=True)
class Clip:
    """The clipper before the decimator: how far it brings the kick's peaks down, and how softly it bends."""

    # Below 0.01 dB the clipper's own anti-aliasing, 0.00002 dB at most, would be most of what it does.
    db: float = number(0.01, 12.0, "off the peaks; 0 = off", off=0.0)
    knee_db: float = number(0.0, 6.0, "0 = hard … 6 = soft")


@dataclass(frozen=True)
class Output:
    """The exported file: its sample rate, low cut, level and polarity."""

    sample_rate: int = choice(SAMPLE_RATES)
    low_cut_hz: float = number(10.0, 1000.0, "high-pass, 12 dB/oct; 0 = off", off=0.0)
    mode: str = choice(("gain", "peak", "true-peak"), '"gain" | "peak" | "true-peak"')
    db: float = number(-60.0, lambda o: 0.0 if o["mode"] in PEAK_MODES else 24.0, "gain in dB, or peak target")
    polarity: int = choice((1, -1))


@dataclass(frozen=True)
class Sub:
    """A clean sub after the drive: the sine at tune_hz that the body's tail meets, under its own amplitude shape."""

    level_db: float = number(-60.0, 24.0, off=-math.inf)
    amp: tuple[tuple[float, float], ...] = knots("gain", (3, 32), MAX_BODY_MS, _shape)


@dataclass(frozen=True)
class Room:
    """The room after the drive and the sub: a seeded tail whose high frequencies fade faster, its delay and level."""

    level_db: float = number(-60.0, 12.0, "re the kick's energy", off=-math.inf)
    decay_ms: float = number(50.0, 4000.0, "to fall 60 dB, at low frequencies")
    damping_hz: float = number(200.0, 20000.0, "fades twice as fast here")
    predelay_ms: float = number(0.0, 200.0)

    @property
    def tail_ms(self) -> float:
        """How long the tail rings after the kick: its delay, then ROOM_TAIL × decay, to -90 dB."""
        return self.predelay_ms + ROOM_TAIL * self.decay_ms


@dataclass(frozen=True)
class Grit:
    """A low-rate sampler after the room: its rate, without a filter against aliasing, its bits, and its input level."""

    rate_hz: float = number(4000.0, 48000.0, "no filter against aliasing")
    bits: int = integer(4, 16, "no dither")
    input_db: float = number(-24.0, 24.0, "into the converter, and back")


@dataclass(frozen=True)
class Play:
    """Softer hits and repeats: what the softest hit becomes (the patch is the hardest), and how repeats differ."""

    soft_db: float = number(-40.0, 0.0, "the softest hit's level")
    soft_click_db: float = number(-24.0, 0.0, "and its click's, further")
    soft_drop: float = number(0.0, 1.0, "of the pitch drop it keeps")
    spread: float = number(0.0, 1.0, "how much each repeat differs")


@dataclass(frozen=True)
class Transient:
    """A sampled transient, layered with the attack: its level, start, pitch and polarity, and its samples."""

    level_db: float = number(-60.0, 24.0, "its peak", off=-math.inf)
    start_ms: float = number(0.0, MAX_ATTACK_MS, "after the trigger")
    pitch: float = number(0.25, 4.0, "playback ratio, 0.25 … 4")
    polarity: int = choice((1, -1))
    rate: int = choice(SAMPLE_RATES, "of the samples, float32 in base64")
    samples: str = sampled()

    @classmethod
    def of(cls, samples: NDArray[np.floating[Any]], rate: int) -> Transient:
        """Take mono samples at `rate` as they sound: stored to peak at exactly 1, with their own peak as the level."""
        peak = float(np.max(np.abs(samples), initial=0.0))
        if not math.isfinite(peak):
            msg = "every sample must be finite"
            raise ValueError(msg)
        if peak == 0.0:
            msg = "the samples are silent"
            raise ValueError(msg)
        return cls(round(20 * math.log10(peak), 2), 0.0, 1.0, 1, rate, encode(samples / peak))

    @property
    def end_ms(self) -> float:
        """When its last sample ends, after the trigger."""
        return self.start_ms + 1000 * decode(self.samples).size / (self.rate * self.pitch)


@dataclass(frozen=True)
class Patch:
    """One kick: the seed of its noise and dither, its six sections, and the optional ones it has."""

    seed: int
    body: Body
    attack: Attack
    drive: Drive
    eq: EQ
    clip: Clip
    output: Output
    sub: Sub | None
    room: Room | None
    grit: Grit | None
    play: Play | None
    transient: Transient | None

    @property
    def length_ms(self) -> float:
        """The source length: the latest source's end, a shape's last knot or the transient's, then the room's tail."""
        ends = [self.body.amp[-1][0], self.attack.amp[-1][0]]
        if self.sub is not None:
            ends.append(self.sub.amp[-1][0])
        if self.transient is not None:
            ends.append(self.transient.end_ms)
        return max(ends) + (0.0 if self.room is None else self.room.tail_ms)


type Section = Body | Attack | Drive | Sub | Room | Grit | EQ | Clip | Output | Play | Transient

SECTIONS: dict[str, type[Section]] = {
    "body": Body,
    "attack": Attack,
    "drive": Drive,
    "sub": Sub,
    "room": Room,
    "grit": Grit,
    "eq": EQ,
    "clip": Clip,
    "output": Output,
    "play": Play,
    "transient": Transient,  # last, since its samples run long
}
OPTIONAL = ("sub", "room", "grit", "play", "transient")  # the sections a patch may leave out, and then has none
FIXED_TONE = ((0.0, 1.0),)  # a pitch curve of one knot: the attack's tone holds tone_hz
EQ_OFF = {  # every band at 0 dB
    "low_hz": 80.0,
    "low_db": 0.0,
    "bell_hz": 200.0,
    "bell_db": 0.0,
    "bell_q": 1.0,
    "high_hz": 8000.0,
    "high_db": 0.0,
    "cutoff_hz": 0.0,
    "resonance": 0.707,
    "sweep": ((0.0, 1.0),),
}
# What each older schema lacks, with the values that render its patches exactly as they always sounded; None for
# an optional section it could not have, so that its patches have none.
ADDED: dict[int, dict[str, dict[str, Any] | None]] = {
    1: {
        "body": {"bright_hz": 0.0, "modes": ()},
        "attack": {"pitch": FIXED_TONE},
        "drive": {"asymmetry": 0.0, "mix": 1.0},
        "clip": {"db": 0.0, "knee_db": 0.0},
        "eq": EQ_OFF,
        "output": {"low_cut_hz": 0.0},
        "sub": None,
        "room": None,
        "grit": None,
        "play": None,
        "transient": None,
    },
    2: {
        "body": {"bright_hz": 0.0, "modes": ()},
        "attack": {"pitch": FIXED_TONE},
        "drive": {"asymmetry": 0.0, "mix": 1.0},
        "clip": {"db": 0.0, "knee_db": 0.0},
        "eq": EQ_OFF,
        "sub": None,
        "room": None,
        "grit": None,
        "play": None,
        "transient": None,
    },
    SCHEMA: {},
}
CURVES = tuple(f"{name}.{f.name}" for name, cls in SECTIONS.items() for f in fields(cls) if f.metadata["key"])


def _schema(data: dict[str, Any]) -> int:
    """Check the schema before anything else: a newer schema may share nothing else with this one."""
    if "schema" not in data:
        raise PatchError(["schema: missing"])
    schema = data["schema"]
    if isinstance(schema, bool) or not isinstance(schema, int):
        raise PatchError([f"schema: expected an integer, found {_kind(schema)}"])
    if schema not in ADDED:
        raise PatchError([f"schema: unsupported version {schema}; this renderer reads schemas 1 to {SCHEMA}"])
    return schema


def _seed(data: dict[str, Any], problems: list[str]) -> None:
    """Check the seed, if there is one: a missing seed is reported with the other missing keys."""
    if "seed" not in data:
        return
    seed = data["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        problems.append(f"seed: expected an integer, found {_kind(seed)}")
    elif not 0 <= seed <= MAX_SEED:
        problems.append(f"seed: {seed} is outside 0 … 2**63 - 1")


def _section(
    name: str,
    cls: type[Section],
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
    added = ADDED[_schema(data)]
    every = {name: {f.name for f in fields(cls)} for name, cls in SECTIONS.items()}
    # The sections the patch's schema did not have at all, which ADDED fills, or leaves out with None.
    whole = {name for name, new in added.items() if new is None or new.keys() >= every[name]}
    problems: list[str] = []
    required = [name for name in SECTIONS if name not in whole and name not in OPTIONAL]
    _keys(problems, "", data, ("schema", "seed", *required), [name for name in OPTIONAL if name not in whole])
    _seed(data, problems)
    sections: dict[str, dict[str, Any]] = {}
    for name, cls in SECTIONS.items():
        new = added.get(name)
        if name in whole:
            if new is not None:
                sections[name] = dict(new)
        elif name not in data:  # reported as missing above, unless the section is optional
            continue
        elif not isinstance(data[name], dict):
            problems.append(f"{name}: expected a table, found {_kind(data[name])}")
        else:
            sections[name] = _section(name, cls, data[name], problems, new or {})
    if problems:
        raise PatchError(problems)
    sub, room, grit, play, transient = (sections.get(name) for name in OPTIONAL)
    patch = Patch(
        data["seed"],
        Body(**sections["body"]),
        Attack(**sections["attack"]),
        Drive(**sections["drive"]),
        EQ(**sections["eq"]),
        Clip(**sections["clip"]),
        Output(**sections["output"]),
        None if sub is None else Sub(**sub),
        None if room is None else Room(**room),
        None if grit is None else Grit(**grit),
        None if play is None else Play(**play),
        None if transient is None else Transient(**transient),
    )
    if patch.play is not None and (unplayable := _unplayable(patch)):
        raise PatchError(unplayable)
    return patch


def to_dict(p: Patch) -> dict[str, Any]:
    """Return the plain data that TOML parses to, so that from_dict(to_dict(p)) == p."""
    data: dict[str, Any] = {"schema": SCHEMA, "seed": p.seed}
    for name, cls in SECTIONS.items():
        section = getattr(p, name)
        if section is not None:  # an optional section the patch leaves out
            data[name] = {f.name: _plain(getattr(section, f.name), f.metadata) for f in fields(cls)}
    return data


def _plain(v: object, meta: Mapping[str, Any]) -> object:
    """Return a field's value as TOML parses it: a curve or a table as an array of tables, a pair as an array."""
    if not isinstance(v, tuple):
        return v
    if meta["key"]:
        return [{"ms": ms, meta["key"]: x} for ms, x in v]
    if meta["columns"]:
        return [dict(zip(meta["columns"], row, strict=True)) for row in v]
    return list(v)


def loads(text: str) -> Patch:
    """Parse a patch from its TOML text, a syntax error included among its problems."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise PatchError([f"TOML: {e}"]) from None
    except RecursionError:  # arrays or tables nested deeper than any patch, made to exhaust the parser
        raise PatchError(["TOML: nested too deeply to be a patch"]) from None
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
        if section is None:  # an optional section the patch leaves out
            continue
        rows: list[tuple[str, str, str]] = []
        curves: list[str] = []
        for f in fields(cls):
            meta, value = f.metadata, getattr(section, f.name)
            if meta["columns"]:
                curves += _table(f.name, value, meta["columns"], meta["note"])
                continue
            if meta["key"] is None:
                rows.append((f.name, (meta["dump"] or _toml)(value), meta["note"]))
                continue
            times = [f"{ms!r}," for ms, _ in value]  # with their commas, so that the values after them line up
            width = max(map(len, times))
            curves += [_note(f"{f.name} = [", meta["note"])]
            curves += [
                f"  {{ ms = {t.ljust(width)} {meta['key']} = {x!r} }}," for t, (_, x) in zip(times, value, strict=True)
            ]
            curves += ["]"]
        width = max(len(key) for key, _, _ in rows)
        lines += ["", f"[{name}]", *(_note(f"{key.ljust(width)} = {text}", note) for key, text, note in rows), *curves]
    return "\n".join(lines) + "\n"


def _table(name: str, rows: Rows, columns: tuple[str, ...], note: str) -> list[str]:
    """Write a table as an array of inline tables, its columns lined up, or as [] when it has no rows."""
    if not rows:
        return [_note(f"{name} = []", note)]
    cells = [[f"{key} = {value!r}," for key, value in zip(columns, row, strict=True)] for row in rows]
    widths = [max(len(row[j]) for row in cells) for j in range(len(columns))]
    lined = (" ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)).rstrip(", ") for row in cells)
    return [_note(f"{name} = [", note), *(f"  {{ {row} }}," for row in lined), "]"]


def _note(line: str, note: str) -> str:
    """Append a comment to a line of canonical TOML, after 28 columns so that the comments line up."""
    return f"{(line + ' ').ljust(28)}# {note}" if note else line


def edit(p: Patch, path: str, value: object) -> Patch:
    """Return the patch with one field, such as "body.tune_hz", or one section replaced, validated like any patch.

    A section's path is its name, such as "transient", and an optional one's value may be None, to leave it out.
    """
    section, _, name = path.partition(".")
    new: Any = value
    if name:
        part = getattr(p, section)
        if part is None:
            raise PatchError([f"{section}: the patch has none"])
        new = dataclasses.replace(part, **{name: value})
    return from_dict(to_dict(dataclasses.replace(p, **{section: new})))


def stretch(p: Patch, curve: str, factor: float, anchor_ms: float | None = None) -> Patch:
    """Rescale a curve's knot times after `anchor_ms` by `factor`, rounded to 1 µs.

    The default anchor is the shape's highest point for an amplitude curve (so
    the rise is kept) and 0 ms for a curve of ratios, a pitch or the cutoff's sweep. This rewrites the points
    themselves: the patch gains no parameter.
    """
    if curve not in CURVES:
        msg = f"unknown curve {curve!r}; expected one of {', '.join(CURVES)}"
        raise ValueError(msg)
    if not (math.isfinite(factor) and factor > 0):
        msg = "the factor must be a positive number"
        raise ValueError(msg)
    section, name = curve.split(".")
    part = getattr(p, section)
    if part is None:
        raise PatchError([f"{section}: the patch has none"])
    pairs = getattr(part, name)
    if anchor_ms is None:
        ratios = next(f for f in fields(type(part)) if f.name == name).metadata["key"] == "ratio"
        anchor_ms = 0.0 if ratios else next(ms for ms, g in pairs if g == 1.0)
    return edit(p, curve, _stretched(pairs, factor, anchor_ms))


def _stretched(pairs: Knots, factor: float, anchor_ms: float) -> Knots:
    """Return the knots with every time after `anchor_ms` rescaled by `factor`, rounded to 1 µs."""
    return tuple((ms if ms <= anchor_ms else round(anchor_ms + factor * (ms - anchor_ms), 3), v) for ms, v in pairs)


# Playing: the hits a [play] section makes, which every patch with one must be able to hold. ballast_synth.play
# draws the repeats; the moves themselves are here, since what [play] may say depends on them.

LEVEL_DB, DROP, LENGTH, CLICK_DB = 1.0, 0.04, 0.04, 2.0  # how far a repeat moves, at a spread of 1


def struck(p: Patch, x: float, seed: int, moves: tuple[float, float, float, float]) -> Patch:
    """Return the hit at strength x with a repeat's seed and its four moves, each in [-1, 1]: one hit, no [play].

    Every source's level moves by `db`, the click's and the transient's by `click_db` more; every pitch ratio's
    distance from 1 is scaled by `keep`; and the body's shape after its peak is stretched by `length`. The hit is
    validated whole, so every problem it has is reported at once.
    """
    play = cast("Play", p.play)
    level, drop, length, click = moves
    soft, spread = 1.0 - x, play.spread
    db = play.soft_db * soft + LEVEL_DB * spread * level
    click_db = db + play.soft_click_db * soft + CLICK_DB * spread * click
    keep = (1 - (1 - play.soft_drop) * soft) * (1 + DROP * spread * drop)
    factor = round(1 + LENGTH * spread * length, 6)

    def moved(value: float, by: float) -> float:
        """Move a level, leaving silence silent, rounded as a patch writes it."""
        return value if not math.isfinite(value) or by == 0.0 else round(value + by, 3)

    b = p.body
    body = dataclasses.replace(
        b,
        level_db=moved(b.level_db, db),
        pitch=b.pitch if keep == 1.0 else tuple((ms, round(1 + (r - 1) * keep, 6)) for ms, r in b.pitch),
        amp=b.amp if factor == 1.0 else _stretched(b.amp, factor, next(ms for ms, g in b.amp if g == 1.0)),
    )
    hit = dataclasses.replace(
        p,
        seed=seed,
        play=None,
        body=body,
        attack=dataclasses.replace(p.attack, level_db=moved(p.attack.level_db, click_db)),
        sub=None if p.sub is None else dataclasses.replace(p.sub, level_db=moved(p.sub.level_db, db)),
        transient=(
            None
            if p.transient is None
            else dataclasses.replace(p.transient, level_db=moved(p.transient.level_db, click_db))
        ),
    )
    return from_dict(to_dict(hit))


def _unplayable(p: Patch) -> list[str]:
    """Return why [play] would take a hit outside what a patch may hold: nothing, when every hit is a patch.

    Every move is linear in the strength and in the repeat's draws, and every limit on a level, a pitch, a length or a
    knot's spacing is a bound on one of them, so the two hits at the ends, the softest with every draw at its lowest
    and the hardest with every draw at its highest, reach every limit any hit can.
    """
    problems = []
    for x, sign, name in ((0.0, -1.0, "softest"), (1.0, 1.0, "hardest")):
        try:
            struck(p, x, p.seed, (sign, sign, sign, sign))
        except PatchError as e:
            problems += [f"play: its {name} hit, as far as spread moves it, would be invalid: {m}" for m in e.problems]
    return problems
