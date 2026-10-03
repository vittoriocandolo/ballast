"""The patch: a complete, readable description of one kick (plan §4, §5).

Every field is required and the renderer holds no defaults, so a future
default can never change an old patch. Each section validates itself when it
is constructed, so every Patch object lies inside the valid domain. The parser
rejects anything else and never clamps or repairs.
"""

from __future__ import annotations

import dataclasses
import math
import re
import tomllib
from dataclasses import dataclass
from decimal import Decimal

SCHEMA = 1
SAMPLE_RATES = (44100, 48000, 88200, 96000)
MAX_BODY_MS = 8000.0
MAX_ATTACK_MS = 1000.0
MIN_SPACING_MS = Decimal("0.01")  # 10 µs, compared on the decimal values as written
MAX_SEED = 2**63 - 1


class PatchError(ValueError):
    """An invalid patch. `problems` lists every reason, each with its field path."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("\n".join(self.problems))


class _Check:
    """Validates and normalizes the fields of one frozen section."""

    def __init__(self, obj, section: str):
        self.obj, self.section, self.problems = obj, section, []

    def fail(self, path: str, message: str) -> None:
        self.problems.append(f"{path}: {message}")

    def done(self) -> None:
        if self.problems:
            raise PatchError(self.problems)

    def _set(self, name, value):
        object.__setattr__(self.obj, name, value)

    def number(self, name, lo, hi, *, hi_open=False, neg_inf=False):
        x = self.value(f"{self.section}.{name}", getattr(self.obj, name), lo, hi, hi_open=hi_open, neg_inf=neg_inf)
        if x is not None:
            self._set(name, x)
        return x

    def value(self, path, v, lo=None, hi=None, *, hi_open=False, neg_inf=False):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            self.fail(path, f"expected a number, found {_kind(v)}")
            return None
        try:
            x = float(v)
        except OverflowError:
            self.fail(path, "is too large")
            return None
        if neg_inf and x == -math.inf:
            return x
        if not math.isfinite(x):
            self.fail(path, "must be finite" + (" (or -inf for silence)" if neg_inf else ""))
            return None
        if (lo is not None and x < lo) or (hi is not None and (x >= hi if hi_open else x > hi)):
            self.fail(path, f"{x:g} is outside {lo:g} … {hi:g}" + (" (exclusive)" if hi_open else ""))
            return None
        return x

    def integer(self, name, allowed=None, lo=None, hi=None):
        path, v = f"{self.section}.{name}", getattr(self.obj, name)
        if isinstance(v, bool) or not isinstance(v, int):
            self.fail(path, f"expected an integer, found {_kind(v)}")
            return None
        if allowed is not None and v not in allowed:
            self.fail(path, f"{v} is not one of {', '.join(map(str, allowed))}")
            return None
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            self.fail(path, f"{v} is outside {lo} … {hi}")
            return None
        return v

    def choice(self, name, allowed):
        path, v = f"{self.section}.{name}", getattr(self.obj, name)
        if not isinstance(v, str) or v not in allowed:
            self.fail(path, f"expected one of {', '.join(repr(a) for a in allowed)}, found {v!r}")
            return None
        return v

    def knots(self, name, key, count, max_ms):
        path, raw = f"{self.section}.{name}", getattr(self.obj, name)
        if not isinstance(raw, (list, tuple)) or not all(isinstance(k, (list, tuple)) and len(k) == 2 for k in raw):
            self.fail(path, "expected a list of (ms, value) knots")
            return None
        counted = count[0] <= len(raw) <= count[1]
        if not counted:
            self.fail(path, f"needs {count[0]} to {count[1]} knots, found {len(raw)}")
        out = []
        for i, (ms, v) in enumerate(raw):
            out.append((self.value(f"{path}[{i}].ms", ms, 0.0, max_ms), self.value(f"{path}[{i}].{key}", v)))
        if not counted or any(m is None or v is None for m, v in out):
            return None
        for i in range(1, len(out)):
            a, b = Decimal(repr(out[i - 1][0])), Decimal(repr(out[i][0]))
            if b == a:
                self.fail(f"{path}[{i}].ms", "repeats the previous knot time")
            elif b < a:
                self.fail(f"{path}[{i}].ms", "is earlier than the previous knot")
            elif b - a < MIN_SPACING_MS:
                self.fail(f"{path}[{i}].ms", f"is {b - a} ms after the previous knot; knots need 0.01 ms (10 µs)")
        out = tuple(out)
        self._set(name, out)
        return out

    def shape(self, name, knots):
        path = f"{self.section}.{name}"
        gains = [g for _, g in knots]
        for i, g in enumerate(gains):
            if not 0.0 <= g <= 1.0:
                self.fail(f"{path}[{i}].gain", f"{g:g} is outside 0 … 1")
        if gains[0] != 0.0:
            self.fail(f"{path}[0].gain", "an amplitude shape must start at 0")
        if gains[-1] != 0.0:
            self.fail(f"{path}[{len(gains) - 1}].gain", "an amplitude shape must end at 0")
        if max(gains) != 1.0:
            self.fail(path, f"the highest gain must be exactly 1, found {max(gains)!r}")


def _kind(v) -> str:
    return {bool: "a boolean", str: "a string", list: "an array", dict: "a table"}.get(type(v), type(v).__name__)


@dataclass(frozen=True)
class Body:
    tune_hz: float
    level_db: float
    h2: float
    phase_deg: float
    phase_ref: str
    pitch: tuple  # ((ms, ratio), ...)
    amp: tuple  # ((ms, gain), ...)

    def __post_init__(self):
        c = _Check(self, "body")
        tune = c.number("tune_hz", 20.0, 1000.0)
        c.number("level_db", -60.0, 24.0, neg_inf=True)
        c.number("h2", 0.0, 0.5)
        c.number("phase_deg", 0.0, 360.0, hi_open=True)
        c.choice("phase_ref", ("onset", "tail"))
        pitch = c.knots("pitch", "ratio", (2, 32), MAX_BODY_MS)
        if pitch is not None:
            if pitch[0][0] != 0.0:
                c.fail("body.pitch[0].ms", "the pitch curve must start at 0 ms")
            if pitch[-1][1] != 1.0:
                c.fail(f"body.pitch[{len(pitch) - 1}].ratio", "the pitch curve must end at ratio 1")
            for i, (_, r) in enumerate(pitch):
                if tune is not None and not 10.0 <= tune * r <= 20000.0:
                    c.fail(f"body.pitch[{i}].ratio", f"body frequency {tune * r:g} Hz is outside 10 Hz … 20 kHz")
                elif tune is None and r <= 0:
                    c.fail(f"body.pitch[{i}].ratio", "must be positive")
        amp = c.knots("amp", "gain", (3, 32), MAX_BODY_MS)
        if amp is not None:
            c.shape("amp", amp)
        c.done()


@dataclass(frozen=True)
class Attack:
    level_db: float
    tone_hz: float
    noise_mix: float
    noise_band_hz: tuple  # (low, high)
    amp: tuple  # ((ms, gain), ...)

    def __post_init__(self):
        c = _Check(self, "attack")
        c.number("level_db", -60.0, 24.0, neg_inf=True)
        c.number("tone_hz", 20.0, 20000.0)
        c.number("noise_mix", 0.0, 1.0)
        band = self.noise_band_hz
        if not isinstance(band, (list, tuple)) or len(band) != 2:
            c.fail("attack.noise_band_hz", "expected [low, high] in Hz")
        else:
            lo = c.value("attack.noise_band_hz[0]", band[0], 20.0, 20000.0)
            hi = c.value("attack.noise_band_hz[1]", band[1], 20.0, 20000.0)
            if lo is not None and hi is not None:
                if lo >= hi:
                    c.fail("attack.noise_band_hz", "the low edge must be below the high edge")
                object.__setattr__(self, "noise_band_hz", (lo, hi))
        amp = c.knots("amp", "gain", (3, 32), MAX_ATTACK_MS)
        if amp is not None:
            c.shape("amp", amp)
        c.done()


@dataclass(frozen=True)
class Drive:
    db: float

    def __post_init__(self):
        c = _Check(self, "drive")
        c.number("db", 0.0, 48.0)
        c.done()


@dataclass(frozen=True)
class Output:
    sample_rate: int
    mode: str
    db: float
    polarity: int

    def __post_init__(self):
        c = _Check(self, "output")
        c.integer("sample_rate", SAMPLE_RATES)
        mode = c.choice("mode", ("gain", "peak"))
        c.number("db", -60.0, 0.0 if mode == "peak" else 24.0)
        c.integer("polarity", (1, -1))
        c.done()


@dataclass(frozen=True)
class Patch:
    seed: int
    body: Body
    attack: Attack
    drive: Drive
    output: Output

    def __post_init__(self):
        problems = _seed_problems(self.seed)
        for name, cls in (("body", Body), ("attack", Attack), ("drive", Drive), ("output", Output)):
            if not isinstance(getattr(self, name), cls):
                problems.append(f"{name}: expected a {cls.__name__} section")
        if problems:
            raise PatchError(problems)

    @property
    def length_ms(self) -> float:
        """The source length: the later of the two amplitude shapes' last knots."""
        return max(self.body.amp[-1][0], self.attack.amp[-1][0])


def _seed_problems(seed) -> list[str]:
    if isinstance(seed, bool) or not isinstance(seed, int):
        return [f"seed: expected an integer, found {_kind(seed)}"]
    if not 0 <= seed <= MAX_SEED:
        return [f"seed: {seed} is outside 0 … 2**63 - 1"]
    return []


_LAYOUT = {
    "body": (Body, ("tune_hz", "level_db", "h2", "phase_deg", "phase_ref", "pitch", "amp")),
    "attack": (Attack, ("level_db", "tone_hz", "noise_mix", "noise_band_hz", "amp")),
    "drive": (Drive, ("db",)),
    "output": (Output, ("sample_rate", "mode", "db", "polarity")),
}
_KNOT_KEY = {("body", "pitch"): "ratio", ("body", "amp"): "gain", ("attack", "amp"): "gain"}


def _keys(problems, prefix, table, expected) -> bool:
    for key in table:
        if key not in expected:
            problems.append(f"{prefix}{key}: unknown field")
    missing = [key for key in expected if key not in table]
    problems.extend(f"{prefix}{key}: missing" for key in missing)
    return not missing


def from_dict(data) -> Patch:
    """Build a patch from parsed TOML, rejecting anything outside the schema."""
    if not isinstance(data, dict):
        raise PatchError(["expected a table"])
    # The schema comes first: a newer schema may share nothing else with this one.
    schema = data.get("schema")
    if schema is None:
        raise PatchError(["schema: missing"])
    if isinstance(schema, bool) or not isinstance(schema, int):
        raise PatchError([f"schema: expected an integer, found {_kind(schema)}"])
    if schema != SCHEMA:
        raise PatchError([f"schema: unsupported version {schema}; this renderer reads schema {SCHEMA}"])
    problems: list[str] = []
    _keys(problems, "", data, ("schema", "seed", *_LAYOUT))
    if "seed" in data:
        problems += _seed_problems(data["seed"])
    sections = {}
    for name, (cls, fields) in _LAYOUT.items():
        table = data.get(name)
        if table is None:
            continue
        if not isinstance(table, dict):
            problems.append(f"{name}: expected a table, found {_kind(table)}")
            continue
        if not _keys(problems, f"{name}.", table, fields):
            continue
        values = {key: table[key] for key in fields}
        for (section, field), key in _KNOT_KEY.items():
            if section == name:
                values[field] = _knot_pairs(problems, f"{name}.{field}", table[field], key)
        if name == "attack" and isinstance(table["noise_band_hz"], list):
            values["noise_band_hz"] = tuple(table["noise_band_hz"])
        if any(v is None for v in values.values()):
            continue
        try:
            sections[name] = cls(**values)
        except PatchError as e:
            problems += e.problems
    if problems:
        raise PatchError(problems)
    return Patch(seed=data["seed"], **sections)


def _knot_pairs(problems, path, knots, key):
    if not isinstance(knots, list):
        problems.append(f"{path}: expected an array of {{ ms, {key} }} tables, found {_kind(knots)}")
        return None
    pairs = []
    for i, knot in enumerate(knots):
        if not isinstance(knot, dict):
            problems.append(f"{path}[{i}]: expected a {{ ms, {key} }} table, found {_kind(knot)}")
            return None
        if not _keys(problems, f"{path}[{i}].", knot, ("ms", key)):
            return None
        pairs.append((knot["ms"], knot[key]))
    return tuple(pairs)


def loads(text: str) -> Patch:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise PatchError([f"TOML: {e}"]) from None
    return from_dict(data)


def load(path) -> Patch:
    with open(path, encoding="utf-8") as f:
        return loads(f.read())


def to_dict(p: Patch) -> dict:
    b, a = p.body, p.attack
    return {
        "schema": SCHEMA,
        "seed": p.seed,
        "body": {
            "tune_hz": b.tune_hz,
            "level_db": b.level_db,
            "h2": b.h2,
            "phase_deg": b.phase_deg,
            "phase_ref": b.phase_ref,
            "pitch": [{"ms": m, "ratio": r} for m, r in b.pitch],
            "amp": [{"ms": m, "gain": g} for m, g in b.amp],
        },
        "attack": {
            "level_db": a.level_db,
            "tone_hz": a.tone_hz,
            "noise_mix": a.noise_mix,
            "noise_band_hz": list(a.noise_band_hz),
            "amp": [{"ms": m, "gain": g} for m, g in a.amp],
        },
        "drive": {"db": p.drive.db},
        "output": {
            "sample_rate": p.output.sample_rate,
            "mode": p.output.mode,
            "db": p.output.db,
            "polarity": p.output.polarity,
        },
    }


_COMMENT_COLUMN = 28


def dumps(p: Patch) -> str:
    """The complete patch as TOML; loads(dumps(p)) == p exactly."""
    b, a, o = p.body, p.attack, p.output
    lines = [f"schema = {SCHEMA}", f"seed = {p.seed}", "", "[body]"]
    lines += _rows(
        [
            ("tune_hz", _f(b.tune_hz), ""),
            ("level_db", _f(b.level_db), ""),
            ("h2", _f(b.h2), "second harmonic, 0 … 0.5"),
            ("phase_deg", _f(b.phase_deg), ""),
            ("phase_ref", f'"{b.phase_ref}"', '"onset" | "tail"'),
        ]
    )
    lines += _curve("pitch", "ratio", b.pitch, "ratio to tune_hz; starts at 0 ms, ends at 1")
    lines += _curve("amp", "gain", b.amp, "shape: 0 at both ends, highest point exactly 1")
    lines += ["", "[attack]"]
    lines += _rows(
        [
            ("level_db", _f(a.level_db), ""),
            ("tone_hz", _f(a.tone_hz), ""),
            ("noise_mix", _f(a.noise_mix), "0 = tone … 1 = noise"),
            ("noise_band_hz", f"[{_f(a.noise_band_hz[0])}, {_f(a.noise_band_hz[1])}]", ""),
        ]
    )
    lines += _curve("amp", "gain", a.amp, "")
    lines += ["", "[drive]"] + _rows([("db", _f(p.drive.db), "small-signal gain; 0 = bypass")])
    lines += ["", "[output]"]
    lines += _rows(
        [
            ("sample_rate", str(o.sample_rate), ""),
            ("mode", f'"{o.mode}"', '"gain" | "peak"'),
            ("db", _f(o.db), "gain in dB, or peak target in dBFS"),
            ("polarity", str(o.polarity), ""),
        ]
    )
    return "\n".join(lines) + "\n"


def _f(x: float) -> str:
    return repr(float(x))


def _note(line: str, note: str) -> str:
    return f"{(line + ' ').ljust(_COMMENT_COLUMN)}# {note}" if note else line


def _rows(rows) -> list[str]:
    width = max(len(k) for k, _, _ in rows)
    return [_note(f"{k.ljust(width)} = {v}", note) for k, v, note in rows]


def _curve(name, key, knots, note) -> list[str]:
    ms = [_f(m) + "," for m, _ in knots]
    width = max(map(len, ms))
    body = [f"  {{ ms = {m.ljust(width)} {key} = {_f(v)} }}," for m, (_, v) in zip(ms, knots)]
    return [_note(f"{name} = [", note), *body, "]"]


CURVES = ("body.pitch", "body.amp", "attack.amp")


def stretch(p: Patch, curve: str, factor: float, anchor_ms: float | None = None) -> Patch:
    """Rescale a curve's knot times after `anchor_ms` by `factor`, rounded to 1 µs.

    The default anchor is the shape's highest point for an amplitude curve (so
    the rise is kept) and 0 ms for the pitch curve. This rewrites the points
    themselves: the patch gains no parameter.
    """
    if curve not in CURVES:
        raise ValueError(f"unknown curve {curve!r}; expected one of {', '.join(CURVES)}")
    if not (math.isfinite(factor) and factor > 0):
        raise ValueError("the factor must be a positive number")
    section, name = curve.split(".")
    knots = getattr(getattr(p, section), name)
    if anchor_ms is None:
        anchor_ms = 0.0 if name == "pitch" else next(ms for ms, g in knots if g == 1.0)
    moved = tuple((ms if ms <= anchor_ms else round(anchor_ms + factor * (ms - anchor_ms), 3), v) for ms, v in knots)
    part = dataclasses.replace(getattr(p, section), **{name: moved})
    return dataclasses.replace(p, **{section: part})


# Tuning vocabulary shared by `kick init`, the render report and the WAV root note.

_NOTES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_LETTERS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def midi_pitch(hz: float) -> float:
    """Fractional MIDI note number (A4 = 69 = 440 Hz)."""
    return 69.0 + 12.0 * math.log2(hz / 440.0)


def note_name(midi: int) -> str:
    return f"{_NOTES[midi % 12]}{midi // 12 - 1}"


def describe_tuning(hz: float) -> str:
    """For example 'G1 +35.3 cents'."""
    p = midi_pitch(hz)
    n = round(p)
    cents = round(100 * (p - n), 1) + 0.0  # + 0.0 turns -0.0 into 0.0
    return f"{note_name(n)} {cents:+.1f} cents"


def parse_tune(text: str) -> float:
    """A note name such as F1, C#2 or Db2 (Hz rounded to 3 decimals), or a frequency in Hz."""
    m = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d+)", text.strip())
    if m:
        midi = _LETTERS[m.group(1).upper()] + {"#": 1, "b": -1, "": 0}[m.group(2)] + 12 * (int(m.group(3)) + 1)
        return round(440.0 * 2.0 ** ((midi - 69) / 12), 3)
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"not a note name or frequency: {text!r}") from None
