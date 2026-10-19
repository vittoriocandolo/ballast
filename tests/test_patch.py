# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The patch: validation with every problem and its path, canonical TOML, edits and tuning."""

import codecs
import datetime as dt
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from ballast_synth import patch
from ballast_synth.patch import PatchError

RATES = "44100, 48000, 88200, 96000, 176400, 192000"


def problems(data: object) -> list[str]:
    """Return every problem from_dict reports for data, which must be invalid."""
    with pytest.raises(PatchError) as caught:
        patch.from_dict(data)
    return caught.value.problems


def test_presets_are_valid_and_canonical(preset: str) -> None:
    """Parse each preset, and write it back as the same text."""
    assert patch.dumps(patch.loads(preset)) == preset


def test_canonical_text_reads_back_to_the_same_patch(tight: patch.Patch) -> None:
    """Write a patch as canonical TOML, and parse the same patch back."""
    assert patch.loads(patch.dumps(tight)) == tight


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("body.tune_hz", 19.0, "body.tune_hz: 19 is outside 20 … 1000"),
        ("body.tune_hz", "50", "body.tune_hz: expected a number, found a string"),
        ("body.level_db", math.inf, "body.level_db: must be finite (or -inf)"),
        ("body.level_db", 30.0, "body.level_db: 30 is outside -60 … 24 (or -inf)"),
        ("output.low_cut_hz", 5.0, "output.low_cut_hz: 5 is outside 10 … 1000 (or 0)"),
        ("body.h2", 0.6, "body.h2: 0.6 is outside 0 … 0.5"),
        ("body.phase_deg", 360.0, "body.phase_deg: 360 is outside 0 … 360 (exclusive)"),
        ("body.phase_ref", "end", "body.phase_ref: expected one of 'onset', 'tail', found 'end'"),
        ("drive.db", True, "drive.db: expected a number, found a boolean"),
        ("drive.db", 10**400, "drive.db: is too large"),
        ("attack.noise_band_hz", ["low", 900.0], "attack.noise_band_hz[0]: expected a number, found a string"),
        ("body.pitch", "fast", "body.pitch: expected an array of { ms, ratio } tables, found a string"),
        ("attack.noise_band_hz", [900.0, 700.0], "attack.noise_band_hz: the low edge must be below the high edge"),
        ("attack.noise_band_hz", [700.0], "attack.noise_band_hz: expected [low, high] in Hz"),
        ("output.sample_rate", 44000, f"output.sample_rate: expected one of {RATES}, found 44000"),
        ("output.sample_rate", 48000.0, f"output.sample_rate: expected one of {RATES}, found 48000.0"),
        ("output.polarity", True, "output.polarity: expected one of 1, -1, found True"),
    ],
)
def test_values_outside_the_domain_are_rejected_not_clamped(
    raw: dict[str, Any],
    path: str,
    value: object,
    message: str,
) -> None:
    """Set one field outside its domain, or to the wrong type, and get exactly one problem."""
    section, key = path.split(".")
    raw[section][key] = value
    assert problems(raw) == [message]


def test_level_accepts_minus_infinity_as_silence(raw: dict[str, Any]) -> None:
    """Accept -inf as a level, outside the range of finite levels."""
    raw["body"]["level_db"] = -math.inf
    assert patch.from_dict(raw).body.level_db == -math.inf


def test_a_peak_target_cannot_exceed_full_scale(raw: dict[str, Any]) -> None:
    """Refuse a peak target above 0 dBFS, which is a valid gain."""
    target = 0.5
    raw["output"].update(mode="peak", db=target)
    assert problems(raw) == ["output.db: 0.5 is outside -60 … 0"]
    raw["output"]["mode"] = "gain"
    assert patch.from_dict(raw).output.db == target


@pytest.mark.parametrize(
    ("curve", "index", "change", "message"),
    [
        ("pitch", 0, {"ms": 1.0}, "body.pitch[0].ms: the pitch curve must start at 0 ms"),
        ("pitch", 2, {"ratio": 1.5}, "body.pitch[2].ratio: the pitch curve must end at ratio 1"),
        ("pitch", 0, {"ratio": 400.0}, "body.pitch[0].ratio: body frequency 22000 Hz is outside 10 … 20000 Hz"),
        ("amp", 0, {"gain": 0.1}, "body.amp[0].gain: an amplitude shape must start at 0"),
        ("amp", 1, {"gain": 0.9}, "body.amp: the highest gain must be exactly 1, found 0.9"),
        ("amp", 2, {"gain": -0.5}, "body.amp[2].gain: -0.5 is outside 0 … 1"),
        ("amp", 1, {"ms": 0.005}, "body.amp[1].ms: is 0.005 ms after the previous knot; knots need 0.01 ms"),
        ("amp", 2, {"ms": 0.1}, "body.amp[2].ms: is earlier than the previous knot"),
        ("amp", 2, {"ms": 0.25}, "body.amp[2].ms: repeats the previous knot time"),
        ("amp", 5, {"gain": 0.1}, "body.amp[5].gain: an amplitude shape must end at 0"),
        ("amp", 5, {"ms": 8000.5}, "body.amp[5].ms: 8000.5 is outside 0 … 8000"),
    ],
)
def test_curve_rules(raw: dict[str, Any], curve: str, index: int, change: dict[str, float], message: str) -> None:
    """Break one rule of a curve with one knot, and get exactly that problem."""
    raw["body"][curve][index].update(change)
    assert problems(raw) == [message]


def test_knot_spacing_is_compared_on_the_decimal_values_as_written(raw: dict[str, Any]) -> None:
    """Space two knots exactly 0.01 ms apart as written, which binary floats would put a hair short."""
    spacing = float(patch.MIN_SPACING_MS)
    raw["body"]["amp"][1]["ms"] = spacing  # 0.01 - 0.0 is exactly the minimum
    assert patch.from_dict(raw).body.amp[1][0] == spacing


def test_a_curve_needs_enough_knots(raw: dict[str, Any]) -> None:
    """Leave an amplitude shape two knots, one too few."""
    del raw["attack"]["amp"][1:3]
    assert problems(raw) == ["attack.amp: needs 3 to 32 knots, found 2"]


def test_missing_and_unknown_fields_are_both_reported(raw: dict[str, Any]) -> None:
    """Misspell a field, which reports the unknown name and the missing one."""
    del raw["body"]["h2"]
    raw["body"]["h3"] = 0.1
    assert problems(raw) == ["body.h3: unknown field", "body.h2: missing"]


def test_every_problem_is_reported_at_once_with_its_path(raw: dict[str, Any]) -> None:
    """Break five things in four sections, and get all five problems in one error."""
    raw["seed"] = -1
    raw["body"]["tune_hz"] = 5000.0
    raw["attack"]["noise_mix"] = 2.0
    raw["output"]["polarity"] = 0
    del raw["drive"]
    assert problems(raw) == [
        "drive: missing",
        "seed: -1 is outside 0 … 2**63 - 1",
        "body.tune_hz: 5000 is outside 20 … 1000",
        "attack.noise_mix: 2 is outside 0 … 1",
        "output.polarity: expected one of 1, -1, found 0",
    ]


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        (None, "schema: missing"),
        ("1", "schema: expected an integer, found a string"),
        (4, "schema: unsupported version 4; this renderer reads schemas 1 to 3"),
    ],
)
def test_the_schema_is_checked_first(raw: dict[str, Any], schema: object, message: str) -> None:
    """Break the schema and another field, and get only the schema's problem."""
    raw["schema"] = schema
    if schema is None:
        del raw["schema"]
    raw["body"]["tune_hz"] = 5000.0  # not reported: another schema may share nothing with this one
    assert problems(raw) == [message]


def test_toml_syntax_errors_are_patch_errors() -> None:
    """Parse broken TOML, and get a patch error that says it is TOML."""
    with pytest.raises(PatchError) as caught:
        patch.loads("schema = 1\n[body\n")
    assert caught.value.problems[0].startswith("TOML: ")


@pytest.mark.parametrize(
    "encode",
    [
        pytest.param(lambda s: s.encode("utf-8"), id="utf-8"),
        pytest.param(lambda s: codecs.BOM_UTF8 + s.encode("utf-8"), id="utf-8-bom"),
        pytest.param(lambda s: codecs.BOM_UTF16_LE + s.encode("utf-16-le"), id="utf-16-le-bom"),  # Windows PowerShell
        pytest.param(lambda s: codecs.BOM_UTF16_BE + s.encode("utf-16-be"), id="utf-16-be-bom"),
    ],
)
def test_read_text_accepts_utf8_and_utf16_with_a_bom(
    tmp_path: Path,
    tight_text: str,
    encode: Callable[[str], bytes],
) -> None:
    """Read a patch saved in each encoding a text editor or a shell may use."""
    path = tmp_path / "kick.toml"
    path.write_bytes(encode(tight_text))
    assert patch.read_text(path) == tight_text


def test_read_text_refuses_other_encodings(tmp_path: Path, tight_text: str) -> None:
    """Refuse a patch in Latin-1, rather than guess at its characters."""
    path = tmp_path / "kick.toml"
    path.write_bytes(tight_text.replace("…", "\xe9").encode("latin-1"))
    with pytest.raises(UnicodeDecodeError):
        patch.read_text(path)


def test_edit_validates_like_any_patch(tight: patch.Patch) -> None:
    """Edit a field to a valid value and to an invalid one."""
    hz = 43.654
    assert patch.edit(tight, "body.tune_hz", hz).body.tune_hz == hz
    with pytest.raises(PatchError, match=r"body\.tune_hz: 2000 is outside 20 … 1000"):
        patch.edit(tight, "body.tune_hz", 2000.0)


def test_stretch_keeps_the_rise_of_an_amplitude_shape(tight: patch.Patch) -> None:
    """Stretch an amplitude shape, which moves only the knots after its peak."""
    before = tight.body.amp
    after = patch.stretch(tight, "body.amp", 2.0).body.amp
    peak = next(ms for ms, gain in before if gain == 1.0)
    assert [gain for _, gain in after] == [gain for _, gain in before]
    assert [ms for ms, _ in after] == [ms if ms <= peak else round(peak + 2.0 * (ms - peak), 3) for ms, _ in before]


def test_stretch_scales_the_pitch_curve_from_its_start(tight: patch.Patch) -> None:
    """Stretch the pitch curve, which moves every knot."""
    after = patch.stretch(tight, "body.pitch", 0.5).body.pitch
    assert [ms for ms, _ in after] == [round(0.5 * ms, 3) for ms, _ in tight.body.pitch]


def test_stretch_after_an_explicit_time(tight: patch.Patch) -> None:
    """Stretch from 0 ms, which moves the knots of the rise too."""
    after = patch.stretch(tight, "attack.amp", 3.0, anchor_ms=0.0).attack.amp
    assert [ms for ms, _ in after] == [round(3.0 * ms, 3) for ms, _ in tight.attack.amp]


@pytest.mark.parametrize(
    ("curve", "factor", "error"),
    [
        ("body.nope", 2.0, "unknown curve"),
        ("body.amp", 0.0, "positive"),
        ("body.amp", math.nan, "positive"),
        ("body.amp", 100.0, r"body\.amp\[3\]\.ms: 9975\.25 is outside 0 … 8000"),
    ],
)
def test_stretch_refuses_what_it_cannot_do(tight: patch.Patch, curve: str, factor: float, error: str) -> None:
    """Refuse an unknown curve, a factor that is not positive, and a stretch past the longest shape."""
    with pytest.raises(ValueError, match=error):
        patch.stretch(tight, curve, factor)


@pytest.mark.parametrize(
    ("text", "hz"),
    [
        ("A4", 440.0),
        ("a4", 440.0),
        ("F1", 43.654),
        ("C#2", 69.296),
        ("Db2", 69.296),
        ("B-1", 15.434),
        ("55", 55.0),
        (" 43.654 ", 43.654),
    ],
)
def test_tunings_are_note_names_or_frequencies(text: str, hz: float) -> None:
    """Parse note names, in either case and with sharps or flats, and plain frequencies."""
    assert patch.parse_tune(text) == hz


def test_an_unknown_tuning_is_refused() -> None:
    """Refuse a note name from another notation."""
    with pytest.raises(ValueError, match="not a note name or frequency: 'H2'"):
        patch.parse_tune("H2")


@pytest.mark.parametrize(
    ("hz", "text"),
    [(440.0, "A4 +0.0 cents"), (43.654, "F1 +0.0 cents"), (50.0, "G1 +35.0 cents"), (60.0, "B1 -49.4 cents")],
)
def test_describe_tuning(hz: float, text: str) -> None:
    """Describe frequencies by their nearest note and their offset in cents."""
    assert patch.describe_tuning(hz) == text


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda raw: raw["body"]["amp"].__setitem__(1, [0.2, 1.0]),
            "body.amp[1]: expected a { ms, gain } table, found an array",
        ),
        (lambda raw: raw["body"]["amp"][1].pop("gain"), "body.amp[1].gain: missing"),
        (lambda raw: raw.__setitem__("drive", 6.0), "drive: expected a table, found a float"),
        (lambda raw: raw.__setitem__("body", None), "body: expected a table, found NoneType"),  # from Python, not TOML
        (lambda raw: raw.__setitem__("schema", None), "schema: expected an integer, found NoneType"),
        (lambda raw: raw.__setitem__("seed", 1.5), "seed: expected an integer, found a float"),
        (lambda raw: raw.__setitem__("seed", dt.date(2026, 1, 1)), "seed: expected an integer, found a date"),
    ],
)
def test_malformed_structure_is_reported_with_its_path(
    raw: dict[str, Any],
    change: Callable[[dict[str, Any]], object],
    message: str,
) -> None:
    """Put the wrong kind of value where a table, a knot or an integer belongs."""
    change(raw)
    assert problems(raw) == [message]


def test_a_ratio_must_be_positive_even_when_the_tuning_is_invalid(raw: dict[str, Any]) -> None:
    """Check the pitch ratios on their own when the tuning they scale is itself invalid."""
    raw["body"]["tune_hz"] = "fifty"
    raw["body"]["pitch"][1]["ratio"] = 0.0
    assert problems(raw) == ["body.tune_hz: expected a number, found a string", "body.pitch[1].ratio: must be positive"]


def test_only_a_table_can_be_a_patch() -> None:
    """Refuse an array as a patch."""
    assert problems([]) == ["expected a table"]


def test_the_seed_is_required(raw: dict[str, Any]) -> None:
    """Leave out the seed, which has no default."""
    del raw["seed"]
    assert problems(raw) == ["seed: missing"]


def test_a_missing_field_does_not_hide_the_problems_of_the_others(raw: dict[str, Any]) -> None:
    """Leave out one field and break another of the same section, and get both problems."""
    del raw["body"]["h2"]
    raw["body"]["tune_hz"] = 5000.0
    assert problems(raw) == ["body.h2: missing", "body.tune_hz: 5000 is outside 20 … 1000"]


def test_every_malformed_knot_is_reported(raw: dict[str, Any]) -> None:
    """Break two knots of one curve in two ways, and get both problems."""
    del raw["body"]["amp"][1]["gain"]
    raw["body"]["amp"][3] = 1.0
    assert problems(raw) == ["body.amp[1].gain: missing", "body.amp[3]: expected a { ms, gain } table, found a float"]


def older(raw: dict[str, Any], schema: int) -> dict[str, Any]:
    """Return the patch as an older schema writes it: without the fields that schema did not have."""
    raw["schema"] = schema
    del raw["clip"], raw["drive"]["asymmetry"], raw["drive"]["mix"]
    if schema == 1:
        del raw["output"]["low_cut_hz"]
    return raw


@pytest.mark.parametrize("schema", [1, 2])
def test_older_schemas_still_read_with_everything_new_off(raw: dict[str, Any], schema: int) -> None:
    """Read a schema 1 or 2 patch, with every later field off, and write it back as schema 3, sounding the same."""
    old = patch.from_dict(older(raw, schema))
    assert (old.drive.asymmetry, old.drive.mix, old.clip.db) == (0.0, 1.0, 0.0)
    assert old.output.low_cut_hz == (0.0 if schema == 1 else raw["output"]["low_cut_hz"])
    assert patch.dumps(old).startswith("schema = 3\n")


def test_an_older_schema_refuses_what_it_did_not_have(raw: dict[str, Any]) -> None:
    """Refuse a low cut in schema 1, and a clipper in schema 2."""
    raw = older(raw, 1)
    raw["output"]["low_cut_hz"] = 30.0
    assert problems(raw) == ["output.low_cut_hz: unknown field"]
    raw["schema"] = 2
    raw["clip"] = {"db": 3.0, "knee_db": 0.0}
    assert problems(raw) == ["clip: unknown field"]


def test_schema_2_requires_the_low_cut(raw: dict[str, Any]) -> None:
    """Leave out the low cut of a schema 2 patch, where it has no default."""
    del raw["output"]["low_cut_hz"]
    assert problems(raw) == ["output.low_cut_hz: missing"]


def test_a_true_peak_target_cannot_exceed_full_scale(raw: dict[str, Any]) -> None:
    """Refuse a true-peak target above 0 dBTP."""
    raw["output"].update(mode="true-peak", db=0.1)
    assert problems(raw) == ["output.db: 0.1 is outside -60 … 0"]


def test_notes_by_number() -> None:
    """Name MIDI notes, number note names, and tune notes to equal temperament."""
    assert [patch.note_name(n) for n in (0, 24, 61, 69)] == ["C-1", "C1", "C#4", "A4"]
    assert [patch.note_number(s) for s in ("C1", "c#1", "Db1", "B-1", "H1")] == [24, 25, 25, 11, None]
    assert [patch.note_hz(69), patch.note_hz(24)] == [440.0, 32.703]
