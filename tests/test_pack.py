# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Packs: names by key or frequency, and the gates every kick must pass."""

import csv
import dataclasses
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from ballast_synth import __version__, pack, patch, render
from ballast_synth.patch import midi_pitch, note_name


@pytest.mark.parametrize(("hz", "name"), [(55.0, "A1"), (43.654, "F1"), (50.0, "50Hz"), (55.016, "55.016Hz")])
def test_a_kick_bears_its_key_only_within_half_a_cent(hz: float, name: str) -> None:
    """Name kicks on a note by the note, and any other by its frequency."""
    assert pack.label(hz) == name


def test_the_presets_pass_every_gate_and_a_click_at_either_edge_does_not(tight: patch.Patch) -> None:
    """Pass the tight preset, then fail it with a loud first or last sample."""
    audio, report = render.render(tight, 24)
    kick = pack.Kick("tight-A1.wav", tight, patch.dumps(tight), audio, report)
    assert pack.gates(kick) == []
    for edge in (0, -1):
        clicked = audio.copy()
        clicked[edge] = 0.01
        problems = pack.gates(dataclasses.replace(kick, audio=clicked))
        assert problems == [f"{'first' if edge == 0 else 'last'} sample at -40.0 dBFS, above -90 dBFS: it would click"]
    assert np.array_equal(kick.audio, audio)


def test_a_true_peak_over_0_dbtp_is_refused(tight: patch.Patch) -> None:
    """Refuse a kick whose true peak is above 0 dBTP, however little, and pass one exactly at it."""
    audio, report = render.render(tight, 24)
    for peak, refused in ((1.0, False), (1.0001, True), (1.5, True)):
        kick = pack.Kick("x.wav", tight, "", audio, dataclasses.replace(report, true_peak=peak))
        assert any(p.startswith("true peak") for p in pack.gates(kick)) == refused


def test_a_single_rate_writes_no_folder_and_keeps_the_text_as_written(tight_text: str) -> None:
    """Pack at one rate, given or the patch's own, into the pack's root with the very text read, comments kept."""
    text = "# my kick\n" + tight_text
    p = patch.loads(text)
    for rates in ([], [p.output.sample_rate]):
        [kick] = pack.kicks([("kick", text, p)], rates, None)
        assert (kick.path, kick.text) == ("kick-A1.wav", text)
    cd, own = pack.kicks([("kick", text, p)], [44100, 48000], None)  # two rates: a folder each, the other rewritten
    assert (cd.path, cd.text) == ("44100/kick-A1.wav", patch.dumps(cd.patch))
    assert (own.path, own.text) == ("48000/kick-A1.wav", text)


@pytest.mark.parametrize("bits", [None, 24])
def test_the_manifest_records_every_file_as_rendered(tight: patch.Patch, tmp_path: Path, bits: int | None) -> None:
    """Write a kit and a kick, and find each manifest row, in JSON and CSV alike, computed from its file and report."""
    played = patch.edit(patch.edit(tight, "play", patch.Play(-18.0, -6.0, 0.6, 0.5)), "body.tune_hz", 51.0)
    kicks = pack.kicks([("kick", "", played)], [], bits, 2, 2)
    pack.write(tmp_path, kicks, bits)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert (manifest["renderer"], manifest["gates"]) == (f"ballast {__version__}", pack.GATES)
    with (tmp_path / "manifest.csv").open(encoding="utf-8", newline="") as f:
        table = list(csv.DictReader(f))
    assert (tmp_path / "manifest.csv").read_bytes().count(b"\n") == len(kicks) + 1  # one line each, and a header
    for k, row, line in zip(kicks, manifest["kicks"], table, strict=True):
        r = k.report
        pitch = midi_pitch(r.tune_hz)
        expected = {
            "file": k.path,
            "patch": k.path.removesuffix(".wav") + ".toml",
            "velocity": f"{k.velocity[0]}-{k.velocity[1]}",
            "round_robin": k.robin,
            "sample_rate": r.sample_rate,
            "format": "float32" if bits is None else "pcm24",
            "tune_hz": r.tune_hz,
            "key": note_name(round(pitch)),
            "cents": round(100 * (pitch - round(pitch)), 1),
            "length_ms": round(1000 * r.samples / r.sample_rate, 3),
            "sample_peak_dbfs": round(20 * math.log10(r.peak), 2),
            "true_peak_dbtp": round(20 * math.log10(r.true_peak), 2),
            "loudness_lufs": round(r.loudness, 1),
            "sha256": hashlib.sha256((tmp_path / k.path).read_bytes()).hexdigest(),
        }
        assert row == expected
        assert line == {key: str(value) for key, value in expected.items()}
        assert (tmp_path / row["patch"]).read_text(encoding="utf-8") == k.text
    assert manifest["kicks"][0]["velocity"] != manifest["kicks"][-1]["velocity"]  # two layers, told apart
    assert manifest["kicks"][0]["cents"] != 0  # off the key, so the cents are written
