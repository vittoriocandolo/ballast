# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Packs: a folder of patches rendered into a finished kit, refused whole unless every kick passes its gates.

Each kick is named after its patch and its key, or its frequency when it is
more than half a cent off every note, and goes into the pack with its patch
file and a line of the manifest: its tuning, length, peaks and loudness, and
the SHA-256 of its WAV. The manifest holds nothing of when or how fast the
pack was made, so the same patches always make the same pack, byte for byte.

The gates refuse a kick whose true peak is above 0 dBTP or whose highest
sample is over the format's ceiling, and one whose first or last sample is
above -90 dBFS, which a sampler would hear as a click where the file starts
or stops.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from . import __version__, render, wav
from .patch import Patch, dumps, edit, midi_pitch, note_name

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

EDGE_DB = -90.0  # how quiet the first and last samples must be: below the smallest step of 16-bit PCM
GATES = [
    "true peak at or below 0 dBTP, and no sample over the format's ceiling",
    f"first and last samples below {EDGE_DB:g} dBFS",
]
ON_KEY = 0.005  # how near a note, in semitones, a kick must be to bear its name: half a cent
COLUMNS = (
    "file",
    "patch",
    "sample_rate",
    "format",
    "tune_hz",
    "key",
    "cents",
    "length_ms",
    "sample_peak_dbfs",
    "true_peak_dbtp",
    "loudness_lufs",
    "sha256",
)


@dataclass(frozen=True)
class Kick:
    """One kick of a pack: where it goes, the patch and text it was rendered from, its audio and its report."""

    path: str  # the WAV's path in the pack, with "/" between folders
    patch: Patch
    text: str
    audio: NDArray[np.float32]
    report: render.Report


def label(hz: float) -> str:
    """Name a tuning by its note, such as "A1", or by its frequency, such as "50Hz", when it is off every note."""
    pitch = midi_pitch(hz)
    n = round(pitch)
    return note_name(n) if abs(pitch - n) < ON_KEY else f"{hz:g}Hz"


def gates(kick: Kick) -> list[str]:
    """Return why a kick cannot go into a pack: nothing, when it passes every gate."""
    r, problems = kick.report, []
    if r.over:
        problems.append(f"over the format's ceiling by {r.over_db:.4f} dB")
    if r.true_peak > 1.0:
        problems.append(f"true peak {render.db(r.true_peak):+.2f} dBTP, above 0 dBTP")
    for edge, sample in (("first", kick.audio[0]), ("last", kick.audio[-1])):
        level = render.db(abs(float(sample))) if kick.audio.size else -math.inf
        if level > EDGE_DB:
            problems.append(f"{edge} sample at {level:.1f} dBFS, above {EDGE_DB:g} dBFS: it would click")
    return problems


def kicks(patches: list[tuple[str, str, Patch]], rates: list[int], bits: int | None) -> list[Kick]:
    """Render every (name, text, patch) at each rate, or at its own rate when `rates` is empty."""
    out = []
    for name, text, p in patches:
        for rate in rates or [p.output.sample_rate]:
            q = p if rate == p.output.sample_rate else edit(p, "output.sample_rate", rate)
            folder = f"{rate}/" if len(rates) > 1 else ""
            audio, report = render.render(q, bits)
            out.append(
                Kick(f"{folder}{name}-{label(q.body.tune_hz)}.wav", q, text if q is p else dumps(q), audio, report),
            )
    return out


def write(folder: str | Path, items: list[Kick], bits: int | None) -> None:
    """Write every kick, its patch file and the manifest into `folder`; every file whole or not at all."""
    root = Path(folder)
    rows = []
    for k in items:
        target = root / k.path
        target.parent.mkdir(parents=True, exist_ok=True)
        render.export(target, k.patch, k.text, k.audio, bits)
        patch_path = Path(k.path).with_suffix(".toml").as_posix()
        wav.save(root / patch_path, k.text.encode("utf-8"))
        r = k.report
        cents = round(100 * (midi_pitch(r.tune_hz) - round(midi_pitch(r.tune_hz))), 1) + 0.0
        rows.append(
            {
                "file": k.path,
                "patch": patch_path,
                "sample_rate": r.sample_rate,
                "format": "float32" if bits is None else f"pcm{bits}",
                "tune_hz": r.tune_hz,
                "key": note_name(round(midi_pitch(r.tune_hz))),
                "cents": cents,
                "length_ms": round(1000 * r.samples / r.sample_rate, 3),
                "sample_peak_dbfs": round(render.db(r.peak), 2),
                "true_peak_dbtp": round(render.db(r.true_peak), 2),
                "loudness_lufs": round(r.loudness, 1),
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            },
        )
    table = io.StringIO()
    writer = csv.DictWriter(table, COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    wav.save(root / "manifest.csv", table.getvalue().encode("utf-8"))
    manifest = {"renderer": f"ballast {__version__}", "gates": GATES, "kicks": rows}
    wav.save(root / "manifest.json", (json.dumps(manifest, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
