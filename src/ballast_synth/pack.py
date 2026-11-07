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
import dataclasses
import hashlib
import io
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from . import __version__, play, render, wav
from .patch import Patch, PatchError, dumps, edit, midi_pitch, note_name

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

EDGE_DB = -90.0  # how quiet the first and last samples must be: below the smallest step of 16-bit PCM
DRUM_KEY = 36  # where a played kit sits: General MIDI's bass drum
GATES = [
    "true peak at or below 0 dBTP, and no sample over the format's ceiling",
    f"first and last samples below {EDGE_DB:g} dBFS",
]
ON_KEY = 0.005  # how near a note, in semitones, a kick must be to bear its name: half a cent
COLUMNS = (
    "file",
    "patch",
    "velocity",
    "round_robin",
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
    velocity: tuple[int, int] = (1, 127)  # the MIDI velocities it plays at
    robin: int = 1  # which repeat of its hit it is, from 1
    kit: str = ""  # the played kit it belongs to, as the path of its SFZ map without ".sfz", or "" for none
    layer: int = 0  # which velocity layer it is, from 0, the softest


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
        level = render.db(abs(float(sample)))  # a render always has samples: at least the decimator's tail
        if level > EDGE_DB:
            problems.append(f"{edge} sample at {level:.1f} dBFS, above {EDGE_DB:g} dBFS: it would click")
    return problems


def velocities(relative_db: list[float]) -> list[tuple[int, int]]:
    """Return each layer's MIDI velocities, softest first, from its loudness against the loudest, in dB.

    Velocity follows a square law, 40·log₁₀(v/127) dB, as Halyard's and
    Brightwork's kits do: a layer sits at the velocity where the law reaches its
    loudness, and neighbours meet halfway in dB, at the geometric mean of their
    velocities. Layers whose loudness does not rise from each to the next cannot
    share the velocities so, and are refused.
    """
    natural = [127 * 10 ** (d / 40) for d in relative_db]
    bounds = [0.0, *(math.sqrt(a * b) for a, b in itertools.pairwise(natural)), 127.0]
    ranges = [(math.floor(bounds[j]) + 1, math.floor(bounds[j + 1])) for j in range(len(natural))]
    for j, (low, high) in enumerate(ranges):
        if low > high or (j and not natural[j] > natural[j - 1]):
            msg = (
                f"layers {j} and {j + 1} are too close in loudness to get velocities of their own: "
                "lower [play] soft_db, or play fewer layers"
            )
            raise ValueError(msg)
    return ranges


def curve(relative_db: float, low: int, high: int) -> tuple[float, list[float]]:
    """Return the volume in dB and the amplitude at each velocity low … high that put a layer on the square law.

    At velocity v the level should be 40·log₁₀(v/127) dB, and the layer's own is
    relative_db, so its gain is (v/127)² / 10^(relative_db/20); the curve is that
    over its largest value, which becomes the group's volume.
    """
    gains = [(v / 127) ** 2 / 10 ** (relative_db / 20) for v in range(low, high + 1)]
    top = max(gains)
    return 20 * math.log10(top), [g / top for g in gains]


def layer_loudness(items: list[Kick]) -> dict[int, float]:
    """Return each layer's loudness against the kit's loudest layer, in dB, by its index: the mean of its repeats."""
    by_layer: dict[int, list[float]] = {}
    for k in items:
        by_layer.setdefault(k.layer, []).append(k.report.loudness)
    mean = {j: sum(v) / len(v) for j, v in by_layer.items()}
    top = max(mean.values())
    return {j: m - top for j, m in mean.items()}


def kicks(
    patches: list[tuple[str, str, Patch]],
    rates: list[int],
    bits: int | None,
    layers: int = 1,
    robins: int = 1,
) -> list[Kick]:
    """Render every (name, text, patch) at each rate, or its own when `rates` is empty, hit at each layer and repeat.

    The layers and repeats of a played patch go through one chain, as the hits of a drum do: its hardest hit sets
    the clipper's threshold, and the kit's loudest hit, whichever it is, sets the output gain they all share.
    """
    played = layers > 1 or robins > 1
    if played and (unplayed := [name for name, _, p in patches if p.play is None]):
        msg = f"{', '.join(unplayed)}: velocity layers and round robins need a [play] section"
        raise ValueError(msg)
    out = []
    for name, text, p in patches:
        for rate in rates or [p.output.sample_rate]:
            q = p if rate == p.output.sample_rate else edit(p, "output.sample_rate", rate)
            stem = f"{f'{rate}/' if len(rates) > 1 else ''}{name}-{label(q.body.tune_hz)}"
            if not played:
                audio, report = render.render(q, bits)
                out.append(Kick(f"{stem}.wav", q, text if q is p else dumps(q), audio, report))
                continue
            out += _kit(q, stem, bits, layers, robins)
    return out


def _kit(p: Patch, stem: str, bits: int | None, layers: int, robins: int) -> list[Kick]:
    """Render every layer and repeat of a played patch through one chain, its loudest hit at the patch's level."""
    played, problems = [], []  # every hit is made before any renders, so that every problem is listed at once
    for j in range(layers):
        x = j / (layers - 1) if layers > 1 else 1.0
        for k in range(robins):
            path = f"{stem}{f'-v{j + 1}' if layers > 1 else ''}{f'-rr{k + 1}' if robins > 1 else ''}.wav"
            try:
                played.append((path, play.again(play.softer(p, x), k), j, k + 1))
            except PatchError as e:
                problems += [f"{path}: {problem}" for problem in e.problems]
    if problems:
        raise PatchError(problems)
    hardest = render.synthesized(p)  # the kit's clipper, set by its hardest hit
    hits = []
    for path, hit, layer, robin in played:
        s = hardest if hit is p else render.synthesized(hit, hardest.threshold)
        hits.append((path, hit, s, layer, robin))
    ceiling = render.ceiling_of(bits)
    chain = render.Chain(hardest.threshold, min(render.output_gain(s, p.output, ceiling) for _, _, s, _, _ in hits))
    kicks = []
    for path, hit, s, layer, robin in hits:
        audio, report = render.finish(hit, s, bits, chain)
        kicks.append(Kick(path, hit, dumps(hit), audio, report, (1, 127), robin, stem, layer))
    # Each layer's velocities follow from how loud its hits came out, on the square law.
    loudness = layer_loudness(kicks)
    ranges = velocities([loudness[j] for j in range(layers)])
    return [dataclasses.replace(k, velocity=ranges[k.layer]) for k in kicks]


def write(folder: str | Path, items: list[Kick], bits: int | None) -> None:
    """Write every kick, its patch file and the manifest into `folder`; every file whole or not at all."""
    root = Path(folder)
    rows = []
    for k in items:
        target = root / k.path
        target.parent.mkdir(parents=True, exist_ok=True)
        render.export(target, k.patch, k.text, k.audio, k.report)
        patch_path = Path(k.path).with_suffix(".toml").as_posix()
        wav.save(root / patch_path, k.text.encode("utf-8"))
        r = k.report
        cents = round(100 * (midi_pitch(r.tune_hz) - round(midi_pitch(r.tune_hz))), 1) + 0.0
        rows.append(
            {
                "file": k.path,
                "patch": patch_path,
                "velocity": f"{k.velocity[0]}-{k.velocity[1]}",
                "round_robin": k.robin,
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
    for kit in dict.fromkeys(k.kit for k in items if k.kit):
        wav.save(root / f"{kit}.sfz", sfz(kit, [k for k in items if k.kit == kit]).encode("utf-8"))
    table = io.StringIO()
    writer = csv.DictWriter(table, COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    wav.save(root / "manifest.csv", table.getvalue().encode("utf-8"))
    manifest = {"renderer": f"ballast {__version__}", "gates": GATES, "kicks": rows}
    wav.save(root / "manifest.json", (json.dumps(manifest, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))


def sfz(kit: str, items: list[Kick]) -> str:
    """Map a played kit for samplers: a group per velocity layer on the drum key, on the square law, repeats in turn.

    Each group keeps its hits' own levels, as played, and adds an amplitude
    curve at every velocity in its range, so the level follows 40·log₁₀(v/127)
    dB across the layers' boundaries, as Halyard's and Brightwork's kits do. A
    kick plays out whatever the note's length, as a drum machine's does.
    """
    name = Path(kit).name
    robins = max(k.robin for k in items)
    loudness = layer_loudness(items)
    lines = [
        f"// {name}: {len(loudness)} velocity layers, {robins} round robins, on key {DRUM_KEY}",
        "// Velocity follows a square law, 40·log10(v/127) dB, across layer boundaries.",
    ]
    for j in sorted(loudness):
        layer = [k for k in items if k.layer == j]
        low, high = layer[0].velocity
        volume, gains = curve(loudness[j], low, high)
        lines.append(
            f"<group> key={DRUM_KEY} lovel={low} hivel={high} amp_veltrack=100 volume={volume:.4f} "
            f"loop_mode=one_shot seq_length={robins}",
        )
        lines += [
            f"amp_velcurve_{v}={g:.9f}" for v, g in zip(range(low, high + 1), gains, strict=True)
        ]  # 10⁻⁴ dB at v=1
        lines += [f"<region> sample={Path(k.path).name} seq_position={k.robin}" for k in layer]
    return "\n".join(lines) + "\n"
