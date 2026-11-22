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
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from . import __version__, play, render, wav
from .patch import Patch, dumps, edit
from .tuning import midi_pitch, note_name

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


TOP_VELOCITY = 127


def law_db(velocity: float) -> float:
    """Return the level the square law gives a velocity, against velocity 127: 40·log₁₀(v/127) dB."""
    return 40 * math.log10(velocity / TOP_VELOCITY)


SFZ_SYNTAX = ("<", ">", "=", "$", "\\", "//", "/*")  # what a sampler reads as syntax even inside a sample's name


def sfz_name_problem(name: str) -> str | None:
    """Return why `name` cannot stand for itself in an SFZ map, or None when it can.

    A sampler reads a map line by line: `<` opens a header, `word=` an opcode,
    `$` a variable, `//` and `/*` a comment and a backslash a folder, and the space
    around a value is trimmed. A name holding any of these, a control
    character or a line break would be read as something else: a crafted file
    name could add regions or opcodes to the map.
    """
    breaks = sorted({f"U+{ord(c):04X}" for c in name if unicodedata.category(c) in {"Cc", "Zl", "Zp"}})
    found = [*breaks, *(s for s in SFZ_SYNTAX if s in name)]
    if found:
        return f"{name!r} cannot name a file in an SFZ map: it holds {' '.join(found)}; rename it"
    if name != name.strip():
        return f"{name!r} cannot name a file in an SFZ map: a sampler trims the space around it; rename it"
    return None


class KitError(ValueError):
    """Kits whose layers cannot share the velocities on the square law. `problems` names each kit and why."""

    def __init__(self, problems: list[str]) -> None:
        """Keep every problem, and say them all."""
        self.problems = problems
        super().__init__("\n".join(problems))


def velocities(relative_db: list[float]) -> list[tuple[int, int]]:
    """Return each layer's MIDI velocities, softest first, from its loudness against the loudest, in dB.

    Velocity follows a square law, 40·log₁₀(v/127) dB, as Halyard's and
    Brightwork's kits do. A layer's velocities run up to its own, the velocity
    where the law reaches its loudness, from just above the softer layer's.
    So the map only ever lowers a hit, never plays it louder than it was
    played, and the law holds at every velocity. Layers whose loudness does
    not rise by a velocity step from each to the next, or a softest layer the
    law cannot reach by velocity 1, cannot be placed so: the ValueError says
    which, counting the layers as their files do, v1 the softest.
    """
    own = [math.floor(TOP_VELOCITY * 10 ** (d / 40) + 1e-9) for d in relative_db]  # the law's inverse
    ranges, low = [], 1
    for j, high in enumerate(own):
        if high < low:
            if not j:
                msg = f"layer v1 is {-relative_db[0]:.1f} dB under the loudest, below the law at velocity 1"
            else:
                msg = f"layers v{j} and v{j + 1} are within a velocity step of each other in loudness"
            raise ValueError(msg)
        ranges.append((low, high))
        low = high + 1
    return ranges


def curve(relative_db: float, low: int, high: int) -> list[float]:
    """Return the amplitude at each velocity low … high that puts a layer on the square law, never above 1.

    At velocity v the level should be 40·log₁₀(v/127) dB, and the layer's own is
    relative_db, so its gain is 10^((law − relative_db)/20): at most 1 up to the
    layer's own velocity, where its range ends.
    """
    return [10 ** ((law_db(v) - relative_db) / 20) for v in range(low, high + 1)]


def layer_loudness(items: list[Kick]) -> dict[tuple[int, int], float]:
    """Return each layer's loudness against the kit's loudest, in dB, by its velocities: the mean of its hits."""
    by_layer: dict[tuple[int, int], list[float]] = {}
    for k in items:
        by_layer.setdefault(k.velocity, []).append(k.report.loudness)
    mean = {v: sum(x) / len(x) for v, x in by_layer.items()}
    top = max(mean.values())
    return {v: m - top for v, m in mean.items()}


def kicks(
    patches: list[tuple[str, str, Patch]],
    rates: list[int],
    bits: int | None,
    layers: int = 1,
    robins: int = 1,
) -> list[Kick]:
    """Render every (name, text, patch) at each rate, or its own when `rates` is empty, hit at each layer and repeat.

    The layers and repeats of a played patch go through one chain: its hardest hit sets
    the clipper's threshold, and the kit's loudest hit, whichever it is, sets the output gain they all share.
    """
    if unnamable := [problem for name, _, _ in patches if (problem := sfz_name_problem(name))]:
        raise ValueError("\n".join(unnamable))
    played = layers > 1 or robins > 1
    refused: list[str] = []  # every kit is tried, so that each one the law cannot map is named at once
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
            try:
                out += _kit(q, stem, bits, layers, robins)
            except ValueError as e:
                refused.append(f"{stem}: {e}; lower [play] soft_db, or play fewer layers")
    if refused:
        raise KitError(refused)
    return out


def _kit(p: Patch, stem: str, bits: int | None, layers: int, robins: int) -> list[Kick]:
    """Render every layer and repeat of a played patch through one chain, its loudest hit at the patch's level."""
    played = []  # every hit is a patch, since the parser checked what [play] can make
    for j in range(layers):
        x = j / (layers - 1) if layers > 1 else 1.0
        for k in range(robins):
            path = f"{stem}{f'-v{j + 1}' if layers > 1 else ''}{f'-rr{k + 1}' if robins > 1 else ''}.wav"
            played.append((path, play.hit(p, x, k), j, k + 1))
    top = play.hit(p, 1.0, 0)
    hardest = render.synthesized(top)  # the kit's clipper, set by its hardest hit
    hits = []
    for path, hit, layer, robin in played:
        s = hardest if hit == top else render.synthesized(hit, hardest.threshold)
        hits.append((path, hit, s, layer, robin))
    ceiling = render.ceiling_of(bits)
    chain = render.Chain(hardest.threshold, min(render.output_gain(s, p.output, ceiling) for _, _, s, _, _ in hits))
    finished = [(path, hit, *render.finish(hit, s, bits, chain), layer, robin) for path, hit, s, layer, robin in hits]
    # Each layer's velocities follow from how loud its hits came out, on the square law.
    loud: dict[int, list[float]] = {}
    for *_, report, layer, _ in finished:
        loud.setdefault(layer, []).append(report.loudness)
    mean = [sum(loud[j]) / len(loud[j]) for j in range(layers)]
    ranges = velocities([m - max(mean) for m in mean])
    return [
        Kick(path, hit, dumps(hit), audio, report, ranges[layer], robin, stem)
        for path, hit, audio, report, layer, robin in finished
    ]


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

    Each group plays its hits as they were played, and its curve lowers them
    below its own velocity, so the level follows 40·log₁₀(v/127) dB across the
    layers' boundaries, as Halyard's and Brightwork's kits do, and no velocity
    plays a hit louder than it was played. A kick plays out whatever the
    note's length, as a one-shot.
    """
    name = Path(kit).name
    robins = max(k.robin for k in items)
    loudness = layer_loudness(items)
    lines = [
        f"// {name}: {len(loudness)} velocity layers, {robins} round robins, on key {DRUM_KEY}",
        "// Velocity follows a square law, 40·log10(v/127) dB, across layer boundaries.",
    ]
    for low, high in sorted(loudness):
        gains = curve(loudness[low, high], low, high)
        lines.append(
            f"<group> key={DRUM_KEY} lovel={low} hivel={high} amp_veltrack=100 loop_mode=one_shot seq_length={robins}",
        )
        # Nine decimals keep even velocity 1, where the curve is near 10⁻⁴, within 10⁻⁴ dB of the law.
        lines += [f"amp_velcurve_{v}={g:.9f}" for v, g in zip(range(low, high + 1), gains, strict=True)]
        lines += [
            f"<region> sample={Path(k.path).name} seq_position={k.robin}" for k in items if k.velocity == (low, high)
        ]
    return "\n".join(lines) + "\n"
