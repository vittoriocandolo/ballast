# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Command line: ballast init | tune | stretch | render | multisample | audition | extract."""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

from . import __version__, wav
from .audition import AuditionError, Watcher, audition
from .patch import (
    CURVES,
    Patch,
    PatchError,
    dumps,
    edit,
    loads,
    note_hz,
    note_name,
    note_number,
    parse_tune,
    read_text,
    stretch,
)
from .render import FORMATS, Report, db, export, render

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray

PRESET_DIR = resources.files("ballast_synth").joinpath("data", "presets")
PRESETS = sorted(f.name.removesuffix(".toml") for f in PRESET_DIR.iterdir() if f.name.endswith(".toml"))


class CliError(Exception):
    """A message for stderr, and the exit status: 2 for bad input, 1 for a refusal or a failure."""

    def __init__(self, message: str, status: int = 2) -> None:
        """Keep the message, and the status to exit with."""
        super().__init__(message)
        self.status = status


def _listing(head: str, e: ValueError) -> CliError:
    """Make an error of `head`, then every problem of a PatchError, or the message of another ValueError, indented."""
    problems = e.problems if isinstance(e, PatchError) else [str(e)]
    return CliError("\n".join([head, *(f"  {m}" for m in problems)]))


def _open(path: str) -> tuple[str, Patch]:
    """Read a patch file: its text, and its patch."""
    try:
        text = read_text(path)
    except OSError as e:
        msg = f"{path}: {e.strerror or e}"
        raise CliError(msg) from None
    except UnicodeDecodeError:
        msg = f"{path}: not UTF-8 or UTF-16 text"
        raise CliError(msg) from None
    try:
        return text, loads(text)
    except PatchError as e:
        msg = f"{path}: invalid patch"
        raise _listing(msg, e) from None


def _emit(text: str, out: str | None) -> None:
    """Write `text` to the file `out`, as UTF-8 with its own newlines, or to stdout."""
    if out is None:
        sys.stdout.write(text)
        return
    try:
        wav.save(out, text.encode("utf-8"))
    except OSError as e:
        msg = f"{out}: {e.strerror or e}"
        raise CliError(msg) from None


def _tuned(p: Patch, note: str, head: str) -> Patch:
    """Retune a patch to a note name or frequency, or explain under `head` why it cannot be."""
    try:
        return edit(p, "body.tune_hz", parse_tune(note))
    except ValueError as e:
        raise _listing(head, e) from None


def cmd_init(a: argparse.Namespace) -> None:
    """Start a patch from a preset, tuned if asked."""
    p = loads(PRESET_DIR.joinpath(f"{a.preset}.toml").read_text(encoding="utf-8"))
    if a.tune is not None:
        p = _tuned(p, a.tune, f"{a.preset}: cannot tune to {a.tune}")
    _emit(dumps(p), a.out)


def cmd_tune(a: argparse.Namespace) -> None:
    """Retune a patch to a note or frequency."""
    _, p = _open(a.patch)
    _emit(dumps(_tuned(p, a.note, f"{a.patch}: cannot tune to {a.note}")), a.out)


def cmd_stretch(a: argparse.Namespace) -> None:
    """Rescale the knot times of one curve."""
    _, p = _open(a.patch)
    try:
        p = stretch(p, a.curve, a.factor, a.after)
    except ValueError as e:
        msg = f"{a.patch}: cannot stretch {a.curve} by {a.factor:g}"
        raise _listing(msg, e) from None
    _emit(dumps(p), a.out)


def cmd_render(a: argparse.Namespace) -> None:
    """Render a patch to WAV, unless it goes over the format's ceiling."""
    bits = FORMATS[a.format]
    if a.allow_over and bits is not None:
        msg = "--allow-over writes float32 only: PCM cannot hold samples beyond full scale"
        raise CliError(msg)
    text, p = _open(a.patch)
    audio, report = render(p, bits)
    sys.stdout.write(f"{report}\n")
    if report.over and bits is not None:
        msg = f"not written: lower output.db by at least {report.over_db:.4f} dB"
        raise CliError(msg, 1)
    if report.over and not a.allow_over:
        msg = (
            f"not written: lower output.db by at least {report.over_db:.2f} dB, "
            "or pass --allow-over to write a float file with overs"
        )
        raise CliError(msg, 1)
    try:
        export(a.out, p, text, audio, bits)
    except OSError as e:
        msg = f"{a.out}: {e.strerror or e}"
        raise CliError(msg) from None
    sys.stdout.write(f"wrote {a.out}\n")


def _kicks(path: str, p: Patch, notes: range, bits: int | None) -> list[tuple[int, Patch, NDArray[np.float32], Report]]:
    """Render the patch tuned to every note, refusing all of them if one cannot be tuned or goes over the ceiling."""
    kicks, problems = [], []
    for n in notes:
        try:
            tuned = edit(p, "body.tune_hz", note_hz(n))
        except PatchError as e:
            problems += [f"{note_name(n)}: {m}" for m in e.problems]
            continue
        kicks.append((n, tuned, *render(tuned, bits)))
    if problems:
        msg = "\n".join([f"{path}: cannot tune to every note", *(f"  {m}" for m in problems)])
        raise CliError(msg)
    if over := [f"  {note_name(n)}: by {report.over_db:.4f} dB" for n, _, _, report in kicks if report.over]:
        msg = "\n".join(["not written: lower output.db; these notes are over the ceiling", *over])
        raise CliError(msg, 1)
    return kicks


def cmd_multisample(a: argparse.Namespace) -> None:
    """Render one sample per key, with an SFZ map, writing nothing unless every note works."""
    low, high = note_number(a.low), note_number(a.high)
    if low is None or high is None or low > high:
        msg = f"expected two note names, the lower first, such as C1 C2; found {a.low} {a.high}"
        raise CliError(msg)
    bits = FORMATS[a.format]
    _, p = _open(a.patch)
    kicks = _kicks(a.patch, p, range(low, high + 1), bits)  # every note renders before any file is written
    stem, folder = Path(a.patch).stem, Path(a.out)
    regions = []
    try:
        folder.mkdir(parents=True, exist_ok=True)
        for n, tuned, audio, report in kicks:
            name = f"{stem}-{note_name(n)}.wav"
            export(folder / name, tuned, dumps(tuned), audio, bits)
            sys.stdout.write(
                f"{note_name(n):<4} {report.tune_hz:9.3f} Hz   peak {db(report.peak):6.2f} dBFS   {name}\n",
            )
            regions.append(f"<region> sample={name} key={n}")
        sfz = [f"// {stem}, from {note_name(low)} to {note_name(high)}: one sample per key, at its root", *regions]
        wav.save(folder / f"{stem}.sfz", ("\n".join(sfz) + "\n").encode("utf-8"))
    except OSError as e:
        msg = f"{e.filename or folder}: {e.strerror or e}"
        raise CliError(msg) from None
    sys.stdout.write(f"wrote {len(kicks)} samples and {folder / stem}.sfz\n")


def cmd_audition(a: argparse.Namespace) -> None:
    """Play a patch in a loop, rendering it again on every edit when watching."""
    text, p = _open(a.patch)
    try:
        audition(p, watch=Watcher(a.patch, text) if a.watch else None, bpm=a.bpm, bed=a.bed)
    except AuditionError as e:
        raise CliError(str(e), 1) from None
    except KeyboardInterrupt:  # how an audition ends
        pass


def cmd_extract(a: argparse.Namespace) -> None:
    """Recover the patch embedded in a rendered WAV, as it was written."""
    try:
        found = wav.chunks(a.wav)
    except OSError as e:
        msg = f"{a.wav}: {e.strerror or e}"
        raise CliError(msg) from None
    except ValueError as e:
        msg = f"{a.wav}: {e}"
        raise CliError(msg) from None
    try:
        text = json.loads(found[wav.PRIVATE])["patch"]
    except KeyError:
        msg = f"{a.wav}: no embedded patch"
        raise CliError(msg) from None
    except (ValueError, TypeError):  # truncated or garbled, or not the table Ballast writes
        text = None
    if not isinstance(text, str):
        msg = f"{a.wav}: the embedded patch is damaged"
        raise CliError(msg)
    _emit(text, a.out)


def bpm(text: str) -> float:
    """Parse a tempo in beats per minute: a positive number."""
    value = float(text)
    if not (math.isfinite(value) and value > 0):
        msg = "must be a positive number"
        raise argparse.ArgumentTypeError(msg)
    return value


def _parser() -> argparse.ArgumentParser:
    """Build the parser of every command and its arguments."""
    ap = argparse.ArgumentParser(prog="ballast", description="A one-voice kick drum synthesizer, rendered to WAV.")
    ap.add_argument("--version", action="version", version=f"ballast {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)
    output = argparse.ArgumentParser(add_help=False)
    output.add_argument("-o", "--out", metavar="FILE", help="write to FILE instead of standard output")
    note = "a note name such as F1 or C#2, or a frequency in Hz"

    command = sub.add_parser("init", parents=[output], help="start a patch from a preset")
    command.add_argument("preset", choices=PRESETS)
    command.add_argument("--tune", metavar="NOTE", help=note)
    command.set_defaults(run=cmd_init)

    command = sub.add_parser("tune", parents=[output], help="retune a patch to a note or frequency")
    command.add_argument("patch")
    command.add_argument("note", help=note)
    command.set_defaults(run=cmd_tune)

    command = sub.add_parser("stretch", parents=[output], help="rescale the knot times of one curve")
    command.add_argument("patch")
    command.add_argument("curve", choices=CURVES)
    command.add_argument("factor", type=float, help="below 1 shortens, above 1 lengthens")
    command.add_argument(
        "--after",
        type=float,
        metavar="MS",
        help="keep knots up to this time (default: the peak of an amplitude shape, 0 ms for pitch)",
    )
    command.set_defaults(run=cmd_stretch)

    command = sub.add_parser("render", help="render a patch to WAV")
    command.add_argument("patch")
    command.add_argument("-o", "--out", required=True, metavar="WAV")
    command.add_argument(
        "--format",
        choices=FORMATS,
        default="float32",
        help="float32 (default), or PCM with seeded TPDF dither for hardware samplers",
    )
    command.add_argument("--allow-over", action="store_true", help="write a gain-mode render over full scale (float32)")
    command.set_defaults(run=cmd_render)

    command = sub.add_parser("multisample", help="render one sample per key, with an SFZ map, to play as an instrument")
    command.add_argument("patch")
    command.add_argument("low", help="the lowest note, such as C1")
    command.add_argument("high", help="the highest note, such as C2")
    command.add_argument("-o", "--out", required=True, metavar="DIR", help="the folder for the samples and the map")
    command.add_argument("--format", choices=FORMATS, default="float32", help="as for render")
    command.set_defaults(run=cmd_multisample)

    command = sub.add_parser("audition", help="play a patch in a loop")
    command.add_argument("patch")
    command.add_argument("--watch", action="store_true", help="re-render whenever the file changes")
    command.add_argument("--bpm", type=bpm, help="hit on the beat instead of after each kick")
    command.add_argument(
        "--with",
        dest="bed",
        metavar="WAV",
        help="loop a bed under the hits, such as the track's bass line: a whole number of bars at --bpm",
    )
    command.set_defaults(run=cmd_audition)

    command = sub.add_parser("extract", parents=[output], help="recover the patch embedded in a rendered WAV")
    command.add_argument("wav")
    command.set_defaults(run=cmd_extract)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line, and return the exit status."""
    for stream in (sys.stdout, sys.stderr):  # not the ANSI code page and CRLF of a Windows redirect
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8", newline="\n")
    ap = _parser()
    args = ap.parse_args(argv)
    if args.command == "audition" and args.bed and args.bpm is None:
        ap.error("--with needs --bpm, so the hits land on the bed's beats")
    try:
        args.run(args)
    except CliError as e:
        sys.stderr.write(f"{e}\n")
        return e.status
    except KeyboardInterrupt:  # Ctrl+C: stop quietly, with the status a shell expects
        return 130
    return 0
