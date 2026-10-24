# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Command line: ballast init | tune | stretch | transient | vary | render | multisample | pack | audition | extract."""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

from . import __version__, pack, vary, wav
from .audition import AuditionError, Watcher, audition
from .patch import (
    CURVES,
    MAX_SEED,
    SAMPLE_RATES,
    Patch,
    PatchError,
    Transient,
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

MAX_VARIATIONS = 999  # how many variations one run writes, at most
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


def _os_error(path: object, e: OSError) -> CliError:
    """Make an error that names the file, and what the system said about it."""
    return CliError(f"{path}: {e.strerror or e}")


def _open(path: str) -> tuple[str, Patch]:
    """Read a patch file: its text, and its patch."""
    try:
        text = read_text(path)
    except OSError as e:
        raise _os_error(path, e) from None
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
        raise _os_error(out, e) from None


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


def cmd_transient(a: argparse.Namespace) -> None:
    """Layer a WAV over the attack as the patch's transient, as it sounds: at its own rate and level."""
    _, p = _open(a.patch)
    try:
        rate, frames = wav.read(a.wav)
    except OSError as e:
        raise _os_error(a.wav, e) from None
    except ValueError as e:
        msg = f"{a.wav}: {e}"
        raise CliError(msg) from None
    if frames.shape[1] not in wav.LAYOUTS:
        msg = f"{a.wav}: {frames.shape[1]} channels; a transient must be mono or stereo"
        raise CliError(msg)
    try:
        p = edit(p, "transient", Transient.of(frames.mean(axis=1), rate))  # stereo as its mid, (L + R) / 2
    except ValueError as e:
        msg = f"{a.wav}: cannot layer it on {a.patch}"
        raise _listing(msg, e) from None
    _emit(dumps(p), a.out)


def cmd_vary(a: argparse.Namespace) -> None:
    """Write variations of a patch into a folder, each a patch of its own: seeded, or along a grid of axes."""
    _, p = _open(a.patch)
    if a.along and (a.count is not None or a.seed is not None):
        msg = "--along sets a grid: leave out -n and --seed, which draw seeded variations"
        raise CliError(msg)
    if a.along:
        axes = [(path, [text for text in steps.split(",") if text]) for path, steps in a.along]
        if (size := math.prod(len(texts) for _, texts in axes)) > MAX_VARIATIONS:
            msg = f"the grid has {size} combinations; at most {MAX_VARIATIONS}"
            raise CliError(msg)
        try:
            made = vary.grid(p, axes)
        except PatchError as e:
            msg = f"{a.patch}: not written, these combinations are invalid"
            raise _listing(msg, e) from None
        what = f"{len(made)} combinations"
    else:
        number, chosen = a.count or 8, 1 if a.seed is None else a.seed
        made = [vary.vary(p, chosen, i) for i in range(1, number + 1)]
        what = f"{number} variations, seed {chosen},"
    stem, folder = Path(a.patch).stem, Path(a.out)
    try:
        folder.mkdir(parents=True, exist_ok=True)
        for i, (q, moves) in enumerate(made, 1):
            name = f"{stem}-v{i}.toml"
            wav.save(folder / name, dumps(q).encode("utf-8"))
            sys.stdout.write(f"{name:<20} {', '.join(moves)}\n")
    except OSError as e:
        raise _os_error(e.filename or folder, e) from None
    sys.stdout.write(f"wrote {what} of {a.patch} to {folder}\n")


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
        raise _os_error(a.out, e) from None
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
        raise _os_error(e.filename or folder, e) from None
    sys.stdout.write(f"wrote {len(kicks)} samples and {folder / stem}.sfz\n")


def _patches(paths: list[str]) -> list[tuple[str, str, Patch]]:
    """Read the patches a pack takes, as (name, text, patch): files, and the .toml files of folders, by name."""
    files: list[Path] = []
    for given in map(Path, paths):
        files += sorted(given.glob("*.toml")) if given.is_dir() else [given]
    if not files:
        msg = "no patches: give patch files, or folders that hold .toml files"
        raise CliError(msg)
    names = [f.stem for f in files]
    if twice := sorted({n for n in names if names.count(n) > 1}):
        msg = f"two patches would share a name in the pack: {', '.join(twice)}"
        raise CliError(msg)
    return [(f.stem, *_open(str(f))) for f in files]


def cmd_pack(a: argparse.Namespace) -> None:
    """Render patches into a finished kit, writing nothing unless every kick passes the gates."""
    bits = FORMATS[a.format]
    kicks = pack.kicks(_patches(a.patches), a.rate or [], bits)  # every kick renders before any file is written
    if refused := [(k, problems) for k in kicks if (problems := pack.gates(k))]:
        lines = ["not written: these kicks fail the pack's gates"]
        lines += [f"  {k.path}: {problem}" for k, problems in refused for problem in problems]
        raise CliError("\n".join(lines), 1)
    try:
        pack.write(a.out, kicks, bits)
    except OSError as e:
        raise _os_error(e.filename or a.out, e) from None
    for k in kicks:
        r = k.report
        sys.stdout.write(f"{k.path:<32} {db(r.true_peak):6.2f} dBTP {r.loudness:6.1f} LUFS\n")
    sys.stdout.write(f"wrote {len(kicks)} kicks, their patches and the manifest to {a.out}\n")


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
        raise _os_error(a.wav, e) from None
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


def count(text: str) -> int:
    """Parse a number of variations: a whole number from 1 to MAX_VARIATIONS."""
    value = int(text)
    if not 1 <= value <= MAX_VARIATIONS:
        msg = f"must be from 1 to {MAX_VARIATIONS}"
        raise argparse.ArgumentTypeError(msg)
    return value


def seed(text: str) -> int:
    """Parse a seed of variations: a whole number from 0 to 2**63 - 1, as a patch's seed."""
    value = int(text)
    if not 0 <= value <= MAX_SEED:
        msg = "must be from 0 to 2**63 - 1"
        raise argparse.ArgumentTypeError(msg)
    return value


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
    _patch_commands(sub)
    _sound_commands(sub)
    return ap


def _patch_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the commands that write patches."""
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

    command = sub.add_parser("transient", parents=[output], help="layer a WAV over the attack, as the transient")
    command.add_argument("patch")
    command.add_argument("wav", help="mono or stereo, at 44.1, 48, 88.2, 96, 176.4 or 192 kHz, ending by 1 s")
    command.set_defaults(run=cmd_transient)

    command = sub.add_parser("vary", help="write seeded variations of a patch, or every combination along axes")
    command.add_argument("patch")
    command.add_argument("-o", "--out", required=True, metavar="DIR", help="the folder for the variations")
    command.add_argument("-n", "--count", type=count, metavar="N", help="how many seeded variations (default: 8)")
    command.add_argument("--seed", type=seed, help="the same seed gives the same variations (default: 1)")
    command.add_argument(
        "--along",
        nargs=2,
        action="append",
        metavar=("FIELD", "STEPS"),
        help="a grid axis, such as drive.db 6,12,18 or body.amp x0.8,x1,x1.25; repeat for more axes",
    )
    command.set_defaults(run=cmd_vary)


def _sound_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the commands that render, play or read sound."""
    output = argparse.ArgumentParser(add_help=False)
    output.add_argument("-o", "--out", metavar="FILE", help="write to FILE instead of standard output")
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

    command = sub.add_parser("pack", help="render patches into a finished kit, with a manifest")
    command.add_argument("patches", nargs="+", metavar="PATCH", help="patch files, or folders of .toml files")
    command.add_argument("-o", "--out", required=True, metavar="DIR", help="the folder for the kit")
    command.add_argument(
        "--rate",
        type=int,
        action="append",
        choices=SAMPLE_RATES,
        help="render at this sample rate, into a folder of its own when given more than once (default: each patch's)",
    )
    command.add_argument("--format", choices=FORMATS, default="pcm24", help="as for render, but 24-bit PCM by default")
    command.set_defaults(run=cmd_pack)

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
