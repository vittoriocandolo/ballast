"""Command line: ballast init | tune | stretch | render | multisample | audition | extract."""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from importlib import resources
from pathlib import Path

from . import __version__, wav
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
from .render import FORMATS, db, export, render

PRESET_DIR = resources.files(__package__).joinpath("data", "presets")
PRESETS = sorted(f.name.removesuffix(".toml") for f in PRESET_DIR.iterdir() if f.name.endswith(".toml"))


class CliError(Exception):
    """A message for stderr, and the exit status: 2 for bad input, 1 for a refusal or a failure."""

    def __init__(self, message: str, status: int = 2):
        super().__init__(message)
        self.status = status


def _listing(head: str, e: ValueError) -> CliError:
    """`head`, then every problem of a PatchError, or the message of another ValueError, indented."""
    problems = e.problems if isinstance(e, PatchError) else [str(e)]
    return CliError("\n".join([head, *(f"  {m}" for m in problems)]))


def _open(path: str) -> tuple[str, Patch]:
    """A patch file's text, and its patch."""
    try:
        text = read_text(path)
    except OSError as e:
        raise CliError(f"{path}: {e.strerror or e}") from None
    except UnicodeDecodeError:
        raise CliError(f"{path}: not UTF-8 or UTF-16 text") from None
    try:
        return text, loads(text)
    except PatchError as e:
        raise _listing(f"{path}: invalid patch", e) from None


def _emit(text: str, out: str | None) -> None:
    """Write `text` to the file `out`, as UTF-8 with its own newlines, or to stdout."""
    if out is None:
        sys.stdout.write(text)
        return
    try:
        wav.save(out, text.encode("utf-8"))
    except OSError as e:
        raise CliError(f"{out}: {e.strerror or e}") from None


def _tuned(p: Patch, note: str, head: str) -> Patch:
    try:
        return edit(p, "body.tune_hz", parse_tune(note))
    except ValueError as e:
        raise _listing(head, e) from None


def cmd_init(a) -> None:
    p = loads(PRESET_DIR.joinpath(f"{a.preset}.toml").read_text(encoding="utf-8"))
    if a.tune is not None:
        p = _tuned(p, a.tune, f"{a.preset}: cannot tune to {a.tune}")
    _emit(dumps(p), a.out)


def cmd_tune(a) -> None:
    _, p = _open(a.patch)
    _emit(dumps(_tuned(p, a.note, f"{a.patch}: cannot tune to {a.note}")), a.out)


def cmd_stretch(a) -> None:
    _, p = _open(a.patch)
    try:
        p = stretch(p, a.curve, a.factor, a.after)
    except ValueError as e:
        raise _listing(f"{a.patch}: cannot stretch {a.curve} by {a.factor:g}", e) from None
    _emit(dumps(p), a.out)


def cmd_render(a) -> None:
    bits = FORMATS[a.format]
    if a.allow_over and bits is not None:
        raise CliError("--allow-over writes float32 only: PCM cannot hold samples beyond full scale")
    text, p = _open(a.patch)
    audio, report = render(p, bits)
    print(report)
    if report.over and bits is not None:
        raise CliError(f"not written: lower output.db by at least {report.over_db:.4f} dB", 1)
    if report.over and not a.allow_over:
        raise CliError(
            f"not written: lower output.db by at least {report.over_db:.2f} dB, "
            "or pass --allow-over to write a float file with overs",
            1,
        )
    try:
        export(a.out, p, text, audio, bits)
    except OSError as e:
        raise CliError(f"{a.out}: {e.strerror or e}") from None
    print(f"wrote {a.out}")


def cmd_multisample(a) -> None:
    low, high = note_number(a.low), note_number(a.high)
    if low is None or high is None or low > high:
        raise CliError(f"expected two note names, the lower first, such as C1 C2; found {a.low} {a.high}")
    bits = FORMATS[a.format]
    _, p = _open(a.patch)
    kicks, problems = [], []
    for n in range(low, high + 1):  # every note renders before any file is written
        try:
            tuned = edit(p, "body.tune_hz", note_hz(n))
        except PatchError as e:
            problems += [f"{note_name(n)}: {m}" for m in e.problems]
            continue
        kicks.append((n, tuned, *render(tuned, bits)))
    if problems:
        raise CliError("\n".join([f"{a.patch}: cannot tune to every note", *(f"  {m}" for m in problems)]))
    if over := [f"  {note_name(n)}: by {report.over_db:.4f} dB" for n, _, _, report in kicks if report.over]:
        raise CliError("\n".join(["not written: lower output.db; these notes are over the ceiling", *over]), 1)
    stem, folder = Path(a.patch).stem, Path(a.out)
    regions = []
    try:
        folder.mkdir(parents=True, exist_ok=True)
        for n, tuned, audio, report in kicks:
            name = f"{stem}-{note_name(n)}.wav"
            export(folder / name, tuned, dumps(tuned), audio, bits)
            print(f"{note_name(n):<4} {report.tune_hz:9.3f} Hz   peak {db(report.peak):6.2f} dBFS   {name}")
            regions.append(f"<region> sample={name} key={n}")
        sfz = [f"// {stem}, from {note_name(low)} to {note_name(high)}: one sample per key, at its root", *regions]
        wav.save(folder / f"{stem}.sfz", ("\n".join(sfz) + "\n").encode("utf-8"))
    except OSError as e:
        raise CliError(f"{e.filename or folder}: {e.strerror or e}") from None
    print(f"wrote {len(kicks)} samples and {folder / stem}.sfz")


def cmd_audition(a) -> None:
    from .audition import AuditionError, audition

    text, p = _open(a.patch)
    try:
        audition(a.patch, text, p, watch=a.watch, bpm=a.bpm, bed=a.bed)
    except AuditionError as e:
        raise CliError(str(e), 1) from None
    except KeyboardInterrupt:  # how an audition ends
        pass


def cmd_extract(a) -> None:
    try:
        found = wav.chunks(a.wav)
    except OSError as e:
        raise CliError(f"{a.wav}: {e.strerror or e}") from None
    except ValueError as e:
        raise CliError(f"{a.wav}: {e}") from None
    try:
        text = json.loads(found[wav.PRIVATE])["patch"]
        if not isinstance(text, str):
            raise TypeError
    except KeyError:
        raise CliError(f"{a.wav}: no embedded patch") from None
    except (ValueError, TypeError):  # truncated or garbled, or not the table Ballast writes
        raise CliError(f"{a.wav}: the embedded patch is damaged") from None
    _emit(text, a.out)


def bpm(text: str) -> float:
    value = float(text)
    if not (math.isfinite(value) and value > 0):
        raise argparse.ArgumentTypeError("must be a positive number")
    return value


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):  # not the ANSI code page and CRLF of a Windows redirect
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8", newline="\n")
    ap = argparse.ArgumentParser(prog="ballast", description="A one-voice kick drum synthesizer, rendered to WAV.")
    ap.add_argument("--version", action="version", version=f"ballast {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)
    output = argparse.ArgumentParser(add_help=False)
    output.add_argument("-o", "--out", metavar="FILE", help="write to FILE instead of standard output")
    note = "a note name such as F1 or C#2, or a frequency in Hz"

    p = sub.add_parser("init", parents=[output], help="start a patch from a preset")
    p.add_argument("preset", choices=PRESETS)
    p.add_argument("--tune", metavar="NOTE", help=note)
    p.set_defaults(run=cmd_init)

    p = sub.add_parser("tune", parents=[output], help="retune a patch to a note or frequency")
    p.add_argument("patch")
    p.add_argument("note", help=note)
    p.set_defaults(run=cmd_tune)

    p = sub.add_parser("stretch", parents=[output], help="rescale the knot times of one curve")
    p.add_argument("patch")
    p.add_argument("curve", choices=CURVES)
    p.add_argument("factor", type=float, help="below 1 shortens, above 1 lengthens")
    p.add_argument(
        "--after",
        type=float,
        metavar="MS",
        help="keep knots up to this time (default: the peak of an amplitude shape, 0 ms for pitch)",
    )
    p.set_defaults(run=cmd_stretch)

    p = sub.add_parser("render", help="render a patch to WAV")
    p.add_argument("patch")
    p.add_argument("-o", "--out", required=True, metavar="WAV")
    p.add_argument(
        "--format",
        choices=FORMATS,
        default="float32",
        help="float32 (default), or PCM with seeded TPDF dither for hardware samplers",
    )
    p.add_argument("--allow-over", action="store_true", help="write a gain-mode render over full scale (float32)")
    p.set_defaults(run=cmd_render)

    p = sub.add_parser("multisample", help="render one sample per key, with an SFZ map, to play as an instrument")
    p.add_argument("patch")
    p.add_argument("low", help="the lowest note, such as C1")
    p.add_argument("high", help="the highest note, such as C2")
    p.add_argument("-o", "--out", required=True, metavar="DIR", help="the folder for the samples and the map")
    p.add_argument("--format", choices=FORMATS, default="float32", help="as for render")
    p.set_defaults(run=cmd_multisample)

    p = sub.add_parser("audition", help="play a patch in a loop")
    p.add_argument("patch")
    p.add_argument("--watch", action="store_true", help="re-render whenever the file changes")
    p.add_argument("--bpm", type=bpm, help="hit on the beat instead of after each kick")
    p.add_argument(
        "--with",
        dest="bed",
        metavar="WAV",
        help="loop a bed under the hits, such as the track's bass line: a whole number of bars at --bpm",
    )
    p.set_defaults(run=cmd_audition)

    p = sub.add_parser("extract", parents=[output], help="recover the patch embedded in a rendered WAV")
    p.add_argument("wav")
    p.set_defaults(run=cmd_extract)

    args = ap.parse_args(argv)
    if args.command == "audition" and args.bed and args.bpm is None:
        ap.error("--with needs --bpm, so the hits land on the bed's beats")
    try:
        args.run(args)
    except CliError as e:
        print(e, file=sys.stderr)
        return e.status
    except KeyboardInterrupt:  # Ctrl+C: stop quietly, with the status a shell expects
        return 130
    return 0
