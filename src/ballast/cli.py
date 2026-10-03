"""Command line: kick init | render | audition | extract (plan §7), and stretch and tune (§8.4)."""

from __future__ import annotations

import argparse
import json
import sys
from importlib import resources

from .patch import CURVES, PatchError, dumps, edit, loads, parse_tune, read_text, stretch

REFERENCES = ("round", "tight", "long", "driven")


def reference_text(name: str) -> str:
    return resources.files(__package__).joinpath("data", "presets", f"{name}.toml").read_text(encoding="utf-8")


def _read(path: str) -> str:
    try:
        return read_text(path)
    except OSError as e:
        raise SystemExit(_fail(f"{path}: {e.strerror}"))


def _fail(message: str, code: int = 2) -> int:
    print(message, file=sys.stderr)
    return code


def _patch(path: str, text: str):
    try:
        return loads(text)
    except PatchError as e:
        raise SystemExit(_fail(f"{path}: invalid patch\n" + "\n".join(f"  {m}" for m in e.problems)))


def cmd_init(args) -> int:
    p = loads(reference_text(args.reference))
    if args.tune is not None:
        try:
            p = edit(p, "body.tune_hz", parse_tune(args.tune))
        except (ValueError, PatchError) as e:
            return _fail(f"--tune {args.tune}: {e}")
    sys.stdout.write(dumps(p))
    return 0


def cmd_tune(args) -> int:
    p = _patch(args.patch, _read(args.patch))
    try:
        p = edit(p, "body.tune_hz", parse_tune(args.note))
    except (ValueError, PatchError) as e:
        problems = e.problems if isinstance(e, PatchError) else [str(e)]
        return _fail(f"{args.patch}: cannot tune to {args.note}\n" + "\n".join(f"  {m}" for m in problems))
    sys.stdout.write(dumps(p))
    return 0


def cmd_stretch(args) -> int:
    p = _patch(args.patch, _read(args.patch))
    try:
        p = stretch(p, args.curve, args.factor, args.after)
    except (ValueError, PatchError) as e:
        problems = e.problems if isinstance(e, PatchError) else [str(e)]
        return _fail(
            f"{args.patch}: cannot stretch {args.curve} by {args.factor:g}\n" + "\n".join(f"  {m}" for m in problems)
        )
    sys.stdout.write(dumps(p))
    return 0


def cmd_render(args) -> int:
    from .render import FORMATS, export, render

    bits = FORMATS[args.format]
    if args.allow_over and bits is not None:
        return _fail("--allow-over writes float32 only: PCM cannot hold samples beyond full scale")
    text = _read(args.patch)
    p = _patch(args.patch, text)
    audio, report = render(p, bits)
    print(report)
    if report.over and bits is not None:
        return _fail(f"not written: lower output.db by at least {report.over_db:.4f} dB", 1)
    if report.over and not args.allow_over:
        return _fail(
            f"not written: lower output.db by at least {report.over_db:.2f} dB, "
            "or pass --allow-over to write a float file with overs",
            1,
        )
    export(args.out, p, text, audio, bits)
    print(f"wrote {args.out}")
    return 0


def cmd_audition(args) -> int:
    from .audition import AuditionError, audition

    text = _read(args.patch)
    p = _patch(args.patch, text)
    try:
        audition(args.patch, text, p, watch=args.watch, bpm=args.bpm, bed=args.bed)
    except AuditionError as e:
        return _fail(str(e), 1)
    except KeyboardInterrupt:
        pass
    return 0


def cmd_extract(args) -> int:
    from . import wav

    try:
        found = wav.chunks(args.wav)
    except (OSError, ValueError) as e:
        return _fail(f"{args.wav}: {e}")
    if wav.PRIVATE not in found:
        return _fail(f"{args.wav}: no embedded kick patch")
    sys.stdout.write(json.loads(found[wav.PRIVATE].decode("utf-8"))["patch"])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="kick", description="A one-voice kick drum synthesizer.")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="print a complete starting patch")
    p.add_argument("reference", choices=REFERENCES)
    p.add_argument("--tune", help="a note name such as F1 or C#2, or a frequency in Hz")
    p.set_defaults(run=cmd_init)

    p = sub.add_parser("tune", help="print the patch retuned to a note or frequency")
    p.add_argument("patch")
    p.add_argument("note", help="a note name such as F1 or C#2, or a frequency in Hz")
    p.set_defaults(run=cmd_tune)

    p = sub.add_parser("stretch", help="print the patch with one curve's knot times rescaled")
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

    p = sub.add_parser("render", help="render a patch to a float WAV")
    p.add_argument("patch")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--format",
        choices=("float32", "pcm24", "pcm16"),
        default="float32",
        help="float32 (default), or PCM with seeded TPDF dither for hardware samplers",
    )
    p.add_argument(
        "--allow-over", action="store_true", help="write a gain-mode render that exceeds full scale (float32)"
    )
    p.set_defaults(run=cmd_render)

    p = sub.add_parser("audition", help="play a patch in a loop")
    p.add_argument("patch")
    p.add_argument("--watch", action="store_true", help="re-render whenever the file changes")
    p.add_argument("--bpm", type=float, help="hit on the beat instead of after each kick")
    p.add_argument(
        "--with",
        dest="bed",
        metavar="WAV",
        help="loop a bed under the hits, such as the track's bass line: a whole number of bars at --bpm",
    )
    p.set_defaults(run=cmd_audition)

    p = sub.add_parser("extract", help="print the patch embedded in an exported WAV")
    p.add_argument("wav")
    p.set_defaults(run=cmd_extract)

    args = ap.parse_args(argv)
    if getattr(args, "bpm", None) is not None and not args.bpm > 0:
        ap.error("--bpm must be positive")
    if getattr(args, "bed", None) and args.bpm is None:
        ap.error("--with needs --bpm, so the hits land on the bed's beats")
    try:
        return args.run(args)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1


if __name__ == "__main__":
    sys.exit(main())
