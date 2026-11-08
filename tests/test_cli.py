# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The command line: every command, its files, and its errors with their exit statuses."""

import hashlib
import io
import json
import runpy
import struct
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.io import wavfile

from ballast_synth import __version__, cli, patch, wav
from ballast_synth.cli import main

BAD_INPUT = 2  # the exit status for input the command cannot use
REFUSED = 1  # the exit status for a refusal or a failure
CD_RATE = 44100
DRIVE_HIGH = 18.0  # the last step of a grid along the drive
INTERRUPTED = 130  # the exit status a shell expects after Ctrl+C

type Run = Callable[..., tuple[int, str, str]]


@pytest.fixture
def run(capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Run:
    """Return a runner of `ballast ARGS…` in an empty directory, giving (status, stdout, stderr)."""
    monkeypatch.chdir(tmp_path)

    def run(*argv: str) -> tuple[int, str, str]:
        status = main(list(argv))
        out, err = capsys.readouterr()
        return status, out, err

    return run


def test_init_prints_a_complete_patch(run: Run, read_preset: Callable[[str], str]) -> None:
    """Start a patch from a preset, on standard output."""
    assert run("init", "tight") == (0, read_preset("tight"), "")


def test_init_tunes_and_writes_a_file(run: Run, tmp_path: Path) -> None:
    """Start a patch from a preset, tuned to a note, in a file."""
    assert run("init", "long", "--tune", "F1", "-o", "kick.toml") == (0, "", "")
    assert "tune_hz   = 43.654" in (tmp_path / "kick.toml").read_text(encoding="utf-8")


def test_tune_and_stretch_rewrite_a_patch_in_place(run: Run, tmp_path: Path) -> None:
    """Retune a patch file and stretch a curve, each writing over the file it read."""
    run("init", "tight", "-o", "kick.toml")
    assert run("tune", "kick.toml", "F#1", "-o", "kick.toml")[0] == 0
    assert run("stretch", "kick.toml", "body.amp", "0.5", "-o", "kick.toml")[0] == 0
    text = (tmp_path / "kick.toml").read_text(encoding="utf-8")
    assert "tune_hz   = 46.249" in text
    assert "{ ms = 120.125, gain = 0.0 }" in text


def test_render_then_extract_gives_back_the_exact_text(
    run: Run,
    tmp_path: Path,
    read_preset: Callable[[str], str],
) -> None:
    """Render a patch with a comment of its own, and extract the very same text from the WAV."""
    text = read_preset("driven").replace("seed = 1", "seed = 1  # kept verbatim, comments and all")
    (tmp_path / "kick.toml").write_text(text, encoding="utf-8")
    status, out, err = run("render", "kick.toml", "-o", "kick.wav", "--format", "pcm24")
    assert status == 0
    assert err == ""
    assert out.splitlines()[-1] == "wrote kick.wav"
    assert "24-bit PCM" in out
    assert run("extract", "kick.wav") == (0, text, "")


def test_transient_layers_a_wav_as_it_sounds(run: Run, tmp_path: Path, click: NDArray[np.float64]) -> None:
    """Layer a mono WAV, and a stereo one of the same click, which both give the click at its own level."""
    wavfile.write(tmp_path / "click.wav", 48000, click.astype(np.float32))
    wavfile.write(tmp_path / "wide.wav", 48000, np.stack([click, click], axis=1).astype(np.float32))
    run("init", "tight", "-o", "kick.toml")
    status, out, err = run("transient", "kick.toml", "click.wav")
    assert (status, err) == (0, "")
    t = patch.loads(out).transient
    assert t is not None
    assert (t.level_db, t.rate, patch.decode(t.samples).size) == (-12.0, 48000, click.size)
    assert run("transient", "kick.toml", "wide.wav") == (0, out, "")


@pytest.mark.parametrize(
    ("frames", "rate", "err"),
    [
        (np.ones((10, 3)), 48000, "click.wav: 3 channels; a transient must be mono or stereo\n"),
        (
            np.ones(10),
            32000,
            (
                "click.wav: cannot layer it on kick.toml\n"
                "  transient.rate: expected one of 44100, 48000, 88200, 96000, 176400, 192000, found 32000\n"
            ),
        ),
        (np.zeros(10), 48000, "click.wav: cannot layer it on kick.toml\n  the samples are silent\n"),
        (
            np.ones(48000 + 1),
            48000,
            (
                "click.wav: cannot layer it on kick.toml\n"
                "  transient.samples: the transient ends at 1000.02 ms, after 1000 ms\n"
            ),
        ),
    ],
)
def test_transient_refuses_what_it_cannot_layer(
    run: Run,
    tmp_path: Path,
    frames: NDArray[np.float64],
    rate: int,
    err: str,
) -> None:
    """Refuse more than two channels, a rate no family has, silence, and a transient that ends after 1 s."""
    wavfile.write(tmp_path / "click.wav", rate, frames.astype(np.float32))
    run("init", "tight", "-o", "kick.toml")
    assert run("transient", "kick.toml", "click.wav") == (BAD_INPUT, "", err)


def test_an_invalid_patch_lists_every_problem(run: Run, tmp_path: Path, read_preset: Callable[[str], str]) -> None:
    """Retune a patch with two broken fields, and get both problems under the file's name."""
    text = read_preset("tight").replace("tune_hz   = 55.0", "tune_hz   = 5.0").replace("h2        = 0.15", 'h2 = "x"')
    (tmp_path / "bad.toml").write_text(text, encoding="utf-8")
    assert run("tune", "bad.toml", "F1") == (
        BAD_INPUT,
        "",
        (
            "bad.toml: invalid patch\n"
            "  body.tune_hz: 5 is outside 20 … 1000\n"
            "  body.h2: expected a number, found a string\n"
        ),
    )


def test_a_render_over_full_scale_is_not_written(run: Run, tmp_path: Path, gain_text: str) -> None:
    """Render 12 dB too loud, which writes nothing, unless overs are allowed in a float file."""
    loud = gain_text.replace("db          = -6.0", "db          = 12.0")
    (tmp_path / "loud.toml").write_text(loud, encoding="utf-8")
    status, out, err = run("render", "loud.toml", "-o", "loud.wav")
    assert status == REFUSED
    assert "over full scale by 12.28 dB" in out
    assert err == (
        "not written: lower output.db by at least 12.28 dB, or pass --allow-over to write a float file with overs\n"
    )
    assert not (tmp_path / "loud.wav").exists()
    assert run("render", "loud.toml", "-o", "loud.wav", "--allow-over")[0] == 0
    assert (tmp_path / "loud.wav").exists()


def test_allow_over_is_float_only(run: Run) -> None:
    """Refuse to allow overs in PCM, which cannot hold them."""
    status, _, err = run("render", "kick.toml", "-o", "k.wav", "--allow-over", "--format", "pcm16")
    assert status == BAD_INPUT
    assert err.startswith("--allow-over writes float32 only")


@pytest.mark.parametrize(
    ("argv", "status", "err"),
    [
        (("tune", "missing.toml", "F1"), BAD_INPUT, "missing.toml: No such file or directory\n"),
        (
            ("init", "tight", "--tune", "C9"),
            BAD_INPUT,
            "tight: cannot tune to C9\n  body.tune_hz: 8372.02 is outside 20 … 1000\n",
        ),
        (("extract", "nothing.wav"), BAD_INPUT, "nothing.wav: No such file or directory\n"),
    ],
)
def test_errors_name_the_file_and_the_problem(run: Run, argv: tuple[str, ...], status: int, err: str) -> None:
    """Fail on a missing file or an impossible tuning, naming the file or the preset and the problem."""
    assert run(*argv) == (status, "", err)


def test_extract_needs_a_wav_with_an_embedded_patch(run: Run, tmp_path: Path) -> None:
    """Refuse to extract from a file that is not a WAV, or from a WAV that Ballast did not write."""
    (tmp_path / "x.wav").write_bytes(b"not a wave file")
    assert run("extract", "x.wav") == (BAD_INPUT, "", "x.wav: not a RIFF WAVE file\n")
    wavfile.write(tmp_path / "foreign.wav", 48000, np.zeros(16, np.float32))
    assert run("extract", "foreign.wav") == (BAD_INPUT, "", "foreign.wav: no embedded patch\n")


def test_audition_refuses_a_render_over_full_scale(run: Run, tmp_path: Path, gain_text: str) -> None:
    """Refuse to audition a render over full scale, which the device would clip."""
    loud = gain_text.replace("db          = -6.0", "db          = 12.0")
    (tmp_path / "loud.toml").write_text(loud, encoding="utf-8")
    status, _, err = run("audition", "loud.toml")
    assert status == REFUSED
    assert err == "over full scale by 12.28 dB: lower output.db to audition it\n"


@pytest.mark.parametrize(
    "argv",
    [
        ("audition", "k.toml", "--with", "bed.wav"),
        ("audition", "k.toml", "--bpm", "0"),
        ("audition", "k.toml", "--bpm", "x"),
    ],
)
def test_audition_arguments_are_checked(run: Run, argv: tuple[str, ...]) -> None:
    """Refuse a bed without a tempo, and a tempo that is not a positive number."""
    with pytest.raises(SystemExit) as caught:
        run(*argv)
    assert caught.value.code == BAD_INPUT


def test_version(run: Run) -> None:
    """Print the version, and exit with success."""
    with pytest.raises(SystemExit) as caught:
        run("--version")
    assert caught.value.code == 0


def test_a_patch_in_another_encoding_is_refused(run: Run, tmp_path: Path, read_preset: Callable[[str], str]) -> None:
    """Refuse a patch file in Latin-1."""
    (tmp_path / "kick.toml").write_bytes(read_preset("tight").replace("…", "\xe9").encode("latin-1"))
    assert run("tune", "kick.toml", "F1") == (BAD_INPUT, "", "kick.toml: not UTF-8 or UTF-16 text\n")


def test_writing_into_a_missing_directory_fails_cleanly(run: Run) -> None:
    """Fail to write a patch or a WAV into a folder that does not exist, naming the file."""
    assert run("init", "tight", "-o", "missing/kick.toml") == (
        BAD_INPUT,
        "",
        "missing/kick.toml: No such file or directory\n",
    )
    run("init", "tight", "-o", "kick.toml")
    status, _, err = run("render", "kick.toml", "-o", "missing/kick.wav")
    assert (status, err) == (BAD_INPUT, "missing/kick.wav: No such file or directory\n")


def test_stretch_lists_every_problem_it_would_cause(run: Run) -> None:
    """Stretch a shape past the longest one allowed, and get the problem of every knot it would move too far."""
    run("init", "tight", "-o", "kick.toml")
    status, out, err = run("stretch", "kick.toml", "body.amp", "100")
    assert status == BAD_INPUT
    assert out == ""
    assert err.splitlines() == [
        "kick.toml: cannot stretch body.amp by 100",
        "  body.amp[3].ms: 9975.25 is outside 0 … 8000",
        "  body.amp[4].ms: 17975.2 is outside 0 … 8000",
        "  body.amp[5].ms: 23975.2 is outside 0 … 8000",
    ]


def test_a_pcm_render_over_its_ceiling_is_not_written(
    run: Run,
    tmp_path: Path,
    gain_text: str,
) -> None:
    """Render 16-bit PCM above its ceiling, which leaves room for dither, and write nothing."""
    hot = gain_text.replace("db          = -6.0", "db          = 0.6")
    (tmp_path / "hot.toml").write_text(hot, encoding="utf-8")
    status, out, err = run("render", "hot.toml", "-o", "hot.wav", "--format", "pcm16")
    assert status == REFUSED
    assert "over the 16-bit ceiling of -0.0005 dBFS, which keeps 2 LSB for dither" in out
    assert err == "not written: lower output.db by at least 0.8786 dB\n"
    assert not (tmp_path / "hot.wav").exists()


def test_an_audition_ends_cleanly_when_interrupted(run: Run, monkeypatch: pytest.MonkeyPatch) -> None:
    """Interrupt an audition, as Ctrl+C ends one, and exit quietly with success."""

    def interrupted(*_args: object, **_kwargs: object) -> NoReturn:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "audition", interrupted)
    run("init", "tight", "-o", "kick.toml")
    assert run("audition", "kick.toml", "--bpm", "128") == (0, "", "")


def test_python_dash_m_runs_the_command_line(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the package as a module, as `python -m ballast_synth --version` does."""
    monkeypatch.setattr(sys, "argv", ["ballast", "--version"])
    with pytest.raises(SystemExit) as caught:
        runpy.run_module("ballast_synth", run_name="__main__")
    assert caught.value.code == 0
    assert capsys.readouterr().out == f"ballast {__version__}\n"


def test_extract_reports_a_damaged_or_foreign_patch_chunk(run: Run, tmp_path: Path) -> None:
    """Extract from a truncated chunk, a chunk without a patch, and one whose patch is not text."""
    run("init", "tight", "-o", "kick.toml")
    run("render", "kick.toml", "-o", "kick.wav")
    data = (tmp_path / "kick.wav").read_bytes()
    start = data.index(b"blst")
    (tmp_path / "cut.wav").write_bytes(data[: start + 48])
    assert run("extract", "cut.wav") == (BAD_INPUT, "", "cut.wav: the embedded patch is damaged\n")
    body = b'{"renderer": "something else"}'
    chunk = b"blst" + len(body).to_bytes(4, "little") + body
    (tmp_path / "other.wav").write_bytes(b"RIFF" + (4 + len(chunk)).to_bytes(4, "little") + b"WAVE" + chunk)
    assert run("extract", "other.wav") == (BAD_INPUT, "", "other.wav: no embedded patch\n")
    body = b'{"patch": 5}'
    chunk = b"blst" + len(body).to_bytes(4, "little") + body
    (tmp_path / "odd.wav").write_bytes(b"RIFF" + (4 + len(chunk)).to_bytes(4, "little") + b"WAVE" + chunk)
    assert run("extract", "odd.wav") == (BAD_INPUT, "", "odd.wav: the embedded patch is damaged\n")


def test_ctrl_c_stops_quietly_with_status_130(run: Run, monkeypatch: pytest.MonkeyPatch) -> None:
    """Interrupt a render, and exit with the status a shell expects after Ctrl+C, saying nothing."""

    def interrupted(*_args: object) -> NoReturn:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "render", interrupted)
    run("init", "tight", "-o", "kick.toml")
    assert run("render", "kick.toml", "-o", "kick.wav") == (INTERRUPTED, "", "")


def test_multisample_renders_one_sample_per_key_with_an_sfz_map(run: Run, tmp_path: Path) -> None:
    """Render three keys, each tuned and mapped to its root, with an SFZ map of all three."""
    run("init", "long", "-o", "bass.toml")
    status, out, err = run("multisample", "bass.toml", "E1", "F#1", "-o", "set", "--format", "pcm24")
    assert (status, err) == (0, "")
    assert out.splitlines()[-1] == f"wrote 3 samples and {Path('set') / 'bass'}.sfz"
    for note, key in (("E1", 28), ("F1", 29), ("F#1", 30)):
        found = wav.chunks(tmp_path / "set" / f"bass-{note}.wav")
        assert struct.unpack("<9I", found[b"smpl"])[3:5] == (key, wav.root_note(patch.note_hz(key))[1])
        assert patch.loads(json.loads(found[wav.PRIVATE])["patch"]).body.tune_hz == patch.note_hz(key)
    assert (tmp_path / "set" / "bass.sfz").read_text(encoding="utf-8").splitlines()[1:] == [
        "<region> sample=bass-E1.wav key=28",
        "<region> sample=bass-F1.wav key=29",
        "<region> sample=bass-F#1.wav key=30",
    ]


def test_multisample_writes_nothing_unless_every_note_works(
    run: Run,
    tmp_path: Path,
    gain_text: str,
) -> None:
    """Refuse notes that cannot be tuned, notes over the ceiling, and keys in the wrong order, writing nothing."""
    run("init", "long", "-o", "bass.toml")
    status, _, err = run("multisample", "bass.toml", "D0", "E0", "-o", "set")
    assert status == BAD_INPUT
    assert err.splitlines()[1:] == [
        "  D0: body.tune_hz: 18.354 is outside 20 … 1000",
        "  D#0: body.tune_hz: 19.445 is outside 20 … 1000",
    ]
    hot = gain_text.replace("db          = -6.0", "db          = 0.0")
    (tmp_path / "hot.toml").write_text(hot, encoding="utf-8")
    status, _, err = run("multisample", "hot.toml", "A1", "B1", "-o", "set")
    assert status == REFUSED
    assert err.startswith("not written: lower output.db; these notes are over the ceiling")
    assert run("multisample", "bass.toml", "G1", "E1", "-o", "set")[0] == BAD_INPUT
    assert not (tmp_path / "set").exists()


def test_multisample_reports_a_folder_it_cannot_make(run: Run, tmp_path: Path) -> None:
    """Fail to make the samples' folder where a file stands, naming it."""
    run("init", "long", "-o", "bass.toml")
    (tmp_path / "set").write_text("a file, not a folder", encoding="utf-8")
    status, out, err = run("multisample", "bass.toml", "E1", "F1", "-o", "set")
    assert (status, out) == (BAD_INPUT, "")
    assert err.startswith("set: ")


def test_any_standard_output_will_do(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Print a patch to a standard output that is not a TextIOWrapper, as when another program embeds Ballast."""
    monkeypatch.chdir(tmp_path)
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    assert main(["init", "tight"]) == 0
    assert out.getvalue().startswith("schema = 3\n")


def test_pack_writes_a_finished_kit_the_same_every_time(run: Run, tmp_path: Path) -> None:
    """Pack two presets at two rates: a folder per rate, named by key or frequency, a manifest, and the same bytes."""
    (tmp_path / "kicks").mkdir()
    for name in ("house", "driven"):
        run("init", name, "-o", f"kicks/{name}.toml")
    status, out, err = run("pack", "kicks", "-o", "kit", "--rate", "44100", "--rate", "48000")
    assert (status, err) == (0, "")
    assert out.splitlines()[-1] == "wrote 4 kicks, their patches and the manifest to kit"
    kit = tmp_path / "kit"
    names = sorted(p.relative_to(kit).as_posix() for p in kit.rglob("*") if p.is_file())
    assert names == [
        "44100/driven-50Hz.toml",
        "44100/driven-50Hz.wav",
        "44100/house-A1.toml",
        "44100/house-A1.wav",
        "48000/driven-50Hz.toml",
        "48000/driven-50Hz.wav",
        "48000/house-A1.toml",
        "48000/house-A1.wav",
        "manifest.csv",
        "manifest.json",
    ]
    manifest = json.loads((kit / "manifest.json").read_text(encoding="utf-8"))
    rows = {row["file"]: row for row in manifest["kicks"]}
    assert (rows["48000/house-A1.wav"]["format"], rows["48000/driven-50Hz.wav"]["cents"]) == ("pcm24", 35.0)
    assert rows["44100/house-A1.wav"]["sha256"] == hashlib.sha256((kit / "44100/house-A1.wav").read_bytes()).hexdigest()
    assert patch.loads((kit / "44100/house-A1.toml").read_text(encoding="utf-8")).output.sample_rate == CD_RATE
    before = {p: p.read_bytes() for p in kit.rglob("*") if p.is_file()}
    assert run("pack", "kicks", "-o", "kit", "--rate", "44100", "--rate", "48000")[0] == 0
    assert {p: p.read_bytes() for p in kit.rglob("*") if p.is_file()} == before


def test_pack_writes_nothing_unless_every_kick_passes_its_gates(run: Run, tmp_path: Path, gain_text: str) -> None:
    """Pack a kick over full scale with a good one, and get both refused, every gate it fails listed."""
    (tmp_path / "kicks").mkdir()
    run("init", "tight", "-o", "kicks/good.toml")
    loud = gain_text.replace("db          = -6.0", "db          = 12.0")
    (tmp_path / "kicks" / "loud.toml").write_text(loud, encoding="utf-8")
    status, out, err = run("pack", "kicks", "-o", "kit")
    assert (status, out) == (REFUSED, "")
    assert err.splitlines()[0] == "not written: these kicks fail the pack's gates"
    assert "  loud-A1.wav: over the format's ceiling by" in err
    assert "  loud-A1.wav: true peak +12.35 dBTP, above 0 dBTP" in err
    assert not (tmp_path / "kit").exists()


def test_commands_name_the_files_they_cannot_read_or_write(run: Run, tmp_path: Path) -> None:
    """Refuse a missing WAV, a file that is not one, and a folder where a file stands, each named."""
    run("init", "tight", "-o", "kick.toml")
    assert run("transient", "kick.toml", "missing.wav") == (BAD_INPUT, "", "missing.wav: No such file or directory\n")
    (tmp_path / "notes.wav").write_text("not a WAV", encoding="utf-8")
    status, out, err = run("transient", "kick.toml", "notes.wav")
    assert (status, out) == (BAD_INPUT, "")
    assert err.startswith("notes.wav: ")
    (tmp_path / "taken").write_text("", encoding="utf-8")
    assert run("vary", "kick.toml", "-o", "taken", "-n", "1") == (BAD_INPUT, "", "taken: File exists\n")
    assert run("pack", "kick.toml", "-o", "taken") == (BAD_INPUT, "", "taken: File exists\n")


def test_pack_and_vary_refuse_kits_too_large_or_unplayable(
    run: Run,
    tmp_path: Path,
    read_preset: Callable[[str], str],
) -> None:
    """Refuse a grid of more than 999 combinations, 17 layers, and hits too soft for any patch to hold."""
    run("init", "tight", "-o", "kick.toml")
    drives, seconds = ",".join(str(db) for db in range(40)), ",".join(str(h / 100) for h in range(30))
    axes = ("--along", "drive.db", drives, "--along", "body.h2", seconds)
    status, _, err = run("vary", "kick.toml", "-o", "grid", *axes)
    assert (status, err) == (BAD_INPUT, "the grid has 1200 combinations; at most 999\n")
    assert not (tmp_path / "grid").exists()
    play = "\n[play]\nsoft_db       = -40.0\nsoft_click_db = -24.0\nsoft_punch    = 0.6\nspread        = 0.5\n"
    (tmp_path / "soft.toml").write_text(read_preset("tight") + play, encoding="utf-8")
    status, out, err = run("pack", "soft.toml", "-o", "kit", "--velocities", "3")
    assert (status, out) == (BAD_INPUT, "")
    assert err == (
        "not written: these hits cannot be played\n"
        "  soft-A1-v1.wav: attack.level_db: -80 is outside -60 … 24 (or -inf)\n"
    )
    with pytest.raises(SystemExit) as caught:
        run("pack", "kick.toml", "-o", "kit", "--velocities", "17")
    assert caught.value.code == BAD_INPUT
    # Layers struck alike are equally loud, and the law cannot give each velocities of its own: every such kit
    # is named, and the pack is refused rather than mapped.
    alike = "\n[play]\nsoft_db       = 0.0\nsoft_click_db = 0.0\nsoft_punch    = 1.0\nspread        = 0.0\n"
    (tmp_path / "alike.toml").write_text(read_preset("tight") + alike, encoding="utf-8")
    (tmp_path / "same.toml").write_text(read_preset("house") + alike, encoding="utf-8")
    status, out, err = run("pack", "alike.toml", "same.toml", "-o", "kit", "--velocities", "2")
    assert (status, out) == (REFUSED, "")
    assert "not written: these kits cannot put their layers on the square law of velocity\n" in err
    assert "alike-A1: layers v1 and v2 are within a velocity step" in err
    assert "same-A1: layers v1 and v2" in err


def test_vary_and_pack_refuse_what_they_cannot_make(run: Run, tmp_path: Path) -> None:
    """Refuse variations whose snap would squeeze two pitch knots closer than 0.01 ms, and a rate given twice."""
    run("init", "tight", "-o", "kick.toml")
    p = patch.loads((tmp_path / "kick.toml").read_text(encoding="utf-8"))
    close = patch.edit(p, "body.pitch", ((0.0, 4.0), (0.01, 3.0), (60.0, 1.0)))
    (tmp_path / "close.toml").write_text(patch.dumps(close), encoding="utf-8")
    status, out, err = run("vary", "close.toml", "-o", "vars", "-n", "8")
    assert (status, out) == (BAD_INPUT, "")
    assert err.startswith("close.toml: not written, a variation is invalid\n  body.pitch[1].ms: ")
    assert not (tmp_path / "vars").exists()
    status, _, err = run("pack", "kick.toml", "-o", "kit", "--rate", "48000", "--rate", "44100", "--rate", "48000")
    assert (status, err) == (BAD_INPUT, "--rate 48000 given more than once\n")


def test_pack_refuses_names_it_cannot_tell_apart(run: Run, tmp_path: Path) -> None:
    """Refuse two patches of one name from two folders, and an empty folder."""
    for folder, name in (("a", "tight"), ("b", "house")):
        (tmp_path / folder).mkdir()
        run("init", name, "-o", f"{folder}/kick.toml")
    assert run("pack", "a", "b", "-o", "kit") == (BAD_INPUT, "", "two patches would share a name in the pack: kick\n")
    (tmp_path / "empty").mkdir()
    assert run("pack", "empty", "-o", "kit")[0] == BAD_INPUT


def test_vary_writes_seeded_variations(run: Run, tmp_path: Path) -> None:
    """Vary a patch twice with one seed and once with another: the same files, then other ones."""
    run("init", "tight", "-o", "kick.toml")
    status, out, _ = run("vary", "kick.toml", "-o", "a", "-n", "3", "--seed", "4")
    assert status == 0
    assert out.splitlines()[0].startswith("kick-v1.toml         punch ×")
    run("vary", "kick.toml", "-o", "b", "-n", "3", "--seed", "4")
    run("vary", "kick.toml", "-o", "c", "-n", "3", "--seed", "5")
    for name in ("kick-v1.toml", "kick-v2.toml", "kick-v3.toml"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
        assert (tmp_path / "a" / name).read_bytes() != (tmp_path / "c" / name).read_bytes()
    for bad in (("-n", "0"), ("--seed", "-1")):
        with pytest.raises(SystemExit) as caught:
            run("vary", "kick.toml", "-o", "d", *bad)
        assert caught.value.code == BAD_INPUT


def test_vary_along_axes_writes_every_combination(run: Run, tmp_path: Path) -> None:
    """Vary a patch along two axes, write nine patches, and refuse a grid mixed with seeded options."""
    run("init", "tight", "-o", "kick.toml")
    axes = ("--along", "drive.db", "6,12,18", "--along", "body.amp", "x0.8,x1,x1.25")
    status, out, _ = run("vary", "kick.toml", "-o", "grid", *axes)
    assert status == 0
    assert out.splitlines()[-1] == "wrote 9 combinations of kick.toml to grid"
    assert patch.loads((tmp_path / "grid" / "kick-v9.toml").read_text(encoding="utf-8")).drive.db == DRIVE_HIGH
    assert run("vary", "kick.toml", "-o", "x", "--along", "drive.db", "6", "--seed", "2")[0] == BAD_INPUT
    status, _, err = run("vary", "kick.toml", "-o", "x", "--along", "drive.db", "6,60")
    assert (status, err.splitlines()[1]) == (BAD_INPUT, "  drive.db 60: drive.db: 60 is outside 0 … 48")
    assert not (tmp_path / "x").exists()


def test_pack_plays_a_kit_with_layers_and_repeats(run: Run, tmp_path: Path, read_preset: Callable[[str], str]) -> None:
    """Pack a patch with [play] in two layers and two repeats, with its SFZ map, and refuse one without [play]."""
    play = "\n[play]\nsoft_db       = -18.0\nsoft_click_db = -6.0\nsoft_punch    = 0.6\nspread        = 0.5\n"
    (tmp_path / "kick.toml").write_text(read_preset("house") + play, encoding="utf-8")
    status, out, err = run("pack", "kick.toml", "-o", "kit", "--velocities", "2", "--round-robins", "2")
    assert (status, err) == (0, "")
    assert out.splitlines()[-1] == "wrote 4 kicks, their patches and the manifest to kit"
    assert (tmp_path / "kit" / "kick-A1.sfz").read_text(encoding="utf-8").startswith("// kick-A1: 2 velocity layers")
    assert (tmp_path / "kit" / "kick-A1-v1-rr2.wav").exists()
    run("init", "tight", "-o", "plain.toml")
    status, _, err = run("pack", "plain.toml", "-o", "kit2", "--velocities", "2")
    assert (status, err) == (BAD_INPUT, "plain: velocity layers and round robins need a [play] section\n")
