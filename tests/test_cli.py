# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The command line: every command, its files, and its errors with their exit statuses."""

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
from scipy.io import wavfile

from ballast_synth import __version__, cli, patch, wav
from ballast_synth.cli import main

BAD_INPUT = 2  # the exit status for input the command cannot use
REFUSED = 1  # the exit status for a refusal or a failure
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


def test_a_render_over_full_scale_is_not_written(run: Run, tmp_path: Path, read_preset: Callable[[str], str]) -> None:
    """Render 12 dB too loud, which writes nothing, unless overs are allowed in a float file."""
    loud = read_preset("tight").replace("db          = -6.0", "db          = 12.0")
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


def test_audition_refuses_a_render_over_full_scale(run: Run, tmp_path: Path, read_preset: Callable[[str], str]) -> None:
    """Refuse to audition a render over full scale, which the device would clip."""
    loud = read_preset("tight").replace("db          = -6.0", "db          = 12.0")
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
    read_preset: Callable[[str], str],
) -> None:
    """Render 16-bit PCM above its ceiling, which leaves room for dither, and write nothing."""
    hot = read_preset("tight").replace("db          = -6.0", "db          = 0.6")
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
    read_preset: Callable[[str], str],
) -> None:
    """Refuse notes that cannot be tuned, notes over the ceiling, and keys in the wrong order, writing nothing."""
    run("init", "long", "-o", "bass.toml")
    status, _, err = run("multisample", "bass.toml", "D0", "E0", "-o", "set")
    assert status == BAD_INPUT
    assert err.splitlines()[1:] == [
        "  D0: body.tune_hz: 18.354 is outside 20 … 1000",
        "  D#0: body.tune_hz: 19.445 is outside 20 … 1000",
    ]
    hot = read_preset("tight").replace("db          = -6.0", "db          = 0.0")
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
