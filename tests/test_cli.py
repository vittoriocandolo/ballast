"""The command line: every command, its files, and its errors with their exit statuses."""

import io
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from ballast_synth import __version__, patch, wav
from ballast_synth.cli import main


@pytest.fixture
def run(capsys, tmp_path, monkeypatch):
    """Run `ballast ARGS…` in an empty directory: (status, stdout, stderr)."""
    monkeypatch.chdir(tmp_path)

    def run(*argv):
        status = main(list(argv))
        out, err = capsys.readouterr()
        return status, out, err

    return run


def test_init_prints_a_complete_patch(run, read_preset):
    assert run("init", "tight") == (0, read_preset("tight"), "")


def test_init_tunes_and_writes_a_file(run, tmp_path):
    assert run("init", "long", "--tune", "F1", "-o", "kick.toml") == (0, "", "")
    assert "tune_hz   = 43.654" in (tmp_path / "kick.toml").read_text(encoding="utf-8")


def test_tune_and_stretch_rewrite_a_patch_in_place(run, tmp_path):
    run("init", "tight", "-o", "kick.toml")
    assert run("tune", "kick.toml", "F#1", "-o", "kick.toml")[0] == 0
    assert run("stretch", "kick.toml", "body.amp", "0.5", "-o", "kick.toml")[0] == 0
    text = (tmp_path / "kick.toml").read_text(encoding="utf-8")
    assert "tune_hz   = 46.249" in text and "{ ms = 120.125, gain = 0.0 }" in text


def test_render_then_extract_gives_back_the_exact_text(run, tmp_path, read_preset):
    text = read_preset("driven").replace("seed = 1", "seed = 1  # kept verbatim, comments and all")
    (tmp_path / "kick.toml").write_text(text, encoding="utf-8")
    status, out, err = run("render", "kick.toml", "-o", "kick.wav", "--format", "pcm24")
    assert status == 0 and err == "" and out.splitlines()[-1] == "wrote kick.wav"
    assert "24-bit PCM" in out
    assert run("extract", "kick.wav") == (0, text, "")


def test_an_invalid_patch_lists_every_problem(run, tmp_path, read_preset):
    text = read_preset("tight").replace("tune_hz   = 55.0", "tune_hz   = 5.0").replace("h2        = 0.15", 'h2 = "x"')
    (tmp_path / "bad.toml").write_text(text, encoding="utf-8")
    assert run("tune", "bad.toml", "F1") == (
        2,
        "",
        "bad.toml: invalid patch\n"
        "  body.tune_hz: 5 is outside 20 … 1000\n"
        "  body.h2: expected a number, found a string\n",
    )


def test_a_render_over_full_scale_is_not_written(run, tmp_path, read_preset):
    (tmp_path / "loud.toml").write_text(read_preset("tight").replace("db          = -6.0", "db          = 12.0"))
    status, out, err = run("render", "loud.toml", "-o", "loud.wav")
    assert status == 1 and "over full scale by 12.32 dB" in out
    assert err == (
        "not written: lower output.db by at least 12.32 dB, or pass --allow-over to write a float file with overs\n"
    )
    assert not (tmp_path / "loud.wav").exists()
    assert run("render", "loud.toml", "-o", "loud.wav", "--allow-over")[0] == 0
    assert (tmp_path / "loud.wav").exists()


def test_allow_over_is_float_only(run):
    status, _, err = run("render", "kick.toml", "-o", "k.wav", "--allow-over", "--format", "pcm16")
    assert status == 2 and err.startswith("--allow-over writes float32 only")


@pytest.mark.parametrize(
    ("argv", "status", "err"),
    [
        (("tune", "missing.toml", "F1"), 2, "missing.toml: No such file or directory\n"),
        (
            ("init", "tight", "--tune", "C9"),
            2,
            "tight: cannot tune to C9\n  body.tune_hz: 8372.02 is outside 20 … 1000\n",
        ),
        (("extract", "nothing.wav"), 2, "nothing.wav: No such file or directory\n"),
    ],
)
def test_errors_name_the_file_and_the_problem(run, argv, status, err):
    assert run(*argv) == (status, "", err)


def test_extract_needs_a_wav_with_an_embedded_patch(run, tmp_path):
    (tmp_path / "x.wav").write_bytes(b"not a wave file")
    assert run("extract", "x.wav") == (2, "", "x.wav: not a RIFF WAVE file\n")
    wavfile.write(tmp_path / "foreign.wav", 48000, np.zeros(16, np.float32))
    assert run("extract", "foreign.wav") == (2, "", "foreign.wav: no embedded patch\n")


def test_audition_refuses_a_render_over_full_scale(run, tmp_path, read_preset):
    (tmp_path / "loud.toml").write_text(read_preset("tight").replace("db          = -6.0", "db          = 12.0"))
    status, _, err = run("audition", "loud.toml")
    assert status == 1 and err == "over full scale by 12.32 dB: lower output.db to audition it\n"


@pytest.mark.parametrize(
    "argv",
    [
        ("audition", "k.toml", "--with", "bed.wav"),
        ("audition", "k.toml", "--bpm", "0"),
        ("audition", "k.toml", "--bpm", "x"),
    ],
)
def test_audition_arguments_are_checked(run, argv):
    with pytest.raises(SystemExit) as caught:
        run(*argv)
    assert caught.value.code == 2


def test_version(run):
    with pytest.raises(SystemExit) as caught:
        run("--version")
    assert caught.value.code == 0


def test_a_patch_in_another_encoding_is_refused(run, tmp_path, read_preset):
    (tmp_path / "kick.toml").write_bytes(read_preset("tight").replace("…", "\xe9").encode("latin-1"))
    assert run("tune", "kick.toml", "F1") == (2, "", "kick.toml: not UTF-8 or UTF-16 text\n")


def test_writing_into_a_missing_directory_fails_cleanly(run, tmp_path):
    assert run("init", "tight", "-o", "missing/kick.toml") == (2, "", "missing/kick.toml: No such file or directory\n")
    run("init", "tight", "-o", "kick.toml")
    status, _, err = run("render", "kick.toml", "-o", "missing/kick.wav")
    assert (status, err) == (2, "missing/kick.wav: No such file or directory\n")


def test_stretch_lists_every_problem_it_would_cause(run, tmp_path):
    run("init", "tight", "-o", "kick.toml")
    status, out, err = run("stretch", "kick.toml", "body.amp", "100")
    assert status == 2 and out == ""
    assert err.splitlines() == [
        "kick.toml: cannot stretch body.amp by 100",
        "  body.amp[3].ms: 9975.25 is outside 0 … 8000",
        "  body.amp[4].ms: 17975.2 is outside 0 … 8000",
        "  body.amp[5].ms: 23975.2 is outside 0 … 8000",
    ]


def test_a_pcm_render_over_its_ceiling_is_not_written(run, tmp_path, read_preset):
    (tmp_path / "hot.toml").write_text(read_preset("tight").replace("db          = -6.0", "db          = 0.6"))
    status, out, err = run("render", "hot.toml", "-o", "hot.wav", "--format", "pcm16")
    assert status == 1 and "over the 16-bit ceiling of -0.0005 dBFS, which keeps 2 LSB for dither" in out
    assert err == "not written: lower output.db by at least 0.9221 dB\n" and not (tmp_path / "hot.wav").exists()


def test_an_audition_ends_cleanly_when_interrupted(run, tmp_path, monkeypatch):
    from ballast_synth import audition

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(audition, "audition", interrupted)
    run("init", "tight", "-o", "kick.toml")
    assert run("audition", "kick.toml", "--bpm", "128") == (0, "", "")


def test_python_dash_m_runs_the_command_line():
    command = [sys.executable, "-m", "ballast_synth", "--version"]
    done = subprocess.run(command, capture_output=True, text=True, check=True)
    assert done.stdout == f"ballast {__version__}\n"


def test_extract_reports_a_damaged_or_foreign_patch_chunk(run, tmp_path):
    run("init", "tight", "-o", "kick.toml")
    run("render", "kick.toml", "-o", "kick.wav")
    data = (tmp_path / "kick.wav").read_bytes()
    start = data.index(b"blst")
    (tmp_path / "cut.wav").write_bytes(data[: start + 48])
    assert run("extract", "cut.wav") == (2, "", "cut.wav: the embedded patch is damaged\n")
    body = b'{"renderer": "something else"}'
    chunk = b"blst" + len(body).to_bytes(4, "little") + body
    (tmp_path / "other.wav").write_bytes(b"RIFF" + (4 + len(chunk)).to_bytes(4, "little") + b"WAVE" + chunk)
    assert run("extract", "other.wav") == (2, "", "other.wav: no embedded patch\n")
    body = b'{"patch": 5}'
    chunk = b"blst" + len(body).to_bytes(4, "little") + body
    (tmp_path / "odd.wav").write_bytes(b"RIFF" + (4 + len(chunk)).to_bytes(4, "little") + b"WAVE" + chunk)
    assert run("extract", "odd.wav") == (2, "", "odd.wav: the embedded patch is damaged\n")


def test_ctrl_c_stops_quietly_with_status_130(run, monkeypatch):
    from ballast_synth import cli

    def interrupted(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "render", interrupted)
    run("init", "tight", "-o", "kick.toml")
    assert run("render", "kick.toml", "-o", "kick.wav") == (130, "", "")


def test_multisample_renders_one_sample_per_key_with_an_sfz_map(run, tmp_path):
    run("init", "long", "-o", "bass.toml")
    status, out, err = run("multisample", "bass.toml", "E1", "F#1", "-o", "set", "--format", "pcm24")
    assert (status, err) == (0, "")
    assert out.splitlines()[-1] == f"wrote 3 samples and {Path('set') / 'bass'}.sfz"
    for note, key in (("E1", 28), ("F1", 29), ("F#1", 30)):
        found = wav.chunks(tmp_path / "set" / f"bass-{note}.wav")
        assert struct.unpack("<9I", found[b"smpl"])[3:5] == (key, wav.root_note(patch.note_hz(key))[1])
        assert patch.loads(json.loads(found[wav.PRIVATE])["patch"]).body.tune_hz == patch.note_hz(key)
    assert (tmp_path / "set" / "bass.sfz").read_text().splitlines()[1:] == [
        "<region> sample=bass-E1.wav key=28",
        "<region> sample=bass-F1.wav key=29",
        "<region> sample=bass-F#1.wav key=30",
    ]


def test_multisample_writes_nothing_unless_every_note_works(run, tmp_path, read_preset):
    run("init", "long", "-o", "bass.toml")
    status, _, err = run("multisample", "bass.toml", "D0", "E0", "-o", "set")
    assert status == 2 and err.splitlines()[1:] == [
        "  D0: body.tune_hz: 18.354 is outside 20 … 1000",
        "  D#0: body.tune_hz: 19.445 is outside 20 … 1000",
    ]
    (tmp_path / "hot.toml").write_text(read_preset("tight").replace("db          = -6.0", "db          = 0.0"))
    status, _, err = run("multisample", "hot.toml", "A1", "B1", "-o", "set")
    assert status == 1 and err.startswith("not written: lower output.db; these notes are over the ceiling")
    assert run("multisample", "bass.toml", "G1", "E1", "-o", "set")[0] == 2
    assert not (tmp_path / "set").exists()


def test_multisample_reports_a_folder_it_cannot_make(run, tmp_path):
    run("init", "long", "-o", "bass.toml")
    (tmp_path / "set").write_text("a file, not a folder")
    status, out, err = run("multisample", "bass.toml", "E1", "F1", "-o", "set")
    assert (status, out) == (2, "") and err.startswith("set: ")


def test_any_standard_output_will_do(capsys, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    out = io.StringIO()  # not a TextIOWrapper, as when another program embeds Ballast
    monkeypatch.setattr(sys, "stdout", out)
    assert main(["init", "tight"]) == 0
    assert out.getvalue().startswith("schema = 2\n")
