# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Audition: the mix of hits and bed, bed loading, the file watcher, the watch loop and the output device."""

import os
import re
import sys
import time
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, NoReturn, Self, cast

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.io import wavfile

from ballast_synth import audition, patch
from ballast_synth.audition import AuditionError, Bed, Hits, Watcher, beat, load_bed
from ballast_synth.render import Report, render

RATE = 48000
WASAPI_DEVICE = 3  # the stand-in WASAPI host's default output device


def onsets(signal: NDArray[np.float32]) -> list[int]:
    """Return the samples where the signal steps up: where each hit starts."""
    return np.flatnonzero(np.diff(np.concatenate(([0.0], signal))) > 0).tolist()


def test_free_running_hits_repeat_after_each_kick_and_half_a_second() -> None:
    """Hit without a tempo, every kick's length plus half a second."""
    hits = Hits(np.ones(1000, np.float32), RATE)
    assert onsets(hits.fill(3 * 25_000)[:, 0]) == [0, 25_000, 50_000]


def test_hits_on_the_beat_are_unchoked_so_their_tails_sum() -> None:
    """Hit on the beat with a kick longer than a beat, so each tail sums with the next hit."""
    hits = Hits(np.ones(30_000, np.float32), RATE, bpm=128.0)  # longer than a beat of 22 500 samples
    out = hits.fill(50_000)[:, 0]
    assert out[[0, 22_499, 22_500, 30_000, 45_000]].tolist() == [1, 1, 2, 1, 2]


def test_a_new_buffer_plays_from_the_next_hit() -> None:
    """Replace the buffer between hits, and hear the old one to its end and the new one from the next hit."""
    hits = Hits(np.ones(100, np.float32), RATE, bpm=120.0)  # a beat is 24 000 samples
    first = hits.fill(1000)
    level = 0.5
    hits.buffer = np.full(100, level, np.float32)
    assert hits.wait() == pytest.approx(23_000 / RATE)
    rest = hits.fill(30_000)[:, 0]
    assert first[0, 0] == 1
    assert rest[23_000] == level


def test_block_sizes_never_change_the_mix() -> None:
    """Mix hits over a bed in blocks of many sizes, and compare every sample with one block."""
    bed = Bed(np.random.default_rng(0).standard_normal((90_495, 2)).astype(np.float32), 4)
    kick = np.random.default_rng(1).standard_normal(30_000).astype(np.float32)
    sizes = [1, 511, 4096, 30_000] * 10
    hits = Hits(kick, RATE, 127.3, bed)
    parts = np.concatenate([hits.fill(size) for size in sizes])
    assert np.array_equal(parts, Hits(kick, RATE, 127.3, bed).fill(sum(sizes)))


def test_the_bed_restarts_on_the_same_beats_as_the_hits() -> None:
    """Loop a bed longer than its bars, and find each loop starting on its beat and cut off by the next."""
    bpm, beats = 127.3, 4
    ramp = np.arange(1, 100_001, dtype=np.float32)[:, None]  # longer than a loop, and each position recognizable
    hits = Hits(np.zeros(1, np.float32), RATE, bpm, Bed(ramp, beats))
    out = hits.fill(300_000)[:, 0]
    for k in range(3):
        start = hits.loop_start(k)
        assert start == beat(k * beats, RATE, bpm)
        assert out[start] == 1.0  # the loop starts over…
    assert out[hits.loop_start(1) - 1] == hits.loop_start(1)  # …cut off where the next one starts


def test_only_a_bed_has_loops() -> None:
    """Refuse to place a loop when there is no bed."""
    with pytest.raises(ValueError, match="there is no bed to loop"):
        Hits(np.zeros(1, np.float32), RATE).loop_start(0)


def write_bed(path: Path, frames: NDArray[Any], rate: int = RATE) -> Path:
    """Write frames as a WAV, and return its path."""
    wavfile.write(path, rate, frames)
    return path


def test_load_bed_reads_whole_bars_in_any_format(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Load the same two bars from 8-bit, 16-bit stereo and float WAVs, as float32 at the same level."""
    bars, level = 2, 0.5
    n = round(bars * 4 * RATE * 60 / 128.0)
    beds: dict[str, NDArray[Any]] = {
        "u8": np.full(n, round(128 + 128 * level), np.uint8),
        "i16": np.full((n, 2), round(level * 2**15), np.int16),
        "f32": np.full(n, level, np.float32),
    }
    for name, data in beds.items():
        bed = load_bed(write_bed(tmp_path / f"{name}.wav", data), RATE, 128.0)
        assert bed.beats == 4 * bars
        assert bed.frames.dtype == np.float32
        assert np.all(bed.frames == level)
    assert capsys.readouterr().out.splitlines()[1].endswith("i16.wav: 2 bars at 128 bpm, stereo")


@pytest.mark.parametrize(
    ("frames", "rate", "error"),
    [
        (np.zeros(round(1.5 * 4 * RATE * 60 / 128.0), np.float32), RATE, "1.500 bars at 128 bpm"),
        (np.zeros(1000, np.float32), RATE, "0.011 bars at 128 bpm"),
        (np.zeros((round(4 * RATE * 60 / 128.0), 3), np.float32), RATE, "3 channels; a bed must be mono or stereo"),
        (np.zeros(round(4 * 44100 * 60 / 128.0), np.float32), 44100, "44100 Hz, but the patch renders at 48000 Hz"),
    ],
)
def test_load_bed_refuses_what_would_drift_or_resample(
    tmp_path: Path,
    frames: NDArray[np.float32],
    rate: int,
    error: str,
) -> None:
    """Refuse a bed that is not whole bars, has more channels than stereo, or is at another rate."""
    with pytest.raises(AuditionError, match=re.escape(error)):
        load_bed(write_bed(tmp_path / "bed.wav", frames, rate), RATE, 128.0)


def test_the_watcher_reports_new_text_only(tmp_path: Path) -> None:
    """Poll a file through an edit, a touch, an unreadable moment mid-save, and another edit."""
    path = tmp_path / "kick.toml"
    path.write_text("a", encoding="utf-8")
    watcher = Watcher(path, "a")
    assert watcher.poll() is None  # the content is what the session started with
    path.write_text("bb", encoding="utf-8")
    assert watcher.poll() == "bb"
    assert watcher.poll() is None
    os.utime(path, ns=(10**18, 10**18))  # touched, not changed
    assert watcher.poll() is None
    path.write_bytes(b"\xff\xfe\xfd")  # caught mid-save, unreadable: skipped
    assert watcher.poll() is None
    path.write_text("ccc", encoding="utf-8")
    assert watcher.poll() == "ccc"


class PortAudioError(Exception):
    """The error PortAudio raises for a stream it cannot open."""


class FakeOutputStream:
    """Opens at 48 kHz, or at any rate with extra settings, and pulls one block when it starts."""

    opened: ClassVar[list[dict[str, Any]]] = []
    status: ClassVar[types.SimpleNamespace] = types.SimpleNamespace(output_underflow=False)
    played: ClassVar[NDArray[np.float32]]

    def __init__(
        self,
        samplerate: int,
        channels: int,
        dtype: str,
        latency: str,
        callback: Callable[[NDArray[np.float32], int, object, types.SimpleNamespace], None],
        **settings: object,
    ) -> None:
        """Open the stream as PortAudio would, after checking the format audition asks for."""
        assert dtype == "float32"
        assert latency == "high"
        FakeOutputStream.opened.append({"samplerate": samplerate, **settings})
        if samplerate != RATE and "extra_settings" not in settings:
            msg = "Invalid sample rate"
            raise PortAudioError(msg)
        self.device, self.channels, self.callback = 0, channels, callback

    def __enter__(self) -> Self:
        """Start the stream, which pulls its first block."""
        self.block = np.zeros((256, self.channels), np.float32)
        self.callback(self.block, 256, None, FakeOutputStream.status)
        FakeOutputStream.played = self.block
        return self

    def __exit__(self, *exc: object) -> None:
        """Stop the stream, letting any exception through."""


@pytest.fixture
def device(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Return a stand-in for sounddevice with one output device, installed on Linux."""
    sd = types.ModuleType("sounddevice")
    vars(sd).update(
        PortAudioError=PortAudioError,
        query_hostapis=lambda: [{"name": "Windows WASAPI", "default_output_device": WASAPI_DEVICE}],
        query_devices=lambda _device: {"name": "Test Device"},
        OutputStream=FakeOutputStream,
        WasapiSettings=lambda exclusive: ("wasapi", exclusive),
        CoreAudioSettings=lambda **flags: ("coreaudio", flags),
    )
    FakeOutputStream.opened.clear()
    monkeypatch.setattr(FakeOutputStream, "status", types.SimpleNamespace(output_underflow=False))
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    monkeypatch.setattr(sys, "platform", "linux")
    return sd


@pytest.fixture
def one_tick(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop an audition at the first tick of its play loop, as Ctrl+C would."""

    def stop(_seconds: float) -> NoReturn:
        raise KeyboardInterrupt

    monkeypatch.setattr(time, "sleep", stop)


@pytest.mark.usefixtures("device")
def test_the_watch_loop_rerenders_every_edit(
    tmp_path: Path,
    gain_text: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Save a valid edit, an invalid one, one too loud, one at another rate, and one saved again while it renders."""
    path = tmp_path / "kick.toml"
    path.write_text(gain_text, encoding="utf-8")
    last = 60.0
    versions = [
        gain_text.replace("tune_hz   = 55.0", "tune_hz   = 49.0"),
        gain_text.replace("h2        = 0.15", 'h2        = "x"'),
        gain_text.replace("db          = -6.0", "db          = 12.0"),
        gain_text.replace("sample_rate = 48000", "sample_rate = 96000"),
        gain_text.replace("tune_hz   = 55.0", f"tune_hz   = {last}"),
    ]
    stamp = [10**18]

    def save(text: str) -> None:  # a new modification time each save, whatever the file system's resolution
        path.write_text(text, encoding="utf-8")
        stamp[0] += 10**9
        os.utime(path, ns=(stamp[0], stamp[0]))

    edits = iter(versions)

    def saved_while_rendering(p: patch.Patch) -> tuple[NDArray[np.float32], Report]:
        result = render(p)
        if p.body.tune_hz == last:  # the last edit is saved again, changed, while it renders
            save(gain_text.replace("tune_hz   = 55.0", "tune_hz   = 61.0"))
        return result

    def sleep(_seconds: float) -> None:  # each tick of the play loop saves the next version, then stops the session
        text = next(edits, None)
        if text is None:
            raise KeyboardInterrupt
        save(text)

    monkeypatch.setattr(audition, "render", saved_while_rendering)
    monkeypatch.setattr(time, "sleep", sleep)
    with pytest.raises(KeyboardInterrupt):
        audition.audition(patch.loads(gain_text), watch=Watcher(path, gain_text))
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("[") or "Hz," in line]
    assert lines == [
        "tuning   55.000 Hz, A1 +0.0 cents",
        lines[1],  # "[1] rendered in … ms; plays at the next hit, in 0 ms"
        "tuning   49.000 Hz, G1 +0.0 cents",
        "[2] invalid patch, still playing the last valid one:",
        "[3] over full scale by 12.28 dB; still playing the last valid one",
        "[4] now at 96000 Hz; restart audition to hear it. Still playing the last valid one",
        "[5] dropped: a newer edit is rendering",
        lines[7],
        "tuning   61.000 Hz, B1 -20.7 cents",
    ]
    assert lines[1].startswith("[1] rendered in")
    assert lines[7].startswith("[6] rendered in")


@pytest.mark.usefixtures("device")
def test_audition_refuses_a_rate_the_device_cannot_play(tight: patch.Patch) -> None:
    """Refuse to audition at a rate the device cannot open, rather than resample."""
    with pytest.raises(AuditionError, match="cannot play 96000 Hz; audition never resamples"):
        audition.audition(patch.edit(tight, "output.sample_rate", 96000))


@pytest.mark.usefixtures("device", "one_tick")
def test_windows_falls_back_to_exclusive_wasapi_for_a_rate_the_mixer_does_not_run_at(
    tight: patch.Patch,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Open WASAPI's default device in shared mode, then in exclusive mode once the shared mixer refuses the rate."""
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(KeyboardInterrupt):
        audition.audition(patch.edit(tight, "output.sample_rate", 96000))
    shared, exclusive = FakeOutputStream.opened  # shared mode refused the rate
    assert shared["device"] == WASAPI_DEVICE
    assert "extra_settings" not in shared
    assert exclusive["device"] == WASAPI_DEVICE
    assert exclusive["extra_settings"] == ("wasapi", True)
    assert capsys.readouterr().out.splitlines()[-1] == "device   Test Device, in exclusive mode"


@pytest.mark.usefixtures("device", "one_tick")
def test_macos_asks_core_audio_to_switch_the_device_rate_or_fail(
    tight: patch.Patch,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ask Core Audio to switch the device to the patch's rate, or to fail rather than convert."""
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(KeyboardInterrupt):
        audition.audition(patch.edit(tight, "output.sample_rate", 96000))
    flags = FakeOutputStream.opened[-1]["extra_settings"][1]
    assert flags == {"change_device_parameters": True, "fail_if_conversion_required": True}


def test_audition_without_sounddevice_says_what_it_needs(tight: patch.Patch, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail to import sounddevice, and say that audition needs it and PortAudio."""
    monkeypatch.setitem(cast("dict[str, object]", sys.modules), "sounddevice", None)  # import fails
    with pytest.raises(AuditionError, match="audition needs the sounddevice package and PortAudio"):
        audition.audition(tight)


def test_a_render_out_of_memory_keeps_the_last_buffer_playing(tight_text: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run out of memory rendering an edit, and keep the last valid buffer playing."""

    def out_of_memory(_p: patch.Patch) -> NoReturn:
        raise MemoryError

    monkeypatch.setattr(audition, "render", out_of_memory)
    hits = Hits(np.zeros(1, np.float32), RATE)
    assert audition.rerender(tight_text, hits) == (
        "render failed: out of memory; still playing the last valid one",
        None,
    )


def test_load_bed_refuses_what_is_not_a_wav(tmp_path: Path) -> None:
    """Refuse a bed that is not a WAV file, naming it."""
    (tmp_path / "bed.wav").write_bytes(b"not a wave file")
    with pytest.raises(AuditionError, match=r"bed\.wav: "):
        load_bed(tmp_path / "bed.wav", RATE, 128.0)


@pytest.mark.usefixtures("one_tick")
def test_windows_without_wasapi_uses_the_default_device(
    tight: patch.Patch,
    device: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Open the default device, with no settings, when Windows has no WASAPI host."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(device, "query_hostapis", lambda: [{"name": "MME", "default_output_device": 0}])
    with pytest.raises(KeyboardInterrupt):
        audition.audition(tight)
    assert FakeOutputStream.opened == [{"samplerate": RATE}]


@pytest.mark.usefixtures("device", "one_tick")
@pytest.mark.parametrize(("level", "warned"), [(0.9, True), (0.1, False)])
def test_a_bed_that_would_clip_under_the_hits_gets_a_warning(
    tmp_path: Path,
    gain: patch.Patch,
    capsys: pytest.CaptureFixture[str],
    *,
    level: float,
    warned: bool,
) -> None:
    """Loop a loud bed and a quiet one under the hits, warning only when they would clip together."""
    bed = write_bed(tmp_path / "bed.wav", np.full((round(4 * RATE * 60 / 128.0), 2), level, np.float32))
    with pytest.raises(KeyboardInterrupt):
        audition.audition(gain, bpm=128.0, bed=bed)
    out = capsys.readouterr().out
    assert ("warning: the hits over the bed peak at +3.0 dBFS, which the device will clip" in out) == warned
    played = FakeOutputStream.played
    assert played.shape == (256, 2)
    assert played[0, 0] == np.float32(level)  # the bed, from sample 0


@pytest.mark.usefixtures("device")
def test_dropouts_are_reported_from_the_play_loop(
    tight: patch.Patch,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report a dropout from the play loop, since the audio thread may not print."""
    monkeypatch.setattr(FakeOutputStream, "status", types.SimpleNamespace(output_underflow=True))
    ticks = iter([None])

    def sleep(_seconds: float) -> None:
        if next(ticks, KeyboardInterrupt) is KeyboardInterrupt:
            raise KeyboardInterrupt

    monkeypatch.setattr(time, "sleep", sleep)
    with pytest.raises(KeyboardInterrupt):
        audition.audition(tight)
    assert "warning: playback dropped out 1 time(s) so far; the device ran out of samples" in capsys.readouterr().out


def test_a_bed_needs_a_tempo(tmp_path: Path, tight: patch.Patch) -> None:
    """Refuse a bed without a tempo, both in the mix and in an audition."""
    with pytest.raises(ValueError, match="a bed needs a tempo"):
        Hits(np.zeros(1, np.float32), RATE, None, Bed(np.zeros((10, 1), np.float32), 4))
    with pytest.raises(AuditionError, match="a bed needs a tempo"):
        audition.audition(tight, bed=str(tmp_path / "bed.wav"))
