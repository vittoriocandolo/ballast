"""Audition: the mix of hits and bed, bed loading, the file watcher, and the watch loop."""

import os
import sys
import types

import numpy as np
import pytest
from scipy.io import wavfile

from ballast import audition, patch
from ballast.audition import AuditionError, Bed, Hits, Watcher, beat, load_bed

RATE = 48000


def onsets(signal: np.ndarray) -> list[int]:
    return np.flatnonzero(np.diff(np.concatenate(([0.0], signal))) > 0).tolist()


def test_free_running_hits_repeat_after_each_kick_and_half_a_second():
    hits = Hits(np.ones(1000, np.float32), RATE)
    assert onsets(hits.fill(3 * 25_000)[:, 0]) == [0, 25_000, 50_000]


def test_hits_on_the_beat_are_unchoked_so_their_tails_sum():
    hits = Hits(np.ones(30_000, np.float32), RATE, bpm=128.0)  # longer than a beat of 22 500 samples
    out = hits.fill(50_000)[:, 0]
    assert out[0] == 1 and out[22_499] == 1 and out[22_500] == 2 and out[30_000] == 1 and out[45_000] == 2


def test_a_new_buffer_plays_from_the_next_hit():
    hits = Hits(np.ones(100, np.float32), RATE, bpm=120.0)  # a beat is 24 000 samples
    first = hits.fill(1000)
    hits.buffer = np.full(100, 0.5, np.float32)
    assert hits.wait() == pytest.approx(23_000 / RATE)
    rest = hits.fill(30_000)[:, 0]
    assert first[0, 0] == 1 and rest[23_000] == 0.5


def test_block_sizes_never_change_the_mix():
    bed = Bed(np.random.default_rng(0).standard_normal((90_495, 2)).astype(np.float32), 4)
    kick = np.random.default_rng(1).standard_normal(30_000).astype(np.float32)
    sizes = [1, 511, 4096, 30_000] * 10
    hits = Hits(kick, RATE, 127.3, bed)
    parts = np.concatenate([hits.fill(size) for size in sizes])
    assert np.array_equal(parts, Hits(kick, RATE, 127.3, bed).fill(sum(sizes)))


def test_the_bed_restarts_on_the_same_beats_as_the_hits():
    bpm, beats = 127.3, 4
    ramp = np.arange(1, 100_001, dtype=np.float32)[:, None]  # longer than a loop, and each position recognizable
    hits = Hits(np.zeros(1, np.float32), RATE, bpm, Bed(ramp, beats))
    out = hits.fill(300_000)[:, 0]
    for k in range(3):
        start = hits.loop_start(k)
        assert start == beat(k * beats, RATE, bpm)
        assert out[start] == 1.0  # the loop starts over…
    assert out[hits.loop_start(1) - 1] == hits.loop_start(1)  # …cut off where the next one starts


def write_bed(path, frames, rate=RATE):
    wavfile.write(path, rate, frames)
    return path


def test_load_bed_reads_whole_bars_in_any_format(tmp_path, capsys):
    n = round(2 * 4 * RATE * 60 / 128.0)
    for name, data in {
        "u8": np.full(n, 192, np.uint8),
        "i16": np.full((n, 2), 16384, np.int16),
        "f32": np.full(n, 0.5, np.float32),
    }.items():
        bed = load_bed(write_bed(tmp_path / f"{name}.wav", data), RATE, 128.0)
        assert bed.beats == 8 and bed.frames.dtype == np.float32
        assert np.all(bed.frames == 0.5)
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
def test_load_bed_refuses_what_would_drift_or_resample(tmp_path, frames, rate, error):
    with pytest.raises(AuditionError, match=error):
        load_bed(write_bed(tmp_path / "bed.wav", frames, rate), RATE, 128.0)


def test_the_watcher_reports_new_text_only(tmp_path):
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


class FakeOutputStream:
    """Opens at 48 kHz, or at any rate with extra settings, and pulls one block when it starts."""

    opened: list[dict] = []

    def __init__(self, samplerate, channels, dtype, latency, callback, **settings):
        FakeOutputStream.opened.append({"samplerate": samplerate, **settings})
        if samplerate != 48000 and "extra_settings" not in settings:
            raise PortAudioError("Invalid sample rate")
        self.device, self.channels, self.callback = 0, channels, callback

    def __enter__(self):
        self.block = np.zeros((256, self.channels), np.float32)
        self.callback(self.block, 256, None, None)
        FakeOutputStream.played = self.block
        return self

    def __exit__(self, *exc):
        return False


class PortAudioError(Exception):
    pass


@pytest.fixture
def device(monkeypatch):
    """A sound device on Linux, through a stand-in for sounddevice."""
    sd = types.SimpleNamespace(
        PortAudioError=PortAudioError,
        query_hostapis=lambda: [{"name": "Windows WASAPI", "default_output_device": 3}],
        query_devices=lambda device=None, kind=None: {"name": "Test Device"},
        OutputStream=FakeOutputStream,
        WasapiSettings=lambda exclusive: ("wasapi", exclusive),
        CoreAudioSettings=lambda **flags: ("coreaudio", flags),
    )
    FakeOutputStream.opened.clear()
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    monkeypatch.setattr(sys, "platform", "linux")
    return sd


def test_the_watch_loop_rerenders_every_edit(tmp_path, tight_text, device, monkeypatch, capsys):
    path = tmp_path / "kick.toml"
    path.write_text(tight_text, encoding="utf-8")
    versions = [
        tight_text.replace("tune_hz   = 55.0", "tune_hz   = 49.0"),
        tight_text.replace("h2        = 0.15", 'h2        = "x"'),
        tight_text.replace("db          = -6.0", "db          = 12.0"),
        tight_text.replace("sample_rate = 48000", "sample_rate = 96000"),
        tight_text.replace("tune_hz   = 55.0", "tune_hz   = 60.0"),
    ]
    stamp = [10**18]

    def save(text):  # a new modification time each save, whatever the file system's resolution
        path.write_text(text, encoding="utf-8")
        stamp[0] += 10**9
        os.utime(path, ns=(stamp[0], stamp[0]))

    edits = iter(versions)
    render = audition.render

    def saved_while_rendering(p, *args):  # the last edit is saved again, changed, while it renders
        result = render(p, *args)
        if p.body.tune_hz == 60.0:
            save(tight_text.replace("tune_hz   = 55.0", "tune_hz   = 61.0"))
        return result

    def sleep(_):  # each tick of the play loop saves the next version, then stops the session
        text = next(edits, None)
        if text is None:
            raise KeyboardInterrupt
        save(text)

    monkeypatch.setattr(audition, "render", saved_while_rendering)
    monkeypatch.setattr(audition.time, "sleep", sleep)
    with pytest.raises(KeyboardInterrupt):
        audition.audition(path, tight_text, patch.loads(tight_text), watch=True)
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("[") or "Hz," in line]
    assert lines == [
        "tuning   55.000 Hz, A1 +0.0 cents",
        lines[1],  # "[1] rendered in … ms; plays at the next hit, in 0 ms"
        "tuning   49.000 Hz, G1 +0.0 cents",
        "[2] invalid patch, still playing the last valid one:",
        "[3] over full scale by 12.32 dB; still playing the last valid one",
        "[4] now at 96000 Hz; restart audition to hear it. Still playing the last valid one",
        "[5] dropped: a newer edit is rendering",
        lines[7],
        "tuning   61.000 Hz, B1 -20.7 cents",
    ]
    assert lines[1].startswith("[1] rendered in") and lines[7].startswith("[6] rendered in")


def test_audition_refuses_a_rate_the_device_cannot_play(tight, device):
    with pytest.raises(AuditionError, match="cannot play 96000 Hz; audition never resamples"):
        audition.audition("kick.toml", "", patch.edit(tight, "output.sample_rate", 96000))


def test_windows_falls_back_to_exclusive_wasapi_for_a_rate_the_mixer_does_not_run_at(device, monkeypatch, capsys):
    monkeypatch.setattr(sys, "platform", "win32")
    audition._stream(96000, 1, lambda frames: None)
    shared, exclusive = FakeOutputStream.opened  # shared mode refused the rate
    assert shared["device"] == 3 and "extra_settings" not in shared
    assert exclusive["device"] == 3 and exclusive["extra_settings"] == ("wasapi", True)
    assert capsys.readouterr().out == "device   Test Device, in exclusive mode\n"


def test_macos_asks_core_audio_to_switch_the_device_rate_or_fail(device, monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    audition._stream(96000, 1, lambda frames: None)
    flags = FakeOutputStream.opened[-1]["extra_settings"][1]
    assert flags == {"change_device_parameters": True, "fail_if_conversion_required": True}


def test_audition_without_sounddevice_says_what_it_needs(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)  # import fails
    with pytest.raises(AuditionError, match="audition needs the sounddevice package and PortAudio"):
        audition._stream(48000, 1, lambda frames: None)


def test_a_failed_render_keeps_the_last_buffer_playing(tight_text, monkeypatch):
    def broken(p, *args):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(audition, "render", broken)
    hits = Hits(np.zeros(1, np.float32), RATE)
    assert audition._rerender(tight_text, RATE, hits) == (
        "render failed, still playing the last valid one: out of memory",
        None,
    )


def test_load_bed_refuses_what_is_not_a_wav(tmp_path):
    (tmp_path / "bed.wav").write_bytes(b"not a wave file")
    with pytest.raises(AuditionError, match="bed.wav: "):
        load_bed(tmp_path / "bed.wav", RATE, 128.0)


def test_windows_without_wasapi_uses_the_default_device(device, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    device.query_hostapis = lambda: [{"name": "MME", "default_output_device": 0}]
    audition._stream(48000, 1, lambda frames: None)
    assert FakeOutputStream.opened == [{"samplerate": 48000}]


@pytest.mark.parametrize(("level", "warned"), [(0.9, True), (0.1, False)])
def test_a_bed_that_would_clip_under_the_hits_gets_a_warning(
    tmp_path, tight, device, monkeypatch, capsys, level, warned
):
    bed = write_bed(tmp_path / "bed.wav", np.full((round(4 * RATE * 60 / 128.0), 2), level, np.float32))

    def stop(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(audition.time, "sleep", stop)
    with pytest.raises(KeyboardInterrupt):
        audition.audition("kick.toml", "", tight, bpm=128.0, bed=bed)
    out = capsys.readouterr().out
    assert ("warning: the hits over the bed peak at +3.0 dBFS, which the device will clip" in out) == warned
    played = FakeOutputStream.played
    assert played.shape == (256, 2) and played[0, 0] == np.float32(level)  # the bed, from sample 0
