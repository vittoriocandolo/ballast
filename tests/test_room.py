# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""The room: a seeded tail of unit energy that darkens as it fades, added after its delay."""

import math

import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from ballast_synth import family, patch, render, room, voice

_, INTERNAL, FACTOR = family.rates(48000)
ROOM = patch.Room(-10.0, 600.0, 4000.0, 8.0)
DECAY_ERROR = 0.1  # how close each third octave's decay comes to the designed one
T30 = (-5.0, -35.0)  # the slope a decay time is read from, as ISO 3382 reads T30, then doubled to 60 dB


def test_the_tail_has_unit_energy_and_is_the_same_every_time() -> None:
    """Build the tail twice and for another seed: unit energy, the same samples, and other ones."""
    h = room.impulse(1, ROOM.decay_ms, ROOM.damping_hz, 48000)
    assert math.fsum(h * h) == pytest.approx(1.0, abs=1e-12)
    room.impulse.cache_clear()
    assert np.array_equal(room.impulse(1, ROOM.decay_ms, ROOM.damping_hz, 48000), h)
    assert not np.array_equal(room.impulse(2, ROOM.decay_ms, ROOM.damping_hz, 48000)[:1000], h[:1000])


@pytest.mark.parametrize("hz", [250.0, 1000.0, 4000.0])
def test_each_frequency_falls_60_db_in_its_designed_time(hz: float) -> None:
    """Measure the tail's decay in a third octave by Schroeder's backward integral, against T(f)."""
    h = room.impulse(1, ROOM.decay_ms, ROOM.damping_hz, 48000)
    band = sosfilt(butter(4, [hz * 2 ** -(1 / 6), hz * 2 ** (1 / 6)], btype="band", fs=INTERNAL, output="sos"), h)
    energy = np.cumsum((band * band)[::-1])[::-1]
    level = 10 * np.log10(energy / energy[0])
    start, end = (int(np.argmax(level < db)) for db in T30)
    t60 = 2 * (end - start) / INTERNAL * 1000
    assert t60 == pytest.approx(ROOM.decay_ms / (1 + hz / ROOM.damping_hz), rel=DECAY_ERROR)


def test_the_room_rings_after_its_delay_and_lengthens_the_kick(tight: patch.Patch) -> None:
    """Add a room: the opening before its delay is untouched, and the kick runs on for the whole tail."""
    plain = render.synthesize(tight)
    roomed = patch.edit(tight, "room", ROOM)
    assert roomed.length_ms == tight.length_ms + ROOM.predelay_ms + 1.5 * ROOM.decay_ms
    out = render.synthesize(roomed)
    delay = round(ROOM.predelay_ms * 48000 / 1000)
    assert np.array_equal(out[:delay], plain[:delay])
    assert out.size > plain.size
    silent = patch.edit(roomed, "room.level_db", -math.inf)
    reach = (voice.sample_count(tight.length_ms / 1000, INTERNAL) - 1) // FACTOR + 1  # outputs from the kick alone
    assert np.array_equal(render.synthesize(silent)[:reach], plain[:reach])
