# Copyright (C) 2026 Vittorio Candolo
# SPDX-License-Identifier: GPL-3.0-or-later

"""Record samples of every preset, the references that tests/test_references.py holds every platform to.

    uv run python tools/references.py            # compare with the stored references
    uv run python tools/references.py --force    # store new ones, after a deliberate change to the sound

Each preset renders at both base rates, and keeps 512 of its float32 samples: the first 128, where the attack
lives, and 384 spread evenly over the rest. Across platforms, rounding may move them by far less than the
tolerance, a millionth of full scale; a change to the sound moves them by more.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ballast_synth import patch, render
from ballast_synth.cli import PRESET_DIR, PRESETS

if TYPE_CHECKING:
    from numpy.typing import NDArray

STORE = Path(__file__).resolve().parent.parent / "tests" / "references.json"
RATES = (44100, 48000)  # one rate of each family
HEAD, SPREAD = 128, 384  # the samples kept from the start, and spread over the rest
TOLERANCE = 1e-6  # how far a sample may move across platforms, of full scale


def picks(size: int) -> NDArray[np.intp]:
    """Return the indices of the samples kept from a render of `size` samples."""
    return np.concatenate((np.arange(HEAD), np.linspace(HEAD, size - 1, SPREAD).round().astype(np.intp)))


def references() -> dict[str, NDArray[np.float32]]:
    """Render every preset at both base rates, and keep its samples, by "preset@rate"."""
    out = {}
    for name in PRESETS:
        p = patch.loads(PRESET_DIR.joinpath(f"{name}.toml").read_text(encoding="utf-8"))
        for rate in RATES:
            audio, _ = render.render(patch.edit(p, "output.sample_rate", rate))
            out[f"{name}@{rate}"] = audio[picks(audio.size)]
    return out


def stored() -> dict[str, NDArray[np.float32]]:
    """Read the stored references."""
    raw = json.loads(STORE.read_text(encoding="utf-8"))
    return {key: np.frombuffer(base64.b64decode(text), dtype="<f4") for key, text in raw.items()}


def main() -> None:
    """Compare the references with the stored ones, or store them with --force."""
    now = references()
    if "--force" in sys.argv[1:]:
        text = {key: base64.b64encode(samples.astype("<f4").tobytes()).decode("ascii") for key, samples in now.items()}
        STORE.write_text(json.dumps(text, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        sys.stdout.write(f"stored {len(now)} references in {STORE.name}\n")
        return
    then = stored()
    worst = (
        max(float(np.max(np.abs(now[key].astype(float) - then[key]))) for key in now)
        if now.keys() == then.keys()
        else np.inf
    )
    sys.stdout.write(f"{len(now)} references, worst difference {worst:.3g} of full scale (tolerance {TOLERANCE:g})\n")
    if worst > TOLERANCE:
        sys.exit(1)


if __name__ == "__main__":
    main()
