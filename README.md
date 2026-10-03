# kick

A one-voice kick drum synthesizer, rendered offline to a mono float WAV. It
implements the *Kick Synth design and implementation plan, revision 2*: a
tonal body and a synthesized attack are summed into one saturator and
band-limited once by a frozen minimum-phase decimator.

    x(t) = b(t) + a(t),    y(t) = S_α(x(t)),    out[m] = p · g_out · Σ_k h[k] · y((mK − k) / F)

Everything before the decimator is a pure function of time, so a render
depends only on the patch, the renderer version and the frozen filter.

## Install

Python 3.11 or later.

```
pip install -e .              # NumPy and SciPy
pip install -e '.[audition]'  # adds sounddevice (needs PortAudio) for `kick audition`
pip install -e '.[test]'      # pytest and soundfile, for the test suite
```

## Use

```
kick init tight --tune F1 > kick.toml      # a complete starting patch (writes tune_hz = 43.654)
kick audition kick.toml --watch --bpm 128  # plays the exact render buffer; re-renders on every save
kick render kick.toml --out kick.wav       # mono float32 WAV with root note and embedded patch
kick verify kick.toml                      # reference comparison for this one patch
kick extract kick.wav > kick.toml          # the exact patch text back from any exported kick
kick stretch kick.toml body.amp 0.6        # rescale a curve's knot times after its peak
kick tune kick.toml F#1                    # the patch retuned to a note or a frequency
```

For hardware samplers, `kick render kick.toml --out kick.wav --format pcm24`
(or `pcm16`) writes integer PCM with triangular dither. The dither is seeded
from the patch, so the file is as reproducible as a float one, and exact
zeros stay zero. The peak is kept 2 LSB below full scale so the dither never
clips: peak mode lowers its target by that much, and a gain-mode render above
it is refused.

To compare in context, as the benchmark does:

```
kick audition kick.toml --watch --bpm 128 --with bass.wav --against favourite.wav [--blind]
```

`--with` loops a bed under the hits, such as the track's bass line. It must
be a whole number of bars at `--bpm`, and it restarts on the tempo grid, so
it never drifts. `--against` alternates the render with a sample, hit by
hit. The sample is scaled to the render's loudness; the render is never
touched. With `--blind` each pair plays in random order, and audition prints
the order when you stop it. Neither file is ever resampled.

The four references are `round`, `tight`, `long` and `driven`. They are
voiced: approved by ear and in the sample-replacement benchmark (see
[Status](#status)).

`kick render` refuses to write a gain-mode render that exceeds full scale and
says by how much to lower `output.db`; `--allow-over` writes it anyway, as a
float file with overs. Nothing is ever clipped. `kick audition` keeps playing
the last valid buffer when an edit is invalid or over full scale, swaps in a
new render at the next hit, and never resamples: it opens the device at the
patch's sample rate or reports that it cannot.

## The patch

A patch is a complete TOML description of one kick. Every field is required
and the renderer has no defaults, so a future default can never change an old
patch. The parser rejects anything outside the valid domain and never clamps
or repairs.

```toml
schema = 1
seed = 1

[body]
tune_hz   = 50.0
level_db  = 0.0
h2        = 0.15            # second harmonic, 0 … 0.5
phase_deg = 0.0
phase_ref = "onset"         # "onset" | "tail"
pitch = [                   # ratio to tune_hz; starts at 0 ms, ends at 1
  { ms = 0.0,  ratio = 8.0 },
  { ms = 8.0,  ratio = 2.0 },
  { ms = 65.0, ratio = 1.0 },
]
amp = [                     # shape: 0 at both ends, highest point exactly 1
  { ms = 0.0,   gain = 0.0 },
  { ms = 0.2,   gain = 1.0 },
  { ms = 45.0,  gain = 0.75 },
  { ms = 160.0, gain = 0.2 },
  { ms = 350.0, gain = 0.0 },
]

[attack]
level_db      = -18.0
tone_hz       = 2000.0
noise_mix     = 0.6         # 0 = tone … 1 = noise
noise_band_hz = [700.0, 9000.0]
amp = [
  { ms = 0.0,  gain = 0.0 },
  { ms = 0.05, gain = 1.0 },
  { ms = 2.0,  gain = 0.25 },
  { ms = 8.0,  gain = 0.0 },
]

[drive]
db = 0.0                    # small-signal gain; 0 = bypass

[output]
sample_rate = 48000         # 44100, 48000, 88200 or 96000
mode        = "gain"        # "gain" | "peak"
db          = -6.0          # gain in dB, or peak target in dBFS
polarity    = 1
```

In brief:

- **Curves.** Curves are monotone cubic Hermite (PCHIP) through their knots.
  Amplitude curves are shapes that start and end at 0 and peak at exactly 1.
  Levels are separate gains in dB.
- **Body.** The body is `sin φ + h2 sin 2φ`. The pitch curve is a ratio to
  `tune_hz`, integrated exactly into phase.
- **Phase modes.** `phase_ref = "onset"` fixes the phase at the trigger.
  `"tail"` makes the body coincide exactly with the free-running sine at
  `tune_hz` from the last pitch knot on, whatever the sweep does before it.
- **Attack noise.** The noise is a seeded continuous-time Gaussian process
  (SHAKE-256 → Rayleigh amplitudes and uniform phases on 1 Hz components). It
  is identical at every sample rate, and editing the band reweights the same
  noise.
- **Drive.** `drive.db` is the saturator's small-signal gain; 0 dB is exact
  bypass.

The valid domain:

| Quantity | Valid range |
|---|---|
| `tune_hz` | 20 – 1000 |
| body frequency at every pitch knot | 10 Hz – 20 kHz |
| `h2` | 0 – 0.5 |
| body and attack `level_db` | −inf (silence), or −60 – +24 |
| `drive.db` | 0 – 48 |
| `tone_hz` | 20 Hz – 20 kHz |
| `noise_mix` | 0 – 1 |
| `noise_band_hz` | 20 ≤ low < high ≤ 20 000 |
| knots per curve | pitch 2 – 32, amplitude 3 – 32 |
| knot spacing | ≥ 10 µs (compared on the decimal values as written) |
| body curves / attack curve | end by 8 s / by 1 s |
| `phase_deg` | 0 ≤ φ < 360 |
| output | 44.1, 48, 88.2 or 96 kHz; gain −60 – +24 dB; peak target −60 – 0 dBFS |
| `seed` | 0 – 2⁶³ − 1 |

## What a render reports

```
tuning   43.654 Hz, F1 +0.0 cents
length   240.0 ms source, 241.9 ms exported (11611 samples at 48000 Hz)
peak     -5.81 dBFS sample, -5.81 dBTP true (4× estimate, information only)
loudness -19.1 LUFS, the largest 400 ms momentary loudness (BS.1770)
domain   inside the validated domain: predicted aliasing -126 dB, margin 25.8 dB
         mostly kinks, 72% at attack.amp[1] (0.08 ms); drive headroom +40.1 dB
render   146 ms
```

- **Loudness** is ITU-R BS.1770 momentary loudness: K-weighted, over 400 ms,
  the largest over every window position. The weighting discounts the lowest
  frequencies, so for a kick it mostly follows the click and the harmonics.
  Matching the level of the tail's fundamental remains a separate check.
- **Domain** is the renderer's own prediction of its aliasing error, relative
  to the output peak. It is inside the validated domain when the prediction
  is at most −100 dB, and the margin is how far it is from that limit.
  Outside the domain the render is still defined and finite, but
  unvalidated: `kick verify` measures the actual error against a converged
  reference render of that patch.
- **The line under it** says what the prediction comes from and what to
  change. It is either the saturation or the curves' kinks, and for kinks,
  the knot that contributes most. The drive headroom is how much more drive
  the patch takes before its status changes. Outside, it is negative, and
  the line says what to lower `drive.db` to. The margin is not the headroom:
  the driven reference has 17 dB of margin, but its status does not change
  until over 25 dB more drive.

## Domains

The plan makes three separate promises:

- **Valid domain.** Everything the parser accepts (the table above). Renders
  there are defined and finite, with bounded time and memory.
- **Validated domain.** Where the numerical criteria are demonstrated. Each
  render states whether it is inside and by what margin, and `kick verify`
  checks any other patch individually. The rule and its evidence are in
  VALIDATION.md.
- **Voiced region.** The four approved references, where musical usefulness
  is promised. Between them they span drive 2–21 dB, `h2` 0.05–0.25 and
  sweep starts of 3.5–7× the tuning.

## How it renders

- **Rates.** The internal rate is 16× the base rate: 705.6 kHz for the
  44.1 kHz family, 768 kHz for 48 kHz. The 88.2 and 96 kHz exports reuse
  their family's rate and filter with factor 8, so a 96 kHz file equals the
  48 kHz file with the in-between samples filled in, bit for bit.
- **Decimator.** One frozen minimum-phase FIR per family. It passes 0–20 kHz
  within 0.001 dB, attenuates at least 110 dB from base/2 upward, and its
  taps sum to exactly 1. It ships as data with a SHA-256 checksum, so the
  sound cannot drift with SciPy versions.
- **Evaluation.** The internal signal is computed in blocks of 2¹⁶ samples
  aligned to the trigger, each source only over its own support. A longer
  render never changes earlier samples.
- **Output.** Output gain or peak normalization, then polarity, then one
  conversion to float32, or one quantization to PCM with seeded TPDF dither.
- **File.** Our own RIFF writer adds a `smpl` chunk, so samplers map the root
  key from `tune_hz`, and a private `kick` chunk with the patch text,
  renderer version, decimator checksum and dependency versions (and, for
  PCM, the dither's key and seed).
- **Report.** Drive leaves the saturator's input unchanged, so one render
  records what the validated-domain rule needs at every drive. It keeps the
  knee slew as a function of α and the jumps at every knot, which are both
  exact. It also keeps the exact output at the loudest positions, a lower
  bound on the output peak. From these it finds where the status changes,
  including narrow outside islands. The estimate can only err on the safe
  side (VALIDATION.md).

## Development

```
python -m pytest                        # the full suite (about a minute)
python tools/design_decimator.py --check
python tools/grid.py run --k 16         # the adversarial grid (minutes on 4 cores)
python tools/grid.py analyze docs/grid/k16.csv docs/grid/k8.csv docs/grid/k32.csv
python tools/headroom.py run --dense    # the report's drive headroom against full renders (minutes)
python tools/headroom.py analyze docs/grid/headroom.csv
python tools/bench.py                   # render time and memory on this machine
python tools/golden.py                  # regenerate golden renders (a deliberate sound change only)
```

[docs/VALIDATION.md](docs/VALIDATION.md) records every numerical criterion of
the plan with its measured value. It also covers the adversarial grid, the
choice of K, the derivation of the validated-domain rule, and performance.
[docs/BENCHMARK.md](docs/BENCHMARK.md) is the protocol for the
sample-replacement benchmark.

| Module | Responsibility |
|---|---|
| `kick/patch.py` | schema, parsing, validation, canonical TOML, tuning helpers |
| `kick/curve.py` | Hermite curves, PCHIP slopes, exact integrals, knot jumps |
| `kick/noise.py` | seeded continuous-time noise |
| `kick/voice.py` | body, attack and drive as pure functions of time; slew, knee slew at every drive, kink bound |
| `kick/render.py` | rates, frozen decimators, output stage, validated-domain rule, cause and drive headroom, report, PCM |
| `kick/loudness.py` | BS.1770 K-weighting at every rate, momentary loudness |
| `kick/reference.py` | the reference renderer and comparison behind `kick verify` |
| `kick/wav.py` | RIFF writing and reading |
| `kick/audition.py` | hit scheduling, beds, A/B and blind order against a sample, watching, playback |
| `kick/cli.py` | commands |
| `kick/data/` | frozen decimators, checksums, the four references |

## Status

| Milestone | State |
|---|---|
| 1. Complete path | Done. Every §8.1 criterion passes (see VALIDATION.md), and renders import into a DAW. |
| 2. Measure and freeze | Done. K = 16, validated-domain rule and decimators frozen from a 737-fixture grid at 8×, 16× and 32×. Golden renders started. |
| 3. Audition | Done. Tested here with a simulated device (exact buffer, stale renders dropped, swaps at hit boundaries) and played through a real one. |
| 4. Voicing | Done. The four references are approved by ear, and their renders are the golden files. No controls were removed. |
| 5. Benchmark | Done. The sample-replacement benchmark passed, so neither pre-registered addition is added (BENCHMARK.md). |

After the benchmark, five additions carried over from the snare/clap plan:

- drive headroom and the cause of the predicted aliasing, in the report;
- loudness, in the report;
- `kick tune`;
- PCM export;
- audition over a bed and against a sample.

None of them changes a render. The golden files, and every domain number of
the references and grid fixtures, are bit-identical to before.
