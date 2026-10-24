# Patch reference

A patch is a complete TOML description of one kick. Every field is required
and the renderer has no defaults, so a future default can never change an old
patch. The sections a patch may leave out are its layers, `[sub]` and
`[transient]`: without one, the patch has none. The parser rejects anything outside the valid domain, reports every
problem with its field path, and never clamps or repairs.

![The tight preset's pitch, body and attack curves through their knots, and the first 60 ms of its output](figures/anatomy-tight.svg)

This is the `tight` preset, drawn above; the [preset gallery](presets.md) draws
every preset the same way:

```toml
schema = 3
seed = 1

[body]
tune_hz   = 55.0
level_db  = 0.0
h2        = 0.15            # second harmonic, 0 … 0.5
bright_hz = 0.0             # harmonics fade above; 0 = sine
phase_deg = 0.0
phase_ref = "onset"         # "onset" | "tail"
pitch = [                   # ratio to tune_hz; starts at 0 ms, ends at 1
  { ms = 0.0,  ratio = 7.0 },
  { ms = 6.0,  ratio = 2.2 },
  { ms = 40.0, ratio = 1.0 },
]
amp = [                     # shape: 0 at both ends, highest point exactly 1
  { ms = 0.0,   gain = 0.0 },
  { ms = 0.25,  gain = 1.0 },
  { ms = 30.0,  gain = 0.8 },
  { ms = 100.0, gain = 0.35 },
  { ms = 180.0, gain = 0.08 },
  { ms = 240.0, gain = 0.0 },
]

[attack]
level_db      = -16.0
tone_hz       = 2500.0
noise_mix     = 0.55        # 0 = tone … 1 = noise
noise_band_hz = [1200.0, 8000.0]
pitch = [                   # ratio to tone_hz; starts at 0 ms, ends at 1
  { ms = 0.0, ratio = 1.0 },
]
amp = [
  { ms = 0.0,  gain = 0.0 },
  { ms = 0.08, gain = 1.0 },
  { ms = 1.5,  gain = 0.35 },
  { ms = 6.0,  gain = 0.0 },
]

[drive]
db        = 6.0             # small-signal gain; 0 = bypass
asymmetry = 0.0             # even harmonics, 0 … 1
mix       = 1.0             # 0 = dry … 1 = driven

[clip]
db      = 0.0               # off the peaks; 0 = off
knee_db = 0.0               # 0 = hard … 6 = soft

[output]
sample_rate = 48000
low_cut_hz  = 0.0           # high-pass, 12 dB/oct; 0 = off
mode        = "gain"        # "gain" | "peak" | "true-peak"
db          = -6.0          # gain in dB, or peak target
polarity    = 1
```

Patch files are UTF-8, with or without a byte-order mark, or UTF-16 with one,
as Windows PowerShell writes.

## Fields

### Top level

| Field | Type | Range | Meaning |
|---|---|---|---|
| `schema` | integer | 1, 2 or 3 | The patch format, checked first: another schema may share nothing else with this one. Schema 3 is current; see [Versions](#versions). |
| `seed` | integer | 0 – 2⁶³ − 1 | Draws the attack's noise and the PCM dither. The same seed gives the same noise at every sample rate. |

### `[body]`: the tonal body

| Field | Type | Range | Meaning |
|---|---|---|---|
| `tune_hz` | number | 20 – 1000 Hz | The note the body settles on. |
| `level_db` | number | −60 – +24 dB, or `-inf` | The body's level; `-inf` silences it. |
| `h2` | number | 0 – 0.5 | The second harmonic: the body is sin φ + h2 sin 2φ, scaled so its power does not change with `h2`. |
| `bright_hz` | number | 50 Hz – 20 kHz, or 0 | Turns the sine into a saw whose harmonics fade above this frequency, by e^(−f / bright_hz), the same in Hz at every pitch; the fundamental stays exactly where it was. 0 is a sine. |
| `phase_deg` | number | 0 ≤ φ < 360 | The starting phase, at the reference point. |
| `phase_ref` | `"onset"` or `"tail"` | | Where `phase_deg` applies: at the trigger, or on the free-running sine at `tune_hz` that the body meets exactly from the last pitch knot on. |
| `pitch` | curve | 2 – 32 knots of `{ ms, ratio }` | The frequency as a ratio to `tune_hz`. It starts at 0 ms and ends at ratio 1, and `tune_hz × ratio` stays within 10 Hz – 20 kHz at every knot. |
| `amp` | curve | 3 – 32 knots of `{ ms, gain }` | The body's amplitude shape, ending by 8 s. |

### `[attack]`: the click

| Field | Type | Range | Meaning |
|---|---|---|---|
| `level_db` | number | −60 – +24 dB, or `-inf` | The attack's level; `-inf` silences it. |
| `tone_hz` | number | 20 Hz – 20 kHz | The frequency the attack's tone settles on. |
| `pitch` | curve | 1 – 32 knots of `{ ms, ratio }` | The tone's frequency as a ratio to `tone_hz`, like the body's. It starts at 0 ms and ends at ratio 1, and `tone_hz × ratio` stays within 20 Hz – 20 kHz at every knot. One knot, `{ ms = 0.0, ratio = 1.0 }`, holds the tone at `tone_hz`; a curve that falls into 1 is a zap. |
| `noise_mix` | number | 0 – 1 | Tone (0) to noise (1), at constant power; both ends are exact. |
| `noise_band_hz` | `[low, high]` | 20 ≤ low < high ≤ 20 000 | The noise band: second-order skirts with their half-power points at the edges. |
| `amp` | curve | 3 – 32 knots of `{ ms, gain }` | The attack's amplitude shape, ending by 1 s. |

### `[drive]`: the saturator

| Field | Type | Range | Meaning |
|---|---|---|---|
| `db` | number | 0 – 48 dB | The small-signal gain of a tanh saturator that keeps full scale at full scale. 0 dB is an exact bypass, of asymmetry and mix too. |
| `asymmetry` | number | 0 – 1 | Even harmonics: warmth that rises as the body decays through the curve. Full scale and the small-signal gain stay exactly put; 1 is as far as the curve stays monotonic. Even harmonics bring DC, which `output.low_cut_hz` removes. |
| `mix` | number | 0 – 1 | The share of the driven signal, against the undriven one: 1 is all driven, 0 all dry. The two line up exactly, since the drive adds no delay. |

### `[sub]`: a clean sub

Optional: a patch without this section has no sub. The sub is a sine at
`tune_hz`, in phase with the free-running sine that the body's tail meets, so
it only adds to the tail. It joins after the drive, which never touches it,
and before the low cut and the clipper.

| Field | Type | Range | Meaning |
|---|---|---|---|
| `level_db` | number | −60 – +24 dB, or `-inf` | The sub's level; `-inf` silences it. |
| `amp` | curve | 3 – 32 knots of `{ ms, gain }` | The sub's own amplitude shape, ending by 8 s. |

### `[clip]`: the clipper

| Field | Type | Range | Meaning |
|---|---|---|---|
| `db` | number | 0 – 12 dB | How far the clipper brings the file's true peak down, exactly; what lies below its knee is untouched. 0 is off. With a peak target in `[output]`, these decibels come back as loudness. |
| `knee_db` | number | 0 – 6 dB | How far below the threshold the clipper starts to bend: 0 is a hard clip, 6 the softest. |

### `[transient]`: a sampled click

Optional: a patch without this section has no transient.
[`ballast transient`](usage.md#ballast-transient-patch-wav--o-file) writes it
from a WAV, and it comes last in the file, since its samples run long.

| Field | Type | Range | Meaning |
|---|---|---|---|
| `level_db` | number | −60 – +24 dB, or `-inf` | The transient's peak level; `-inf` silences it. |
| `start_ms` | number | 0 – 1000 ms | When its first sample plays, after the trigger. |
| `pitch` | number | 0.25 – 4 | How fast, and so how high, it plays: 2 is an octave up and half as long. |
| `polarity` | integer | 1 or −1 | −1 inverts it against the body and the attack. |
| `rate` | integer | 44100, 48000, 88200, 96000, 176400 or 192000 | The sample rate of its samples. A transient of either family plays in a patch of either. |
| `samples` | base64 | at least one sample | Little-endian float32, peaking at exactly 1, so that `level_db` alone sets the level. Played from `start_ms` at `pitch`, they must end by 1 s, like the attack. |

### `[output]`

| Field | Type | Range | Meaning |
|---|---|---|---|
| `sample_rate` | integer | 44100, 48000, 88200, 96000, 176400 or 192000 | The file's sample rate. |
| `low_cut_hz` | number | 10 – 1000 Hz, or 0 | The corner of a 12 dB/octave Butterworth high-pass after the drive; 0 turns it off. |
| `mode` | `"gain"`, `"peak"` or `"true-peak"` | | Apply `db` as a gain, or scale so the highest sample sits at `db` dBFS, or the true peak at `db` dBTP. |
| `db` | number | gain −60 – +24 dB; peak target −60 – 0 | The gain, or the peak target. In both peak modes the highest sample also stays within the format's ceiling. |
| `polarity` | integer | 1 or −1 | −1 inverts the output. |

## Curves

A curve is an array of knots, each a table of `ms` and a value: `ratio` for
a pitch curve, `gain` for an amplitude shape.

- **Times** start at 0 ms or later and strictly increase, at least 0.01 ms
  (10 µs) apart. The spacing is compared on the decimal values as written, so
  `0.01` after `0.0` is exactly enough.
- **Shape.** Ballast draws a monotone cubic (PCHIP) through the knots: it
  passes through every knot and, unlike a cubic spline, never overshoots
  between two of them ([Design](design.md#curves) draws both). Before its first
  knot and after its last, a curve holds its end value.
- **Amplitude shapes** have gains from 0 to 1, start and end at 0, and peak at
  exactly 1. Levels belong in `level_db`, so a shape only says how the sound
  moves. Their ends are flat, so they leave and join silence smoothly.
- **Pitch curves** end flat on ratio 1, so the body lands on its note, and
  the attack's tone on `tone_hz`, without a kink.

The kick lasts until the latest of the amplitude shapes' last knots, the
sub's included, and the transient's end.

## Validation

The parser reports every problem at once, each with its path:

```
body.tune_hz: 5000 is outside 20 … 1000
body.amp[2].ms: is earlier than the previous knot
attack.noise_mix: expected a number, found a string
output.sample_rate: expected one of 44100, 48000, 88200, 96000, 176400, 192000, found 44000
```

Unknown fields are refused as well as missing ones, so a typo never passes
silently. A TOML syntax error is reported with its line and column.

## Canonical form

`ballast init`, `tune`, `stretch` and `transient` write the canonical form:
every field in the order of this reference, with numbers as Python writes
them, so they read back exactly, and the standard comments shown above. A
transient's samples are one TOML string written 64 base64 characters a line,
with every line break escaped away. Reading a canonical patch
and writing it again gives the same text.

`ballast render` embeds the patch file as you wrote it, comments included, and
`ballast extract` returns it unchanged.

## Versions

A patch states its schema, so a later renderer can read it as it was meant,
and a renderer refuses a schema it does not know rather than guess.

- **Schema 3**, the current one, adds `body.bright_hz`, `attack.pitch`, `drive.asymmetry`,
  `drive.mix`, the `[clip]` section, and the optional `[sub]` and
  `[transient]`.
- **Schema 2** adds `output.low_cut_hz` to schema 1.
- **Schema 1** has no low cut.

Ballast still reads schemas 1 and 2 and renders them exactly as they always
sounded: what they lack is off, with a sine body, a fixed tone, no asymmetry, all of the
drive, and no clipper, low cut, sub or transient. `tune` and `stretch` write them back as schema 3 with
those values, which sound the same.
