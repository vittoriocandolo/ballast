# Patch reference

A patch is a complete TOML description of one kick. Every field is required
and the renderer has no defaults, so a future default can never change an old
patch. The parser rejects anything outside the valid domain, reports every
problem with its field path, and never clamps or repairs.

![The tight preset's pitch, body and attack curves through their knots, and the first 60 ms of its output](figures/anatomy.svg)

This is the `tight` preset, drawn above:

```toml
schema = 2
seed = 1

[body]
tune_hz   = 55.0
level_db  = 0.0
h2        = 0.15            # second harmonic, 0 … 0.5
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
amp = [
  { ms = 0.0,  gain = 0.0 },
  { ms = 0.08, gain = 1.0 },
  { ms = 1.5,  gain = 0.35 },
  { ms = 6.0,  gain = 0.0 },
]

[drive]
db = 6.0                    # small-signal gain; 0 = bypass

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
| `schema` | integer | 1 or 2 | The patch format, checked first: another schema may share nothing else with this one. Schema 2 is current; see [Versions](#versions). |
| `seed` | integer | 0 – 2⁶³ − 1 | Draws the attack's noise and the PCM dither. The same seed gives the same noise at every sample rate. |

### `[body]`: the tonal body

| Field | Type | Range | Meaning |
|---|---|---|---|
| `tune_hz` | number | 20 – 1000 Hz | The note the body settles on. |
| `level_db` | number | −60 – +24 dB, or `-inf` | The body's level; `-inf` silences it. |
| `h2` | number | 0 – 0.5 | The second harmonic: the body is sin φ + h2 sin 2φ, scaled so its power does not change with `h2`. |
| `phase_deg` | number | 0 ≤ φ < 360 | The starting phase, at the reference point. |
| `phase_ref` | `"onset"` or `"tail"` | | Where `phase_deg` applies: at the trigger, or on the free-running sine at `tune_hz` that the body meets exactly from the last pitch knot on. |
| `pitch` | curve | 2 – 32 knots of `{ ms, ratio }` | The frequency as a ratio to `tune_hz`. It starts at 0 ms and ends at ratio 1, and `tune_hz × ratio` stays within 10 Hz – 20 kHz at every knot. |
| `amp` | curve | 3 – 32 knots of `{ ms, gain }` | The body's amplitude shape, ending by 8 s. |

### `[attack]`: the click

| Field | Type | Range | Meaning |
|---|---|---|---|
| `level_db` | number | −60 – +24 dB, or `-inf` | The attack's level; `-inf` silences it. |
| `tone_hz` | number | 20 Hz – 20 kHz | The frequency of the attack's sine. |
| `noise_mix` | number | 0 – 1 | Tone (0) to noise (1), at constant power; both ends are exact. |
| `noise_band_hz` | `[low, high]` | 20 ≤ low < high ≤ 20 000 | The noise band: second-order skirts with their half-power points at the edges. |
| `amp` | curve | 3 – 32 knots of `{ ms, gain }` | The attack's amplitude shape, ending by 1 s. |

### `[drive]`: the saturator

| Field | Type | Range | Meaning |
|---|---|---|---|
| `db` | number | 0 – 48 dB | The small-signal gain of a tanh saturator that keeps full scale at full scale. 0 dB is an exact bypass. |

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
the pitch, `gain` for an amplitude shape.

- **Times** start at 0 ms or later and strictly increase, at least 0.01 ms
  (10 µs) apart. The spacing is compared on the decimal values as written, so
  `0.01` after `0.0` is exactly enough.
- **Shape.** Ballast draws a monotone cubic (PCHIP) through the knots: it
  passes through every knot and never overshoots between two of them. Before
  its first knot and after its last, a curve holds its end value.
- **Amplitude shapes** have gains from 0 to 1, start and end at 0, and peak at
  exactly 1. Levels belong in `level_db`, so a shape only says how the sound
  moves. Their ends are flat, so they leave and join silence smoothly.
- **The pitch curve** ends flat on ratio 1, so the body lands on its note
  without a kink.

The kick lasts until the later of the two amplitude shapes' last knots.

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

`ballast init`, `tune` and `stretch` write the canonical form: every field in
the order of this reference, with numbers as Python writes them, so they read
back exactly, and the standard comments shown above. Reading a canonical patch
and writing it again gives the same text.

`ballast render` embeds the patch file as you wrote it, comments included, and
`ballast extract` returns it unchanged.

## Versions

A patch states its schema, so a later renderer can read it as it was meant,
and a renderer refuses a schema it does not know rather than guess.

- **Schema 2**, the current one, adds `output.low_cut_hz`.
- **Schema 1** has no low cut. Ballast still reads it, and renders it exactly
  as it always did; `tune` and `stretch` write it back as schema 2 with
  `low_cut_hz = 0.0`, which sounds the same.
