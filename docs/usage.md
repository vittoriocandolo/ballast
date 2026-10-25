# Usage

- [A session](#a-session)
- [Commands](#commands)
- [Rendering for DAWs and samplers](#rendering-for-daws-and-samplers)
- [Audition](#audition)
- [Designing a kick](#designing-a-kick)
- [Python API](#python-api)

## A session

Start from the preset closest to what you want, tuned to the key of your track:

```
ballast init tight --tune F1 -o kick.toml
```

Play it on the beat, and keep it playing while you edit:

```
ballast audition kick.toml --watch --bpm 128
```

Open `kick.toml` in any editor. Every save renders again, and the new sound
takes over at the next hit; an invalid edit is reported and the last valid
sound keeps playing. When it sounds right, render it:

```
ballast render kick.toml -o kick.wav --format pcm24
```

The WAV carries the patch. `ballast extract kick.wav -o kick.toml` gives it
back, so a folder of rendered kicks is also a library of patches.

## Commands

Every command prints `--help`. `init`, `tune`, `stretch`, `transient` and
`extract` print the patch unless given `-o FILE`, which writes it as UTF-8 with plain
newlines. On Windows, prefer `-o` to shell redirection: PowerShell re-encodes
what it redirects, so comments may not survive.

### `ballast init PRESET [--tune NOTE] [-o FILE]`

Writes a complete patch from a preset: `house`, `tech-house` or `trance`; one
of `sampled`, `deep`, `fat` and `kit`, from machine to band, the last two
with a `[play]` section; or one of the four shapes `tight`, `driven`, `round`
and `long`. Every preset peaks at −1 dBTP.
`--tune` takes a note name such as `F1`, `C#2` or `Db2`, or a frequency in
Hz. A note name sets `tune_hz` to its equal-tempered frequency (A4 = 440 Hz),
rounded to 3 decimals.

### `ballast tune PATCH NOTE [-o FILE]`

Retunes a patch. The pitch curve is a ratio to `tune_hz`, so the whole sweep
moves with it, and every knot must stay within 10 Hz – 20 kHz.

### `ballast stretch PATCH CURVE FACTOR [--after MS] [-o FILE]`

Rescales the knot times of `body.pitch`, `body.amp`, `attack.pitch` or
`attack.amp`. Below 1 shortens, above 1 lengthens. Knots up to `--after` stay
put; by default, that is the peak of an amplitude shape, so the rise keeps its
shape, and 0 ms for a pitch curve. Times are rounded to 1 µs, and the result is checked like
any patch.

### `ballast transient PATCH WAV [-o FILE]`

Layers a WAV over the attack as the patch's `[transient]`, replacing any it
had. The WAV must be mono, or stereo, which is taken as its mid, (L + R) / 2,
at one of the six sample rates, and end within 1 s. It goes in as it sounds:
at its own rate, with its own peak as `level_db`, from the trigger, at pitch 1.
Then `start_ms`, `pitch`, `level_db` and `polarity` place it; deleting the
section removes it.

`init`, `tune`, `stretch` and `transient` write the canonical form of the
patch, with its standard comments: your own comments in the file are not
kept.

### `ballast vary PATCH -o DIR [-n N] [--seed SEED] [--along FIELD STEPS]...`

Writes N variations of a patch, 8 by default, into the folder `DIR`, as
`kick-v1.toml`, `kick-v2.toml` and so on: patches near yours, to audition and
keep. Each makes seven bounded moves, drawn from the seed:

| Move | What changes | Reach |
|---|---|---|
| punch | the first pitch ratio | × 2^±0.5 |
| snap | the pitch curve's times | × 2^±0.5 |
| length | the body's shape after its peak | × 2^±0.4 |
| click | the attack's level | ± 4 dB |
| tone | the attack's tone | × 2^±0.5 |
| drive | the drive | ± 4 dB |
| bright | the body's brightness, when it has any | × 2^±1 |

Each move stays within what the patch allows, and the tuning, the levels of
the body, the sub and the transient, and the output stay as they were. The
same seed gives the same variations on every computer and in every version
of Ballast, and each line printed says what moved, such as `punch ×1.08,
snap ×1.01, length ×1.222, …`. `ballast pack` renders the folder.

With `--along`, `vary` sets fields along axes instead, and writes every
combination, the first axis outermost:

```
ballast vary kick.toml -o grid --along drive.db 6,12,18 --along body.amp x0.8,x1,x1.25
```

Each axis is a field and its steps, separated by commas. A number sets the
field; `x` and a factor stretches a curve, as `ballast stretch` does, or
multiplies a number; any other word sets a choice, such as
`--along body.phase_ref onset,tail`. Every combination is checked before any
file is written, and each invalid one is listed. A grid takes neither `-n`
nor `--seed`, and makes at most 999 patches.

### `ballast render PATCH -o WAV [--format FORMAT] [--allow-over]`

Renders to a mono WAV at the patch's sample rate, prints a report and writes
the file. `--format` is `float32` (the default), `pcm24` or `pcm16`. See
[Rendering](#rendering-for-daws-and-samplers).

### `ballast multisample PATCH LOW HIGH -o DIR [--format FORMAT]`

Renders the patch at every semitone from `LOW` to `HIGH`, such as `C1 C2`,
into the folder `DIR`: one WAV per key, named after the patch and the note,
with its root key set, and an SFZ map that assigns each sample to its key. A
sampler then plays the patch as an instrument, at every pitch, with the real
length and character of each note rather than a transposed copy, which is what
an 808 bassline wants. Every note must be a valid tuning of the patch and fit
full scale; otherwise nothing is written and every problem is listed.

### `ballast pack PATCH... -o DIR [--rate RATE]... [--format FORMAT] [--velocities N] [--round-robins M]`

Renders patches, given as files or as folders of `.toml` files, into a
finished kit in the folder `DIR`:

- one WAV per patch, named after the patch and its key, such as
  `house-A1.wav`, or its frequency, such as `driven-50Hz.wav`, when it is more
  than half a cent off every note;
- the patch it was rendered from beside each WAV;
- `manifest.csv` and `manifest.json`, with each kick's sample rate, format,
  tuning, key and cents, length, sample and true peaks, loudness, and the
  SHA-256 of its WAV.

`--format` is `pcm24` by default, as sample packs and samplers expect. Each
patch renders at its own sample rate unless `--rate` says otherwise; given
more than once, each rate gets a folder of its own, such as `44100/` and
`48000/`. The manifest holds nothing of when or how fast the kit was made, so
the same patches always make the same kit.

`--velocities` and `--round-robins` play each patch as a drum, for patches
with a `[play]` section: N velocity layers, softest to hardest, each in M
repeats, named such as `house-A1-v2-rr3.wav`, and an SFZ map per patch, such
as `house-A1.sfz`, on key 36, General MIDI's bass drum, whose groups split
the velocities among the layers and whose regions take the repeats in turn.
The levels are in the samples, so the map tells the sampler not to scale
them again. The hits of a kit go through one chain, as a drum's do: its
hardest hit sets the clipper's threshold, and its loudest hit, whichever it
is, sets the gain they all share, so a soft hit stays soft after the output
stage and every repeat keeps its own small difference.

![The house preset played in three velocity layers, and two repeats of its hardest hit](figures/play.svg)

Every kick must pass two gates: its true peak at or below 0 dBTP, with no
sample over the format's ceiling, and its first and last samples below
−90 dBFS, so no sampler clicks where the file starts or stops. If one fails,
nothing is written and every failure is listed, as `multisample` does.

### `ballast audition PATCH [--watch] [--bpm BPM] [--with WAV]`

Plays the patch until you press Ctrl+C. `--watch` renders again on every
save, `--bpm` hits on the beat, and `--with` loops a bed, such as your bass
line, under the hits. See [Audition](#audition).

### `ballast extract WAV [-o FILE]`

Recovers the exact patch text from a WAV that Ballast rendered.

### Exit status

| Status | Meaning |
|---|---|
| 0 | done |
| 1 | refused or failed: a render over full scale, or an audition that cannot play |
| 2 | bad input: an invalid patch, a file that cannot be read or written, or bad arguments |
| 130 | interrupted with Ctrl+C |

Every file Ballast writes is written whole: to a temporary file beside it,
then renamed over it. A failed or interrupted write leaves the previous file
as it was, so retuning a patch in place can never destroy it.

Errors go to standard error and name the file and the problem. An invalid
patch lists every problem with its field path:

```
kick.toml: invalid patch
  body.tune_hz: 5 is outside 20 … 1000
  body.h2: expected a number, found a string
```

## Rendering for DAWs and samplers

### Formats

| Format | Use | Largest sample |
|---|---|---|
| `float32` | DAWs; keeps any level exactly | full scale, or above with `--allow-over` |
| `pcm24` | hardware samplers, archives | 2 LSB below full scale |
| `pcm16` | older samplers | 2 LSB below full scale |

PCM exports add triangular (TPDF) dither of ±1 LSB before rounding. The
dither is seeded from the patch, so a PCM file is as reproducible as a float
one, and exact zeros stay zero. Peaks are kept 2 LSB below full scale so the
dither can never clip: in `peak` mode the target comes down by that much
(0.0005 dB at 16 bits), and a `gain`-mode render above it is refused.

### Levels and overs

`output.mode = "gain"` applies `output.db` as a plain gain, which keeps
relative levels between kicks. `output.mode = "peak"` scales the kick so its
highest sample sits at `output.db` dBFS. `output.mode = "true-peak"` scales it
so its true peak, the peak between samples, sits at `output.db` dBTP, for files
that must not exceed a level after conversion; the highest sample still stays
within the format's ceiling. A gain-mode render over full scale
is never clipped: `ballast render` refuses it, says by how much to lower
`output.db`, and exits with status 1. `--allow-over` writes it anyway as a
float file with overs, for a DAW to bring down.

### The report

```
tuning   43.654 Hz, F1 +0.0 cents
length   240.0 ms source, 241.9 ms exported (11612 samples at 48000 Hz, 24-bit PCM)
peak     -1.09 dBFS sample, -1.00 dBTP true
loudness -14.4 LUFS momentary, at its loudest
render   66 ms
```

- **tuning**: `tune_hz`, and the nearest note with its offset in cents.
- **length**: the patch's length, and the exported length, which adds the
  decimator's tail: 1.9 ms at 48 or 96 kHz, 3.7 ms at 44.1 or 88.2 kHz.
- **peak**: the highest sample, and the true peak between samples, read on
  the file's own reconstruction at 32 times the base rate, to within
  0.000001 dB: a converter cannot play the file louder. A render over full
  scale or over the PCM ceiling says by how much.
- **loudness**: the highest momentary loudness, in LUFS: EBU R128's 400 ms
  window, slid one sample at a time over the kick played once, with the
  K-weighting of ITU-R BS.1770 derived for the file's sample rate. It
  compares kicks as a loudness meter would, at their loudest moment.
- **render**: the wall-clock time.

### What the file holds

A standard RIFF WAVE file, mono, which any DAW or sampler reads. Besides the
audio it holds:

- a `smpl` chunk with the MIDI root note and its fine tuning, from `tune_hz`,
  so samplers map the kick to its key. A pitch less than half a cent flat of a
  note counts as that note, so samplers that ignore the fine tuning are never
  a semitone off;
- a `blst` chunk with JSON: the patch text, the renderer version, the
  decimator's checksum and rates, the platform and the Python, NumPy and SciPy
  versions, and for PCM the dither's bit depth, key and seed.

Readers skip chunks they do not know.

## Audition

`ballast audition` plays the exact buffer `ballast render` would write, at the
patch's sample rate.

![Audition of the long preset at 128 bpm: each hit plays to its end and the tails sum, while a bed loops on the bar](figures/audition.svg)

- **Hits.** Without `--bpm`, the kick repeats after its own length plus
  half a second. With `--bpm`, it hits on every beat. Hits are unchoked: each
  plays to its end, so long tails overlap and sum, as in a DAW.
- **A bed.** `--with bass.wav --bpm 128` loops a mono or stereo WAV under the
  hits. It must be at the patch's sample rate and last a whole number of bars
  at the tempo, within 1 ms. Each loop starts on the beat grid, so it never
  drifts from the kick. If the kick over the bed would clip, audition warns
  you.
- **Watching.** With `--watch`, every save renders again and the new sound
  takes over at the next hit. An invalid patch, a render over full scale or a
  new sample rate is reported, and the last valid sound keeps playing. Edits
  saved while a render runs are rendered next, and the stale render is
  dropped.
- **Dropouts.** If the device ever runs out of samples, for example while a
  long render keeps the computer busy, audition says so, with a count.
- **No resampling.** Audition opens the device at the patch's sample rate or
  says it cannot. On Windows it plays through WASAPI, in exclusive mode when
  the shared mixer runs at another rate. On macOS it asks Core Audio to switch
  the device to the patch's rate, or fails. On Linux, a sound server such as
  PipeWire or PulseAudio may still resample to its own rate; set it to the
  patch's rate to hear the exact samples.

## Designing a kick

Sixteen of the moves below are made on the `tight` preset, each with a link
to its sound, to compare with [tight itself](audio/tight.wav); the figure
shows what they do.

![The tight preset and sixteen moves made on it: retuned to E1, a faster pitch drop, a longer body, a louder click, a brighter body, a zap, a sampled click, a sub, a room, grit, more drive, a knock, a low-pass, a low cut, warmth and clipping](figures/design.svg)

- **Tune it to the track.** A kick in the key of the bass sits in the mix. Use
  `ballast tune kick.toml E1`, or `--tune` with `init`.
  [Hear it in E1](audio/tight-e1.wav).
- **Punch is the pitch drop.** A higher first ratio and a faster fall give a
  harder hit; a slower fall sounds rounder. The `tight` preset falls from 7
  times the tuning to 2.2 in 6 ms, then to the note by 40 ms;
  `ballast stretch kick.toml body.pitch 0.5` makes it fall twice as fast.
  [Hear the faster fall](audio/tight-snap.wav).
- **Length is the body's amplitude shape.** Stretch it after the peak with
  `ballast stretch kick.toml body.amp 1.5`, which keeps the attack intact.
  [Hear the longer body](audio/tight-long.wav).
- **Brightness carries the weight to small speakers.** `body.bright_hz`
  turns the body's sine into a saw whose harmonics fade above it, the same in
  Hz at every pitch, while the fundamental stays put. Its harmonics, unlike
  the drive's, last as long as the tail. `body.bright_hz = 1000.0` puts 15%
  of the tail's power above 150 Hz, where a laptop can play it.
  [Hear the brighter body](audio/tight-bright.wav).
- **An acoustic body is modes.** `body.modes` adds partials at fixed ratios
  to the body, each fading at its own rate: a drum head instead of a sweep.
  With a slight glide, `pitch = [{ ms = 0.0, ratio = 1.12 }, { ms = 40.0,
  ratio = 1.0 }]`, a felt beater of noise, and four membrane modes and a
  shell mode:

  ```toml
  modes = [
    { ratio = 1.594, db = -5.0,  decay_ms = 140.0 },
    { ratio = 2.136, db = -9.0,  decay_ms = 90.0 },
    { ratio = 2.296, db = -12.0, decay_ms = 80.0 },
    { ratio = 2.653, db = -14.0, decay_ms = 60.0 },
    { ratio = 7.3,   db = -16.0, decay_ms = 35.0 },
  ]
  ```

  [Hear the drum](audio/drum.wav).
- **The click is the attack.** A tone around 1.5–3 kHz with band-limited
  noise, a few milliseconds long, cuts through a dense mix; `noise_mix` blends
  the two at constant power. `attack.level_db = -8.0` brings the click 8 dB
  forward. [Hear the louder click](audio/tight-click.wav).
- **A zap is the click's own sweep.** `attack.pitch` is a pitch curve for the
  attack's tone, a ratio to `tone_hz` like the body's. Falling into 1, it
  dives into `tone_hz` and brightens the click:
  `pitch = [{ ms = 0.0, ratio = 4.0 }, { ms = 2.0, ratio = 1.0 }]` dives two
  octaves, from 10 kHz, in 2 ms. [Hear the zap](audio/tight-zap.wav).
- **Layer a sampled click.** `ballast transient kick.toml click.wav` layers
  any short WAV over the attack, before the drive, which glues it to the body.
  Try it with [this click](audio/click.wav), three decaying modes of a struck
  block. `transient.pitch` tunes it as a sampler would, `start_ms` places it,
  and `polarity = -1` flips it if it fights the body's first swing.
  [Hear it layered](audio/tight-sample.wav).
- **Weight is the drive.** `drive.db` adds odd harmonics and density while
  full scale stays at full scale; `h2` adds the even second harmonic.
  `drive.db = 18.0`, up from 6, squares the body off.
  [Hear the heavier drive](audio/tight-drive.wav).
- **A sub carries the weight on, clean.** A `[sub]` section adds the sine
  the body's tail meets, under its own shape, after the drive: however hard
  the body is driven, the sub stays a sine, in phase with the tail. A sub at
  -6 dB that outlasts the body by 160 ms:

  ```toml
  [sub]
  level_db = -6.0
  amp = [
    { ms = 0.0,   gain = 0.0 },
    { ms = 5.0,   gain = 1.0 },
    { ms = 150.0, gain = 0.6 },
    { ms = 400.0, gain = 0.0 },
  ]
  ```

  [Hear the sub](audio/tight-sub.wav).
- **Depth is a room.** A `[room]` section puts the kick in a space: a
  seeded tail that darkens as it fades, after the drive, so it stays clean.
  `decay_ms` is how long it rings, `damping_hz` how dark it gets, and
  `predelay_ms` how far away the walls are:

  ```toml
  [room]
  level_db    = -12.0         # re the kick's energy
  decay_ms    = 600.0         # to fall 60 dB, at low frequencies
  damping_hz  = 4000.0        # fades twice as fast here
  predelay_ms = 8.0
  ```

  [Hear the room](audio/tight-room.wav).
- **Warmth is asymmetry.** `drive.asymmetry` adds even harmonics that grow as
  the body decays, and leaves full scale where it was; `drive.mix` below 1
  keeps some of the undriven kick, for density without losing the hit. Even
  harmonics bring a little DC, so pair them with a low cut.
  [Hear `drive.asymmetry = 1.0`](audio/tight-warm.wav).
- **Loudness is the clipper.** `clip.db` takes exactly that many decibels off
  the kick's true peak and leaves everything below the knee alone; `knee_db`
  softens the edge. The presets normalize to −1 dBTP, so the decibels come
  back as loudness: here `tight` goes from −13.4 to −11.4 LUFS. In gain
  mode the kick just peaks lower.
  [Hear `clip.db = 4.0`](audio/tight-clip.wav).
- **Long 808 tails want `phase_ref = "tail"`.** The tail is then exactly the
  sine at `tune_hz` with phase `phase_deg`, whatever the sweep does before it.
  Editing the sweep never shifts the tail, and kicks that share a tuning share
  a tail, so layered and edited kicks stay in phase.
- **Watch the true peak.** For files that must not exceed a level after
  conversion, use `output.mode = "true-peak"`, or read the dBTP figure in the
  report rather than the sample peak.
- **Knock is a bell.** `eq.bell_db = 6.0` lifts 200 Hz by 6 dB, where the
  body sweeps through at full strength in its first milliseconds: the
  tech house knock. It raises the kick's peak more than its loudness, so at
  −1 dBTP `tight` comes out quieter, −17.1 LUFS against −13.4; let the
  clipper take that peak back, as `tech-house` does.
  [Hear the knock](audio/tight-knock.wav).
- **Grit is a vintage sampler.** A `[grit]` section samples the kick as the
  drum samplers of the 1980s did, at a low rate with no filter against
  aliasing, in 12 bits, through their staircase. Pair it with the EQ's
  low-pass, which plays their output filter:

  ```toml
  [grit]
  rate_hz  = 26040.0          # no filter against aliasing
  bits     = 12               # no dither
  input_db = 6.0              # into the converter, and back
  ```

  [Hear the grit](audio/tight-grit.wav).
- **Dark is a low-pass.** `eq.cutoff_hz = 1200.0` closes the kick down
  above 1.2 kHz at 12 dB per octave: the filtered thud of a sampled disco
  kick. `eq.resonance` above 0.707 rings at the corner, as an analog filter
  does. [Hear the dark kick](audio/tight-dark.wav).
- **Cut what you cannot hear.** `output.low_cut_hz = 30.0` high-passes the
  kick at 30 Hz with a 12 dB/octave Butterworth filter, which removes DC and
  sub-sonic energy. The presets leave it off, so what you design is what you
  get: on a one-shot, a high-pass also shifts the body's phase and raises its
  peak against its loudness.
  [Hear the low cut](audio/tight-cut.wav).
- **Play it like a drum.** A `[play]` section says how the kick sounds hit
  softer, and how much each repeat differs; `ballast pack` then writes
  velocity layers and round robins, with an SFZ map:

  ```toml
  [play]
  soft_db       = -18.0       # the softest hit's level
  soft_click_db = -6.0        # and its click's, further
  soft_punch    = 0.6         # of the pitch drop it keeps
  spread        = 0.5         # how much each repeat differs
  ```

  `ballast pack kick.toml -o kit --velocities 4 --round-robins 3` writes 12
  hits and `kick-A1.sfz` in `kit`.
- **Play it as an instrument.** `ballast multisample bass.toml E1 E2 -o bass`
  gives an 808 bassline a sample for every note.

## Python API

The modules are importable for scripting renders:

```python
from ballast_synth import patch, render

text = patch.read_text("kick.toml")  # UTF-8, or UTF-16 with a BOM
kick = patch.loads(text)  # raises patch.PatchError, with .problems
kick = patch.edit(kick, "body.tune_hz", patch.parse_tune("G1"))
audio, report = render.render(kick, 24)  # float32 at kick.output.sample_rate
print(report)
if not report.over:
    render.export("kick-g1.wav", kick, patch.dumps(kick), audio, 24)
```

`render.render` returns the audio as a float32 NumPy array and a report;
`render.export` writes it, as float32 when `bits` is `None` and as dithered
PCM otherwise, with the patch text embedded.
