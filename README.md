# Ballast

A kick drum synthesizer that renders a short text patch to a WAV file, the same
way every time.

![Waveforms of the seven presets, tech-house, tight, house, driven, trance, round and long, rendered at 48 kHz](docs/figures/presets.svg)

Ballast synthesizes one voice: a tonal body with an exact pitch sweep, and a
short attack of a tone, which can sweep too, and noise, with any sampled click
layered over them. It computes them at 32 times the sample rate,
saturates their sum, and band-limits it once with a frozen minimum-phase
filter. The files it writes keep every alias at least 104 dB down, are ready
for any sampler, and carry the patch that made them.

Listen to the presets: [house](docs/audio/house.wav) ·
[tech-house](docs/audio/tech-house.wav) · [trance](docs/audio/trance.wav) ·
[tight](docs/audio/tight.wav) · [driven](docs/audio/driven.wav) ·
[round](docs/audio/round.wav) · [long](docs/audio/long.wav). Each peaks at
−1 dBTP, and the [preset gallery](docs/presets.md) shows its curves,
spectrum, loudness and patch.

## Features

- **Clean at any drive.** Synthesis runs at 1.4112 or 1.536 MHz, and the
  drive and the clipper average their curves around each sample, by
  second-order antiderivative anti-aliasing, so even a full-scale 20 kHz tone
  at the maximum drive keeps every alias 104 dB down. A minimum-phase FIR passes 0–20 kHz within
  ±0.000014 dB and rejects everything from half the base rate up by at least
  116.4 dB, without ringing before the kick.
- **Character by design.** `body.bright_hz` turns the body into a saw whose
  harmonics fade above it, by the same Hz at every pitch, so its weight
  carries to small speakers. A `[sub]` adds the sine the tail meets, clean,
  after the drive, and `[eq]` shelves and a bell for weight, knock and air.
  Warmth is `drive.asymmetry`: even harmonics that
  grow as the body decays, with full scale and the drive's gain left exactly
  where they were. `drive.mix` blends in the undriven kick, sample-aligned.
  Loudness is the clipper: `clip.db` takes exactly that many decibels off the
  file's true peak, from a hard to a soft knee. A zap is `attack.pitch`: the
  click's tone dives into its own frequency.
- **Sampled transients.** `ballast transient` layers any short WAV over the
  attack, stored inside the patch. The same frozen filter brings it up to the
  internal rate, it plays at any pitch and from either rate family within 140
  dB of exact interpolation, and the drive glues it to the body.
- **Exact pitch.** Pitch curves are integrated exactly into phase. With
  `phase_ref = "tail"`, an 808-style tail meets the free-running sine at its
  note exactly, whatever the sweep did before.
- **Deterministic.** Every sample is a pure function of the patch and its
  time, and the filter ships as checksummed data. The same patch renders the
  same samples, run after run.
- **The patch travels with the sound.** Every WAV embeds its patch, and
  `ballast extract` gives it back byte for byte, comments included.
- **Strict patches.** Every field is required and nothing is clamped. An
  invalid patch is refused with every problem and its field path.
- **Live audition.** Plays the exact render buffer, renders again on every
  save, and loops your bass line under the hits in tempo, without resampling.
- **Sampler-ready.** The root key sits in the `smpl` chunk; 16- and 24-bit
  exports use seeded TPDF dither that never clips. `ballast multisample`
  renders one sample per key with an SFZ map, to play an 808 bassline.
- **Mastering-aware.** Normalize to a sample peak or to a true peak, clip
  the transient for loudness, high-pass the kick with an optional low cut, and
  read every render's loudness: its loudest momentary moment, by EBU R128.

## Install

Ballast needs Python 3.12 or later. From a checkout of this repository:

```
uv tool install ".[audition]"   # the ballast command, with live audition
pip install ".[audition]"       # or with pip, into the current environment
```

Audition plays through PortAudio, which `sounddevice` bundles on Windows and
macOS. On Linux, install it from your distribution, for example
`sudo apt install libportaudio2`. Leave out `[audition]` to render only.

## Quick start

```
ballast init tight --tune F1 -o kick.toml       # a complete patch to start from
ballast audition kick.toml --watch --bpm 128    # hear every save of kick.toml, on the beat
ballast render kick.toml -o kick.wav --format pcm24
ballast extract kick.wav                        # the patch, back out of the file
ballast multisample kick.toml C1 C2 -o kick     # one sample per key, with an SFZ map
```

`ballast render` prints what it made:

```
tuning   43.654 Hz, F1 +0.0 cents
length   240.0 ms source, 241.9 ms exported (11612 samples at 48000 Hz, 24-bit PCM)
peak     -1.08 dBFS sample, -1.00 dBTP true
loudness -14.4 LUFS momentary, at its loudest
render   66 ms
wrote kick.wav
```

## Specifications

| Parameter | Value |
|---|---|
| Output | mono WAV: float32, or 24- or 16-bit PCM with TPDF dither |
| Sample rates | 44.1, 48, 88.2, 96, 176.4 and 192 kHz |
| Internal rate | 32 × the base rate: 1411.2 or 1536 kHz |
| Passband | 0–20 kHz within ±0.000014 dB (specified ±0.001 dB) |
| Stopband | at least 116.4 dB down from half the base rate (specified 110 dB) |
| Aliasing | at least 104 dB down for a full-scale tone at any frequency up to 20 kHz, at any drive and asymmetry, and through the clipper; more than 130 dB down with up to 24 dB of drive |
| Levels | a gain, or a sample or true peak target; nothing is ever clipped |
| Loudness | the highest momentary loudness, EBU R128's 400 ms window slid sample by sample, K-weighted at every rate |
| PCM headroom | peaks stay 2 LSB below full scale, so dither never clips |
| Low cut | optional 12 dB/octave Butterworth high-pass, 10 Hz – 1 kHz |
| Tuning | 20 Hz – 1 kHz, by frequency or note name; the sweep spans 10 Hz – 20 kHz |
| Length | body up to 8 s, attack and transient up to 1 s |
| Render time | about 60 ms for a 240 ms kick, about 0.25 s for a 1.45 s one, on a 12-core laptop; the clipper adds up to half a second |

## Documentation

- [Usage](docs/usage.md): the commands, rendering for DAWs and samplers,
  audition, designing a kick, and the Python API.
- [Presets](docs/presets.md): every preset, with its sound, curves, spectrum
  and patch.
- [Patch reference](docs/patch.md): every field, its range and what it does.
- [Design](docs/design.md): every stage of the signal path, drawn from the
  code, and what makes a render reproducible.
- [Changelog](CHANGELOG.md).

## Contributing

Contributions are welcome. [CONTRIBUTING.md](CONTRIBUTING.md) explains how to
set up, test and propose a change.

## License

Ballast is free software, under the [GNU General Public License](LICENSE),
version 3 or later.

The sounds you make with it are yours. An additional permission in
[NOTICE](NOTICE) lets you use the sounds Ballast renders, the patches and SFZ
maps it writes, and its presets under any terms you choose, in a track or in a
sample pack you sell.
