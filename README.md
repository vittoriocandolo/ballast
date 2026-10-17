# Ballast

A kick drum synthesizer that renders a short text patch to a WAV file, the same
way every time.

![Waveforms of the four presets, tight, driven, round and long, rendered at 48 kHz](docs/figures/presets.svg)

Ballast synthesizes one voice: a tonal body with an exact pitch sweep, and a
short attack of tone and noise. It computes them at 16 times the sample rate,
saturates their sum, and band-limits it once with a frozen minimum-phase
filter. The files it writes are alias-free, ready for any sampler, and carry
the patch that made them.

Listen to the presets: [tight](docs/audio/tight.wav) ·
[driven](docs/audio/driven.wav) · [round](docs/audio/round.wav) ·
[long](docs/audio/long.wav)

## Features

- **Alias-free.** Synthesis runs at 705.6 or 768 kHz. A minimum-phase FIR
  passes 0–20 kHz within ±0.00002 dB and rejects everything from half the base
  rate up by at least 115.7 dB, without ringing before the kick.
- **Exact pitch.** The pitch curve is integrated exactly into phase. With
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
  exports use seeded TPDF dither that never clips.

## Install

Ballast needs Python 3.11 or later. From a checkout of this repository:

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
```

`ballast render` prints what it made:

```
tuning   43.654 Hz, F1 +0.0 cents
length   240.0 ms source, 241.9 ms exported (11611 samples at 48000 Hz, 24-bit PCM)
peak     -5.76 dBFS sample, -5.73 dBTP true
render   36 ms
wrote kick.wav
```

## Specifications

| Parameter | Value |
|---|---|
| Output | mono WAV: float32, or 24- or 16-bit PCM with TPDF dither |
| Sample rates | 44.1, 48, 88.2 and 96 kHz |
| Internal rate | 16 × the base rate: 705.6 or 768 kHz |
| Passband | 0–20 kHz within ±0.00002 dB (specified ±0.001 dB) |
| Stopband | at least 115.7 dB down from half the base rate (specified 110 dB) |
| PCM headroom | peaks stay 2 LSB below full scale, so dither never clips |
| Tuning | 20 Hz – 1 kHz, by frequency or note name; the sweep spans 10 Hz – 20 kHz |
| Length | body up to 8 s, attack up to 1 s |
| Render time | tens of milliseconds per kick on a laptop |

## Documentation

- [Usage](docs/usage.md): the commands, rendering for DAWs and samplers,
  audition, designing a kick, and the Python API.
- [Patch reference](docs/patch.md): every field, its range and what it does.
- [Design](docs/design.md): the signal path, the filters, and what makes a
  render reproducible.
- [Changelog](CHANGELOG.md).

## Contributing

Contributions are welcome. [CONTRIBUTING.md](CONTRIBUTING.md) explains how to
set up, test and propose a change.

## License

[MIT](LICENSE)
