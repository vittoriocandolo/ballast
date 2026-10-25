# Ballast

A kick drum synthesizer that renders a short text patch to a WAV file, the same
way every time.

![Waveforms of the eleven presets, from tech-house to deep, rendered at 48 kHz](docs/figures/presets.svg)

Ballast synthesizes one voice: a tonal body with an exact pitch sweep, or a
drum head's modes, and a short attack of a tone and noise, with a clean sub,
a sampled click, a room and a vintage sampler's grit when you want them. It
computes them at 32 times the sample rate and band-limits them once with a
frozen minimum-phase filter. The files it writes keep every alias at least
104 dB down, unless you ask for grit, are ready for any sampler, and carry the
patch that made them.

Listen to the presets: [house](docs/audio/house.wav) ·
[tech-house](docs/audio/tech-house.wav) · [trance](docs/audio/trance.wav);
from machine to band, [sampled](docs/audio/sampled.wav) ·
[deep](docs/audio/deep.wav) · [fat](docs/audio/fat.wav) ·
[kit](docs/audio/kit.wav); and the shapes [tight](docs/audio/tight.wav) ·
[driven](docs/audio/driven.wav) · [round](docs/audio/round.wav) ·
[long](docs/audio/long.wav). Each peaks at −1 dBTP, and the
[preset gallery](docs/presets.md) shows its curves, spectrum, loudness and
patch.

## Features

- **Weight, by design.** The body lands exactly on its note, and a `[sub]`
  adds the sine its tail meets, clean, after the drive. `body.bright_hz`
  turns the body into a saw whose harmonics fade above a frequency, the same
  at every pitch, so its weight carries to small speakers. The drive adds
  density, `drive.asymmetry` warmth that grows as the body decays, and the
  clipper loudness: `clip.db` takes exactly that many decibels off the file's
  true peak. `[eq]` has shelves, a bell for knock and a resonant low-pass.
- **From machine to band.** `[grit]` samples the kick as the drum samplers of
  the 1980s did, aliasing and staircase included, computed exactly: the only
  place Ballast aliases, and only when asked. At the other end, `body.modes`
  gives the body a drum head's partials, `[room]` puts it in a space that
  darkens as it fades, and `[play]` plays it like a drum.
- **Clicks.** `attack.pitch` sweeps the click's tone into a zap, and
  `ballast transient` layers any short WAV over the attack, stored in the
  patch, played at any pitch and from either rate family within 140 dB of
  exact interpolation.
- **Clean at any drive.** Synthesis runs at 1.4112 or 1.536 MHz, and the
  drive and the clipper average their curves around each sample, by
  second-order antiderivative anti-aliasing, so even a full-scale 20 kHz tone
  at the maximum drive keeps every alias 104 dB down. A minimum-phase FIR
  passes 0–20 kHz within ±0.000014 dB and rejects everything from half the
  base rate up by at least 116.4 dB, without ringing before the kick.
- **Exact pitch.** Pitch curves are integrated exactly into phase. With
  `phase_ref = "tail"`, an 808-style tail meets the free-running sine at its
  note exactly, whatever the sweep did before.
- **Deterministic.** Every sample is a pure function of the patch and its
  time, and the filters ship as checksummed data. The same patch renders the
  same samples bit for bit, run after run, on one platform with the same NumPy
  and SciPy, and the same sound on any other: only rounding can differ, under
  a millionth of full scale, which the tests check on every platform.
- **The patch travels with the sound.** Every WAV embeds its patch, and
  `ballast extract` gives it back byte for byte, comments included.
- **Strict patches.** Every field is required and nothing is clamped. An
  invalid patch is refused with every problem and its field path.
- **Finished kits.** `ballast vary` writes seeded variations of a patch, the
  same on every computer, or every combination along axes you choose.
  `ballast pack` renders a folder of patches into a kit: WAVs named by key,
  their patches beside them, a manifest of tuning, peaks and loudness, and
  nothing written unless every kick passes its gates. With `[play]`, it writes
  velocity layers and round robins, with an SFZ map.
- **Live audition.** Plays the exact render buffer, renders again on every
  save, and loops your bass line under the hits in tempo, without resampling.
- **Sampler-ready.** The root key sits in the `smpl` chunk; 16- and 24-bit
  exports use seeded TPDF dither that never clips. `ballast multisample`
  renders one sample per key with an SFZ map, to play an 808 bassline.
- **Mastering-aware.** Normalize to a sample peak or to a true peak, high-pass
  the kick with an optional low cut, and read every render's loudness: its
  loudest momentary moment, by EBU R128.

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
ballast vary kick.toml -n 8 -o vars             # eight seeded variations to audition
ballast vary kick.toml -o grid --along drive.db 6,12,18  # or every step along axes
ballast pack vars -o kit --rate 44100 --rate 48000  # a finished kit, with a manifest
```

`ballast render` prints what it made:

```
tuning   43.654 Hz, F1 +0.0 cents
length   240.0 ms source, 241.9 ms exported (11612 samples at 48000 Hz, 24-bit PCM)
peak     -1.09 dBFS sample, -1.00 dBTP true
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
| Aliasing | at least 104 dB down for a full-scale tone at any frequency up to 20 kHz, at any drive and asymmetry, and through the clipper; more than 130 dB down with up to 24 dB of drive. `[grit]` aliases on purpose, as the sampler it models |
| Levels | a gain, or a sample or true peak target; nothing is ever clipped |
| True peak | read on the file's own reconstruction at 32 times the base rate, to within 0.000001 dB, where a 4× meter can read tenths of a decibel low; the clipper takes its `clip.db` off it within 0.0000001 dB |
| Loudness | the highest momentary loudness, EBU R128's 400 ms window slid sample by sample, K-weighted at every rate |
| PCM headroom | peaks stay 2 LSB below full scale, so dither never clips |
| Low cut | optional 12 dB/octave Butterworth high-pass, 10 Hz – 1 kHz |
| Tuning | 20 Hz – 1 kHz, by frequency or note name; the sweep spans 10 Hz – 20 kHz |
| Length | body up to 8 s, attack and transient up to 1 s, and a room's tail up to 6.2 s after them |
| Render time | about 60 ms for a 240 ms kick and a third of a second for a 1.45 s one, on a 12-core laptop; the clipper, a room or grit each add a few tenths of a second |

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
