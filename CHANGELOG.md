# Changelog

All notable changes to Ballast are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Within a version, renders are bit
for bit the same; a change to the sound of an existing patch is a new version.

## [0.1.0] - Unreleased

The first public release.

### Added

- A one-voice kick drum synthesizer: a body with an exact pitch sweep and a
  second harmonic, an attack of tone and seeded band-limited noise, and a tanh
  saturator anti-aliased to the second order, computed at 32 times the sample
  rate: every setting keeps its aliases at least 104 dB down.
- Frozen minimum-phase decimators for the 44.1 and 48 kHz families, verified
  by checksum: flat to ±0.000014 dB up to 20 kHz and at least 116.4 dB down
  from half the base rate.
- Patches in TOML, with every field required, every problem reported with its
  field path, and a canonical form. Schema 3 is current; patches of schemas 1
  and 2 still read and render exactly as they always sounded.
- Character by design: the drive's asymmetry, even harmonics that leave full
  scale and the drive's gain where they were, and its mix with the undriven
  kick; and a clipper that takes exactly `clip.db` off the file's true peak,
  from a hard to a soft knee, anti-aliased like the drive.
- Exports at 44.1, 48, 88.2, 96, 176.4 and 192 kHz, as float32 or as 24- or
  16-bit PCM with seeded TPDF dither, normalized to a gain, a sample peak or a
  true peak, with the root key in a `smpl` chunk and the patch and its
  provenance in a `blst` chunk. Every file is written whole or not at all.
- The `ballast` command: `init` from four presets, `tune`, `stretch`,
  `render`, `multisample` for one sample per key with an SFZ map, `audition`
  with live re-rendering, tempo, a bed and dropout warnings, and `extract`.
- Documentation with figures and sounds generated from the code: a figure for
  every stage of the signal path, a gallery of the presets with their sounds,
  curves, spectra and patches, and the sound of every design move in the usage
  guide.
- A test suite that checks the documented properties on every platform.
- Type annotations throughout, with a `py.typed` marker so that type checkers
  use them.
