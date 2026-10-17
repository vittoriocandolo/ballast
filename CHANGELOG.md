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
  saturator, computed at 16 times the sample rate.
- Frozen minimum-phase decimators for the 44.1 and 48 kHz families, verified
  by checksum: flat to ±0.00002 dB up to 20 kHz and at least 115.7 dB down
  from half the base rate.
- Patches in TOML (schema 1), with every field required, every problem
  reported with its field path, and a canonical form.
- Exports at 44.1, 48, 88.2 and 96 kHz, as float32 or as 24- or 16-bit PCM
  with seeded TPDF dither, with the root key in a `smpl` chunk and the patch
  and its provenance in a `blst` chunk.
- The `ballast` command: `init` from four presets, `tune`, `stretch`,
  `render`, `audition` with live re-rendering, tempo and a bed, and `extract`.
- Documentation, figures and example sounds generated from the code, and a
  test suite that checks the documented properties on every platform.
