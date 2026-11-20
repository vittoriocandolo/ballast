# Contributing

Thank you for helping with Ballast.

## Set up

Ballast uses [uv](https://docs.astral.sh/uv/):

```
uv sync --all-extras
```

## Check a change

```
uv run coverage run -m pytest
uv run coverage report
uv run ruff check
uv run ruff format --check
uv run mypy
uv run typos
uv run zizmor --persona auditor .
```

Coverage runs the tests and requires every line and every branch of
`ballast_synth` to run, ruff runs every stable rule, mypy checks all the code
strictly, as Python 3.12 would run it, pytest turns any warning into an error,
typos checks the spelling of code and documentation, and zizmor audits the
GitHub workflows. `uv run pytest` alone runs the tests faster.

`uv run ruff check --fix`, then `uv run ruff format`, fix most of what ruff
finds; in that order, the formatter keeps the trailing commas the linter
adds. The formatter warns that the trailing comma rule, COM812, may conflict
with it: on this code they agree, so the rule stays on.

CI runs mypy and the tests on every platform and Python that Ballast
supports, and the tests again with the oldest NumPy and SciPy supported
versions.

## Rules

- **Correct, simple, minimal, elegant, in that order.** A change that makes
  the code shorter but less obviously correct is not an improvement.
- **Renders stay bit for bit the same.** On one platform with the same NumPy
  and SciPy, a change to any rendered sample changes the sound of existing
  patches. It needs a reason, a changelog entry, a new version and new
  references (`uv run python tools/references.py --force`). Across platforms
  and library versions, samples may differ by rounding, which is no change:
  the references allow a millionth of full scale.
- **Patches have no defaults.** A new field means a new schema version, and
  older schemas keep rendering exactly as before.
- **Seeds and names are forever.** The keys that seed the noise, the dither,
  the room, the variations and the repeats, `ballast/noise/v1`,
  `ballast/dither/v1`, `ballast/room/v1`, `ballast/vary/v1` and
  `ballast/play/v1`, and the WAV chunk `blst` never change: a new behavior
  takes a new key.
- **Every behavior has a test,** written as a property that holds on every
  platform, such as "rendering in blocks never changes a sample", rather than
  as a stored hash of a render, which rounding would break. The one stored
  reference, samples of every preset, allows for rounding. Coverage checks
  that the tests run every line and every branch; code no test can reach is
  removed, not excluded.
- **Errors say where and what.** They name the file or the field path, and the
  problem.
- **NumPy and SciPy are the only dependencies,** and `sounddevice` only for
  audition.
- **Every check stays on.** Fix what a linter finds instead of silencing it.
  The one exception is `assert` in tests, which is how pytest checks.

## Documentation

The figures in `docs/figures`, the sounds in `docs/audio` and the preset
gallery `docs/presets.md` are generated from the code:

```
uv run python docs/build.py
```

Commit them with the change that alters them, and keep the documentation in
`docs/` and the README true to the code. The tests check that every link in
the documentation leads somewhere, that every figure and sound is shown, and
that the patch reference's example is the tight preset as it ships.

## Frozen filters

The decimators ship as data in `src/ballast_synth/data`, checked by SHA-256.
`uv run python tools/decimators.py` shows how they were designed, and whether
a design with the current SciPy still matches them. `--force` replaces them,
which changes every render.

## Releasing

1. Set the version in `pyproject.toml` and `src/ballast_synth/__init__.py`; a
   test checks that they agree. The build points the README's links on PyPI at
   the repository at the release's tag (see `hatch_build.py`), so the page
   shows the documentation of that very release.
2. Move the changelog's entries under the new version, with the date.
3. Tag the commit `vX.Y.Z` and push the tag.

The release workflow then checks that the tag matches the version, runs the
tests, builds the package and publishes it to PyPI as `ballast-synth`, through
PyPI's trusted publishing: configure this repository and its `pypi`
environment as a trusted publisher on PyPI once, and no token is needed. The
tests and the build run in a job that never holds the publishing token; a
second job, which runs no project code, uploads what the first built, with
attestations of where it was built. Give the `pypi` environment required
reviewers, so that a pushed tag publishes only once someone approves it. The
package is `ballast_synth`; the command stays `ballast`.

## Commits and pull requests

Write commit subjects in the imperative, such as "Add a 24-bit export", with a
body that explains why. Keep a pull request to one change, with its tests and
documentation.

## License

Ballast is licensed under the GNU GPL version 3 or later, with the additional
permission in [NOTICE](NOTICE) for the files it writes. By contributing, you
agree to license your contribution under the same terms, so that the sounds
people make with Ballast stay theirs.
