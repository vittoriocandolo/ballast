# Contributing

Thank you for helping with Ballast. Its priorities, in order, are: correct,
simple, minimal, elegant. A change that makes the code shorter but less
obviously correct is not an improvement.

## Set up

Ballast uses [uv](https://docs.astral.sh/uv/):

```
uv sync --all-extras    # the package in editable mode, audition, pytest, ruff and mypy
```

## Check a change

```
uv run pytest
uv run ruff check
uv run ruff format --check
uv run mypy
```

CI runs the tests on Linux, macOS and Windows with Python 3.11 to 3.14, and
once with the oldest NumPy and SciPy that Ballast supports.

## Rules

- **Renders stay bit for bit the same.** A change to any rendered sample
  changes the sound of existing patches. It needs a reason, a changelog entry
  and a new version.
- **Patches have no defaults.** A new field means a new schema version, and
  older schemas keep rendering exactly as before.
- **Every behavior has a test,** written as a property that holds on every
  platform, such as "rendering in blocks never changes a sample", rather than
  as a stored hash of a render.
- **Errors say where and what.** They name the file or the field path, and the
  problem.
- **NumPy and SciPy are the only dependencies,** and `sounddevice` only for
  audition.

## Documentation

The figures in `docs/figures` and the sounds in `docs/audio` are generated
from the code:

```
uv run python docs/build.py
```

Commit them with the change that alters them, and keep the documentation in
`docs/` and the README true to the code.

## Releasing

1. Set the version in `pyproject.toml` and `src/ballast_synth/__init__.py`; a test
   checks that they agree.
2. Move the changelog's entries under the new version, with the date.
3. Tag the commit `vX.Y.Z` and push the tag.

The release workflow then checks that the tag matches the version, runs the
tests, builds the package and publishes it to PyPI as `ballast-synth`, through
PyPI's trusted publishing: configure this repository and its `pypi`
environment as a trusted publisher on PyPI once, and no token is needed. The
name `ballast` on PyPI belongs to another project, so the package you import
is `ballast_synth`; the command stays `ballast`.

## Commits and pull requests

Write commit subjects in the imperative, such as "Add a 24-bit export", with a
body that explains why. Keep a pull request to one change, with its tests and
documentation.
