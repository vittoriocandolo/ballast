# Contributing

Thank you for helping with Ballast. Its priorities, in order, are: correct,
simple, minimal, elegant. A change that makes the code shorter but less
obviously correct is not an improvement.

## Set up

Ballast uses [uv](https://docs.astral.sh/uv/):

```
uv sync --all-extras    # the package in editable mode, audition, pytest and ruff
```

## Check a change

```
uv run pytest
uv run ruff check
uv run ruff format --check
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

## Commits and pull requests

Write commit subjects in the imperative, such as "Add a 24-bit export", with a
body that explains why. Keep a pull request to one change, with its tests and
documentation.
