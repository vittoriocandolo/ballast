# Contributing

Thank you for helping with Ballast.

## Set up

Ballast uses [uv](https://docs.astral.sh/uv/):

```
uv sync --all-extras
```

## Check a change

```
uv run pytest
uv run ruff check
uv run ruff format --check
uv run mypy
uv run typos
uv run zizmor --persona auditor .
```

Ruff runs every stable rule, mypy checks all the code strictly, as Python
3.11 would run it, pytest turns any warning into an error, typos checks the
spelling of code and documentation, and zizmor audits the GitHub workflows.

`uv run ruff check --fix`, then `uv run ruff format`, fix most of what ruff
finds; in that order, the formatter keeps the trailing commas the linter
adds. The formatter warns that the trailing comma rule, COM812, may conflict
with it: on this code they agree, so the rule stays on.

CI runs mypy and the tests on every platform and Python that Ballast
supports, and the tests again with the oldest NumPy and SciPy supported
versions.

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
- **Every check stays on.** Fix what a linter finds instead of silencing it.
  The one exception is `assert` in tests, which is how pytest checks.

## Documentation

The figures in `docs/figures` and the sounds in `docs/audio` are generated
from the code:

```
uv run python docs/build.py
```

Commit them with the change that alters them, and keep the documentation in
`docs/` and the README true to the code.

## Releasing

1. Set the version in `pyproject.toml` and `src/ballast_synth/__init__.py`.
2. Move the changelog's entries under the new version, with the date.
3. Tag the commit `vX.Y.Z` and push the tag.

The release workflow then checks that the tag matches the version, runs the
tests, builds the package and publishes it to PyPI as `ballast-synth`, through
PyPI's trusted publishing: configure this repository and its `pypi`
environment as a trusted publisher on PyPI once, and no token is needed.
The package is `ballast_synth`; the command stays `ballast`.

## Commits and pull requests

Write commit subjects in the imperative, such as "Add a 24-bit export", with a
body that explains why. Keep a pull request to one change, with its tests and
documentation.

## License

Ballast is licensed under the GNU GPL version 3 or later, with the additional
permission in [NOTICE](NOTICE) for the files it writes. By contributing, you
agree to license your contribution under the same terms, so that the sounds
people make with Ballast stay theirs.
