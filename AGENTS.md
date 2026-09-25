# pyLife – Agent instructions

pyLife is an Open Source Python library (Bosch Research) for fatigue and
reliability lifetime assessment of mechanical components. Core numerical
computations use `numpy`/`scipy`, with `pandas` as the primary data container
via a custom accessor framework (see Architecture below). Two Cython
extensions (`rainflow_ext`, `_fkm_linear_functions`) provide performance
critical code.

## Build / test / lint

The project uses `uv` for dependency management (see `pyproject.toml`).
Since pyLife is a library, not an application, `uv.lock` is gitignored and
not committed — don't rely on it being present or up to date in a fresh
checkout. There is no `tox.ini` or `Makefile`.

```sh
# Install deps (dev + optional extras used in CI)
uv sync --dev --extra all --all-groups

# Run the full test suite (pytest config lives in [tool.pytest.ini_options] in pyproject.toml)
uv run pytest -n auto --cov --cov=term-missing

# Run a single test file / test / class
uv run pytest tests/strength/test_meanstress.py
uv run pytest tests/strength/test_meanstress.py::test_something
uv run pytest tests/strength/test_meanstress.py -k "some_expr"

# Doctests are collected too (`--doctest-modules`), testpaths = tests + src/pylife
uv run pytest src/pylife/strength/meanstress.py --doctest-modules
```

Notes:
- Default test run excludes markers `slow_acceptance` and `demos` (see
  `addopts` in `pyproject.toml`). Demo notebook tests run separately via
  `uv run pytest -m demos` (needs the `dev`/`visuals` extras, e.g. `testbook`).
- A few modules are excluded from doctest collection and coverage
  (`strength/helpers.py`, `strength/sn_curve.py`,
  `materialdata/woehler/bayesian.py`) — don't be surprised they're skipped.
- Building the Cython extensions requires `cythonize`; if you change
  `src/pylife/stress/rainflow/extension.pyx` or
  `src/pylife/strength/fkm_linear/extension.pyx.in` you need to rebuild
  (`uv sync` / `pip install -e .` triggers the build via `setup.py`).
- Linting/formatting tools (run via `.pre-commit-config.yaml`): `isort`,
  `black`, `flake8` (max line length 132 in pre-commit, 120 in
  `[tool.flake8]`/CODINGSTYLE guidance of ~90-125 chars). Run
  `pre-commit run --all-files` if available.
- CI (`.github/workflows/pytest.yml`) runs the matrix across Python 3.9–3.14
  on Ubuntu and Windows; keep changes compatible with that range.

## Architecture: the pandas accessor "Signal" pattern

Most of pyLife's public API is exposed as **pandas DataFrame/Series
accessors**, not as objects you instantiate directly. This is the single most
important pattern to understand before touching code in `stress/`,
`strength/`, `mesh/`, `materialdata/`, `materiallaws/`.

- `pylife.core.PylifeSignal` (in `src/pylife/core/pylifesignal.py`) is the
  base class for all "signal" accessors. Subclasses are registered with
  `@pd.api.extensions.register_dataframe_accessor('name')` (or
  `register_series_accessor`) and implement `_validate(self)` to check the
  required columns/index entries are present (raising `AttributeError`/
  `ValueError` otherwise). Use `fail_if_key_missing()` / `get_missing_keys()`
  from `PylifeSignal`/`DataValidator` for that validation.
- Once registered, a plain `pandas.DataFrame`/`Series` with the right
  columns/index gains the accessor automatically, e.g. `df.stress_signal...`
  or `series.woehler...` — grep for `register_dataframe_accessor` /
  `register_series_accessor` to find all signal types.
- `PylifeSignal.from_parameters(**kwargs)` builds a signal directly from
  scalar/array parameters instead of an existing pandas object.
- `register_method(cls, name)` (decorator) lets you attach extra methods to an
  existing accessor class from a different module without modifying it —
  used to keep optional/heavy functionality (e.g. certain FKM or Woehler
  extensions) decoupled from the core class.
- `pylife.core.Broadcaster` (base of `PylifeSignal`) aligns two pandas
  operands (e.g. a load collective and a material parameter table) onto a
  common index before elementwise operations — central to how signals
  combine with each other across the library.
- `DataValidator` (`data_validator.py`) implements the actual missing-key
  checks used by `_validate()` methods.

When adding a new kind of physical data (e.g. a new stress/strain
representation), the idiomatic approach is: create a `PylifeSignal` subclass,
register it as a DataFrame/Series accessor, implement `_validate`, and expose
computations as accessor methods/properties rather than free functions.

## Key conventions

- **Physical quantity variable names are mandated**, not just suggested — see
  `docs/variable_names.rst` (e.g. `S11`/`S22`/... for stress tensor
  components, `amplitude`, `meanstress`, `R` for cyclic stress, Greek letters
  spelled out like `alpha`). Match these names in new code/columns instead of
  inventing new ones.
- Full style guide: `docs/CODINGSTYLE.rst`. Beyond PEP8, it asks for:
  - Lines should usually stay under ~90 characters; *never* exceed 125.
  - Module names: short, single words if possible (`rainflow`, `gradient`);
    if multiple words are needed use `snake_case` — never dashes or capital
    letters in module names.
  - Class names: `CamelCase`, short nouns.
  - Function/variable names: `lowercase_with_underscores`; can be longer and
    more sentence-like than class names; local variables may be short
    (e.g. `res`) but should be descriptive if the surrounding code is more
    than a glance long.
  - Leading underscore `_` for anything not part of the public API
    (methods and attributes).
  - Prefer `@property` over manual getters/setters; avoid setters unless
    truly needed — prefer creating a modified copy over mutating an object
    in place.
  - Keep functions/methods short and single-purpose; if a function no
    longer fits your editor window or you need to scroll to match up a
    loop/`if`, split it up.
  - Minimize comments — code and numpy-style docstrings documenting the
    public API should make the intent clear on their own; extract
    well-named helper functions instead of explanatory comments. Remove any
    commented-out code before opening a PR.
  - Don't group functions into a class just because they're semantically
    related — that's what modules are for. Use a class only when callers
    need to keep state around across multiple related calls.
- Tests live under `tests/`, mirroring the `src/pylife/` package layout
  (`tests/strength`, `tests/stress`, `tests/mesh`, ...); doctest examples
  embedded in `src/pylife/**` are also executed as tests (see pytest config
  above), so keep docstring examples runnable and accurate.
- `tests/conftest.py` toggles pandas Copy-on-Write mode for the whole session
  — be aware of CoW semantics when writing tests that mutate DataFrames.
- Every functional change (bugfix or feature) requires accompanying pytest
  unit tests (see `CONTRIBUTING.md`); PRs without tests are rejected.
- Add a short entry to `CHANGELOG.md` under "Unreleased" for any non-cosmetic
  change, referencing the GitHub issue number if applicable.
- Commits require a DCO `Signed-off-by:` trailer (`git commit -s`); PRs
  target the `develop` branch (not `master`) except urgent bugfixes.
