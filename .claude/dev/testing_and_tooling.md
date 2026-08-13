# Testing & tooling

ARIEL uses **uv** for environments, **pytest** for tests, and **pre-commit**
for commit-time gates. Source of truth: [noxfile.py](../../noxfile.py),
[pyproject.toml](../../pyproject.toml),
[.pre-commit-config.yaml](../../.pre-commit-config.yaml),
[CONTRIBUTING.md](../../CONTRIBUTING.md).

Note: `noxfile.py` currently defines only a `docs` session (live-reload docs
build via `sphinx-autobuild`). There is no `tests`, `pre-commit`, or
`tests_compiled` nox session despite older docs/comments referring to them —
run `pytest` and the pre-commit hooks directly (below), not through `nox`.

## Environment

`pyproject.toml` requires **Python ≥ 3.12**. Set up with uv:

```bash
uv venv
uv sync
uv run examples/re_book/1_brain_evolution.py   # run anything via `uv run`
```

(`CONTRIBUTING.md` also shows `uv install`; the README's `uv venv` + `uv sync`
is the current flow.)

## Tests & coverage

```bash
uv run pytest tests/unit                                   # unit tests
uv run pytest tests/unit --cov=src/ariel --cov-report=term-missing
```

- Tests live in [tests/](../../tests/), written with `pytest`
  (`pytest`, `pytest-cov`, `pytest-modern`). `tests/functional/` is a set of
  `.ipynb` notebooks, not `pytest`-collected by default — no wired-up nox/CI
  session currently executes them automatically; run them manually (e.g. via
  Jupyter or `nbclient`) if you need to exercise that path.
- **Coverage must be 100%** — `pyproject.toml` sets
  `[tool.coverage.report] fail_under = 100`. New code needs new tests or the
  suite fails. Excluded lines: `pragma: no cover`, `if TYPE_CHECKING:`. In
  practice this is only enforced against whatever `pytest` invocation you run
  it with — since `tests/functional/` isn't included above, coverage numbers
  from the command above will read lower than the "true" number the notebooks
  would contribute.
- Asserts are allowed in test files (ruff `S101` carve-out for `tests/`,
  `test_*.py`, `noxfile.py`).

When adding a feature, add the test in the matching `tests/…` path **in the same
change** — don't defer it, or coverage drops below 100% and CI blocks.

## Pre-commit

`pre-commit` is not itself a project dependency (not in any
`[dependency-groups]` in `pyproject.toml`) — install it separately, then
install the git hook once:

```bash
uv tool install pre-commit   # or: pipx install pre-commit
uv tool run pre-commit install
```

The hooks (see [.pre-commit-config.yaml](../../.pre-commit-config.yaml)) run
`ruff check` + `ruff format`, `pydoclint` (NumPy docstring linting), `prettier`,
and TOML/YAML validators. Code that isn't ruff-clean and numpydoc-compliant will
be rejected at commit time — write it to standard up front (see
[coding_standards.md](coding_standards.md)).

**Careful with `ruff check` run standalone (outside pre-commit):**
`ruff.toml` sets `fix = true` and `unsafe-fixes = true`, so a bare
`ruff check <path>` silently rewrites files — there is no separate `--fix`
opt-in in this repo. Use `ruff check --no-fix <path>` (or `--diff`) for a
read-only report.

## mypyc compiled build

ARIEL can be compiled with `mypyc` for speed (opt-in, off by default),
triggered by the `ARIEL_COMPILE_MYPYC=1` env var via [setup.py](../../setup.py).
There is currently no nox session for this (`tests_compiled`, referenced by
older docs, does not exist in `noxfile.py`) — you won't need this for normal
development, only when validating the compiled path manually.
