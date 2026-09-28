# Development

[README](../README.md) · [Architecture](architecture.md) · [Troubleshooting](troubleshooting.md)

## Environment

Run commands from the repository root. Use uv for dependency management and execution:

```console
uv sync --locked
uv run --locked python -m restaurant_booking --help
```

`uv sync --locked` installs the default development dependencies as well as the application. `.python-version` selects Python 3.12; `pyproject.toml` supports Python 3.12 and 3.13. The application relies on the root-level `tools.py` and `restaurants.json`, so a standalone wheel is not the supported runtime distribution.

The interactive CLI requires `GROQ_API_KEY`, provided through `.env` or the environment. Langfuse tracing is optional and activates only when both project keys are configured; leaving either key blank runs without tracing. The Langfuse dependency is still installed by the same `uv sync --locked` command. Linting, formatting, help, version output, and tests do not require credentials. Installing or updating dependencies may require network access; the checks themselves run locally.

## Source layout

```text
src/restaurant_booking/
    __init__.py          Package version
    __main__.py          python -m entry point
    cli.py               Terminal interaction and tracing lifecycle
    config.py            Environment-backed settings
    graph.py             Nodes, routing, and model instructions
    restaurant_tools.py  Validated adapters around the supplied tools
    responses.py         Reference matching and verified response templates
    state.py             Typed conversation and booking state
tests/
    __init__.py          Test package
    helpers.py           Scripted model and shared test support
    test_*.py            Unit and graph integration tests
docs/
    architecture.md      Exact runtime graph and behavior
    development.md       Contributor workflow
    troubleshooting.md   Operational guidance
    assessment/          Challenge documents
```

Keep application code, prompts, tests, and documentation in English. Add behavior to the existing modules when it fits; new infrastructure should solve a concrete need.

## Checks

Run the same checks before submitting a change:

```console
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked python -m unittest discover -s tests -t . -v
```

To apply formatting to application and test code:

```console
uv run --locked ruff format .
```

Ruff's configured scope covers `src/` and `tests/`; the supplied `tools.py` is excluded. The editor and Git configuration establish UTF-8 text, consistent indentation, and line endings without changing the supplied assessment inputs.

The [CI workflow](../.github/workflows/ci.yml) is configured to run lint, formatting, the offline test suite, and CLI smoke checks on Windows and Linux with Python 3.12 and 3.13. It requires no API secrets and makes no inference or tracing requests. Workflow setup and dependency installation need network access; the checks run locally in each runner.

For a focused test while developing:

```console
uv run --locked python -m unittest tests.test_booking_flow -v
```

Use `tests.helpers` for the shared scripted model and tool-call builder. Tests exercise the real graph and adapters while isolating catalogue mutations in memory. Extend coverage around a changed boundary, such as ambiguity, unavailable slots, confirmation, or rate-limit recovery. Scripted responses do not validate Groq's interpretation of arbitrary natural language.

`tests.test_assessment_integrity` verifies the SHA-256 hashes of the original tools and restaurant data. The Git attributes also disable text normalization for those files, preserving their bytes across checkouts.

## Change boundaries

- Keep root-level `tools.py` and `restaurants.json` unchanged, including their contents and encoding. Extend the adapters in `src/restaurant_booking/` instead.
- Keep booking execution behind the explicit confirmation boundary. A model tool request prepares a candidate; a later affirmative user reply executes it.
- Render names, times, and prices only from returned records. Refresh availability before offering alternatives after a conflict.
- Keep `.env`, credentials, traces, generated caches, and virtual environments out of version control. `.env.example` contains placeholders only.
- Update [architecture.md](architecture.md) whenever graph routes or state responsibilities change. Compare diagram node IDs and edges with `build_graph(...).get_graph()`; helper calls are not separate graph nodes.

The PDFs in [assessment/](assessment/) are reference documents, not generated build output. Repository cleanup relocates them without changing their contents.

## Dependency updates

Direct application and development dependencies are pinned in `pyproject.toml`; `uv.lock` records their resolved environment. Normal setup uses `--locked` so an unexpected dependency change fails visibly.

For an intentional update, change the relevant pin, run `uv lock`, then `uv sync --locked` and the full checks above. Review both `pyproject.toml` and `uv.lock` together. Avoid unrelated dependency upgrades when fixing conversation behavior.
