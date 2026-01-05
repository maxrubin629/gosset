# Repository Guidelines

## Project Structure & Module Organization
`gosset/` is the Python package with the CLI (`__main__.py`, `cli.py`), analysis helpers (`analyze.py`), and backends in `gosset/backends/` for Transformers and llama.cpp. `viewer/` is a static HTML/CSS/JS app that renders token-entropy heatmaps. `examples/` holds sample log JSON. `logs/` is the default output directory for generated runs. `requirements.txt` lists Python deps.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` - install dependencies.
- `python -m gosset generate ...` - run a Transformers-backed generation and write a log (see README for flags).
- `python -m gosset generate-llamacpp ...` - hit a llama.cpp server and log entropy lower bounds.
- `python -m gosset analyze --log logs/foo.json` - compute top tokens by entropy.
- `cd viewer && python -m http.server 8000` - serve the UI locally.

## Coding Style & Naming Conventions
Python uses 4-space indentation, snake_case, and type hints are used in core modules; keep new code aligned with `gosset/` style. CLI flags use `--kebab-case` and match `gosset/cli.py`. Viewer code uses 2-space indentation and semicolons; keep DOM/JS in `viewer/app.js` simple and dependency-free. No formatter or linter is configured, so follow existing patterns.

## Testing Guidelines
There is no automated test suite or coverage requirement yet. If you add tests, place them under `tests/` and use `test_*.py` naming so they can be run with a standard pytest workflow; include small JSON fixtures in `examples/` rather than `logs/`.

## Commit & Pull Request Guidelines
Git history only shows simple, sentence-style subjects (e.g., "Initial commit: ..."), so keep commit titles short and descriptive. PRs should state intent, list key commands run (or explain if none), and include a screenshot or short clip for viewer changes. If the log schema changes, call it out and update `examples/` accordingly.

## Configuration & Data Notes
Python 3.10+ is recommended. If you use mamba/conda, the local env name is `gpt-oss-scope`. Model weights are fetched by `transformers` at runtime; avoid committing large generated logs.
