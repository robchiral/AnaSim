# Contributing

## Setup and checks

```bash
git clone https://github.com/robchiral/AnaSim.git
cd AnaSim
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
ruff check .
python -m pytest -q
```

The hosted app runs the same code in Pyodide. Test it there too:

```bash
npm install --no-save "pyodide@$(python scripts/build_web.py --pyodide-version)"
node scripts/pyodide_tests.mjs -q
```

To measure performance, run `python scripts/run_benchmarks.py --bench engine` (add
`--profile` for a breakdown).

## Interface changes

The local and hosted versions share `anasim/web_assets/` and `anasim/web.py`.
Try changes with `anasim`, and serve the hosted build with
`python scripts/build_web.py --serve`.

Regenerate the README demo if its appearance or controls change:

```bash
python -m pip install playwright pillow
playwright install chromium
python scripts/capture_demo.py
```

## Guidelines

- Keep changes focused, with short imperative commit subjects.
- Test simulation changes by running `SimulationEngine` through a clinical
  sequence. Use published data to set expected ranges where available.
- Cite sources for model changes in `docs/REFERENCES.md`, noting what AnaSim
  adapts or calibrates.
- Add user-visible changes to `CHANGELOG.md`.
- Write user-facing text plainly and state model limits.
