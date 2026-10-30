# Contributing

## Setup

Use Python 3.10 or later. Browser tests and Pyodide also require Node.js 22.

```bash
git clone https://github.com/robchiral/AnaSim.git
cd AnaSim
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

See the [architecture guide](docs/ARCHITECTURE.md#main-modules) for the code
structure and [CLI and Python usage](docs/CLI_USAGE.md) for scripted runs.

## Checks

Run these commands for changes to shared Python code. They include the native
Python and Pyodide test suites. Pyodide is the Python runtime of the hosted app.

```bash
ruff check .
mypy
python -m pytest -q -n auto
npm install --no-save "pyodide@$(python scripts/build_web.py --pyodide-version)"
node scripts/pyodide_tests.mjs -q
```

Each type of change also needs the checks below.

| Change | Relevant checks |
|--------|-----------------|
| Physiology, pharmacology, or ventilation | Clinical scenarios and tests of the affected subsystem at several step sizes |
| Browser commands or state fields | `tests/test_web.py`, `tests/test_state_semantics.py`, and local and hosted browser previews |
| Settings controls or monitor display | `node --test tests/*.mjs` and browser checks |
| Guided scenarios | `tests/test_scenarios.py`, `tests/test_action_log.py`, and a complete browser run of the changed scenario |
| Local server | `tests/test_local.py` in native Python |
| Documentation | Run changed examples and check links |

Use published data to set expected clinical responses where available. Check
conservation of gas or drug when a change affects either one.

## Guided scenarios

Build a `Scenario` from `ScenarioStep` objects using the [requirement
helpers](anasim/scenarios/base.py). Register its builder, starting mode, and
maintenance technique with a `ScenarioSpec` in the
[registry](anasim/scenarios/__init__.py). The [hemorrhage
example](anasim/scenarios/hemorrhage.py) shows a complete scenario.

Use `tci_instruction` for guidance that changes with target-controlled infusion
(TCI). The scenario tests complete each registered scenario with TCI enabled and
disabled.

## Interface changes

Run the local interface and a preview of the hosted app in separate terminals.

```bash
anasim
python scripts/build_web.py --serve
```

The hosted preview runs at `http://localhost:8000/` and needs internet access to
load Pyodide. Rebuild it after Python changes. In both versions, check the
changed interface, settings edits, recordings, and starting a new session.

Regenerate the README demo after interface changes that affect it.

```bash
python -m pip install playwright pillow
playwright install chromium
python scripts/capture_demo.py
```

## Performance changes

```bash
python scripts/run_benchmarks.py --bench engine,tci,monitor --repeat 3
```

Add a workload that exercises the change if none exists. Compare timings,
numerical results, and waveforms with the same seeds, settings, and step sizes.
Report native Python and Pyodide timings separately. Use `--profile` to find
slow functions, but measure timings without it.

## Documentation

Keep each kind of information in one place:

- Model behavior and main modeling choices go in the [architecture
  guide](docs/ARCHITECTURE.md#model-notes).
- Inputs, units, and defaults go in [CLI and Python
  usage](docs/CLI_USAGE.md#fields).
- Citations with brief topic labels go in [Model
  references](docs/REFERENCES.md).
- User-visible changes go in the [changelog](CHANGELOG.md).

## License

Under Section 5 of the [license](LICENSE), you retain ownership of your original
contributions and grant Robert Chen permission to use, modify, and sublicense
them, including under paid licenses. Submit only work that you have the
authority to license, and identify any third-party material and its license.
