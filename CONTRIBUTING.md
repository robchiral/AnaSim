# Contributing

## Setup

Use Python 3.10 or later. Node.js 22 is required for browser tests and Pyodide.

```bash
git clone https://github.com/robchiral/AnaSim.git
cd AnaSim
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

In Windows Command Prompt, use `python` in place of `python3` and activate with
`.venv\Scripts\activate.bat`.
See the [architecture guide](docs/ARCHITECTURE.md#main-modules) for the code
structure and [CLI and Python usage](docs/CLI_USAGE.md) for scripted runs.

## Validation

Run native Python and Pyodide test suites for shared Python changes. Pyodide
is the hosted app's Python runtime.

| Change | Relevant checks |
|--------|-----------------|
| Physiology, pharmacology, or ventilation | Clinical scenarios and affected subsystem tests at several step sizes |
| Browser commands or state fields | `tests/test_web.py`, `tests/test_state_semantics.py`, and local and hosted browser previews |
| Settings controls | `node --test tests/test_settings.mjs` and browser checks |
| Guided scenarios | `tests/test_scenarios.py`, `tests/test_action_log.py`, and a complete browser run of the changed scenario |
| Local server | `tests/test_local.py` in native Python |
| Documentation | Run changed examples, check links, and preserve citations and model limits |

```bash
ruff check .
mypy
python -m pytest -q -n auto
npm install --no-save "pyodide@$(python scripts/build_web.py --pyodide-version)"
node scripts/pyodide_tests.mjs -q
```

The Pyodide runner accepts pytest arguments, runs up to eight workers, and skips local HTTP server tests.

Use published data to set expected clinical responses where available. Check
gas or drug conservation when a change affects either.

## Guided scenarios

Build a `Scenario` from `ScenarioStep` objects using the
[requirement helpers](anasim/scenarios/base.py). Register its builder, starting
mode, and maintenance technique with a `ScenarioSpec` in the
[registry](anasim/scenarios/__init__.py). See the
[hemorrhage example](anasim/scenarios/hemorrhage.py) and
[objective checks](docs/ARCHITECTURE.md#scenario-objectives).
Use `tci_instruction` for guidance that changes with target-controlled infusion
(TCI). The scenario tests complete every registered scenario through browser
commands with TCI enabled and disabled.

## Interface changes

Run the local interface and a hosted-app preview in separate terminals.

```bash
anasim
python scripts/build_web.py --serve
```

The hosted preview runs at `http://localhost:8000/`; change the port with `--port`.
It requires internet access to load Pyodide packages. Rebuild after Python
changes. Check the changed interface, settings edits, recordings, and starting
a new session in both versions.

Regenerate the README demo after interface changes that affect it.

```bash
python -m pip install playwright pillow
playwright install chromium
python scripts/capture_demo.py
```

## Performance changes

```bash
python scripts/run_benchmarks.py --bench engine --repeat 3
```

The engine benchmark runs awake and steady-state simulations at `dt=0.1`.
Add a workload to exercise the change if needed. Compare timings, numerical
results, and waveforms with the same seeds, settings, and step sizes.
Report native Python and Pyodide timings separately. Use `--profile` to locate
slow functions; measure timings without the profiler.

## Documentation and pull requests

Keep model behavior, assumptions, calibrations, and limits in
[the architecture guide](docs/ARCHITECTURE.md#model-notes), inputs, units, and
defaults in [CLI and Python usage](docs/CLI_USAGE.md#fields), and citations with
brief topic labels in [Model references](docs/REFERENCES.md). Add user-visible
changes to the [changelog](CHANGELOG.md).