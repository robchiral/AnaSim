# Contributing

## Setup

Use Python 3.10 or later and Node.js 22.

```bash
git clone https://github.com/robchiral/AnaSim.git
cd AnaSim
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

See [Architecture](docs/ARCHITECTURE.md#main-modules) for modules and step order,
and [CLI and Python usage](docs/CLI_USAGE.md) for scripts.

## Choose checks

Run native and Pyodide suites for shared Python changes.

| Change | Relevant checks |
|--------|-----------------|
| Physiology, PK/PD, or ventilation | Engine scenarios and affected subsystem tests at several step sizes |
| Browser commands or state fields | `tests/test_web.py`, `tests/test_state_semantics.py`, and both local and hosted previews |
| Settings controls | `node --test tests/test_settings.mjs` and browser checks |
| Guided scenarios | `tests/test_scenarios.py`, `tests/test_action_log.py`, and completing the scenario in the browser |
| Local server | `tests/test_local.py` in native Python |
| Documentation | Run changed examples, check links, and preserve citations and model limits |

```bash
ruff check .
python -m pytest -q
npm install --no-save "pyodide@$(python scripts/build_web.py --pyodide-version)"
node scripts/pyodide_tests.mjs -q
```

The Pyodide runner accepts pytest arguments and skips local HTTP server tests.

Base clinical response tests on published data where available; check affected
gas or drug conservation.

## Guided scenarios

Build a `Scenario` from `ScenarioStep` objects using the
[requirement helpers](anasim/scenarios/base.py). Register its builder, starting
mode, and maintenance technique with a `ScenarioSpec` in the
[registry](anasim/scenarios/__init__.py). See the
[hemorrhage example](anasim/scenarios/hemorrhage.py) and
[objective checks](docs/ARCHITECTURE.md#scenario-objectives).
Use `tci_instruction` for dosing guidance that changes when TCI is enabled.
The scenario tests complete every registered scenario through browser commands
with TCI enabled and disabled.

## Interface changes

Preview local and hosted interfaces in separate terminals.

```bash
anasim
python scripts/build_web.py --serve
```

The hosted preview runs at `http://localhost:8000/` (change it with `--port`)
and loads Pyodide packages from the CDN. Rebuild after Python changes.
Check settings, recordings, starting a new session, and the changed display.

Update the README demo with:

```bash
python -m pip install playwright pillow
playwright install chromium
python scripts/capture_demo.py
```

## Performance changes

```bash
python scripts/run_benchmarks.py --bench engine --repeat 3
```

The default benchmark uses awake and steady-state runs at `dt=0.1`; add a
scenario when needed. Compare runs with the same seeds, settings, and step
sizes, including numerical results and waveforms. Report native and Pyodide
timings separately. Use `--profile` to find slow functions and unprofiled runs
for timing.

## Documentation and changes

Keep model behavior, assumptions, calibrations, and limits in
[Architecture](docs/ARCHITECTURE.md#model-notes), inputs, units, and defaults in
[CLI usage](docs/CLI_USAGE.md#fields), and citations with brief topic labels in
[References](docs/REFERENCES.md). Add user-visible changes to [Changelog](CHANGELOG.md).

Use plain, direct language. Omit metaphors, promotional wording, obvious
explanations, and unnecessary contrasts. Document each fact once and link to it.

Describe behavior changes and validation in PRs. Use short, imperative commit subjects.
