# Contributing

## Local setup

Use the setup and validation commands in the [README](README.md#development).

For UI changes, also launch `anasim` and inspect the affected workflow
interactively.

Measure the complete simulation loop, including monitors and TCI, with:

```bash
python scripts/run_benchmarks.py --bench engine --steps 10000 --repeat 5
```

These benchmarks use a fixed seed and 0.1-second steps. Initialization and warmup
are excluded from the elapsed time. Use `--profile` to find where time is spent.

## Guidelines

- Keep changes focused and use short imperative commit subjects.
- Add behavior-level tests for simulation changes. Prefer driving
  `SimulationEngine` through a clinical sequence over testing single equations
  or setters, and anchor bounds to a cited source where one exists.
- Cite primary literature for physiology or pharmacology changes.
- Document material model adaptations and their clinical rationale.
- Update `docs/REFERENCES.md` when model sources or material adaptations change.
- Update `CHANGELOG.md` for user-visible changes.
- Use direct technical language in user-facing text. Define unfamiliar terms and
  state model limits explicitly.
