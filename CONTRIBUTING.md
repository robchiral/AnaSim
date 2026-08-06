# Contributing

## Local setup

Follow the development setup in the [README](README.md#development), then run:

```bash
QT_QPA_PLATFORM=offscreen python -m pytest -q
```

For UI changes, also launch `anasim` and inspect the affected workflow
interactively.

## Guidelines

- Keep changes focused and use short imperative commit subjects.
- Add behavior-level tests for simulation changes.
- Cite primary literature for physiology or pharmacology changes.
- Document deliberate model deviations and their clinical rationale.
- Update `docs/REFERENCES.md` when model sources or calibrated deviations change.
- Update `CHANGELOG.md` for user-visible changes.
