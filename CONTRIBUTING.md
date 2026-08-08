# Contributing

## Local setup

Follow the development setup in the [README](README.md#development), then run:

```bash
ruff check .
QT_QPA_PLATFORM=offscreen python -m pytest -q
```

For UI changes, also launch `anasim` and inspect the affected workflow
interactively.

## Guidelines

- Keep changes focused and use short imperative commit subjects.
- Add behavior-level tests for simulation changes.
- Cite primary literature for physiology or pharmacology changes.
- Document material model adaptations and their clinical rationale.
- Update `docs/REFERENCES.md` when model sources or material adaptations change.
- Update `CHANGELOG.md` for user-visible changes.
- Use direct technical language in user-facing text. Define unfamiliar terms and
  state model limits without promotional wording.
