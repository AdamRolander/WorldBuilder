## What / why

<!-- one paragraph; link the issue this addresses -->

## Checklist

- [ ] `pytest` and `ruff check .` pass
- [ ] Geometry / heuristic changes have a test that fails without the change
- [ ] GPU-affecting changes: before/after numbers (photo, detector, object counts, wall-clock, `scripts/audit_scene.py` scores) and a screenshot of the viewer or `plan_view.png`
- [ ] Prompt changes: `scripts/bench_detectors.py` before/after
- [ ] Docs updated (README / ARCHITECTURE / CHANGELOG as appropriate)
- [ ] No model weights, SAM upstream code, or API keys added to the repo
