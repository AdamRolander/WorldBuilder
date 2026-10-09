## What / why

<!-- one paragraph; link the ROADMAP ticket id (PIPE-#, CODE-#, PUB-#, DCC-#, RB-#) -->

## Checklist

- [ ] `pytest` and `ruff check .` pass
- [ ] Geometry / heuristic changes have a test that fails without the change
- [ ] GPU-affecting changes: before/after numbers per `docs/VALIDATION.md` §2 (photo, detector, counts, wall-clock, layout notes) and a screenshot of the viewer or `plan_view.png`
- [ ] Prompt changes: `scripts/bench_detectors.py` before/after
- [ ] Docs updated (README / ARCHITECTURE / CHANGELOG as appropriate)
- [ ] No model weights, SAM upstream code, or API keys added to the repo
