# Contributing

Thanks for looking at WorldBuilder. The project is small and opinionated;
here is how to work on it without fighting it.

## Ground rules

* One pull request per change. Explain *why* in the description; the code
  should already say *what*.
* Anything that touches geometry or a placement heuristic needs a test in
  `tests/` that fails without the change. CPU-only tests run everywhere;
  GPU behaviour is verified manually per `docs/VALIDATION.md` and the
  before/after numbers go in the PR.
* Prompt changes go in `src/prompts.py` only — both detector backends read
  it. Run `scripts/bench_detectors.py` before and after.
* Keep the heavy stacks out of `pyproject.toml`; they are installed per
  README §2 because they need specific indexes and source builds.

## Dev setup (CPU, no models)

```bash
conda create -n wb-dev python=3.11 && conda activate wb-dev
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest
ruff check .
```

## Reporting bugs

Please include the photo (or a similar one), `detected_objects.json`,
`segmentation_results.json`, `3d_models/layout.json` and the console log.
Those four files answer most questions without a GPU.

## Code of conduct

Be kind, assume good faith, and keep discussions about the work. The
[Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/)
applies.
