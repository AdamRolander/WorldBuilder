# Contributing

Thanks for looking at WorldBuilder. Contributions of every size are welcome, and a lot of
useful work needs no GPU at all: placement, the room and its textures, the viewer, exporters,
plugins and docs all run on a laptop from scenes on disk.

## Ways to Help

- **Try it on your own photo** and open an [issue](https://github.com/AdamRolander/WorldBuilder/issues)
  with what broke. Reports from cards other than an RTX 5090 are especially useful
  ([Hardware](docs/HARDWARE.md#reports-from-other-machines)).
- **Pick up an issue** labelled `good first issue` or `no-gpu`.
- **Build or test a plugin** for a tool you use. [Plugins](docs/plugins/README.md) lists the
  targets and what each needs.
- **Improve the docs.** If a setup step tripped you up, a PR that fixes the step is the best
  kind of bug report.

## Ground Rules

- One pull request per change. Explain *why* in the description; the code should already
  say *what*.
- Prefer general rules grounded in the photo's evidence (depth, masks, normals, colour
  statistics) over special cases for a label or a scene. A fix that only helps one photo
  usually hurts another.
- Anything that touches geometry or a placement heuristic needs a test in `tests/` that fails
  without the change. CPU tests run in CI; GPU behaviour is checked by re-running a set of
  photos and comparing the runs with `scripts/benchmark.py` ([Benchmarks](docs/BENCHMARKS.md)).
  Put the before/after numbers in the PR.
- Prompt changes go in `src/prompts.py` only; both detector backends read it. Run
  `scripts/bench_detectors.py` before and after.
- Keep the heavy stacks out of `pyproject.toml`; they are installed per
  [Setup](docs/SETUP.md) because they need specific indexes and source builds.
- Never commit model weights, photos you don't have the rights to share, or `.env`.

## Development Setup (CPU, No Models)

```bash
conda create -n wb-dev python=3.11 && conda activate wb-dev
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,cpu,robotics]"
pytest
ruff check .
```

The full GPU install is in [Setup](docs/SETUP.md). How the code is organised is in
[Architecture](docs/ARCHITECTURE.md).

## Reporting Bugs

Please include the photo (or a similar one), `detected_objects.json`,
`segmentation_results.json`, `3d_models/layout.json` and the console log.
Those four files answer most questions without a GPU.

Security problems: please report them privately, see [SECURITY.md](SECURITY.md).

## Code of Conduct

Be kind, assume good faith, and keep discussions about the work. The
[Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/)
applies.
