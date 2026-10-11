# Benchmarks: how a change is judged

WorldBuilder has no ground-truth 3D for the photos it runs on, so quality is
measured against the one thing that is known: the photo itself. Speed,
memory and size are recorded by the pipeline as it runs. One script turns
both into a table and compares two runs.

```bash
python scripts/benchmark.py outputs/<run>                               # every scene in a run folder
python scripts/benchmark.py outputs/<run> --appearance                  # + re-render the photo's view
python scripts/benchmark.py outputs/<run> --baseline outputs/<earlier>  # compare; exit 1 on a regression
```

A *run* is a folder of scene directories written by `main.py`. The script
writes `benchmark.json` and `benchmark.md` next to them.

## What is measured

### Placement (CPU, from `scripts/audit_scene.py`)

| column | meaning | good |
| --- | --- | --- |
| IoU mean / med | each placed mesh is projected through the recovered camera and its silhouette compared with the object's own segmentation mask; mean and median intersection-over-union over the scene's objects | higher. 0.5–0.6 is typical; 1.0 would mean every mesh covers exactly the pixels of its object |
| IoU<0.3 | objects whose mesh mostly misses their mask | lower |

The masks come from the photo and are independent of where stage 4 puts
things, so this catches a placement rule that moves objects somewhere
"more physical" but away from where the photo shows them.

### Speed (`<scene>/metrics.json`, written by `src/profiling.py`)

| column | meaning |
| --- | --- |
| detect s, segment s | stages 1 and 2 |
| first obj s | seconds from the start of the run until the first reconstructed object exists (includes model loading) |
| obj p50 / p90 s | SAM 3D seconds per object: median and 90th percentile |
| SAM 3D s | all objects |
| stage 4 s | layout + placement + room + world export (CPU) |
| exports s | viewer, `scene.glb`, `scene_lite.glb` (CPU) |
| total s | wall clock for the scene |

`metrics.json` has the full breakdown (`seconds` per phase, including
`assemble.layout`, `assemble.placement`, `assemble.room`,
`assemble.export_world`, `finalize.*`). Replaying stage 4 on a finished
scene updates only the phases it re-ran.

### Memory

| column | meaning |
| --- | --- |
| VRAM peak GB | highest memory in use **on the device** while the scene was processed, sampled four times a second. Includes other processes on the card (the local VLM server is one) and PyTorch's cache, so it is the number that decides whether a card is big enough |
| this proc GB | highest amount this process had *allocated* (PyTorch's own counter); lower than the device figure by the allocator's cache |

`metrics.json` also has the device peak per phase.

### Size

`lite MB` is `scene_lite.glb`, the file viewers and DCC tools load.

### Appearance (`--appearance`; needs `pyrender` and EGL or a display)

The finished scene is rendered, unlit, from the photo's own camera and
compared with the photo:

| column | meaning | good |
| --- | --- | --- |
| coverage | share of the frame where the scene has any geometry | higher |
| colour err | mean absolute RGB difference between render and photo over covered pixels, 0–1 | lower |
| obj sharp | gradient energy of the render inside the object masks, relative to the photo's | closer to 1 |
| room sharp | the same outside the object masks | closer to 1 |

These are from the photo's viewpoint only; they say nothing about how the
scene looks from behind. They are the right check for texture work, and a
coarse one for everything else.

## Comparing two runs

With `--baseline` every cell shows the change, and the exit status is 1 if a
guarded metric got worse by more than its tolerance:

| metric | tolerance |
| --- | --- |
| IoU mean | −0.02 |
| IoU<0.3 | +2 objects |
| stage 4 s | +50 % |
| obj p50 s | +25 % |
| VRAM peak GB | +1.0 |
| lite MB | +50 % |
| obj sharp | −0.05 |
| colour err | +0.01 |

Override with `--tolerances '{"mean_iou": 0.05}'`. Timings on a shared
machine vary by tens of percent between runs, which is why those tolerances
are loose; treat a time regression as a prompt to re-run, not as proof.

To try a change to stage 4 without a GPU and without overwriting a run:

```bash
python scripts/clone_scene.py outputs/<run>/* --into outputs/<run>-try   # a few MB: meshes are symlinked
python scripts/replay_assembly.py outputs/<run>-try/*
python scripts/benchmark.py outputs/<run>-try --baseline outputs/<run> --appearance
python scripts/make_gallery.py outputs/<run>-try --before outputs/<run>   # pictures, side by side
```

## What the numbers do not tell you

* A mesh can match its mask and still be the wrong shape from the side.
  Look at the side and top views in `scripts/render_scene.py`'s contact
  sheet.
* The IoU is computed per object and averaged, so a scene with many small
  objects is dominated by them.
* Nothing here measures detection recall (objects the detector never
  listed). `scripts/bench_detectors.py` does that against a second detector.
