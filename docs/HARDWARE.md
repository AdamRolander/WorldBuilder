# Hardware: what you need to run WorldBuilder

Short version: **an NVIDIA card with 24 GB of VRAM** is the realistic
minimum for the full pipeline; 32 GB is what it was developed on. Stage 4,
the viewer, the exports and the Blender/MuJoCo integrations need no GPU at
all.

## Measured (RTX 5090, 32 GB, 2026-10-08)

Peak memory per stage. "Allocated" is PyTorch's own high-water mark for
the process (`metadata.vram_peak_gb` in `reconstruction_results.json`, also
printed at the end of every run); "device" is what `nvidia-smi` showed,
which includes PyTorch's cache and ~0.4 GB of desktop.

| stage | model | peak allocated | peak on device | notes |
| --- | --- | ---: | ---: | --- |
| 1 detect (local) | Qwen3-VL-8B, bf16 | — (other process) | ~21.5 GB | parked in CPU RAM afterwards; 0 GB with the Gemini detector |
| 2 segment | SAM 3 | 3.9 GB | — | |
| 3 reconstruct | SAM 3D Objects (+ MoGe) | 19.2 GB | 22.8–23.7 GB | the stage that sets the floor; independent of object count |
| 3b metric scale (optional) | MoGe-2 ViT-L, subprocess | ~2 GB (est.) | — | <1 s on GPU; falls back to CPU when under 4 GB is free |
| 4 assemble + exports | none | 0 | 0 | CPU, 20–100 s (pose search and instance sharing dominate) + ~25 s for the lite GLB |

Only one large model is on the GPU at a time (the README's "VRAM dance").
System RAM: 62 GB on the dev box; the parked models take roughly 30 GB of
it, so treat 48 GB as the practical minimum and 64 GB as comfortable.
Disk: ~15 GB of weights plus the Hugging Face cache for the VLM (~18 GB),
and 0.8–4 GB per scene with the full-resolution meshes.

### Per phase, twelve photos (2026-10-10)

Every run now writes `metrics.json` (seconds per phase, time to the first
object, seconds per object, device-wide and per-process memory peaks; see
[BENCHMARKS.md](BENCHMARKS.md)). On twelve photos with 4–77 object
instances each (industrial halls, rooms, outdoor scenes):

| phase | seconds | device peak | notes |
| --- | ---: | ---: | --- |
| detect (local VLM) | 11–35, typically 30 | 18–22.5 GB | independent of scene size |
| segment (SAM 3) | 1–10 | 6–10 GB | includes the first load when cold |
| reconstruct (SAM 3D) | **4.5–6.7 per object** (median), 5–11 at the 90th percentile | 22–26.4 GB (17.8–20.2 GB allocated) | linear in the number of objects: 26 s for 4, 503 s for 77. This is the bottleneck |
| metric scale (MoGe-2) | 4–13 | — | subprocess |
| stage 4: layout, placement, room, world export | 4–90 | 0 | CPU; placement dominates, then photo re-projection |
| exports: viewer, GLB, lite GLB | 4–45 | 0 | CPU; the lite GLB bake dominates |
| **time to first object** | 23–85, typically 40 | | detection + segmentation + model wake-up |
| **total** | 70 s (4 objects) to 870 s (77 objects) | **26.4 GB** | |

The device peak during SAM 3D is higher than in the table above because it
includes PyTorch's cache, which grows into whatever memory is free on a
32 GB card; the allocated figure (≤ 20.2 GB) is what a smaller card has to
fit.

## What that means for cards

| VRAM | examples | verdict |
| --- | --- | --- |
| 32 GB+ | RTX 5090, A6000 (48), L40S (48), A100 | Works. This is the tested configuration. |
| 24 GB | RTX 3090 / 3090 Ti, RTX 4090, A5000, L4 (cloud), A10G (cloud, 22.3 GB usable) | **Should work, untested.** SAM 3D's 19.2 GB allocated peak fits; the local VLM at ~21 GB is tight, so close other GPU apps or use the Gemini detector. Expect it to be slower than a 5090 (a 3090 maybe 2×). The first person to run this owes the table a row (task T23). |
| 16 GB | RTX 4080, 4060 Ti 16 GB, 5070 Ti, A4000 | **Not as is.** SAM 3D does not fit. Would need half-precision or offloading inside SAM 3D (unexplored), plus the Gemini detector or a 4-bit VLM. |
| ≤ 12 GB | RTX 3060, 4070, laptops | No for the models. Fine for everything downstream of them. |

**It has to be NVIDIA.** SAM 3 hard-codes CUDA tensors, and SAM 3D depends
on CUDA-only libraries (spconv, kaolin, gsplat, pytorch3d's CUDA ops). AMD,
Intel and Apple Silicon cannot run stages 2–3. Linux x86-64 only for the
same upstream reasons; WSL2 on Windows is plausible but untested.

**The cheapest viable card** is a used RTX 3090 (24 GB). If you are
borrowing a machine: a 3090 or 4090 desktop is the target; a 4080 is not
enough.

## If the card is borderline

* Use `--detector gemini` (needs an API key and network): removes the
  ~21 GB VLM entirely, so only SAM 3D's ~19 GB matters.
* Close everything else on the GPU (browsers with hardware acceleration,
  other model servers). On the dev box a vLLM container holding 29 GB was
  the cause of every early OOM.
* Photo resolution barely matters for VRAM: SAM 3 resizes to 1008 px and
  SAM 3D crops per object.
* If stage 3 dies with CUDA OOM on a 24 GB card, set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and retry before
  concluding it does not fit; fragmentation, not the peak, is the usual
  culprit near the limit.

## No GPU at all

Everything after stage 3 runs on a laptop from a finished scene directory:

```bash
python scripts/replay_assembly.py outputs/<scene>     # layout, placement, room, exports
python scripts/audit_scene.py outputs/<scene>         # placement vs. the photo
python -m src.mujoco_export outputs/<scene> --stabilize
```

and the viewer, the Blender add-on and the MuJoCo environment only need the
files a server produced. Issues labelled `no-gpu` in the tracker fit this setup, and rent-a-GPU options are a few dollars for
an afternoon if you want to run the models once.

## Reports from other machines

| date | card | driver / CUDA | detector | photo → objects | wall-clock | peak allocated (seg / recon) | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-10-08 | RTX 5090 32 GB | 580.x / 12.8 | local Qwen3-VL-8B | bedroom 625×350 → 34 | 348 s | 3.9 / 19.2 GB | via the Blender add-on test |
| | | | | | | | *add yours* |
