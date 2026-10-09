# WorldBuilder

Turn a single photo of a room into a navigable 3D scene.

An image goes in; a VLM lists the object types it sees, SAM 3 segments every
instance by text prompt, SAM 3D Objects lifts each mask into a posed,
textured mesh, and the scene point map (MoGe, computed once) gives the
floor, gravity direction, walls and ceiling — textured from the photo — into
which the objects are placed with support-aware snapping. You get per-object
meshes, a single GLB, and a self-contained WebXR viewer; Blender and Unreal
bridges pull the same scene over HTTP.

```
image ─▶ [1] detect ─────▶ [2] segment ─────▶ [3] reconstruct ─────▶ [4] layout + assemble ─▶ viewer.html
          Gemini 2.5 Flash    SAM 3 (text +      SAM 3D Objects         floor/gravity/walls/     scene.glb
          or local Qwen3-VL   box prompts;       (mask → mesh + pose;   ceiling from the shared   room.ply
                              floor/wall/        one shared MoGe        point map; support graph;
                              ceiling masks)     point map)             snap; collisions
```

Two ways to run it: a CLI (`main.py`) for batch/single images, and a web app
(`webapp/`) that takes an upload, streams progress, and serves scenes to the
viewer and to DCC plugins.

> **State of the project (Sept 2026).** The `audit-and-roadmap` branch is a
> substantial clean-up and extension of the demo-day pipeline. Start with
> [docs/AUDIT.md](docs/AUDIT.md) (what changed and why),
> [docs/ROADMAP.md](docs/ROADMAP.md) (what's next, with ticket ids) and
> [docs/VALIDATION.md](docs/VALIDATION.md) (how to verify on the GPU).

---

## Contents

- [Requirements](#requirements)
- [Setup](#setup) — the long part, ~1–2 hours end to end
- [Running it](#running-it)
- [Detector routing: Gemini vs. local Qwen3-VL](#detector-routing-gemini-vs-local-qwen3-vl)
- [Outputs](#outputs)
- [Blender / Unreal](#blender--unreal)
- [Remote access via ngrok](#remote-access-via-ngrok-for-a-vr-headset-off-lan)
- [Development](#development)
- [Repo layout](#repo-layout)
- [Troubleshooting](#troubleshooting)
- [Documentation index](#documentation-index)

---

## Requirements

- Linux x86-64 (upstream SAM 3D Objects is linux-64 only).
- One NVIDIA GPU. Developed on an RTX 5090 (32 GB, driver 580.x, CUDA
  12.8). Measured peak is 19 GB allocated for SAM 3D and ~21 GB for the
  local VLM, one at a time, so a **24 GB card (RTX 3090/4090) should work
  but is untested**; 16 GB is not enough. Details, measurements and what
  to do on a borderline card: [docs/HARDWARE.md](docs/HARDWARE.md).
  Everything after reconstruction (placement, room, viewer, exports,
  Blender/MuJoCo) runs without a GPU.
- ~60 GB free disk: ~15 GB of model checkpoints, and outputs run 0.5–1.5 GB
  per scene (the shareable `scene_lite.glb` is 7–23 MB).
- Conda or mamba.
- A Hugging Face account, because **both** SAM checkpoint repos are gated and
  need manual access approval (can take a day — request these first).
- A Gemini API key, if you want the cloud detector. Not needed for the
  fully local path.

### The VRAM dance

Three large models (Qwen3-VL 8B, SAM 3, SAM 3D Objects) cannot fit on one
card at once, so the pipeline keeps exactly one of them on the GPU at a time and
parks the others in CPU RAM between stages. You'll see `Moving model: cuda →
cpu` in the logs — that's working as intended, and it costs a few seconds per
stage transition. This is why the VLM lives in a separate process behind an
HTTP server with `/unload` and `/reload` endpoints, and why `main.py` wraps
each stage in `_to_device('cuda')` / `_to_device('cpu')`.

---

## Setup

### 0. Request checkpoint access first

Both are gated on Hugging Face and approval is not instant:

- https://huggingface.co/facebook/sam3
- https://huggingface.co/facebook/sam-3d-objects

Then authenticate locally:

```bash
pip install 'huggingface-hub[cli]<1.0'
hf auth login
```

### 1. Clone WorldBuilder and its vendored dependencies

The three upstream repos are **not** committed to this repo (they're large,
have their own git history and their own licences) — clone them into the
project root by these exact names, since the code imports them by path:

```bash
git clone git@github.com:AdamRolander/WorldBuilder.git
cd WorldBuilder

git clone https://github.com/facebookresearch/sam3.git             sam3_repo
git clone https://github.com/facebookresearch/sam-3d-objects.git   sam3d_objects_repo
git clone https://github.com/facebookresearch/pytorch3d.git        pytorch3d
```

Commits this was last known to work against:

| directory            | upstream                                | commit    |
| -------------------- | --------------------------------------- | --------- |
| `sam3_repo`          | facebookresearch/sam3                   | `7b89b8f` |
| `sam3d_objects_repo` | facebookresearch/sam-3d-objects         | `e19b169` |
| `pytorch3d`          | facebookresearch/pytorch3d              | `f5f6b78` |

If something breaks after a fresh clone, `git -C <dir> checkout <commit>` is
the first thing to try.

### 2. Environment A — `worldbuilder-main` (pipeline + webapp)

This runs everything except the local VLM: SAM 3, SAM 3D Objects, the layout
and assembly stages, the viewer, and the web app.

> **Heads-up on the deviation from upstream.** SAM 3D Objects'
> `environments/default.yml` pins torch 2.5.1 + CUDA 12.1. That does **not**
> work on an RTX 5090 (sm_120 requires CUDA 12.8), so this env runs
> **torch 2.8.0+cu128** with `pytorch3d` built from source against it.
> `docs/torch_versions_cu121_backup.txt` records the old cu121 pin set.
> If your GPU is Ada or older, the upstream cu121 path is likely smoother —
> follow `sam3d_objects_repo/doc/setup.md` verbatim instead of the versions
> below.

```bash
conda create -n worldbuilder-main python=3.11
conda activate worldbuilder-main

# PyTorch first — everything else compiles against it.
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

# SAM 3 (editable, imported from src/segmentation.py via sys.path)
pip install -e ./sam3_repo

# SAM 3D Objects + its inference deps. See sam3d_objects_repo/doc/setup.md;
# on cu128 you install these without the cu121 PIP_EXTRA_INDEX_URL pins.
pip install -e './sam3d_objects_repo[dev]'
pip install -e './sam3d_objects_repo[inference]'
./sam3d_objects_repo/patching/hydra     # upstream hydra patch, still required

# pytorch3d from source (no cu128 wheels exist)
pip install -e ./pytorch3d

# WorldBuilder's own thin layer (+ dev tools)
pip install -r requirements.txt
pip install -e ".[dev]"
```

Verify:

```bash
python -c "import torch, sam3, sam3d_objects, pytorch3d, kaolin; \
print(torch.__version__, torch.cuda.is_available())"
# → 2.8.0+cu128 True
pytest            # CPU-only test suite, ~5 s
```

Known-good versions in this env: `numpy 1.26.4` (do **not** upgrade to 2.x —
open3d and kaolin break), `transformers 4.39.3`, `open3d 0.19.0`,
`kaolin 0.18.0`, `gsplat 1.5.3`, `spconv-cu121 2.3.8`.

### 3. Environment B — `worldbuilder-vlm` (local VLM server, optional)

Separate env because Qwen3-VL needs `transformers` 5.x, which conflicts with
the 4.39 that SAM 3D Objects pins. Skip this entirely if you only ever use
the Gemini detector.

```bash
conda create -n worldbuilder-vlm python=3.11
conda activate worldbuilder-vlm
pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -r vlm_server/requirements.txt
```

The Qwen3-VL-8B weights (~17 GB) download from Hugging Face on first server
start — no access request needed, but budget the time.

### 4. Download checkpoints

**SAM 3** → `checkpoints/sam3.pt` (~3.4 GB):

```bash
mkdir -p checkpoints
hf download facebook/sam3 sam3.pt --local-dir checkpoints
```

**SAM 3D Objects** → `sam3d_objects_repo/checkpoints/hf/` (~12 GB):

```bash
cd sam3d_objects_repo
TAG=hf
hf download --repo-type model --local-dir checkpoints/${TAG}-download \
  --max-workers 1 facebook/sam-3d-objects
mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
rm -rf checkpoints/${TAG}-download
cd ..
```

The result must contain `sam3d_objects_repo/checkpoints/hf/pipeline.yaml` —
that's the path `src/reconstruction_3d.py` loads. MoGe (`Ruicheng/moge-vitl`)
is fetched automatically by SAM 3D on first run.

### 5. Configure `.env`

Create `.env` in the project root (it's gitignored — never commit your key):

```bash
GEMINI_API_KEY=your-key-from-https://aistudio.google.com/apikey   # only for the cloud detector
WORLDBUILDER_DETECTOR=qwen          # "qwen" = fully local, "gemini" = cloud
TORCH_HOME=/path/to/torch/cache     # optional
TORCH_HUB_OFFLINE=1                 # optional: avoid hub checks on every run
```

| variable                       | default                 | meaning                                              |
| ------------------------------ | ----------------------- | ---------------------------------------------------- |
| `GEMINI_API_KEY`               | —                       | Required when the detector is `gemini`.               |
| `WORLDBUILDER_DETECTOR`        | `gemini`                | `gemini` \| `qwen` (aliases: `vlm`, `local`).         |
| `WORLDBUILDER_VLM_URL`         | `http://127.0.0.1:8765` | Where the local VLM server lives.                     |
| `WORLDBUILDER_GEMINI_MODEL`    | `gemini-2.5-flash`      | Cloud model id.                                       |
| `WORLDBUILDER_SHARED_POINTMAP` | `1`                     | `0` recomputes MoGe per object (old behaviour).       |
| `WORLDBUILDER_STRETCH_STRUCTURAL` | unset                | `1` stretches pillars/beams to ceiling height (opt-in). |
| `WORLDBUILDER_OUTPUTS`         | `outputs/`              | Where the webapp reads/writes scenes.                 |

### 6. Smoke test

```bash
conda activate worldbuilder-main
python main.py --image living_room.jpg --detector gemini     # or start the VLM server and use --detector qwen
```

First run takes a few minutes (model loads dominate). Success looks like a
new `outputs/living_room_g/` containing `viewer.html`, `scene.glb` and
`3d_models/layout.json`.

---

## Running it

### CLI

```bash
conda activate worldbuilder-main

python main.py                                   # batch: every image in test_images/
python main.py --image path/to/photo.jpg         # single image
python main.py --image photo.jpg --detector qwen # override .env for this run
python main.py --input-dir my_photos --output my_outputs
python main.py --force                           # reprocess instead of skipping done scenes
python main.py --image photo.jpg --no-room       # objects only, no floor/walls/ceiling
python main.py --image photo.jpg --no-structural # skip SAM 3 floor/wall/ceiling masks (geometry-only layout)
```

Output goes to `outputs/<image_stem>_<g|q>/`, where the suffix records which
detector produced it — so a Gemini run and a Qwen run of the same photo sit
side by side and `--force` only clobbers the matching one. Batch mode skips
scenes that already have all three result JSONs and writes a
`timing_report_*.csv` with per-image timings and failure counts.

View a finished scene:

```bash
python -m http.server -d outputs/living_room_g 8000
# → http://localhost:8000/viewer.html
```

### Web app

Terminal 1 — the local VLM server (skip if using Gemini):

```bash
conda activate worldbuilder-vlm
python vlm_server/server.py                          # Qwen3-VL-8B, binds 127.0.0.1:8765
python vlm_server/server.py --model Qwen/Qwen2.5-VL-7B-Instruct   # any HF image-text model
```

Terminal 2 — the web app:

```bash
conda activate worldbuilder-main
python -m webapp.server                # binds 0.0.0.0:5174
# → http://localhost:5174/
```

Reach it from:

- **this machine:** http://localhost:5174
- **a headset/phone on the same LAN:** `http://<lan-ip>:5174` — find the IP with
  `hostname -I | awk '{print $1}'`. (Off-LAN? See
  [ngrok](#remote-access-via-ngrok-for-a-vr-headset-off-lan).)

Upload an image; the frontend polls `/api/jobs/<id>` and advances a 5-stage
progress display, then drops you into the generated viewer. Only one job runs
at a time (the GPU can't be shared), and job state is in memory — restart the
server and in-flight jobs are lost, though finished scenes on disk are still
served.

API surface (`webapp/server.py`):

| endpoint                       | purpose                                                        |
| ------------------------------ | -------------------------------------------------------------- |
| `GET  /api/health`             | liveness                                                        |
| `GET  /api/vlm-status`         | proxies the VLM server's `/health`                              |
| `GET  /api/demos`              | pre-baked scenes in `outputs/` with a `viewer.html`             |
| `POST /api/upload`             | form fields `image` (+ optional `detector`) → `{job_id, scene_id}` |
| `GET  /api/jobs/<id>`          | `{status, stage 1–5, scene_id, error}`                          |
| `GET  /api/scenes/<id>`        | manifest: objects, layout, failures, file URLs                  |
| `GET  /api/scenes/<id>/glb`    | the whole scene as one GLB (built on demand)                    |
| `GET  /outputs/<path>`         | static access to generated scenes                               |

Passing a known `scene_id` as a `job_id` returns `complete` immediately —
that's the demo fallback seam for showing a pre-baked scene
(`http://localhost:5174/?demo=your_scene_g`).

### Utilities

```bash
python scripts/regen_viewer.py outputs/scene_a outputs/scene_b   # rebuild viewer.html + scene.glb after viewer edits
python -m src.scene_export outputs/scene_a                       # just the GLB
python -m src.debug_masks --image photo.jpg --output-dir outputs/scene_a   # overlay masks on the photo
python scripts/bench_detectors.py --offline outputs outputs_og2  # local-VLM recall vs Gemini on past runs
python scripts/replay_layout.py demo_day/lr2.webp --masks outputs/lr2_q/masks   # layout stage only, CPU, seconds
python scripts/replay_assembly.py outputs/scene_a                # redo placement + room + exports on CPU (no GPU run)
python scripts/audit_scene.py outputs/scene_a                    # score placement against the photo, write audit.png
python scripts/render_scene.py outputs/scene_a                   # headless contact sheet: photo view, orbit, top, side
python -m src.scene_export outputs/scene_a --lite                # decimated GLB with baked textures
python -m src.mujoco_export outputs/scene_a --stabilize --check  # MuJoCo model (see integrations/mujoco)
python main.py --image photo.jpg --resume                        # reuse detection + masks already in the scene dir
python scripts/bench_detectors.py --live demo_day --save bench.json        # same, live against the VLM server
```

---

## Detector routing: Gemini vs. local Qwen3-VL

Stage 1 asks a vision model to list the object types worth reconstructing.
Two interchangeable backends implement the same one-method interface
(`detect_objects(image_path) -> List[Dict]`), so swapping is a single env var
or `--detector`:

```bash
WORLDBUILDER_DETECTOR=gemini   # src/object_detection.py     — Gemini 2.5 Flash (default)
WORLDBUILDER_DETECTOR=qwen     # src/local_vlm_detection.py  — local Qwen3-VL-8B via vlm_server/
```

|                | `gemini`                                 | `qwen`                                       |
| -------------- | ---------------------------------------- | -------------------------------------------- |
| Model          | `gemini-2.5-flash` via `google-genai`, JSON mode | `Qwen/Qwen3-VL-8B-Instruct` via transformers (`--model` to swap) |
| Runs where     | Google's API                             | `vlm_server/server.py`, localhost:8765        |
| Needs          | `GEMINI_API_KEY`, network                | ~17 GB VRAM while active, second conda env    |
| Latency        | ~2–5 s                                   | ~15–40 s incl. GPU wake                       |
| Failure mode   | 429/503 → 5 retries, exponential backoff | malformed JSON → 3 attempts (greedy, then sampled), salvage parse |
| Privacy        | image leaves the machine                 | fully offline                                 |

**Both backends read the same prompt** from `src/prompts.py`; edit it there.
It asks for short everyday labels (they become SAM 3's text prompts), a
short description, `expected_instances`, and an optional `bbox_2d` per type
(Qwen3-VL's boxes are used as a SAM 3 fallback prompt when text fails).
`src/detection_postprocess.py` then normalises labels, drops people,
architecture and object *parts*, collapses duplicates, and caps the list —
the safety net for a local model that loops.

---

## Outputs

Each scene directory holds:

```
outputs/<scene>/
├── detected_objects.json          # stage 1 after clean-up (detected_objects_raw.json = as returned)
├── segmentation_results.json      # stage 2: per-instance masks, scores, boxes, prompt used
├── masks/                         # stage 2: mask PNGs (+ structural_floor/wall/ceiling.png)
├── reconstruction_results.json    # objects (pose, support, fit to the photo), removed/failed with reasons, timings, peak VRAM
├── 3d_models/
│   ├── stage3/                    # SAM 3D's raw output: model-space meshes, raw.json poses, pointmap.npz (replay input)
│   ├── <id>_<label>.ply           # one world-space mesh per object instance (vertex colours)
│   ├── room.glb                   # textured floor/walls/ceiling quads + photo relief of built-ins (named nodes)
│   ├── room.ply                   # the same shell as vertex colours (fallback for PLY-only consumers)
│   ├── layout.json                # floor, gravity, yaw, ceiling, wall evidence, scale estimate, placement diagnostics
│   ├── scene_pointmap.npz         # MoGe scene points (aligned frame) for debugging / viewers
│   └── scene_combined.ply         # all objects composed into one mesh
├── scene.glb                      # full-resolution scene (hundreds of MB): named nodes, vertex colours
├── scene_lite.glb                 # same scene, 8k triangles per object with baked textures (7–23 MB) — use this one
├── mujoco/                        # optional: scene.xml + assets (python -m src.mujoco_export)
├── scene_points.json              # small point-cloud sidecar the viewer can toggle on
└── viewer.html                    # self-contained Three.js / WebXR viewer
```

Coordinates are **scale-invariant** (consistent within a scene, not across
scenes). `layout.json → estimated_metric_scale` is a prior-based factor to
metres (camera at ~1.5 m); proper metric scale is roadmap item `PIPE-6`.
`viewer.html` loads Three.js from a CDN, so viewing needs network access
(not the GPU).

---

## Blender / Unreal / MuJoCo

`integrations/blender_worldbuilder/` is a Blender 4.2+ extension (also
installs as a legacy add-on) and `integrations/unreal/WorldBuilderBridge/`
an Unreal 5.4+ editor plugin. Both are thin HTTP clients: pick a photo, the
running web app does the work, the scene comes back as a GLB and is imported
with named objects. Setup, publishing requirements (extensions.blender.org,
Fab) and status are in [docs/INTEGRATIONS.md](docs/INTEGRATIONS.md). The
Blender add-on passes Blender's extension validation and a headless
end-to-end test (`scripts/test_blender_addon.py`); the Unreal plugin is
still an untested scaffold.

For robotics, `python -m src.mujoco_export outputs/<scene> --stabilize`
(or `GET /api/scenes/<scene>/mujoco`) writes a MuJoCo model: metric, Z-up,
textured, with collision geometry, free joints on movable objects and the
photo's camera. `integrations/mujoco/` has a Gymnasium wrapper and a
one-call way to stand a robot in the room; `--mjx` targets MJX. See
[integrations/mujoco/README.md](integrations/mujoco/README.md).

---

## Remote access via ngrok (for a VR headset off-LAN)

When the headset can't reach the GPU box directly (different networks),
tunnel the webapp:

```bash
# one-time on the GPU machine
ngrok config add-authtoken <your-token-from-ngrok.com>

# each session, alongside the webapp
./ngrok http --url=<your-static-domain>.ngrok-free.dev 5174
```

Open `https://<your-static-domain>.ngrok-free.dev` on the Quest. The first
load shows an ngrok interstitial — click "Visit Site" once (the cookie lasts
7 days). The iframe shows it again on first scene load; click through the
same way. WebXR works over the tunnel because ngrok terminates HTTPS, so no
LAN certificate setup is needed.

Free tier gives 1 GB bandwidth and 20k requests/month. A 20-object scene can
be ~100 MB, so that's roughly 10 scene loads per month. Cloudflare Tunnel is
the free-and-unlimited alternative if you have a domain.

The `ngrok` binary is gitignored — grab it from
[ngrok.com/download](https://ngrok.com/download).

---

## Development

```bash
pytest                      # CPU-only: geometry vs pytorch3d, synthetic-room layout, placement, parsing
ruff check .                # lint (config in pyproject.toml)
```

The geometry and heuristics are unit-tested without any model weights;
anything that touches SAM 3 / SAM 3D is verified manually per
[docs/VALIDATION.md](docs/VALIDATION.md) with before/after numbers.
See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Repo layout

```
main.py                     CLI entrypoint + batch runner + timing reports
requirements.txt            WorldBuilder's own deps (not the SAM stacks)
pyproject.toml              package metadata, pytest/ruff config

src/
  prompts.py                the one detection prompt both backends use
  object_detection.py       stage 1a: Gemini detector (JSON mode)
  local_vlm_detection.py    stage 1b: HTTP client for the local VLM server
  detection_postprocess.py  stage 1c: normalise, dedupe, drop people/architecture/parts
  segmentation.py           stage 2:  SAM 3 text + box prompts, NMS, mask hygiene
  reconstruction_3d.py      stage 3:  SAM 3D Objects with a shared point map; writes the stage-3 cache
  stage3_cache.py           what stage 3 leaves behind so stage 4 can be replayed on a CPU
  scene_assembly.py         stage 4:  layout → placement → room → exports, from the cache
  room_layout.py            stage 4a: floor, gravity, yaw, walls, ceiling from the point map
  placement.py              stage 4b: depth refit, contact support, IoU-checked moves, duplicate removal
  room_texture.py           stage 4c: textured shell quads + photo relief of built-ins
  room_generator.py         legacy box/label placement (fallback when there is no point map)
  mesh_bake.py              decimation + vertex-colour → texture baking (scene_lite.glb)
  mujoco_export.py          MuJoCo / MJX model export with a stability check
  geometry.py               pose math shared by stage 4 (checked against pytorch3d)
  scene_export.py           GLB scene export
  viewer_generator.py       emit the Three.js / WebXR viewer
  debug_masks.py            dev tool: overlay masks on the source image

vlm_server/server.py        local VLM detection server (separate conda env)
webapp/
  server.py                 Flask API + job queue + scene/GLB endpoints
  static/                   React (UMD + Babel standalone, no build step)
integrations/
  blender_worldbuilder/     Blender extension (manifest + add-on)
  unreal/WorldBuilderBridge Unreal editor plugin (uplugin, C++ stub, Python bridge)
  mujoco/                   Gymnasium wrapper + robot attachment for exported scenes
scripts/
  regen_viewer.py           rebuild viewer.html + scene.glb for existing output dirs
  bench_detectors.py        local-VLM recall benchmark vs Gemini
  replay_layout.py          layout stage only (CPU MoGe) for fast tuning
  plot_scene.py             plan-view diagnostic PNG of a finished scene
  replay_assembly.py        re-run stage 4 on a finished scene (CPU, seconds)
  audit_scene.py            score placement against the photo's masks and point map
  render_scene.py           headless contact-sheet renders (pyrender + EGL)
  test_blender_addon.py     end-to-end add-on test, run inside `blender --background`
tests/                      CPU-only pytest suite
docs/                       ARCHITECTURE, AUDIT, ROADMAP, VALIDATION, INTEGRATIONS, HARDWARE, CONTRIBUTOR_TASKS

checkpoints/                SAM 3 weights            (gitignored)
sam3_repo/                  upstream clone           (gitignored)
sam3d_objects_repo/         upstream clone + weights (gitignored)
pytorch3d/                  upstream clone           (gitignored)
outputs/                    generated scenes         (gitignored)
```

---

## Troubleshooting

**`CUDA out of memory` mid-pipeline.** Usually a model failed to park on CPU,
or another process holds the GPU (`nvidia-smi`). Restart the webapp/CLI
process; if it's reproducible, check the `Moving model: cuda → cpu` lines in
the log to see which stage didn't release.

**`VLM server unreachable at http://127.0.0.1:8765`.** The server isn't
running, or is still loading the model (~1 min). `curl
localhost:8765/health` — `{"ok": true}` means it's ready.

**`GEMINI_API_KEY not found`.** `.env` is missing, or you started the process
from a directory other than the project root (`load_dotenv()` resolves `.env`
relative to the working directory) — or you meant `--detector qwen`.

**`KeyError: 'CONDA_PREFIX'` from `notebook/inference.py`.** Fixed on this
branch (the path is derived from the interpreter), but if you run the
upstream notebook directly you need an activated conda shell.

**"Textured room failed … using box room".** The layout stage fell back;
`3d_models/layout.json` and the console say why (usually no floor visible
or no valid intrinsics). The scene is still complete.

**SAM 3 checkpoint 404 / gated repo error.** Access wasn't approved yet, or
`hf auth login` hasn't run in this shell.

**`RuntimeError: Not compiled with GPU support` from pytorch3d.** It was built
on a machine without a visible GPU. Rebuild `pip install -e ./pytorch3d` on the
GPU box.

**`pipeline.yaml` not found.** The SAM 3D download landed in the wrong place —
the `mv` step in setup §4 is easy to miss.

**numpy 2.x errors after installing something new.** Something upgraded numpy.
`pip install 'numpy==1.26.4'` and reinstall whatever pulled it.

---

## Documentation index

| document | what it is for |
| --- | --- |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | data flow, file formats, frames/units, extension points |
| [docs/AUDIT.md](docs/AUDIT.md) | the September 2026 code audit: findings, fixes, evidence |
| [docs/ROADMAP.md](docs/ROADMAP.md) | every planned item with ids, blockers and a suggested order |
| [docs/VALIDATION.md](docs/VALIDATION.md) | GPU validation procedure and before/after measurements |
| [docs/INTEGRATIONS.md](docs/INTEGRATIONS.md) | Blender extension and Unreal plugin: setup, publishing rules, status |
| [integrations/mujoco/README.md](integrations/mujoco/README.md) | MuJoCo / MJX export, Gymnasium wrapper, limits |
| [docs/HARDWARE.md](docs/HARDWARE.md) | measured VRAM per stage, which cards work, what runs without a GPU |
| [docs/CONTRIBUTOR_TASKS.md](docs/CONTRIBUTOR_TASKS.md) | a menu of tasks that need no GPU, with setup |
| [CONTRIBUTING.md](CONTRIBUTING.md) | how to work on the repo |

## Licence

WorldBuilder's code is MIT (see `LICENSE`). Model weights and the upstream
SAM repositories are obtained separately under their own terms (Meta's SAM
License for SAM 3 / SAM 3D Objects; MIT for MoGe; BSD for PyTorch3D;
Apache-2.0 for Qwen3-VL) and are never redistributed by this project.
