# WorldBuilder

Turn a single photo of a room into a navigable 3D scene.

An image goes in; a VLM lists the objects it sees, SAM 3 segments each one by
text prompt, SAM 3D Objects lifts every mask into a posed, textured 3D asset,
and a procedural room box + floor snapping assembles them into a scene you can
open in a browser or a VR headset.

```
image ──▶ [1] object detection ──▶ [2] segmentation ──▶ [3] 3D reconstruction ──▶ [4] scene assembly
           Gemini 2.5 Flash          SAM 3                SAM 3D Objects            room box, floor snap,
           or local Qwen3-VL         (text-prompted)      (mask → mesh + pose)      collision resolve, viewer.html
```

Two ways to run it: a CLI (`main.py`) for batch/single images, and a web app
(`webapp/`) that takes an upload and streams progress while the pipeline runs.

---

## Contents

- [Requirements](#requirements)
- [Setup](#setup) — the long part, ~1–2 hours end to end
- [Running it](#running-it)
- [Detector routing: Gemini vs. local Qwen3-VL](#detector-routing-gemini-vs-local-qwen3-vl)
- [Outputs](#outputs)
- [Remote access via ngrok](#remote-access-via-ngrok-for-a-vr-headset-off-lan)
- [Repo layout](#repo-layout)
- [Troubleshooting](#troubleshooting)

---

## Requirements

- Linux x86-64 (upstream SAM 3D Objects is linux-64 only).
- One NVIDIA GPU with **≥32 GB VRAM**. Developed on an RTX 5090 (32 GB,
  driver 580.x, CUDA 12.8). 32 GB is the floor, not comfortable headroom —
  see [the VRAM dance](#the-vram-dance) below for why.
- ~60 GB free disk: ~15 GB of model checkpoints, and outputs run 100 MB–1 GB
  per scene.
- Conda or mamba.
- A Hugging Face account, because **both** SAM checkpoint repos are gated and
  need manual access approval (can take a day — request these first).
- A Gemini API key, if you want the cloud detector.

### The VRAM dance

Three large models (Qwen3-VL 8B, SAM 3, SAM 3D Objects) cannot fit in 32 GB
at once, so the pipeline keeps exactly one of them on the GPU at a time and
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

The three upstream repos are **not** committed to this repo (they're large and
have their own git history) — clone them into the project root by these exact
names, since the code imports them by path:

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

This runs everything except the local VLM: SAM 3, SAM 3D Objects, the room
generator, the viewer, and the web app.

> **Heads-up on the deviation from upstream.** SAM 3D Objects'
> `environments/default.yml` pins torch 2.5.1 + CUDA 12.1. That does **not**
> work on an RTX 5090 (sm_120 requires CUDA 12.8), so this env runs
> **torch 2.8.0+cu128** with `pytorch3d` built from source against it.
> `torch_versions_backup.txt` records the old cu121 pin set for reference.
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

# WorldBuilder's own thin layer
pip install -r requirements.txt
```

Verify:

```bash
python -c "import torch, sam3, sam3d_objects, pytorch3d, kaolin; \
print(torch.__version__, torch.cuda.is_available())"
# → 2.8.0+cu128 True
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

`src/segmentation.py` looks for `checkpoints/sam3.pt` by default; pass
`SAM3Segmenter(checkpoint_path=...)` to override.

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
that's the path `src/reconstruction_3d.py` loads.

### 5. Configure `.env`

Create `.env` in the project root (it's gitignored — never commit your key):

```bash
GEMINI_API_KEY=your-key-from-https://aistudio.google.com/apikey
WORLDBUILDER_DETECTOR=gemini        # or "qwen" for the local VLM
TORCH_HOME=/path/to/torch/cache     # optional
TORCH_HUB_OFFLINE=1                 # optional: avoid hub checks on every run
```

| variable                | default                 | meaning                                              |
| ----------------------- | ----------------------- | ---------------------------------------------------- |
| `GEMINI_API_KEY`        | —                       | Required when the detector is `gemini`.               |
| `WORLDBUILDER_DETECTOR` | `gemini`                | `gemini` \| `qwen` (aliases: `vlm`, `local`).         |
| `WORLDBUILDER_VLM_URL`  | `http://127.0.0.1:8765` | Where the local VLM server lives.                     |

### 6. Smoke test

```bash
conda activate worldbuilder-main
python main.py --image test_images/living_room.jpg
```

First run takes a few minutes (model loads dominate). Success looks like a
new `outputs/living_room_g/` containing `viewer.html`.

---

## Running it

### CLI

```bash
conda activate worldbuilder-main

python main.py                                   # batch: every image in test_images/
python main.py --image path/to/photo.jpg         # single image
python main.py --input-dir my_photos --output my_outputs
python main.py --force                           # reprocess instead of skipping done scenes
```

Output goes to `outputs/<image_stem>_<g|q>/`, where the suffix records which
detector produced it — so a Gemini run and a Qwen run of the same photo sit
side by side and `--force` only clobbers the matching one. Batch mode skips
scenes that already have all three result JSONs and writes a
`timing_report_*.csv` with per-image timings.

View a finished scene:

```bash
python -m http.server -d outputs/living_room_g 8000
# → http://localhost:8000/viewer.html
```

### Web app

Terminal 1 — the local VLM server (skip if using Gemini):

```bash
conda activate worldbuilder-vlm
python vlm_server/server.py            # binds 127.0.0.1:8765
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

| endpoint                 | purpose                                          |
| ------------------------ | ------------------------------------------------ |
| `GET  /api/health`       | liveness                                          |
| `GET  /api/vlm-status`   | proxies the VLM server's `/health`                |
| `GET  /api/demos`        | pre-baked scenes in `outputs/` with a `viewer.html` |
| `POST /api/upload`       | accepts `image`, queues a job → `{job_id, scene_id}` |
| `GET  /api/jobs/<id>`    | `{status, stage 1–5, scene_id, error}`            |
| `GET  /outputs/<path>`   | static access to generated scenes                 |

Passing a known `scene_id` as a `job_id` returns `complete` immediately —
that's the demo fallback seam for showing a pre-baked scene.

**Pre-baking demo scenes.** Any `outputs/` directory containing a
`viewer.html` shows up automatically as a demo chip in the webapp (via
`/api/demos`). So `python main.py --image test_images/your_scene.jpg` is all
it takes to add one. To jump straight to it — useful as a break-glass during
a live demo — use the URL param:

```
http://localhost:5174/?demo=your_scene_g
```

### Utilities

```bash
python regen_viewer.py outputs/scene_a outputs/scene_b   # rebuild viewer.html after viewer edits
python -m src.debug_masks                                # overlay masks on source image to check alignment
```

---

## Detector routing: Gemini vs. local Qwen3-VL

Stage 1 asks a vision model to list the objects worth reconstructing. Two
interchangeable backends implement the same one-method interface
(`detect_objects(image_path) -> List[Dict]`), so swapping is a single env var:

```bash
WORLDBUILDER_DETECTOR=gemini   # src/object_detection.py     — Gemini 2.5 Flash (default)
WORLDBUILDER_DETECTOR=qwen     # src/local_vlm_detection.py  — local Qwen3-VL-8B
```

`main.py` reads it at the top of `process_image()`; the webapp reads it in
`/api/upload` to pick the scene-name suffix. Set it in `.env`, or override
per-run:

```bash
WORLDBUILDER_DETECTOR=qwen python main.py --image photo.jpg
```

|                | `gemini`                                 | `qwen`                                       |
| -------------- | ---------------------------------------- | -------------------------------------------- |
| Model          | `gemini-2.5-flash` via `google-genai`    | `Qwen/Qwen3-VL-8B-Instruct` via transformers  |
| Runs where     | Google's API                             | `vlm_server/server.py`, localhost:8765        |
| Needs          | `GEMINI_API_KEY`, network                | ~17 GB VRAM while active, second conda env    |
| Latency        | ~2–5 s                                   | ~15–40 s incl. GPU wake                       |
| Failure mode   | 429/503 → 5 retries, exponential backoff | malformed JSON → 3 retries, then salvage parse|
| Privacy        | image leaves the machine                 | fully offline                                 |

Both prompts ask for the same JSON schema (`id`, `label`, `description`,
`expected_instances`) and both are deliberately steered toward plain everyday
labels — "sofa", not "modular seating unit" — because the label becomes SAM
3's text prompt in stage 2, and SAM 3 segments common nouns far more reliably
than jargon. If you tune one prompt, tune the other to match, or the two
backends drift apart. They live in `src/object_detection.py` and
`vlm_server/server.py`.

Whichever backend runs, `src/detection_postprocess.py` then drops people from
the list (reconstructed humans come out as uncanny static blobs).

**Gemini specifics.** Transient `503`/`429`/`UNAVAILABLE`/`RESOURCE_EXHAUSTED`
errors get 5 attempts with 1/2/4/8/16 s backoff; anything else raises
immediately. Responses wrapped in markdown fences are unwrapped before
parsing.

**Qwen specifics.** The server holds the model in CPU RAM and moves it to the
GPU only for the duration of a `/detect` call, then the client immediately
calls `/unload` to hand VRAM back to SAM 3D — including when detection raised
mid-flight. Generation retries up to 3 times (greedy first, then
`temperature=0.3`), and the parser salvages truncated output by trimming back
to the last complete object. To swap models, edit `MODEL_NAME` at the top of
`vlm_server/server.py`; `Qwen/Qwen2.5-VL-7B-Instruct` is a drop-in fallback.

---

## Outputs

Each scene directory holds:

```
outputs/<scene>/
├── detected_objects.json          # stage 1: what the VLM saw
├── segmentation_results.json      # stage 2: per-instance masks, scores, boxes
├── masks/                         # stage 2: mask PNGs
├── reconstruction_results.json    # stage 3: per-object mesh paths, pose, scale
├── 3d_models/
│   ├── <label>_<n>.ply            # one mesh per object instance
│   ├── room.ply                   # procedural floor/walls box
│   └── scene.ply                  # everything composed into one mesh
└── viewer.html                    # self-contained Three.js viewer
```

`viewer.html` pulls Three.js from unpkg at runtime, so viewing needs network
access (not the GPU). Scene assembly — floor snapping, XZ collision
resolution, the room box — lives in `src/room_generator.py`;
`FLOOR_SUPPORTED_KEYWORDS` there is the whitelist of categories forced onto
the floor plane when SAM 3D's depth estimate comes back wrong.

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

## Repo layout

```
main.py                     CLI entrypoint + batch runner + timing reports
regen_viewer.py             rebuild viewer.html for existing output dirs
requirements.txt            WorldBuilder's own deps (not the SAM stacks)

src/
  object_detection.py       stage 1a: Gemini detector
  local_vlm_detection.py    stage 1b: HTTP client for the local VLM server
  detection_postprocess.py  stage 1c: drop people, renumber ids
  segmentation.py           stage 2:  SAM 3 text-prompted segmentation
  reconstruction_3d.py      stage 3:  SAM 3D Objects → meshes, pose, scale
  room_generator.py         stage 4:  floor snap, collision resolve, room box
  viewer_generator.py       stage 4:  emit self-contained Three.js viewer
  debug_masks.py            dev tool: overlay masks on the source image

vlm_server/server.py        Qwen3-VL detection server (separate conda env)
webapp/
  server.py                 Flask API + job queue
  static/                   React (UMD + Babel standalone, no build step)

checkpoints/                SAM 3 weights            (gitignored)
sam3_repo/                  upstream clone           (gitignored)
sam3d_objects_repo/         upstream clone + weights (gitignored)
pytorch3d/                  upstream clone           (gitignored)
outputs/                    generated scenes         (gitignored)
test_images/                input photos             (gitignored)
```

---

## Troubleshooting

**`CUDA out of memory` mid-pipeline.** Usually a model failed to park on CPU.
Restart the webapp/CLI process; if it's reproducible, check the
`Moving model: cuda → cpu` lines in the log to see which stage didn't release.
`GPU free after move: …` in the VLM server log tells you whether Qwen actually
let go.

**`VLM server unreachable at http://127.0.0.1:8765`.** The server isn't
running, or is still loading the model (~1 min). `curl
localhost:8765/health` — `{"ok": true}` means it's ready.

**`GEMINI_API_KEY not found`.** `.env` is missing, or you started the process
from a directory other than the project root. `load_dotenv()` resolves `.env`
relative to the working directory.

**SAM 3 checkpoint 404 / gated repo error.** Access wasn't approved yet, or
`hf auth login` hasn't run in this shell.

**`RuntimeError: Not compiled with GPU support` from pytorch3d.** It was built
on a machine without a visible GPU. Rebuild `pip install -e ./pytorch3d` on the
GPU box.

**`pipeline.yaml` not found.** The SAM 3D download landed in the wrong place —
the `mv` step in setup §4 is easy to miss. You want
`sam3d_objects_repo/checkpoints/hf/pipeline.yaml`.

**numpy 2.x errors after installing something new.** Something upgraded numpy.
`pip install 'numpy==1.26.4'` and reinstall whatever pulled it.
