# Setup

Everything needed to run the full pipeline on your own GPU. Plan on one to two hours end to
end, most of it downloads and the wait for Hugging Face to approve the two gated SAM
checkpoints, so start with [step 0](#0-request-checkpoint-access-first).

Only want to work on placement, the room, the viewer or an exporter? None of that needs a
GPU or model weights; see [CPU-Only Development](#cpu-only-development) at the end.

**On this page:** [Requirements](#requirements) · [Setup](#setup-steps) ·
[CPU-Only Development](#cpu-only-development) · [Troubleshooting](TROUBLESHOOTING.md)

## Requirements

- Linux x86-64 (upstream SAM 3D Objects is linux-64 only).
- One NVIDIA GPU. Developed on an RTX 5090 (32 GB, driver 580.x, CUDA
  12.8). Measured peak is 19 GB allocated for SAM 3D and ~21 GB for the
  local VLM, one at a time, so a **24 GB card (RTX 3090/4090) should work
  but is untested**; 16 GB is not enough. Details, measurements and what
  to do on a borderline card: [Hardware](HARDWARE.md).
  Everything after reconstruction (placement, room, viewer, exports,
  Blender/MuJoCo) runs without a GPU.
- ~60 GB free disk: ~15 GB of model checkpoints, and outputs run 0.8–2 GB
  per scene (the shareable `scene_lite.glb` is 7–23 MB).
- Conda or mamba.
- A Hugging Face account, because **both** SAM checkpoint repos are gated and
  need manual access approval (can take a day, so request these first).
- A Gemini API key, if you want the cloud detector. Not needed for the
  fully local path.

### The VRAM Dance

Three large models (Qwen3-VL 8B, SAM 3, SAM 3D Objects) cannot fit on one
card at once, so the pipeline keeps exactly one of them on the GPU at a time and
parks the others in CPU RAM between stages. You'll see `Moving model: cuda →
cpu` in the logs; that's working as intended, and it costs a few seconds per
stage transition. This is why the VLM lives in a separate process behind an
HTTP server with `/unload` and `/reload` endpoints, and why `main.py` wraps
each stage in `_to_device('cuda')` / `_to_device('cpu')`.

---

## Setup Steps

### 0. Request Checkpoint Access First

Both are gated on Hugging Face and approval is not instant:

- https://huggingface.co/facebook/sam3
- https://huggingface.co/facebook/sam-3d-objects

Then authenticate locally:

```bash
pip install 'huggingface-hub[cli]<1.0'
hf auth login
```

### 1. Clone WorldBuilder and Its Upstream Dependencies

The three upstream repos are **not** committed to this repo (they're large,
have their own git history and their own licences). Clone them into the
project root by these exact names, since the code imports them by path:

```bash
git clone https://github.com/AdamRolander/WorldBuilder.git
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

### 2. Environment A: `worldbuilder-main` (Pipeline and Web App)

This runs everything except the local VLM: SAM 3, SAM 3D Objects, the layout
and assembly stages, the viewer, and the web app.

> **Heads-up on the deviation from upstream.** SAM 3D Objects'
> `environments/default.yml` pins torch 2.5.1 + CUDA 12.1. That does **not**
> work on an RTX 5090 (sm_120 requires CUDA 12.8), so this env runs
> **torch 2.8.0+cu128** with `pytorch3d` built from source against it.
> If your GPU is Ada or older, the upstream cu121 path is likely smoother:
> follow `sam3d_objects_repo/doc/setup.md` verbatim instead of the versions
> below.

```bash
conda create -n worldbuilder-main python=3.11
conda activate worldbuilder-main

# PyTorch first: everything else compiles against it.
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
pytest            # CPU-only test suite
```

Known-good versions in this env: `numpy 1.26.4` (do **not** upgrade to 2.x;
open3d and kaolin break), `transformers 4.39.3`, `open3d 0.19.0`,
`kaolin 0.18.0`, `gsplat 1.5.3`, `spconv-cu121 2.3.8`.

### 3. Environment B: `worldbuilder-vlm` (Local VLM Server, Optional)

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
start. No access request is needed, but budget the time.

### 4. Download Checkpoints

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

The result must contain `sam3d_objects_repo/checkpoints/hf/pipeline.yaml`;
that's the path `src/reconstruction_3d.py` loads. MoGe (`Ruicheng/moge-vitl`)
is fetched automatically by SAM 3D on first run.

### 4b. Optional: Metric Scale (MoGe-2)

Without this, scenes are converted to metres by assuming the camera was
1.5 m above the floor. With it, a metric depth model measures the scale.
MoGe-2 needs a newer `moge` package than SAM 3D pins, so it goes in its own
folder and is only ever imported by one script in a subprocess:

```bash
pip install --no-deps --target third_party/moge2 "git+https://github.com/microsoft/MoGe.git"
pip install --no-deps --target third_party/moge2 \
    "git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37"
python scripts/estimate_metric_scale.py outputs/<scene>     # also runs automatically after stage 3
```

The weights (~1.3 GB) download from Hugging Face on first use; check their
licence on the model card before commercial use.

### 5. Configure `.env`

Create `.env` in the project root (it's gitignored; never commit your key):

```bash
GEMINI_API_KEY=your-key-from-https://aistudio.google.com/apikey   # only for the cloud detector
WORLDBUILDER_DETECTOR=qwen          # "qwen" = fully local, "gemini" = cloud
TORCH_HOME=/path/to/torch/cache     # optional
TORCH_HUB_OFFLINE=1                 # optional: avoid hub checks on every run
```

| variable                       | default                 | meaning                                              |
| ------------------------------ | ----------------------- | ---------------------------------------------------- |
| `GEMINI_API_KEY`               | (none)                  | Required when the detector is `gemini`.               |
| `WORLDBUILDER_DETECTOR`        | `gemini`                | `gemini` \| `qwen` (aliases: `vlm`, `local`).         |
| `WORLDBUILDER_VLM_URL`         | `http://127.0.0.1:8765` | Where the local VLM server lives.                     |
| `WORLDBUILDER_GEMINI_MODEL`    | `gemini-2.5-flash`      | Cloud model id.                                       |
| `WORLDBUILDER_SHARED_POINTMAP` | `1`                     | `0` recomputes MoGe per object (old behaviour).       |
| `WORLDBUILDER_STRETCH_STRUCTURAL` | unset                | `1` stretches pillars/beams to ceiling height (opt-in). |
| `WORLDBUILDER_OUTPUTS`         | `outputs/`              | Where the webapp reads/writes scenes.                 |
| `WORLDBUILDER_HOST`            | `127.0.0.1`             | Interface the webapp listens on (`0.0.0.0` for the LAN). |
| `WORLDBUILDER_COMBINED`        | unset                   | `1` also writes `3d_models/scene_combined.ply` (all objects merged). |
| `WORLDBUILDER_PHOTO_TEXTURE`   | `1`                     | photo pixels on flat or facade-like sides that face the camera; `all` on every object; `0` keeps SAM 3D's own colours. |
| `WORLDBUILDER_RELIEF`, `WORLDBUILDER_BUILTINS`, `WORLDBUILDER_SHARE_INSTANCES`, `WORLDBUILDER_LITE`, `WORLDBUILDER_METRIC` | `1` | `0` switches the feature off (for comparisons). |

### 6. Smoke Test

```bash
conda activate worldbuilder-main
python main.py --image living_room.jpg --detector gemini     # or start the VLM server and use --detector qwen
```

First run takes a few minutes (model loads dominate). Success looks like a
new `outputs/living_room_g/` containing `viewer.html`, `scene.glb` and
`3d_models/layout.json`.

Next: [Usage](USAGE.md) for the CLI, the web app and the viewer.

---

## CPU-Only Development

Placement, the room and its textures, the viewer, the GLB, MuJoCo and Blender exports all run
on a CPU from what stage 3 leaves on disk. To work on them without a GPU:

```bash
conda create -n wb-dev python=3.11 && conda activate wb-dev
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,cpu,robotics]"
pytest
ruff check .
```

To replay stage 4 on a real scene you need that scene's folder from someone with a GPU
(`3d_models/stage3/` is the part that matters), then
`python scripts/replay_assembly.py outputs/<scene>`. See [Hardware](HARDWARE.md#no-gpu-at-all).
