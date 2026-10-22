# Usage

How to run WorldBuilder once it is installed ([Setup](SETUP.md)): the command line, the web
app, reviewing and correcting a scene in the viewer, and what ends up on disk.

**On this page:** [Running It](#running-it) · [Detector Routing](#detector-routing-gemini-vs-local-qwen3-vl) ·
[Outputs](#outputs) · [Remote Access](#remote-access-via-ngrok-vr-headset-off-the-lan)

## Running It

### CLI

```bash
conda activate worldbuilder-main

python main.py                                   # batch: every image in test_images/
python main.py --image path/to/photo.jpg         # single image
python main.py --image photo.jpg --detector qwen # override .env for this run
python main.py --input-dir my_photos --output my_outputs
python main.py --force                           # reprocess instead of skipping done scenes
python main.py --image photo.jpg --no-room       # objects only, no floor/walls/ceiling
python main.py --image photo.jpg --no-structural # skip SAM 3 floor/wall/ceiling/window masks (geometry-only layout)
```

Output goes to `outputs/<image_stem>_<g|q>/`, where the suffix records which
detector produced it, so a Gemini run and a Qwen run of the same photo sit
side by side and `--force` only clobbers the matching one. Batch mode skips
scenes that already have all three result JSONs and writes a
`timing_report_*.csv` with per-image timings and failure counts.

View a finished scene:

```bash
python -m http.server -d outputs/living_room_g 8000
# → http://localhost:8000/viewer.html
```

### Web App

Terminal 1, the local VLM server (skip if using Gemini):

```bash
conda activate worldbuilder-vlm
python vlm_server/server.py                          # Qwen3-VL-8B, binds 127.0.0.1:8765
python vlm_server/server.py --model Qwen/Qwen2.5-VL-7B-Instruct   # any HF image-text model
```

Terminal 2, the web app:

```bash
conda activate worldbuilder-main
python -m webapp.server                # binds 127.0.0.1:5174 (this machine only)
# → http://localhost:5174/
python -m webapp.server --host 0.0.0.0 # also reachable from other devices on your network
```

The server has no login: anyone who can reach it can upload photos and
start GPU jobs. It therefore listens on localhost unless you pass
`--host 0.0.0.0` (or set `WORLDBUILDER_HOST`), which you need for a headset,
or for Blender / Unreal running on another computer. Only do that on a
network you trust.

Reach it from:

- **this machine:** http://localhost:5174
- **a headset/phone on the same LAN** (with `--host 0.0.0.0`): `http://<lan-ip>:5174`; find the IP with
  `hostname -I | awk '{print $1}'`. (Off-LAN? See
  [ngrok](#remote-access-via-ngrok-vr-headset-off-the-lan).)

Upload an image; the frontend polls `/api/jobs/<id>` and advances a 5-stage
progress display, then drops you into the generated viewer. Only one job runs
at a time (the GPU can't be shared), and job state is in memory: restart the
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
| `GET  /api/scenes/<id>/glb`    | the whole scene as one GLB (`?lite=1`: decimated, baked textures); manual edits applied unless `?raw=1` |
| `GET/POST /api/scenes/<id>/edits` | manual corrections saved from the viewer (`edits.json`)      |
| `GET  /api/scenes/<id>/mujoco` | zip with a MuJoCo model of the scene                            |
| `GET  /outputs/<path>`         | static access to generated scenes                               |

Passing a known `scene_id` as a `job_id` returns `complete` immediately;
that's the demo fallback seam for showing a pre-baked scene
(`http://localhost:5174/?demo=your_scene_g`).

### Reviewing and Correcting a Scene in the Viewer

`viewer.html` loads `scene_lite.glb` (tens of MB; add `?full=1` for the
dense meshes). Move the mouse over the object list, or over the scene, and
the matching object lights up; untick it to remove it. Click an object or
its name to type its position, rotation and scale (position is the object's
centre: Y is height above the floor, X and Z are relative to where the
photo was taken, in metres when the scale is known). Every piece of
built-in structure (relief surfaces, solid blocks; outdoors, the ground) has
its own checkbox.

The selected object shows which way each value moves it: arrows for +X, +Y and +Z, or, after
clicking **Rotation°** (or a rotation field), a ring showing which way a positive angle
turns it. **Shift-click** objects in the scene or the list to select several; a change then
applies to all of them, and turning or scaling acts about the middle of the group (so typing
180 in Y turns that part of the room around). **Esc** clears the selection.

Scenes made before this viewer existed keep their old `viewer.html`; refresh them with
`python scripts/regen_viewer.py --viewer-only outputs/<scene>`.

**Save edits** writes `edits.json` next to the scene. Served by the web app
it is saved through the API, and every GLB the server hands out (the
download button, the Blender add-on, the Unreal bridge) and the MuJoCo
export has the edits applied. Opened from a plain file server the file is
downloaded instead: put it in the scene folder and run
`python -m src.scene_edits outputs/<scene>` to write `scene_lite.edited.glb`.
The pipeline's own output is never modified; delete `edits.json` to undo
everything.

### Utilities

```bash
python scripts/benchmark.py outputs/run_b --baseline outputs/run_a --appearance   # quality, speed, memory, size; exit 1 on a regression
python scripts/clone_scene.py outputs/run_a/* --into outputs/run_b    # light copy (meshes symlinked) to replay with other code or settings
python -m src.scene_edits outputs/scene_a                         # apply edits.json to the GLBs
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
python scripts/estimate_metric_scale.py outputs/scene_a          # metres per scene unit from MoGe-2, then replay
python scripts/make_gallery.py outputs/run_b --before outputs/run_a   # before/after page for two runs
python main.py --image photo.jpg --resume                        # reuse detection + masks already in the scene dir
python scripts/bench_detectors.py --live demo_day --save bench.json        # same, live against the VLM server
```

---

## Detector Routing: Gemini vs. Local Qwen3-VL

Stage 1 asks a vision model to list the object types worth reconstructing.
Two interchangeable backends implement the same one-method interface
(`detect_objects(image_path) -> List[Dict]`), so swapping is a single env var
or `--detector`:

```bash
WORLDBUILDER_DETECTOR=gemini   # src/object_detection.py     Gemini 2.5 Flash (default)
WORLDBUILDER_DETECTOR=qwen     # src/local_vlm_detection.py  local Qwen3-VL-8B via vlm_server/
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
architecture and object *parts*, collapses duplicates, and caps the list:
the safety net for a local model that loops.

---

## Outputs

Each scene directory holds:

```
outputs/<scene>/
├── detected_objects.json          # stage 1 after clean-up (detected_objects_raw.json = as returned)
├── segmentation_results.json      # stage 2: per-instance masks, scores, boxes, prompt used
├── masks/                         # stage 2: mask PNGs (+ structural_floor/wall/ceiling/window.png)
├── reconstruction_results.json    # objects (pose, support, fit to the photo), removed/failed with reasons
├── metrics.json                   # seconds per phase, time to first object, seconds per object, peak GPU memory, output sizes
├── edits.json                     # optional: manual corrections saved from the viewer (applied by the GLB endpoints)
├── 3d_models/
│   ├── stage3/                    # SAM 3D's raw output: model-space meshes, raw.json poses, pointmap.npz, metric_scale.json
│   ├── <id>_<label>.ply           # one world-space mesh per object instance (vertex colours; the side the photo saw is re-coloured from the photo)
│   ├── room.glb                   # textured floor/walls/ceiling + photo relief in separate pieces + solid built-in boxes (named nodes); open scenes: ground or terrain only
│   ├── room.ply                   # the same shell as vertex colours (fallback for PLY-only consumers)
│   ├── layout.json                # floor, gravity, yaw, ceiling, wall evidence, scale estimate, placement diagnostics
│   └── scene_pointmap.npz         # MoGe scene points (aligned frame) for debugging / viewers
├── scene.glb                      # full-resolution scene (hundreds of MB): named nodes, vertex colours
├── scene_lite.glb                 # same scene, 8k triangles per object with baked textures (7–30 MB): what the viewer and DCC tools load
├── mujoco/                        # optional: scene.xml + assets (python -m src.mujoco_export)
├── scene_points.json              # small point-cloud sidecar the viewer can toggle on
└── viewer.html                    # self-contained Three.js / WebXR viewer
```

Coordinates are **scale-invariant** (consistent within a scene, not across
scenes). `layout.json → estimated_metric_scale` converts to metres: measured
by MoGe-2 when the [metric scale step](SETUP.md#4b-optional-metric-scale-moge-2) is done, otherwise a camera-height prior
(`metric_scale_source` says which).
`viewer.html` loads Three.js from a CDN, so viewing needs network access
(not the GPU).

**Open scenes.** When the upper part of the photo is sky, or neither a
ceiling nor a wall was observed, the scene is treated as open
(`layout.json → scene_type: "open"`): no walls or ceiling are invented, and
ground that is not flat is meshed from the point map as terrain. Outdoor
photos are an early use case: objects and the nearby ground come out, the
distance does not.

---

## Remote Access via ngrok (VR Headset Off the LAN)

When the headset can't reach the GPU box directly (different networks),
tunnel the webapp:

```bash
# one-time on the GPU machine
ngrok config add-authtoken <your-token-from-ngrok.com>

# each session, alongside the webapp
./ngrok http --url=<your-static-domain>.ngrok-free.dev 5174
```

Open `https://<your-static-domain>.ngrok-free.dev` on the Quest. The first
load shows an ngrok interstitial; click "Visit Site" once (the cookie lasts
7 days). The iframe shows it again on first scene load; click through the
same way. WebXR works over the tunnel because ngrok terminates HTTPS, so no
LAN certificate setup is needed.

Free tier gives 1 GB bandwidth and 20k requests/month. A 20-object scene can
be ~100 MB, so that's roughly 10 scene loads per month. Cloudflare Tunnel is
the free-and-unlimited alternative if you have a domain.

The `ngrok` binary is gitignored; grab it from
[ngrok.com/download](https://ngrok.com/download).

For Blender, Unreal and MuJoCo, see [Plugins](plugins/README.md).
