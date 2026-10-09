# Tasks that need no GPU

WorldBuilder's models need a 24 GB NVIDIA card (see
[HARDWARE.md](HARDWARE.md)), but most of the code around them does not.
Since October 2026 the pipeline saves everything SAM 3D produced for a
scene (`3d_models/stage3/`), and everything after that — layout,
placement, room textures, exports, viewers, the MuJoCo and Blender
integrations — runs on a laptop CPU in seconds from that folder.

This page is a menu for contributors without a GPU. Each task says what
you need, what "done" looks like, and roughly how big it is
(**S** an evening, **M** a weekend, **L** a couple of weeks part-time).
Ids refer to [ROADMAP.md](ROADMAP.md).

## 0. Setup without a GPU (do this first, ~20 min)

```bash
git clone https://github.com/AdamRolander/WorldBuilder && cd WorldBuilder
python -m venv .venv && source .venv/bin/activate
pip install numpy pillow scipy trimesh requests python-dotenv opencv-python-headless \
            pytest ruff matplotlib fast-simplification xatlas pyrender mujoco gymnasium
pytest                      # 70 tests, all CPU, ~40 s
```

Then ask Adam for a **sample scene pack** (a zipped scene directory with
its `stage3/` cache, masks and the photo; the smallest validation scene, `cl5_q`, is ~800 MB until T2 exists) and unpack it
under `outputs/`. With that you can run the whole back half of the
pipeline:

```bash
python scripts/replay_assembly.py outputs/k1_q      # layout + placement + room + exports
python scripts/audit_scene.py outputs/k1_q          # scores placement against the photo
python scripts/render_scene.py outputs/k1_q         # contact sheet (needs EGL or a display)
python -m http.server -d outputs/k1_q 8000          # then open http://localhost:8000/viewer.html
python -m src.mujoco_export outputs/k1_q --stabilize --check
```

If the first command prints a placement summary and the second a JSON line
with `mean_reproj_iou`, you are set up. **Writing this setup down as you go
and fixing whatever was wrong in these instructions is itself the first
contribution** (see T1).

## 1. Packaging and developer experience

| id | size | task | done when |
| --- | --- | --- | --- |
| T1 | S | **CPU-only install path.** Add a `cpu` extra to `pyproject.toml` with the packages above, make `pip install -e ".[cpu,dev]"` work on Linux, macOS and Windows, document it in the README. Note every stumble. | A fresh clone on your OS goes from zero to green `pytest` and a replayed scene using only the README. |
| T2 | S | **Sample scene pack.** Script (`scripts/make_scene_pack.py`) that zips a scene with decimated `stage3` meshes (use `src/mesh_bake.decimate`, ~20k faces each) so the pack is tens of MB, not hundreds; attach it to a GitHub release; teach `replay_assembly.py --fetch-sample` to download it. | Anyone can replay a scene without asking for a file. |
| T3 | M | **Docker image for the server** (`CODE-8`). You can write and build the Dockerfile without a GPU (CUDA base image, both conda envs, weights mounted as a volume, never baked in — SAM licence). Adam or the roommate's machine does the final GPU smoke test. | `docker build` succeeds in CI; `docker run --gpus all` instructions in the README. |
| T4 | S | **CI for the new code paths.** Extend `.github/workflows/ci.yml`: install the `cpu` extra, run ruff + pytest on Linux and macOS, cache pip. Add a job that runs `blender --command extension validate integrations/blender_worldbuilder`. | Green badge on both OSes; a manifest error fails CI (one already slipped through once). |
| T5 | M | **Vendor three.js** (`CODE-5`). Copy the pinned three.js modules the viewer imports into `webapp/static/vendor/`, switch the import map to relative paths, keep a CDN fallback. | `viewer.html` loads with the network cable unplugged. |
| T6 | S | **Type checking** (`CODE-2`). Add pyright (basic mode) on `src/`, fix or annotate what it finds, add to CI. | `pyright src` clean. |
| T7 | M | **`Pipeline` class with progress callbacks** (`CODE-3`). `main.process_image` → a class with `on_stage(name, fraction)` hooks; the webapp uses them instead of polling the filesystem. Can be developed against a fake pipeline; one GPU run to confirm. | Webapp progress bar driven by callbacks; no behaviour change in outputs. |

## 2. Stage 4 quality (all replayable on CPU)

These are measurable: `scripts/audit_scene.py` prints the numbers, and
`docs/VALIDATION.md` §4 has the current ones to beat.

| id | size | task | done when |
| --- | --- | --- | --- |
| T8 | M | **Non-rectangular rooms** (`PIPE-8`). Fit several vertical planes in the aligned XZ plane (sequential RANSAC on wall points), build the shell as an extruded polygon, keep the box as fallback. Synthetic L-shaped room test in `tests/`. | L-shaped synthetic room recovered within 5 cm; no regression on the six validation scenes. |
| T9 | M | **Doors and windows** (`PIPE-9`). Masks for "door"/"window" already come out of SAM 3 if prompted (needs a GPU run by someone else — ask for masks). Cut openings into shell quads, export them as named nodes. | `room.glb` has `room_door_*` / `room_window_*` nodes; RoomBuilder can read a door position. |
| T10 | M | **Better hidden-area fill** (`PIPE-10`). `room_texture.fill_hidden` tiles the largest visible rectangle. Try exemplar-based synthesis or LaMa (runs on CPU for 512² textures) and compare on the validation scenes. | Side-by-side renders in the PR; no visible mirrored-tile pattern on the bathroom floor. |
| T11 | M | **Built-ins clean-up.** `src/builtin_boxes.py` now puts solid boxes behind the relief. Remaining: snap the relief's vertices to those planes (counters flat, fronts vertical), merge the boxes of one cabinet run, give box sides the front's material instead of a flat colour. | Kitchen counter top is planar to < 5 mm in the relief; side views of the cabinet run look like cabinets; renders attached. |
| T12 | S | **Yaw snapping for large furniture.** Sofas, beds and cabinets are almost always parallel to a wall. Snap an object's heading to the room axes when it is within ~10° and the silhouette IoU does not drop (the guard already exists in `placement.py`). | Unit test + audit numbers unchanged or better. |
| T13 | M | **Instance sharing, second half.** `pose_fit.share_instances` replaces fragments and look-alike repeats, but conservatively (3 of 11 classroom chairs). Improve the same-model test (a DINO/CLIP embedding of the mask crop instead of hue + 16×16 correlation), and write shared meshes once in the GLB (glTF instancing). | More chairs shared with no false merges on the library's picture frames; lite GLB smaller. |
| T14 | S | **Audit report as HTML.** One page per scene: photo, overlay, renders, per-object table, layout notes. Makes reviewing a pipeline change a click. | `scripts/audit_scene.py --html` writes it. |

## 3. Robotics and DCC integrations

| id | size | task | done when |
| --- | --- | --- | --- |
| T15 | S | **MuJoCo Menagerie example.** `integrations/mujoco/examples/` with a script that drops a Franka or a Unitree Go2 into an exported scene, runs a scripted motion and saves a video. | Runs from a clean checkout with the sample pack. |
| T16 | M | **MJX throughput benchmark.** Measure steps/s for the `--mjx` export at batch sizes 1…4096 on whatever accelerator you have (CPU jax counts), document which contact settings matter. | Table in `integrations/mujoco/README.md`. |
| T17 | M | **USD export** for Isaac Sim / Isaac Lab (`ROB-3`). From `scene_lite.glb` + layout to a `.usda` with `UsdPhysics` rigid bodies and collision approximations (`pxr` is on PyPI as `usd-core`). | Scene opens in usdview with physics schemas; Isaac test by someone with an RTX card. |
| T18 | M | **URDF / SDF export** for Gazebo and PyBullet (`ROB-4`): one static world file + one model per movable object. PyBullet runs on CPU, so you can test it end to end. | `pybullet.loadSDF` shows the scene; objects rest. |
| T19 | S | **Blender add-on polish** (`DCC-2`). The add-on passes `extension validate` and an end-to-end headless test (`scripts/test_blender_addon.py`). Needs: icon, preferences UI review, a scene-list dropdown for "Import existing scene" (the server has `/api/scenes`), screenshots for extensions.blender.org. Blender itself runs fine without a GPU; point it at Adam's server or a mock. | Submitted for review on extensions.blender.org. |
| T20 | M | **Mock server.** A 100-line Flask app that serves a sample pack through the same `/api/*` routes, so DCC plug-in developers need no GPU and no models. | Blender test passes against the mock. |
| T21 | M | **Unreal plugin on Windows** (`DCC-5`) if you have a Windows machine with UE 5.4+: build the scaffold, import `scene_lite.glb` through Interchange, report what breaks. | Notes + fixes in a PR. |

## 4. Evaluation and docs

| id | size | task | done when |
| --- | --- | --- | --- |
| T22 | M | **Evaluation set** (`PUB-4`). Collect 30 permissively licensed room photos (own photos or CC0), write per-photo object lists by hand. This is the ground truth the JOSS paper needs and it is pure legwork. | `eval/` with photos, licences and JSON lists. |
| T23 | S | **Run the pipeline on the roommate's machine** and write down everything: card, driver, install time, every error, wall-clock per stage, peak VRAM (the pipeline prints it). | A filled-in row in `docs/HARDWARE.md` and fixes to the README. |
| T24 | S | **README pass as a newcomer.** Follow it literally; open an issue for every sentence that assumed knowledge you did not have. | Issues filed; quick ones fixed. |

## How to pick

* New to the code: **T1 → T24 → T14**. They force a full read-through and
  every one ends in a merged PR.
* Packaging person: **T1, T2, T3, T4**.
* Graphics person: **T10, T11, T8**.
* Robotics person: **T15, T18, T17**.

Conventions: branch from `main`, `ruff check . && pytest` before pushing,
one task per PR, put before/after numbers or renders in the PR text when
the task changes output.
