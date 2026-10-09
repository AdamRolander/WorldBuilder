# WorldBuilder roadmap

One place for everything that is planned, blocked, or half-done, with a
suggested order. Tickets have stable ids so commits, PRs and docs can refer
to them (`PIPE-3`, `PUB-1`, …). Status as of **2026-10-08**.

Related documents: [AUDIT.md](AUDIT.md) (what was wrong and what was fixed),
[VALIDATION.md](VALIDATION.md) (how to verify a change on the GPU, with
before/after numbers), [INTEGRATIONS.md](INTEGRATIONS.md) (Blender/Unreal/MuJoCo),
[HARDWARE.md](HARDWARE.md) (what card you need),
[CONTRIBUTOR_TASKS.md](CONTRIBUTOR_TASKS.md) (work that needs no GPU); publishing,
hosting and RoomBuilder planning notes are kept privately.

Legend: ☐ todo · ◐ in progress · ☑ done · ⛔ blocked (by what)

---

## 0. Suggested order of work

The dependencies below drive this order more than priority does.

1. **Look at the October scenes and merge** (`audit-and-roadmap` → `main`):
   the galleries in `outputs/10-08-validation/` (second pass vs first) and
   `outputs/10-08-newphotos/` (photos nothing was tuned on), the Blender
   add-on and the MuJoCo export; then merge and tag `v0.2.0` (`PUB-2`).
2. **Rotate the test photos.** Six photos have now been looked at twice;
   the next pass should start from a fresh set (and keep these as a
   regression set) so fixes are judged on pictures they were not written
   for. `PUB-4` (an evaluation set with ground truth) is the durable form.
3. **Local-VLM recall** (`PIPE-1`, `PIPE-21`): two-pass or tiled detection,
   and let the detector name built-in furniture so counters and cabinet
   runs become real objects rather than boxes behind a relief.
4. **Object appearance** (`PIPE-11`, `PIPE-24`): SAM 3D's own texture
   baking, and re-projecting the photo onto the visible side of each
   object. With the room now sharp, object textures are the soft part.
5. **Real inpainting for what tiling cannot predict** (`PIPE-10`).
6. **Verify the metric scale** (`PIPE-6`) against a photo with a known
   dimension, then use it for thresholds that are still relative.
7. **Doors/windows + non-box rooms** (`PIPE-8`, `PIPE-9`): CPU-only, good
   contributor tasks.
8. **Packaging** (`CODE-8` Docker, sample scene pack, CPU install extra):
   what turns "works on Adam's machine" into something a stranger can run;
   mostly doable without a GPU (CONTRIBUTOR_TASKS T1–T4).
9. **Publishing clock** (`PUB-*`): tag, changelog, Zenodo DOI, a static
   demo site. Zero-risk and time-driven (JOSS wants six months of history).
10. **RoomBuilder phase 1** (`RB-*`) once the UCSD data question is
   answered; independent of the pipeline until phase 3.

---

## 1. Pipeline quality — `PIPE-*`

| id | status | item | notes / blockers |
| --- | --- | --- | --- |
| PIPE-1 | ◐ | **Local VLM detection recall.** Old prompt: recall 0.53 vs Gemini (19 photos), 10/42 looping. New prompt: 0.52, no loops, outdoor photos work, 14 photos hit the 25-entry cap (VALIDATION §3.6) — raise the cap / second pass next. Benchmark Qwen3-VL-8B with the new prompt against Gemini on the 20 `_g` scenes (`scripts/bench_detectors.py --live demo_day`), then try: (a) two-pass detection (list, then "what did you miss?" with the first list in context), (b) tiled detection for wide/cluttered scenes (2×2 crops + merge), (c) a larger local model — Qwen3-VL-32B-AWQ (~20 GB, fits after SAM 3D is parked) or Gemma-3-27B-it (4-bit). | Needs GPU. Criteria: mean recall vs Gemini ≥ 0.8 without exceeding 60 s/photo. |
| PIPE-2 | ☑ | Repetition loops, prompt drift, dead 0.35 threshold, hard `single` cap, sentence prompts — fixed (AUDIT §1–2). | Validate on more scenes (VALIDATION §2). |
| PIPE-3 | ◐ | **Box-prompt path validation.** Fired 2× on lr2 and recovered a mask both times (VALIDATION §3.6); hit rate across many photos still unmeasured. Log how often it fires and whether the resulting masks survive NMS. | Needs GPU; one afternoon. |
| PIPE-4 | ☐ | **Cluttered scenes: tiled segmentation.** SAM 3 runs at 1008 px; small objects in a 4K kitchen photo are a handful of pixels. Run SAM 3 on 2×2 overlapping crops for labels that returned nothing at full resolution, merge with NMS. | Straightforward; expect ~2× stage-2 time only when it fires. |
| PIPE-5 | ☐ | **Lighting/low-contrast robustness.** Try CLAHE / exposure normalisation before detection and segmentation on the dark demo photos (`demo_day/lib*`, `bl*`). Measure recall change. | Cheap experiment. |
| PIPE-6 | ◐ | **Metric scale.** `scripts/estimate_metric_scale.py` runs MoGe-2 (isolated install under `third_party/`, called in a subprocess after stage 3) and writes metres-per-unit to the stage-3 cache; layout, viewer, Blender add-on and MuJoCo export use it. Pixel-wise agreement with the scale-free map is ±1–5 %; ceilings moved from 3.5–5 m to 2.1–3.2 m. | Not yet checked against a measured room. Check MoGe-2's weight licence before commercial use. Nothing on disk is pre-multiplied. |
| PIPE-7 | ☑ | Shared point map, gravity alignment, textured floor/walls/ceiling, support-aware placement (AUDIT §3–5). | Superseded in October by PIPE-17/18 (same ideas, driven by the photo instead of boxes and labels). |
| PIPE-8 | ☐ | **Non-rectangular rooms.** Fit multiple vertical planes (sequential RANSAC in the aligned XZ plane) and build the room as an extruded polygon instead of a box; keep the box as fallback. | After PIPE-6 so thresholds are in metres. CPU-only (CONTRIBUTOR_TASKS T8). |
| PIPE-9 | ☐ | **Doors and windows.** SAM 3 prompts "door", "window" already work; cut openings into the shell quads and export them as named nodes (RoomBuilder needs door swing). What is seen *through* a detected wall is already painted onto it (PIPE-18), so windows show as pictures; real openings are the remaining step. | Medium; T9. |
| PIPE-10 | ◐ | **Predicting unseen room surfaces.** Walls: per-row dominant colour plus sideways-tiled detail; floors/ceilings: lighting-free mirrored tiling; unseen walls borrow the best-seen wall's make-up; texels seen at a grazing angle are predicted instead of smeared; flush objects (tiles, panelling) stay in the wall texture. Next: a learned inpainter (LaMa) where a surface is not texture-like, and openings (windows, doors) which this model cannot invent. | CPU-only; T10. |
| PIPE-11 | ◐ | **Object texture baking.** `scene_lite.glb` bakes vertex colours into a per-object texture after decimation (`src/mesh_bake.py`). Still to try: SAM 3D's own `with_texture_baking=True`, which samples the Gaussians and should be sharper than vertex colours. Measure time cost. | Needs GPU for the SAM 3D variant. |
| PIPE-12 | ☑ | **Mesh decimation.** `scene_lite.glb`: 8k triangles per object with baked textures; the kitchen went from 394 MB / 19 M triangles to 16.6 MB. Default download for DCC clients. | LOD chains (several budgets per object) not done. |
| PIPE-13 | ☐ | **Permissive-model variant.** Evaluate TRELLIS / Hunyuan3D 2.x / SPAR3D as a SAM 3D replacement for a SAM-licence-free stack (matters for a commercial RoomBuilder backend). | Research; see LICENSE notes. |
| PIPE-14 | ☐ | **Outdoor/open scenes.** Layout stage assumes a room; add a "no ceiling / ground plane only" mode when the ceiling is not found and the point map spans > ~15 m. | Small. |
| PIPE-15 | ☐ | **Batch throughput.** Keep all three models resident with SAM 3D quantised (fp8/int8) or on a 48 GB card; removes the park/wake cost (~10 s/scene). | Hardware-dependent. |
| PIPE-16 | ☑ | **Depth refit against the point map.** Each object is slid along its viewing rays until its visible surface is at the observed depth (`placement.refit_depth`); fires on 6–10 objects per scene. | — |
| PIPE-17 | ☑ | **Evidence-based placement** (`src/placement.py`). Support comes from the surface under each mask, not from bounding boxes or label lists; every move is checked by silhouette/mask IoU; part/group masks, structural duplicates and meshes that miss their mask are removed. Mean silhouette IoU on the six validation scenes 0.30 → 0.52 (VALIDATION §4). | The label lists in `room_generator.py` remain only for the no-point-map fallback. |
| PIPE-18 | ☑ | **Textured shell + photo relief** (`src/room_texture.py`). Room planes are UV-textured quads at photo resolution with material-continuing fill; built-ins come from the point map as a textured relief. | Relief is 2.5D and unsimplified: PIPE-22. |
| PIPE-19 | ◐ | **Instance sharing** (`src/pose_fit.share_instances`). Fragments (short, and hidden or cut by the frame) and look-alike repeats take the best-observed instance's mesh, re-posed at its true size by a heading/position search on an occlusion-aware silhouette score. Conservative on purpose: a complete instance must be explained better by the exemplar and look like it. | Fires on a minority of candidates (classroom: 3 of 11 chairs). Appearance matching is crude (hue + a 16×16 crop correlation); an embedding (DINO) would be the real test. Meshes are not yet instanced in the GLB. |
| PIPE-20 | ☐ | **VLM as a second opinion.** (a) Ask the detector for a `rests_on` hint per type and use it *only* where the photo gives no contact evidence (bottom edge occluded); (b) after placement, show the VLM a render next to the photo and ask which objects look wrong. (a) is cheap; (b) costs a second VLM pass (~30 s, ~17 GB) and an 8B model's spatial judgement is unproven — benchmark on the audit script's known-bad objects before wiring it in. | Needs GPU. The audit's IoU check already catches what (b) would, for free, whenever a mask exists. |
| PIPE-21 | ☐ | **Built-in furniture as objects.** The prompt forbids countertops and cabinets, so they only exist as relief. Let the detector name them ("kitchen counter", "cabinets", "shelving unit") and let SAM 3D build them; the structural-duplicate filter already drops what is really wall or floor. | Needs GPU; measure SAM 3D quality on large built-ins first. |
| PIPE-22 | ◐ | **Solid built-ins** (`src/builtin_boxes.py`). Axis-aligned faces in the background geometry are covered with rectangles and extruded away from their normal to the shell (counter top → floor, cabinet front → wall), thinned to slabs where the photo saw through them; the visible face is textured like a room plane. The relief stays on top as the photo-exact skin. | Boxes are only as good as the faces seen: 4–8 per scene, flat-coloured sides. Still to do: snap the relief's vertices to the fitted planes; merge boxes of one cabinet run. T11. |
| PIPE-24 | ☐ | **Re-project the photo onto objects.** For vertices the photo saw (inside the mask, depth-consistent), take the colour from the photo instead of SAM 3D's texture; keep SAM 3D's colours elsewhere. The room is now sharper than the objects in it. | CPU-only; needs care at mask edges. |
| PIPE-25 | ☐ | **Placement leftovers.** (a) Objects left floating when the gap exceeds the settling bound (0.75 × height) are neither hung nor mounted; decide by looking at what is behind them. (b) The visibility rule moves the *small* object when two disagree; when the large one is wrong (a sofa mesh that is too shallow) it should be the one refitted. (c) Search heading as well as position in the silhouette refinement. | CPU-only. |
| PIPE-26 | ☐ | **Validate fitted walls in the layout stage.** Placement now refuses a "wall" that a quarter of the objects stand beyond, but the layout stage should not offer it: check for observed points a little beyond a candidate wall (cabinet fronts and partitions have room behind them; a real wall has nothing, or far-away outdoors through a window). | Found on a new photo (VALIDATION §5.3). CPU-only. |
| PIPE-23 | ☐ | **Tilt-shift / cropped photos.** Architectural photos with corrected verticals break the centred-principal-point assumption; the point map is then slightly sheared. Detect it (wall normals not perpendicular to the floor normal) and warn or correct. | Not yet observed for certain; a risk for magazine-style photos. |

## 2. Code health — `CODE-*`

| id | status | item |
| --- | --- | --- |
| CODE-1 | ☑ | Test suite (70 CPU tests), ruff, CI workflow, pyproject, LICENSE, CITATION. |
| CODE-2 | ☐ | Type-check with `pyright`/`mypy` on `src/` (mostly annotated already). |
| CODE-3 | ☐ | Split `main.process_image` into a `Pipeline` class with stage hooks (progress callbacks for the webapp instead of polling the filesystem). |
| CODE-4 | ◐ | Webapp renamed, detector picker and GLB download added; the "photo point cloud" toggle exists in the viewer only. |
| CODE-5 | ☐ | Vendor three.js into `webapp/static/vendor/` so viewers work offline (the "fully local" claim currently stops at the browser). |
| CODE-6 | ☑ | Stage 3 writes `3d_models/stage3/` (model-space meshes, raw poses, point map); `scripts/replay_assembly.py` re-runs all of stage 4 on CPU in ~10 s (plus ~25 s for the lite GLB). `main.py --resume` reuses detection/segmentation. |
| CODE-7 | ☐ | Structured logging (JSON lines per stage) instead of prints; the timing CSV becomes a view over it. |
| CODE-8 | ☐ | Docker image for the server side (CUDA 12.8 base, both envs) — the biggest setup-time reducer for external users. T3. |
| CODE-9 | ☑ | `scripts/audit_scene.py` (placement scored against masks and point map) and `scripts/render_scene.py` (headless contact sheets): a pipeline change can be judged without opening a viewer. |
| CODE-10 | ☐ | Scenes are large on disk (stage-3 cache + world PLYs + full GLB: 0.8–2.1 GB per scene, 1.6 GB for the 35-object kitchen). Add `--keep minimal` that drops the full-resolution world PLYs and `scene.glb` once the lite GLB exists, and have the viewer load the lite GLB. |
| CODE-11 | ☐ | `requirements.txt` pins numpy 1.26 but the working env has numpy 2.4; reconcile (the pipeline runs on 2.x). |

## 3. Publishing — `PUB-*`

| id | status | item |
| --- | --- | --- |
| PUB-1 | ☑ | Paper drafts (kept privately until submission). Verify SAM 3 / SAM 3D citations against their official reports; add an ORCID. |
| PUB-2 | ☐ | Tag `v0.2.0` after merging this branch; enable Zenodo GitHub integration; add the DOI badge. |
| PUB-3 | ☑ | `docs/ARCHITECTURE.md`: data flow, per-stage JSON schemas, frame conventions. |
| PUB-4 | ☐ | Evaluation set: 30 room photos with per-photo object lists (consented), a results table (recall, time, failures) in the docs. |
| PUB-5 | ☐ | Changelog + monthly tagged releases for ≥ 6 months (JOSS history requirement). |
| PUB-6 | ☐ | One documented research use (own preprint / course project / external lab). |
| PUB-7 | ☐ | Submit to JOSS (or SoftwareX if the "used in research" bar is not met by then). |
| PUB-8 | ☐ | Static demo site: five scenes' `viewer.html` + lite GLB on GitHub/Cloudflare Pages. Free, no server, and the best advert the project can have. |

## 4. DCC integrations — `DCC-*` (details in INTEGRATIONS.md)

| id | status | item |
| --- | --- | --- |
| DCC-0 | ☑ | GLB scene export + `/api/scenes/<id>[/glb]`; Blender extension scaffold; Unreal plugin scaffold. |
| DCC-1 | ☑ | Blender end-to-end test, headless: `scripts/test_blender_addon.py` registers the add-on in Blender 5.0.1, uploads a photo to a local server, waits, imports the lite GLB and renders (VALIDATION §4.4). Not yet done: a human clicking through the UI panel. |
| DCC-2 | ◐ | `blender --command extension validate` passes (two manifest errors fixed) and `extension build` produces the zip. Remaining: account, icon, screenshots, submit. T19. |
| DCC-3 | ◐ | Blender: metric-scale checkbox (applies the server's estimate to the scene root), lite/full mesh choice, "import existing scene". Unreal: not yet. Real scale needs PIPE-6. |
| DCC-4 | ☐ | Unreal: Editor Utility Widget with progress instead of the blocking call. |
| DCC-5 | ⛔ | Fab publisher onboarding + Win64 build for UE 5.4/5.5. Blocked by: no Windows/UE machine. |
| DCC-6 | ☐ | Vertex-colour material + decimation option shipped with the UE plugin. |
| DCC-7 | ☐ | Streaming import (objects appear as they finish). Needs a per-object job endpoint. |

## 5. Robotics — `ROB-*` (details in `integrations/mujoco/README.md`)

| id | status | item |
| --- | --- | --- |
| ROB-0 | ☑ | MuJoCo export (`src/mujoco_export.py`, `GET /api/scenes/<id>/mujoco`): Z-up metric MJCF, textured visual meshes, convex or CoACD collision, free joints for supported objects, support pads, photo camera, `--stabilize`. Gymnasium wrapper and robot attachment in `integrations/mujoco/`. |
| ROB-1 | ◐ | MJX variant (`--mjx`): primitive collisions; loads and steps in MJX on CPU JAX. GPU throughput and batch limits unmeasured (T16). |
| ROB-2 | ◐ | Collision for built-ins: the boxes from PIPE-22 are exported as static MuJoCo box geoms (`builtin_NN`), so counters and cabinet runs are solid. Pads remain for surfaces no box covers. Coverage is whatever faces the photo showed. |
| ROB-3 | ☐ | USD export with `UsdPhysics` for Isaac Sim / Isaac Lab (T17). MJCF import works there meanwhile. |
| ROB-4 | ☐ | URDF/SDF export for Gazebo and PyBullet (T18). |
| ROB-5 | ☐ | Worked examples with MuJoCo Menagerie robots (Franka pick, Go2 walk-through) and a recorded video (T15). |
| ROB-6 | ☐ | Articulation: doors, drawers and cabinet fronts as hinged/sliding bodies. Needs part segmentation (SAM 3 "drawer", "cabinet door" prompts) and an axis estimate; research-grade. |
| ROB-7 | ☐ | Physical properties: per-object mass and friction from the label/material (a VLM lookup is good enough to beat uniform density). |
| ROB-8 | ◐ | Metric scale from MoGe-2 (PIPE-6) is used by the export; `--scale` still overrides. Verify once against a measured scene. |

## 6. RoomBuilder — `RB-*` (sister repo)

| id | status | item |
| --- | --- | --- |
| RB-0 | ☑ | Brief, competitor check (Dormscape), sister repo with a working phase-0 prototype. |
| RB-1 | ☐ | Confirm whether Dormscape covers UCSD; decide the wedge accordingly. |
| RB-2 | ☐ | Data: 10–20 UCSD room types as YAML from HDH plans + measurements; ask HDH for architectural dimensions and permission. |
| RB-3 | ☐ | Deploy the static prototype (Cloudflare Pages / GitHub Pages); add cross-device layout sharing (Worker + KV). |
| RB-4 | ☐ | Apply to Amazon Associates once the site is live; then Impact/CJ for Target, Wayfair, IKEA. |
| RB-5 | ☐ | Product import tool (retailer URL → dimensions, photo, price). |
| RB-6 | ☐ | Roommate co-editing (y.js over WebSocket). |
| RB-7 | ⛔ | Photo capture via WorldBuilder. Blocked by PIPE-6 (metric scale) and PIPE-9 (doors). |
| RB-8 | ☐ | Licence/business decision for the RoomBuilder repo. |

---

## 7. Known nuances worth remembering

* **Units.** Everything from MoGe-1/SAM 3D is scale-invariant. Within a scene
  it is consistent, across scenes it is not. `layout.estimated_metric_scale`
  converts to metres; `layout.metric_scale_source` says whether it came
  from MoGe-2 or from the old camera-height prior (used when
  `third_party/moge2` is not installed). Nothing on disk is pre-multiplied.
* **Frames.** SAM 3D world = OpenCV camera with X and Y negated (X left, Y
  up, Z forward). The layout stage rotates the whole scene (gravity + yaw);
  `layout.json` stores that rotation so anything else (point cloud, future
  panoramas) can be brought into the same frame.
* **The SAM licence** is not OSI; keep weights and upstream code out of
  anything you distribute (extensions, Fab, Docker images on public hubs).
* **VRAM.** 32 GB is the floor. The local VLM adds ~17 GB on the GPU only
  during detection; the park/wake dance costs ~10 s per scene.
* **Determinism.** SAM 3D uses `seed=42`; SAM 3 is deterministic; Gemini is
  not (temperature 0.2 now). Bench numbers vary by ±1 object between runs.
* **The photo is the ground truth for placement.** `scripts/audit_scene.py`
  scores every object against its mask and the point map; a placement
  change that lowers the mean silhouette IoU is a regression even if it
  "looks more physical". Replay with `--placement legacy|raw` to compare.
* **Replaying needs the cache.** Scenes made before 2026-10-08 have no
  `3d_models/stage3/` and cannot be replayed; rerun them with `--resume`.
* **The relief is 2.5D.** It is exactly right from the photo's viewpoint
  and has no back side. Good for context and for holding things up, not a
  substitute for modelled furniture.
* **GPU sharing on the dev box.** A Docker vLLM container (`routing-vllm`)
  auto-restarts and takes 29 GB; `docker stop routing-vllm` before runs.
