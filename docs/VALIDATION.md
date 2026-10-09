# Validation: how to verify a pipeline change, and what was measured

The CPU test suite (`pytest`) pins geometry and heuristics, but the models
only run on the GPU. This page is the manual procedure plus the numbers
recorded on **2026-09-17** for the `audit-and-roadmap` branch against the
demo-day code (`MigrateSAM @ a47ffff`).

## 1. Ground rules

* Run baseline and candidate on the **same photos**, same detector, same
  machine, nothing else on the GPU (`nvidia-smi`; on the dev box
  `docker stop routing-vllm`).
* Keep the original code in a git worktree and symlink the vendored repos
  and checkpoints into it, so the two runs cannot share modified modules:

  ```bash
  git worktree add /tmp/wb_base MigrateSAM
  for d in sam3_repo sam3d_objects_repo pytorch3d checkpoints; do ln -s $PWD/$d /tmp/wb_base/$d; done
  cp .env /tmp/wb_base/
  (cd /tmp/wb_base && CONDA_PREFIX=$CONDA_PREFIX python main.py --image ../photo.jpg --output /tmp/base_out)
  python main.py --image photo.jpg --output /tmp/new_out --detector qwen
  ```
* Compare: detected types, segmented instances, reconstructed objects,
  failures, wall-clock, `layout.json` notes, and **look at both viewers**.
  Numbers catch regressions; eyes catch nonsense.

## 2. Recommended check-list per change

| what | how | pass when |
| --- | --- | --- |
| Detection | `scripts/bench_detectors.py --live demo_day --save bench.json` | mean recall vs Gemini not lower than the last recorded run |
| Segmentation | count instances in `segmentation_results.json`; open `masks/` | no label lost that was found before; no mask > 60 % of frame |
| Layout | `3d_models/layout.json` → `notes`, `wall_sources`, `plane_texture_stats` | floor from mask/geometric (not fallback) on indoor photos; ≥1 wall detected when a wall is visible; textured fraction of the far wall > 0.4 |
| Assembly | `python scripts/audit_scene.py outputs/<scene>` (+ `render_scene.py` for a contact sheet); console lines `contact:`, `removed:` | mean silhouette IoU not lower than the last recorded run (§4.1); nothing removed that is plainly a good object |
| Timing | `reconstruction_results.json → metadata.timings_sec` | per-object reconstruction time not worse than baseline |
| Exports | `scene.glb` opens in <https://gltf-viewer.donmccurdy.com>; `viewer.html` loads | — |

## 3. Results recorded 2026-09-17

Photos: `demo_day/lr2.webp` (living room, 1200×800) and `demo_day/k1.jpg`
(cluttered kitchen, 1200×1136). Detector `qwen` = local Qwen3-VL-8B;
`gemini` = gemini-2.5-flash. RTX 5090.

### 3.1 Baseline (demo-day code)

| photo | detector | types | instances | reconstructed | wall-clock | room |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| lr2 | gemini | 20 | 27 | 27 | 238 s | box from object bounds |
| lr2 | qwen | 16 | 21 | 21 | 187 s | box |
| k1 | qwen | 22 | 75 | 75 | 535 s | box |

### 3.2 New pipeline (this branch)

| photo | detector | types | instances | reconstructed | wall-clock | layout |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| lr2 | gemini | 19 | 33 | 33 | 286 s | floor from mask (98 % inliers, tilt 0.8°), yaw 0.2°, ceiling measured; walls: back 68 %, right 18 % textured (after wall-trust fix, offline replay) |
| lr2 | qwen | 16 | 30 | 30 | 244 s | same floor/yaw; textured room (see §3.4 for the rerun) |
| k1 | qwen | 22 | 71 | 71 | 524 s | floor from mask (100 %), yaw 3.3°, ceiling measured; right wall detected |

Reading the table:

* **More instances from the same photos** (27→33, 21→30): the dead 0.35
  threshold, label+description union and the soft `single` prior recover
  objects SAM 3 was moderately confident about. Whether every extra
  instance is a real object needs eyes on the viewer (§2); duplicates are
  suppressed by NMS at IoU 0.5, so the extras are mostly small items.
* **Wall-clock is per-object dominated** (~8 s per object in SAM 3D). The
  shared point map removes one MoGe pass per object (≈0.3 s each on the
  5090) — a few percent, not a step change. It matters more for the layout
  stage, which now costs 0.3–2 s for the whole scene.
* **The kitchen exposed a label problem**: the local VLM listed `handle`,
  `drawer`, `knob` as objects and the support graph chained them. The
  post-processor now drops part-of-object labels (`PART_LABELS`); the
  run above predates that filter.

### 3.3 Layout stage, offline on real point maps (CPU, MoGe ViT-L)

`scripts`-free replay via `room_layout.estimate_layout` on the same photos:

| photo | floor | yaw | walls detected (votes) | far-wall textured |
| --- | --- | --- | --- | ---: |
| lr2 | mask, 98 % inliers, 0.8° tilt | 0.2° (support 0.69) | x_min (2.4 k), x_max (30 k), z_max (302 k) | 68 % |
| k1 | mask, 100 %, 1.1° tilt | 3.3° (support 0.82) | x_max (62 k) | (open kitchen, no far wall) |

Before the wall-trust fix the pipeline run of lr2 rejected the detected
back wall because two object meshes poked 0.4 units through it, grew the
room to the objects, and textured only 8 % of the far wall. Objects are
now nudged back inside strong walls (≤35 % of their own extent) instead.

### 3.4 Final-code reruns

`after3` (wall trust + part filter + intrinsics recovery + containment), lr2 with `qwen`:
30/30 objects, 245 s; floor from mask (98 %), yaw 0.2°; **back wall 58 % / right wall 16 % textured**
(was 7 % / 0 % before the wall-trust fix); 5 objects nudged back inside detected walls;
3 objects kept in their container (throw blanket ⊂ sofa, book ⊂ side table); 26 pair-pushes
(was 37). One regression found and fixed afterwards: the floor plane was lowered to the
lowest object's bottom, which zeroed the floor texture — the fitted floor is now fixed and
sunk objects are raised (`after4` rerun below).

`after3` k1 (kitchen) with `qwen`: 22 types → 18 after the part filter dropped `drawer`, `handle`
→ 48 instances (was 71/75: the part labels had been spawning dozens of tiny masks) → 48/48 reconstructed
in **388 s (was 524–535 s)**; floor from mask (100 %), yaw 3.2°, ceiling measured 3.09 m; 6 objects
nudged inside the detected right wall; 23 pair-pushes (was 48).

`after4` lr2 with `qwen` (floor fix): 30/30 in 245 s; **floor 20 %, back wall 61 %, right wall 17 %
textured**; 3 contained, 12 on supports, 6 nudged; `plan_view.png` shows pillows on the sofa (not
behind it), the rug under the seating group and the far wall coincident with the point cloud.

**Caveat discovered afterwards:** the local VLM server process used by every run above had been
started before `vlm_server/server.py` was rewritten, so stage 1 ran the *old* prompt without
repetition control. The new post-processor absorbed the damage (dedupe, part filter), but the new
prompt, `bbox_2d` fallback and loop protection are validated only by the `after5`/`bench_live2`
runs queued at the very end of the session (see §3.6).

### 3.6 Local-VLM detection benchmark (`scripts/bench_detectors.py --live demo_day`)

Old prompt/server, Qwen3-VL-8B, 42 photos, 19 with a Gemini reference: **mean recall 0.53**,
median 6–9 s per photo, but 10 photos degenerated into 100+ entry loops at ~65 s each, and the
three outdoor photos (forest, playground, maze) returned nothing. This is the number PIPE-1 starts from.

**New prompt/server** (same 42 photos, same model): **mean recall 0.52** (unchanged), but
**zero looping photos** (was 10), a flat ~23 s per photo (was 7 s normally / 65 s when looping),
and the three outdoor photos now return objects (forest 10, playground 14, maze 1). Fourteen photos
hit the 25-entry cap, which likely costs recall on cluttered scenes — raising the cap or a
"what did you miss?" second pass is the next PIPE-1 experiment. The longer prompt plus `bbox_2d`
output is what makes simple photos slower.

`after5` lr2 with the new server: 19 types (was 16), **box fallback fired twice and recovered a
mask each time** (PIPE-3 evidence), 35 instances, 35/35 reconstructed in 283 s; layout identical
to `after4` (floor 20 %, back wall 61 %, right wall 17 % textured).

### 3.7 First multi-image batch (`9-17-validation/`, 6 photos, local VLM)

Adam's first run: `bathroom3` completed (36/36 objects, floor from mask, back and right walls
detected, 41 % of the far wall textured) and every later photo failed with CUDA OOM in the VLM
server. Root cause: SAM 3D was never parked (AUDIT §3.4); fixed the same evening. Batch mode skips
finished scenes, so rerunning the same command resumes from the second photo.
Visual notes from the bathroom: object recall good; textured room "interesting, needs refinement",
ghosted room preferred (now the default); the viewer's 45° corner start position read as a tilted
scene next to the face-on photo (now starts from the photo's viewpoint).

### 3.8 Pending

Still to do: a 5–6 photo sweep including a dark scene
(`demo_day/lib2.webp`), an outdoor one (`demo_day/o2.webp`) and a
portrait-orientation phone photo, with screenshots of each viewer.

## 4. Results recorded 2026-10-08 (evidence-based placement, textured room)

Same six photos as §3.7 (`9-17-validation/`), same detections and masks
(copied over, `main.py --resume`), SAM 3D rerun once to fill the stage-3
cache, then stage 4 replayed on the CPU. "Before" is the September output
as it was on disk; "after" is this branch. Numbers from
`scripts/audit_scene.py`:

* **silhouette IoU** — each placed mesh projected back through the photo's
  camera, compared with its SAM 3 mask (mean over objects; 1.0 = the object
  is exactly where and how big the photo shows it);
* **< 0.3** — objects that are badly off;
* **offset** — distance between the object's visible surface and the point
  map under its mask, in units of the object's own size.

### 4.1 Placement

| scene | objects | silhouette IoU | objects < 0.3 | offset |
| --- | ---: | ---: | ---: | ---: |
| bathroom3 | 36 → 34 | 0.35 → **0.51** | 18 → 6 | 0.55 → 0.30 |
| cl5 (classroom) | 30 → 24 | 0.35 → **0.49** | 16 → 6 | 0.35 → 0.21 |
| k1 (kitchen) | 38 → 35 | 0.29 → **0.50** | 20 → 6 | 0.47 → 0.30 |
| lib2 (library) | 44 → 38 | 0.28 → **0.58** | 22 → 2 | 0.61 → 0.17 |
| lr2 (living room) | 35 → 32 | 0.23 → **0.48** | 22 → 8 | 0.47 → 0.34 |
| ucsd-basement lab | 18 → 17 | 0.28 → **0.58** | 11 → 2 | 0.52 → 0.24 |
| **mean** | | **0.30 → 0.52** | 109 → 30 of ~200 | 0.50 → 0.26 |

For reference, SAM 3D's poses with *no* placement at all score 0.48 on the
bathroom: the September placement (0.35) was making scenes worse than
leaving them alone, mostly through box-based "support" relations (the
shower tray's box carried the sink, the rug and a side table) and a floor
plane tilted 3.3° by a mask that spanned two floor levels.

What the three reported problems turned out to be:

* **Bathroom "standing on a ramp".** Floor mask = carpet + raised shower
  tray; one plane through both is tilted 4.2° from the camera's up. Gravity
  from every horizontal and vertical surface in the point map
  (`room_layout.refine_gravity`, 81 % of normals agree) differs from that
  fit by 3.3° and is right. The floor height is now the lowest
  well-supported level, not the plane's average.
* **Classroom desks inside each other.** SAM 3 returned a two-desk pod and
  each desk top as separate "desk" masks (IoU < 0.5, so NMS kept all), and
  two masks for one desk further back. `placement.contained_parts` removes
  masks that are ≥ 85 % inside a larger same-label mask;
  `find_duplicates` removes same-label objects sharing a volume. Four
  duplicates went (two desk tops, one doubled desk, and a "presentation
  screen" that was the picture area of the other one), plus two meshes
  that did not match their masks.
* **Kitchen items off the counter.** Three causes: (1) support assigned by
  box overlap ("knife block on appliance", "pot on stove on oven") moved
  items up to a metre; (2) depth errors per object, now refit against the
  point map (10 objects, factors 0.91–1.35; 5–10 objects in every scene); (3) there was no counter — the
  detector is told not to list countertops, so correctly placed fruit
  floated over nothing. The relief (§4.2) is what they stand on now.

Other removals per scene are listed in `reconstruction_results.json →
failed` with a reason (`structural: floor`, `part or group of #20`,
`removed: mesh does not match its mask (IoU 0.03)`, `duplicate of #23`).

### 4.2 Room

| | September | October |
| --- | --- | --- |
| representation | 96×96 vertex-coloured grid per plane | one UV-textured quad per plane, texture long side 1.5× the photo's (512–2048 px), in `room.glb` |
| hidden areas | nearest visible colour dragged sideways, blended with the median | largest visible rectangle tiled with mirror symmetry and feathered in; median colour when the visible part is not texture-like; unseen walls take the seen walls' colour |
| objects baked into the room | yes (a second rug on the floor) | no (object pixels are excluded) |
| built-ins (counters, cabinets, beams, sloped ceilings) | absent | photo relief from the point map: 3k–73k triangles, 3–42 % of the frame (most in the kitchen, least in the living room) |

Fill mode chosen per plane on the six scenes: floors 2× tiled / 4× median;
far walls 1× tiled / 5× median. So the material-continuing fill fires on a
minority of planes today (bathroom carpet, classroom wood floor); the rest
are either too occluded to offer a large enough visible rectangle or not
texture-like (walls with windows, screens, shelving) and get the median
colour, which is flat but clean. Never-seen planes take the seen walls'
colour.

### 4.3 Exports

| scene | `scene.glb` | `scene_lite.glb` | triangles full → lite |
| --- | ---: | ---: | --- |
| k1 | 394 MB | 16.6 MB | 19.4 M → ~0.35 M |
| lib2 | 491 MB | 15.0 MB | |
| lr2 | 381 MB | 14.3 MB | |
| ucsd lab | 209 MB | 7.5 MB | |
| bathroom3 | 462 MB | 22.8 MB | |
| cl5 | 186 MB | 9.4 MB | |

MuJoCo (`python -m src.mujoco_export outputs/10-08-validation/k1_q --stabilize --check`):
35 bodies, 16 free; unstabilised, 6 of them drift > 5 cm in 2 s (faucet,
a bottle, bowl, cutting board, a fruit, rolling pin); after `--stabilize`
10 remain free with ≤ 4 cm drift. The `--mjx` variant loads and steps in
`mujoco.mjx` (CPU JAX, 100 steps in 1.4 s after compile).

### 4.4 End to end from Blender, on a photo not used for tuning

`blender --background --python scripts/test_blender_addon.py -- --photo test_images/bedroom.jpeg`
against a local server with the local VLM (Blender 5.0.1):

| | |
| --- | --- |
| pipeline | 348 s: detect 34 s, segment 13 s, reconstruct + assemble 275 s (40 instances → 34 objects kept) |
| removed | "wooden floor" (structural: floor), 4 nested "window shutter" masks, 1 pillow (part of another mask) |
| placement | silhouette IoU 0.57 (median 0.63), 4 objects < 0.3, 11 on the floor |
| peak VRAM (allocated) | segment 3.9 GB, reconstruct 19.2 GB |
| import | 13.7 MB lite GLB, 0.4 s, 34 objects + 7 room parts, 41/41 meshes textured, 277k polygons |

`blender --command extension validate` passes; it failed before on two
over-long manifest strings. The viewer was checked in headless Chrome
(textured room, relief toggle, object list).

### 4.5 Reproducing

```bash
# once per scene, GPU: fills 3d_models/stage3/
python main.py --input-dir 9-17-validation --output outputs/10-08-validation --detector qwen --resume
# afterwards, CPU only:
python scripts/replay_assembly.py outputs/10-08-validation/k1_q                      # this branch
python scripts/replay_assembly.py outputs/10-08-validation/k1_q --placement legacy   # September rules
python scripts/replay_assembly.py outputs/10-08-validation/k1_q --placement raw      # SAM 3D untouched
python scripts/audit_scene.py outputs/10-08-validation/k1_q                          # table + overlay
python scripts/render_scene.py outputs/10-08-validation/k1_q                         # contact sheet
```

## 5. Known limitations seen in these runs

* **Partially visible objects stay partial.** A chair seen as a seat back
  behind a desk becomes a small floating slab: SAM 3D reconstructed what
  the mask showed. Placement leaves it where the photo has it. `PIPE-19`.
* **The relief is 2.5D** and raw: wavy counter tops, no back sides, holes
  where large objects stood in front. `PIPE-22`.
* **Tiling shows on strongly lit floors** (bathroom: mirrored shadow
  blobs). `PIPE-10`.
* **Faint object ghosts on walls** where a mask was a few pixels too small
  (lr2's pendant lamps). Dilating blocked pixels further trades this
  against losing real wall.
* **Six of 16 free kitchen objects do not rest** in MuJoCo without
  `--stabilize`: their supporter's convex hull is not under them.

* Ceiling height is "measured" when a downward-facing plane is visible,
  otherwise a prior (2.6 m at the camera-height scale). Both living-room
  runs report ~4.1 m because the camera-height prior (1.5 m) is too high
  for a photo taken from sofa height — metric scale is `PIPE-6`.
* Floor texture coverage is 15–25 % in furnished rooms: most of the floor
  is occluded. The fill is nearest-sample + median colour; real inpainting
  is `PIPE-10`.
* Side walls are often weak (few votes) because they are foreshortened;
  the box falls back to point-cloud extents there, which can include
  flying points beyond a doorway.
