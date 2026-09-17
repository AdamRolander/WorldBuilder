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
| Assembly | console: `support relations`, `snapped`, `collision resolution`, `nudged … inside detected walls` | no hanging object snapped; supports look physical; pushes < number of objects |
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

### 3.7 Pending

Still to do: a 5–6 photo sweep including a dark scene
(`demo_day/lib2.webp`), an outdoor one (`demo_day/o2.webp`) and a
portrait-orientation phone photo, with screenshots of each viewer.

## 4. Known limitations seen in these runs

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
