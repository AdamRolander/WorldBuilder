# Architecture

How a photo becomes a scene, what each stage writes, and the conventions
that hold it together. Module docstrings carry the details; this page is
the map.

## 1. Data Flow

```
photo.jpg
   │
   ▼  stage 1  src/object_detection.py | src/local_vlm_detection.py → vlm_server/server.py
detected_objects_raw.json  ── src/detection_postprocess.py ──▶ detected_objects.json
   │        [{id, label, description, expected_instances, bbox_2d?}]
   ▼  stage 2  src/segmentation.py (SAM 3)
segmentation_results.json + masks/*.png (+ masks/structural_{floor,wall,ceiling,window}.png)
   │        [{id, label, mask_path, bbox, confidence, prompt_used, source, area_px}]
   ▼  stage 3  src/reconstruction_3d.py (SAM 3D Objects; MoGe point map computed once)
3d_models/stage3/  (src/stage3_cache.py; everything below this line is CPU and replayable)
   │        <id>_<label>.ply   model-space mesh as SAM 3D returned it
   │        raw.json           untouched pose per object {translation, rotation_quaternion[w,x,y,z], scale}
   │        pointmap.npz       scene point map (camera-frame world), validity, intrinsics
   ▼  stage 4  src/scene_assembly.py
   │   4a layout     src/room_layout.py    floor, gravity (all surfaces), yaw, walls, ceiling → layout.json
   │       clean-up  src/scene_assembly.py  parts and group masks, structural detections (floor, ceiling,
   │                                       views through a window), glass
   │   4b placement  src/placement.py      depth refit, silhouette pose search (src/pose_fit.py), contact
   │                                       support (proxy boxes for failed supporters), settling, upright,
   │                                       squaring, instance sharing, walls (contact, gap closing),
   │                                       restacking, visibility, duplicates
   │   4c room       src/room_texture.py   textured shell quads (predicted where unseen) + photo relief,
   │                 src/builtin_boxes.py  solid boxes behind the relief → room.glb (+ room.ply)
   ▼  exports  reconstruction_results.json · world-space PLYs · metrics.json
              scene.glb / scene_lite.glb (src/scene_export.py, src/mesh_bake.py)
              viewer.html (src/viewer_generator.py) · mujoco/scene.xml (src/mujoco_export.py)
```

`main.py` orchestrates the CLI; `webapp/server.py` runs the same
`process_image` in a worker thread and derives progress from which files
exist. Stage 4 has no GPU or model dependency, is unit-tested on synthetic
scenes, and can be re-run on any finished scene with
`scripts/replay_assembly.py` (seconds instead of a GPU run).

## 2. Frames and Units

* **Camera-frame world** (what SAM 3D returns): OpenCV camera with X and Y
  negated: X left, Y up, Z into the scene, camera at the origin.
  `room_layout.OPENCV_TO_WORLD = diag(-1,-1,1)`.
* **Aligned world** (what is on disk after stage 4): camera-frame world
  rotated by `R_total = Yaw(yaw_deg) · G` where `G` maps the fitted floor
  normal to +Y and the yaw aligns the dominant wall direction with the
  axes. Stored in `layout.json` (`gravity_rotation`, `yaw_deg`).
* **Object pose composition** (row-vector form, matches upstream
  `compose_transform`): `world = (v_gaussian · scale) @ R(q) + t`, where
  `v_gaussian` is the GLB vertex after the Y/Z swap
  `(x, y, z) → (x, −z, y)`. `src/geometry.world_vertices` is the single
  implementation; `tests/test_geometry.py` checks it against pytorch3d.
* **Units** are MoGe-1's scale-invariant units, consistent within a scene.
  `layout.estimated_metric_scale` converts to metres: from MoGe-2's metric
  point map when `3d_models/stage3/metric_scale.json` exists, otherwise a
  camera-height prior (1.5 m); `layout.metric_scale_source` says which.
  Nothing on disk is pre-multiplied by it.
* **Intrinsics** are normalised (MoGe convention): `u = fx·x/z + cx` with
  `u ∈ [0,1]`, recovered from the point map when the depth model's are not
  passed through.

## 3. Key Design Decisions

| decision | why |
| --- | --- |
| One prompt file for all detectors | the label becomes SAM 3's text prompt; backends must be steered identically |
| Detector output is treated as untrusted | local models loop, invent parts, bake counts into labels; `clean_detections` is the safety net |
| Our threshold is pushed into `Sam3Processor` | its default 0.5 pre-filter made every lower threshold dead code |
| Point map computed once, passed to every SAM 3D call | saves N−1 MoGe passes and is the input to the layout stage |
| Layout before placement | gravity alignment must precede any "up"-based heuristic; the floor plane replaces "lowest object" as the floor |
| Placement asks the photo, not the boxes | per object: depth from the point map under its mask, support from the surface under the mask's bottom edge, and every move kept only if the silhouette still matches the mask. Box-overlap and label-list rules (the September version, still in `room_generator.py` as the no-point-map fallback) moved more objects wrong than right |
| Gravity from all surfaces | a floor mask that spans a rug, a step or a shower tray tilts a fitted plane; walls and table tops outvote it |
| Strong walls override object bounds | thousands of point-map votes beat one noisy per-object depth; objects are nudged back inside (capped) |
| Room = textured shell + relief | planes get real textures at photo resolution (unseen parts continue the visible material: regular patterns by whole periods, others by their fine detail, with the local colour of the seen surface around each hidden patch); what is neither object nor plane (counters, cabinets, beams) is meshed from the point map. A prior-placed plane never inherits pixels from geometry behind it; an observed wall does (windows), but a window never teaches a wall its material |
| Stage 3 output is immutable | meshes, raw poses and the point map are cached, so stage 4 is a pure function of files and can be replayed, compared (`--placement legacy|raw`) and developed without a GPU |
| Thin DCC clients over HTTP | SAM licence, GPU requirements, one server for every consumer |

## 4. Extension Points

* **New detector**: implement `detect_objects(image_path) -> List[Dict]`
  emitting the schema in `src/prompts.py`, add a kind in
  `main.build_detector`, done; everything downstream is detector-agnostic.
* **New 3D generator** (TRELLIS, Hunyuan3D…): produce the same pose dict
  plus a GLB; `reconstruction_3d` is the only file that knows SAM 3D.
* **Better textures**: `room_texture.fill_hidden` is where an inpainter
  would replace mirrored tiling; `room_texture.build_relief` is where plane
  fitting would turn the relief into clean geometry.
* **New placement evidence** (a VLM hint, a second view): add it as another
  correction in `placement.place_objects`; the silhouette-IoU guard there
  is what keeps a wrong hint from doing damage.
* **New export target**: read `scene_lite.glb` + `layout.json` +
  `reconstruction_results.json` (see `src/mujoco_export.py`, ~300 lines).
* **Metric scale**: anything that writes `stage3/metric_scale.json`
  (`scripts/estimate_metric_scale.py` does, with MoGe-2) sets it;
  everything else reads it from `layout.json`.
* **Progress hooks**: the webapp polls the filesystem; a callback in
  `process_image` would remove that.

## 5. Files That Matter When Debugging

`detected_objects_raw.json` (what the model said), `masks/`
(`debug_masks.py` overlays them), `3d_models/layout.json` (`notes`,
`wall_sources`, `plane_texture_stats`, `relief`, `assembly`),
`reconstruction_results.json` (per object: `fit_iou_raw`/`fit_iou`,
`depth_refit`, `support`, `support_source`; `failed` with reasons),
`audit.png` (`scripts/audit_scene.py`: masks in green, placed meshes in
red), `renders.png` (`scripts/render_scene.py`), `plan_view.png`
(`scripts/plot_scene.py`), and the console log: every stage prints its
decisions with the reason.

## 6. Repository Layout

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
  placement.py              stage 4b: depth refit, contact support, settling, walls, visibility, duplicates
  pose_fit.py               stage 4b: silhouette pose search and instance sharing
  room_texture.py           stage 4c: textured shell quads (unseen parts predicted) + photo relief
  builtin_boxes.py          stage 4c: solid boxes for counters/cabinets behind the relief
  room_generator.py         legacy box/label placement (fallback when there is no point map)
  object_texture.py         stage 4d: re-project the photo onto the visible side of each object
  mesh_bake.py              decimation + vertex-colour → texture baking (scene_lite.glb)
  scene_edits.py            manual corrections from the viewer (edits.json) applied to the GLB exports
  profiling.py              seconds and GPU memory per phase → metrics.json
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
  estimate_metric_scale.py  metres per scene unit from MoGe-2 (isolated install)
  make_gallery.py           before/after HTML page for two runs of the same photos
  make_hero.py              README animation: the photo next to a camera sweep through the scene
  replay_assembly.py        re-run stage 4 on a finished scene (CPU, seconds)
  audit_scene.py            score placement against the photo's masks and point map
  render_scene.py           headless contact-sheet renders (pyrender + EGL)
  benchmark.py              one table per run: placement, speed, memory, size, appearance; compares two runs
  clone_scene.py            light copy of scenes for replaying stage 4 without touching the originals
  test_blender_addon.py     end-to-end add-on test, run inside `blender --background`
tests/                      CPU-only pytest suite
docs/                       setup, usage, architecture, benchmarks, hardware, plugins (plugins/)

checkpoints/                SAM 3 weights            (gitignored)
sam3_repo/                  upstream clone           (gitignored)
sam3d_objects_repo/         upstream clone + weights (gitignored)
pytorch3d/                  upstream clone           (gitignored)
outputs/                    generated scenes         (gitignored)
```


