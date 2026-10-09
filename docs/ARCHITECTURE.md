# Architecture

How a photo becomes a scene, what each stage writes, and the conventions
that hold it together. Module docstrings carry the details; this page is
the map.

## 1. Data flow

```
photo.jpg
   │
   ▼  stage 1  src/object_detection.py | src/local_vlm_detection.py → vlm_server/server.py
detected_objects_raw.json  ── src/detection_postprocess.py ──▶ detected_objects.json
   │        [{id, label, description, expected_instances, bbox_2d?}]
   ▼  stage 2  src/segmentation.py (SAM 3)
segmentation_results.json + masks/*.png (+ masks/structural_{floor,wall,ceiling}.png)
   │        [{id, label, mask_path, bbox, confidence, prompt_used, source, area_px}]
   ▼  stage 3  src/reconstruction_3d.py (SAM 3D Objects; MoGe point map computed once)
3d_models/stage3/  (src/stage3_cache.py — everything below this line is CPU and replayable)
   │        <id>_<label>.ply   model-space mesh as SAM 3D returned it
   │        raw.json           untouched pose per object {translation, rotation_quaternion[w,x,y,z], scale}
   │        pointmap.npz       scene point map (camera-frame world), validity, intrinsics
   ▼  stage 4  src/scene_assembly.py
   │   4a layout     src/room_layout.py    floor, gravity (all surfaces), yaw, walls, ceiling → layout.json
   │   4b placement  src/placement.py      depth refit, silhouette pose search (src/pose_fit.py), contact
   │                                       support, settling, instance sharing, walls, visibility, duplicates
   │   4c room       src/room_texture.py   textured shell quads (predicted where unseen) + photo relief,
   │                 src/builtin_boxes.py  solid boxes behind the relief → room.glb (+ room.ply)
   ▼  exports  reconstruction_results.json · world-space PLYs · scene_combined.ply
              scene.glb / scene_lite.glb (src/scene_export.py, src/mesh_bake.py)
              viewer.html (src/viewer_generator.py) · mujoco/scene.xml (src/mujoco_export.py)
```

`main.py` orchestrates the CLI; `webapp/server.py` runs the same
`process_image` in a worker thread and derives progress from which files
exist. Stage 4 has no GPU or model dependency, is unit-tested on synthetic
scenes, and can be re-run on any finished scene with
`scripts/replay_assembly.py` (seconds instead of a GPU run).

## 2. Frames and units

* **Camera-frame world** (what SAM 3D returns): OpenCV camera with X and Y
  negated — X left, Y up, Z into the scene, camera at the origin.
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

## 3. Key design decisions

| decision | why |
| --- | --- |
| One prompt file for all detectors | the label becomes SAM 3's text prompt; backends must be steered identically (AUDIT §1.2) |
| Detector output is treated as untrusted | local models loop, invent parts, bake counts into labels; `clean_detections` is the safety net |
| Our threshold is pushed into `Sam3Processor` | its default 0.5 pre-filter made every lower threshold dead code (AUDIT §2.1) |
| Point map computed once, passed to every SAM 3D call | saves N−1 MoGe passes and is the input to the layout stage |
| Layout before placement | gravity alignment must precede any "up"-based heuristic; the floor plane replaces "lowest object" as the floor |
| Placement asks the photo, not the boxes | per object: depth from the point map under its mask, support from the surface under the mask's bottom edge, and every move kept only if the silhouette still matches the mask. Box-overlap and label-list rules (the September version, still in `room_generator.py` as the no-point-map fallback) moved more objects wrong than right (VALIDATION §4.1) |
| Gravity from all surfaces | a floor mask that spans a rug, a step or a shower tray tilts a fitted plane; walls and table tops outvote it |
| Strong walls override object bounds | thousands of point-map votes beat one noisy per-object depth; objects are nudged back inside (capped) |
| Room = textured shell + relief | planes get real textures at photo resolution (unseen parts continue the visible material); what is neither object nor plane (counters, cabinets, beams) is meshed from the point map. A prior-placed plane never inherits pixels from geometry behind it; an observed wall does (windows) |
| Stage 3 output is immutable | meshes, raw poses and the point map are cached, so stage 4 is a pure function of files and can be replayed, compared (`--placement legacy|raw`) and developed without a GPU |
| Thin DCC clients over HTTP | SAM licence, GPU requirements, one server for every consumer |

## 4. Extension points

* **New detector**: implement `detect_objects(image_path) -> List[Dict]`
  emitting the schema in `src/prompts.py`, add a kind in
  `main.build_detector`, done — everything downstream is detector-agnostic.
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
  `process_image` (CODE-3) would remove that.

## 5. Files that matter when debugging

`detected_objects_raw.json` (what the model said), `masks/`
(`debug_masks.py` overlays them), `3d_models/layout.json` (`notes`,
`wall_sources`, `plane_texture_stats`, `relief`, `assembly`),
`reconstruction_results.json` (per object: `fit_iou_raw`/`fit_iou`,
`depth_refit`, `support`, `support_source`; `failed` with reasons),
`audit.png` (`scripts/audit_scene.py`: masks in green, placed meshes in
red), `renders.png` (`scripts/render_scene.py`), `plan_view.png`
(`scripts/plot_scene.py`), and the console log — every stage prints its
decisions with the reason.
