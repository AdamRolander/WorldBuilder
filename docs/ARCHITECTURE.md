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
3d_models/<id>_<label>.ply (model space, vertex colours) + pose per object
   │        {translation[3], rotation_quaternion[w,x,y,z], scale[3]}   (camera frame)
   ▼  stage 4a src/room_layout.py            stage 4b src/room_generator.py
layout.json (gravity R, yaw, floor, ceiling,  poses updated in place: upright, supports,
walls, scale estimate) + room.ply (textured)  containment, snap, collisions, wall clamp
   │
   ▼  exports  reconstruction_results.json · PLYs baked to world space · scene_combined.ply
              scene.glb (src/scene_export.py) · viewer.html (src/viewer_generator.py)
```

`main.py` orchestrates the CLI; `webapp/server.py` runs the same
`process_image` in a worker thread and derives progress from which files
exist. Stage 4 has no GPU or model dependency and is fully unit-tested.

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
* **Units** are MoGe's scale-invariant units, consistent within a scene.
  `layout.estimated_metric_scale` converts to metres under a camera-height
  prior (1.5 m). Nothing on disk is pre-multiplied by it.
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
| Support and containment relations | AABB-only reasoning that still captures lamp-on-table and pillow-in-sofa; avoids the two classic failure modes (stacked items pushed off, floor-snapping things that sit on furniture) |
| Strong walls override object bounds | thousands of point-map votes beat one noisy per-object depth; objects are nudged back inside (capped) |
| Room textured by projection with a surface-consistency test | a prior-placed plane must not inherit pixels from geometry behind it |
| Thin DCC clients over HTTP | SAM licence, GPU requirements, one server for every consumer |

## 4. Extension points

* **New detector**: implement `detect_objects(image_path) -> List[Dict]`
  emitting the schema in `src/prompts.py`, add a kind in
  `main.build_detector`, done — everything downstream is detector-agnostic.
* **New 3D generator** (TRELLIS, Hunyuan3D…): produce the same pose dict
  plus a GLB; `reconstruction_3d` is the only file that knows SAM 3D.
* **Better textures**: `room_layout.textured_plane` is where a UV atlas +
  inpainting would replace vertex colours.
* **Metric scale**: replace `estimated_metric_scale` in `estimate_layout`;
  everything else reads it from `layout.json`.
* **Progress hooks**: the webapp polls the filesystem; a callback in
  `process_image` (CODE-3) would remove that.

## 5. Files that matter when debugging

`detected_objects_raw.json` (what the model said), `masks/`
(`debug_masks.py` overlays them), `3d_models/layout.json` (`notes`,
`wall_sources`, `plane_texture_stats`, `assembly`), `plan_view.png`
(`scripts/plot_scene.py`), and the console log — every stage prints its
decisions with the reason.
