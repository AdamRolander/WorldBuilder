# Changelog

All notable changes to WorldBuilder. Format follows [Keep a Changelog](https://keepachangelog.com/);
versions follow [SemVer](https://semver.org/) (0.x: minor = features, patch = fixes).

## [Unreleased]

### Added
- Documentation split into Setup, Usage, Troubleshooting and Plugins pages (Blender, MuJoCo,
  Unreal, planned targets); the README is now an overview with a before/after animation
  and example stills (`scripts/make_hero.py`: several scenes, objects pop in one at a time
  with their masks lit on the photo; WebP or GIF).
- `SECURITY.md` (private vulnerability reports) and `.github/CODEOWNERS`.
- Job progress: `GET /api/jobs/<id>` reports the step (1-4), objects reconstructed so far and
  an ETA (`3d_models/stage3/progress.json` is written after every object); scene listings
  and manifests report the lite and full GLB sizes and object counts.
- Viewer: axis hints on the selection (±X/±Y/±Z arrows, rotation direction rings, scale
  axes) and multi-select with shift-click (moves, turns and scales apply to the group).
- Blender add-on 0.4.0: step and progress bar with time left; **Review in Browser First**
  (default) opens the finished scene in the browser and imports it after review; a scene
  list with object counts and sizes for **Import Existing Scene**; **Full Detail** import
  option; **Edit in Browser** uses the selected object's scene; Help panel; Title Case labels.
- `scripts/regen_viewer.py --viewer-only`.

### Changed
- The Unreal bridge imports the lite GLB (baked textures) instead of the dense scene.
- The Blender add-on (`integrations/blender_worldbuilder/`) is now licensed
  GPL-3.0-or-later, as extensions.blender.org requires. The rest of the project stays MIT.

## [0.3.0] — 2026-10-17

### Added
- Photo re-projection onto objects (`src/object_texture.py`): the side of each mesh the
  photo saw takes the photo's pixels (own mask, front surface, depth agreement).
- Open scenes: sky in the upper frame, or no ceiling and no wall observed, means no
  invented walls; uneven ground is meshed as terrain; the viewer uses the sky's colour.
- `metrics.json` per scene (`src/profiling.py`): seconds per phase, time to first object,
  seconds per object, device-wide and per-process GPU memory peaks, output sizes.
- `scripts/benchmark.py` and `docs/BENCHMARKS.md`: one table per run and a baseline
  comparison that fails on regressions; `scripts/clone_scene.py`.
- Viewer: loads `scene_lite.glb` by default, hover highlighting, typed position /
  rotation / scale, one checkbox per piece of built-in structure, edits saved to
  `edits.json` (`src/scene_edits.py`) and applied by the GLB endpoints and MuJoCo export.
- Blender add-on 0.3.0: "Review in browser".
- A `window` structural mask (SAM 3). Windows are painted on the wall where seen but never
  teach a wall its material; detections that are the view through a window (trees,
  buildings, blinds in the frame) are left to the wall; window regions that are the glass
  of a cabinet or a display case are not windows. `scripts/add_structural_masks.py` adds
  the mask to scenes segmented before it existed.
- Glass: an object the photo sees through and that the detector describes as glass is
  exported translucent, and the walls behind it keep their texture.
- Placement: resting objects stand upright; objects follow a supporter that later moved;
  objects against a wall are squared to it, and long furniture to the room's axes; wall
  contact climbs over an object's own (unsegmented) top surface; a short mesh whose mask
  reaches down to its support is lengthened down to it.
- A detected supporter whose reconstruction misses its mask is replaced by a box fitted to
  the photo's points instead of being dropped, so what stands on it does not float.
- Segmentation tries a label's head noun ("chair" for "dining chair") before the box fallback.
- `scripts/detect_only.py`: stage 1 for a batch, so the VLM server can be stopped before
  SAM 3 / SAM 3D load.

### Changed

- The detector is told that built-in furniture (islands, cabinet runs, shelving, vanities,
  range hoods) is an asset too; up to 24 instances per label (was 12).
- Room fill: hidden patches take the local colour of the seen surface around them (no
  silhouettes of lamps and chairs on the walls); regular floor and ceiling patterns tile
  by whole periods in phase with the seen part; other materials tile only their fine
  detail; an unseen ceiling takes the colour of the walls' top band instead of white.
- Built-in blocks: a vertical face is extruded no deeper than it is tall or wide, nor
  deeper than its top was seen to run (a bar counter no longer reaches the back wall).
- Photo re-projection onto objects only where the photographed side is flat or a shallow
  facade facing the camera (bookcases keep their books; sofas keep SAM 3D's texture).
- Same-label masks: a mask containing two or more same-label masks is a group and is
  dropped for its members (a row of chairs is no longer one long bar); instance sharing
  checks colour make-up, and a flat object's face, before copying a mesh.
- Detections lying on the observed ceiling plane (tiles, grilles, flush panels) are the
  ceiling's surface.
- The relief is exported as one mesh per connected piece (`room_relief_00`, …), each with
  its own cropped texture.
- Placement is 2–3× faster (silhouette scoring works on the object's bounding box);
  results are identical.
- The web server listens on `127.0.0.1` unless `--host` / `WORLDBUILDER_HOST` says otherwise.
- Headless renders are unlit, matching the browser viewer.
- `scene_combined.ply` is no longer written unless `WORLDBUILDER_COMBINED=1` (nothing read it;
  it doubled each scene's size on disk).

### Fixed
- Spikes in the relief: sheets along the viewing rays at depth edges and across the holes
  left by small objects.

## [0.2.0] — 2026-10-08

### Added (October 2026, second pass)
- Silhouette pose search with an occlusion- and depth-aware score (`src/pose_fit.py`);
  gravity settling of free-standing objects (shift / ray slide / downward stretch);
  ray-slide wall clamp; visibility ordering that ignores transparent objects.
- Instance sharing: fragments and look-alike repeats take the best-observed
  instance's mesh at its true size.
- Predictive room textures: walls row by row from the dominant seen material,
  lighting-free tiling, grazing-angle texels predicted, flush objects kept in the
  wall, unseen walls borrow the best-seen wall's make-up.
- Solid built-ins behind the relief (`src/builtin_boxes.py`), exported as MuJoCo
  collision boxes.
- Metric scale from MoGe-2 (`scripts/estimate_metric_scale.py`, isolated under
  `third_party/`); `layout.metric_scale_source`.
- `scripts/make_gallery.py`; 70 CPU tests.

### Fixed (October 2026, second pass)
- Room box trimmed 1 % of object extents on undetected sides (objects on the boundary).
- A tall object raised an observed ceiling and blanked its texture.
- Audit read the original scene's meshes when run on a copied scene.

### Added (October 2026)
- Evidence-based placement (`src/placement.py`): depth refit along viewing rays
  against the point map, support from the surface under each mask (floor / object /
  unmodelled surface), axis-agnostic upright correction, every move kept only if the
  silhouette still matches the mask; removal of part/group masks, structural
  duplicates, same-volume duplicates and meshes that miss their mask.
- Gravity from all horizontal and vertical surfaces (`room_layout.refine_gravity`)
  and floor height from the lowest supported level (`floor_offset`).
- Textured room shell and photo relief of built-ins (`src/room_texture.py`,
  `3d_models/room.glb`); viewer loads it, with a relief toggle.
- Stage-3 cache (`3d_models/stage3/`) and CPU replay of stage 4
  (`scripts/replay_assembly.py`, `--placement evidence|legacy|raw`); `main.py --resume`.
- `scripts/audit_scene.py` (placement scored against masks and point map) and
  `scripts/render_scene.py` (headless contact sheets).
- `scene_lite.glb`: decimated objects with baked textures (`src/mesh_bake.py`).
- MuJoCo / MJX export (`src/mujoco_export.py`), Gymnasium wrapper and robot
  attachment (`integrations/mujoco/`), `GET /api/scenes/<id>/mujoco`.
- `GET /api/scenes`, `GET /api/scenes/<id>/glb?lite=1`; peak VRAM per stage in the
  results metadata.
- Blender add-on 0.2.0: lite download, import existing scene, metric-scale option;
  `scripts/test_blender_addon.py` end-to-end test.
- Docs: `HARDWARE.md`, `integrations/mujoco/README.md`;
  60 CPU tests.

### Fixed (October 2026)
- Scenes tilted by a floor mask spanning two levels (bathroom: 3.3°).
- Objects lifted or moved by box-overlap "support" relations; countertop items
  displaced by up to a metre; duplicate desks from part masks.
- Blender extension manifest failed validation (tagline and permission strings
  over 64 characters).

### Added
- Structural layout stage (`src/room_layout.py`): floor plane, gravity alignment,
  Manhattan yaw, wall/ceiling detection and photo-projected room textures from a
  single shared MoGe point map; `layout.json`, `scene_pointmap.npz`.
- Support, containment and wall-clamp relations in scene assembly; objects rest
  on what they stand on, stay inside detected walls, and hanging fixtures are
  never floor-snapped.
- SAM 3 hand-off improvements: label+description prompt union, soft `single`
  prior, box-prompt fallback from VLM `bbox_2d`, mask hygiene, floor/wall/ceiling masks.
- Shared detection prompt (`src/prompts.py`), Gemini JSON mode, detector-agnostic
  clean-up (runaway-loop dedupe, count stripping, people/architecture/part filters).
- GLB scene export, `GET /api/scenes/<id>` and `/glb` endpoints, webapp detector
  picker and GLB download; viewer shows textured room, supports, failures, point cloud.
- Blender extension and Unreal plugin scaffolds (`integrations/`).
- CPU test suite (39 tests), ruff config, CI workflow, `pyproject.toml`, LICENSE (MIT),
  CITATION.cff, CONTRIBUTING.md, docs (architecture, integrations).
- Scripts: `bench_detectors.py`, `plot_scene.py`, `regen_viewer.py` (moved).

### Changed
- MoGe runs once per scene instead of once per object.
- Pose math is numpy (`src/geometry.py`), verified against pytorch3d; no GPU in stage 4.
- `Sam3Processor` confidence threshold is now the pipeline's (was a dead 0.35 behind a 0.5 pre-filter).
- Floor-category matching is token-based (no more "table lamp" on the floor).
- Collision resolution requires real interpenetration and caps displacement.
- Local VLM server: repetition penalty, lower token cap, `--model` flag, bbox output.
- `main.py` gained `--detector`, `--quality`, `--no-room`, `--no-structural`; batch `--force` no longer monkey-patches.
- Webapp path checks use `Path.relative_to`; stage 4 is reported; jobs are pruned.

### Fixed
- `KeyError: 'CONDA_PREFIX'` when running outside an activated conda shell.
- Upstream drops MoGe intrinsics; recovered from the point map by least squares.
- Failed reconstructions are recorded instead of silently skipped.

## [0.1.0] — 2026-05 (demo-day pipeline, branch `MigrateSAM`)
- Gemini / Qwen3-VL detection → SAM 3 → SAM 3D Objects → box room → WebXR viewer; Flask webapp.
