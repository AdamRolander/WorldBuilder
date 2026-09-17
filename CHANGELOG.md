# Changelog

All notable changes to WorldBuilder. Format follows [Keep a Changelog](https://keepachangelog.com/);
versions follow SemVer once `v0.2.0` is tagged.

## [Unreleased] — branch `audit-and-roadmap`

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
  CITATION.cff, CONTRIBUTING.md, docs (architecture, audit, roadmap, validation,
  integrations).
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
