# WorldBuilder roadmap

One place for everything that is planned, blocked, or half-done, with a
suggested order. Tickets have stable ids so commits, PRs and docs can refer
to them (`PIPE-3`, `PUB-1`, …). Status as of **2026-09-17**.

Related documents: [AUDIT.md](AUDIT.md) (what was wrong and what was fixed),
[VALIDATION.md](VALIDATION.md) (how to verify a change on the GPU, with
before/after numbers), [PUBLISHING.md](PUBLISHING.md) (JOSS),
[INTEGRATIONS.md](INTEGRATIONS.md) (Blender/Unreal),
[ROOMBUILDER.md](ROOMBUILDER.md) (the sister app).

Legend: ☐ todo · ◐ in progress · ☑ done · ⛔ blocked (by what)

---

## 0. Suggested order of work

The dependencies below drive this order more than priority does.

1. **Land and re-validate this branch** (`audit-and-roadmap`): run
   VALIDATION §2 on 5–6 photos, look at the viewers, merge. Everything
   else builds on the shared point map and the layout stage.
2. **Local-VLM quality** (`PIPE-1..3`): the cheapest large win for the
   "fully local" claim; it is mostly prompt/decoding/benchmark work and can
   be done in an evening each.
3. **Metric scale** (`PIPE-6`): unblocks RoomBuilder phase 3, the Unreal
   importer's units, and makes the ceiling prior unnecessary. One
   dependency swap (MoGe-2 metric variant) plus a scale-propagation pass.
4. **Doors/windows + non-box rooms** (`PIPE-8`, `PIPE-9`): the visible
   quality jump for interiors.
5. **Publishing groundwork in parallel** (`PUB-*`): commits, tags,
   changelog, Zenodo DOI are zero-risk and clock-driven (JOSS wants six
   months of history). Start the clock now.
6. **Blender end-to-end test** (`DCC-1`) as soon as any machine with
   Blender can reach the server; it is the on-ramp for external users,
   which JOSS needs.
7. **RoomBuilder phase 1** (`RB-*`) once the UCSD data question is
   answered; independent of the pipeline until phase 3.

---

## 1. Pipeline quality — `PIPE-*`

| id | status | item | notes / blockers |
| --- | --- | --- | --- |
| PIPE-1 | ◐ | **Local VLM detection recall.** Benchmark Qwen3-VL-8B with the new prompt against Gemini on the 20 `_g` scenes (`scripts/bench_detectors.py --live demo_day`), then try: (a) two-pass detection (list, then "what did you miss?" with the first list in context), (b) tiled detection for wide/cluttered scenes (2×2 crops + merge), (c) a larger local model — Qwen3-VL-32B-AWQ (~20 GB, fits after SAM 3D is parked) or Gemma-3-27B-it (4-bit). | Needs GPU. Criteria: mean recall vs Gemini ≥ 0.8 without exceeding 60 s/photo. |
| PIPE-2 | ☑ | Repetition loops, prompt drift, dead 0.35 threshold, hard `single` cap, sentence prompts — fixed (AUDIT §1–2). | Validate on more scenes (VALIDATION §2). |
| PIPE-3 | ☐ | **Box-prompt path validation.** Qwen3-VL `bbox_2d` → SAM 3 geometric fallback is implemented but its hit rate is unmeasured. Log how often it fires and whether the resulting masks survive NMS. | Needs GPU; one afternoon. |
| PIPE-4 | ☐ | **Cluttered scenes: tiled segmentation.** SAM 3 runs at 1008 px; small objects in a 4K kitchen photo are a handful of pixels. Run SAM 3 on 2×2 overlapping crops for labels that returned nothing at full resolution, merge with NMS. | Straightforward; expect ~2× stage-2 time only when it fires. |
| PIPE-5 | ☐ | **Lighting/low-contrast robustness.** Try CLAHE / exposure normalisation before detection and segmentation on the dark demo photos (`demo_day/lib*`, `bl*`). Measure recall change. | Cheap experiment. |
| PIPE-6 | ☐ | **Metric scale.** Replace the camera-height prior (`layout.estimated_metric_scale`) with (a) MoGe-2's metric head or Depth-Pro, or (b) known object sizes (a door is ~2.03 m, a twin XL bed 2.03 m) as anchors. Propagate to poses, room and GLB units. | Unblocks RB-phase 3, DCC-3. Check MoGe-2 licence + VRAM. |
| PIPE-7 | ☑ | Shared point map, gravity alignment, textured floor/walls/ceiling, support-aware placement (AUDIT §3–5). | ◐ textured room needed the intrinsics fix (this branch, rerun pending in VALIDATION). |
| PIPE-8 | ☐ | **Non-rectangular rooms.** Fit multiple vertical planes (sequential RANSAC in the aligned XZ plane) and build the room as an extruded polygon instead of a box; keep the box as fallback. | After PIPE-6 so thresholds are in metres. |
| PIPE-9 | ☐ | **Doors and windows.** SAM 3 prompts "door", "window" already work; cut openings into the wall planes and export them as named nodes (RoomBuilder needs door swing). Windows: emissive quad + the photo crop as texture. | Medium. |
| PIPE-10 | ☐ | **Inpainting for hidden room texture.** Replace nearest-sample fill with a proper inpainter (LaMa is small and permissive; SD-inpaint if quality matters) on the unwrapped plane textures. | Needs a texture atlas per plane (currently vertex colours). |
| PIPE-11 | ☐ | **Object texture baking.** SAM 3D can bake a UV texture (`with_texture_baking=True`); we export vertex colours. Baked textures are far better in Blender/Unreal. Measure time cost. | Cheap to try. |
| PIPE-12 | ☐ | **Mesh decimation/LOD.** SAM 3D meshes are dense (a 30-object scene is ~100 MB). Quadric decimation to a target triangle budget at export, optional. | `trimesh`/`open3d` have it; one flag. |
| PIPE-13 | ☐ | **Permissive-model variant.** Evaluate TRELLIS / Hunyuan3D 2.x / SPAR3D as a SAM 3D replacement for a SAM-licence-free stack (matters for a commercial RoomBuilder backend). | Research; see PUBLISHING §2. |
| PIPE-14 | ☐ | **Outdoor/open scenes.** Layout stage assumes a room; add a "no ceiling / ground plane only" mode when the ceiling is not found and the point map spans > ~15 m. | Small. |
| PIPE-15 | ☐ | **Batch throughput.** Keep all three models resident with SAM 3D quantised (fp8/int8) or on a 48 GB card; removes the park/wake cost (~10 s/scene). | Hardware-dependent. |
| PIPE-16 | ☐ | **Pose refinement across objects.** Use the point map to re-fit each object's depth (median of masked point-map depth vs. mesh depth) — SAM 3D's per-object depth is the main source of "floating" objects. | Promising; medium effort. |

## 2. Code health — `CODE-*`

| id | status | item |
| --- | --- | --- |
| CODE-1 | ☑ | Test suite (39 CPU tests), ruff, CI workflow, pyproject, LICENSE, CITATION. |
| CODE-2 | ☐ | Type-check with `pyright`/`mypy` on `src/` (mostly annotated already). |
| CODE-3 | ☐ | Split `main.process_image` into a `Pipeline` class with stage hooks (progress callbacks for the webapp instead of polling the filesystem). |
| CODE-4 | ◐ | Webapp renamed, detector picker and GLB download added; the "photo point cloud" toggle exists in the viewer only. |
| CODE-5 | ☐ | Vendor three.js into `webapp/static/vendor/` so viewers work offline (the "fully local" claim currently stops at the browser). |
| CODE-6 | ☐ | `scripts/replay_assembly.py`: re-run layout + assembly on an existing scene without re-running the models (needs model-space PLYs to be kept — write `*.model.ply` alongside the baked ones). Makes heuristic tuning a 5-second loop. |
| CODE-7 | ☐ | Structured logging (JSON lines per stage) instead of prints; the timing CSV becomes a view over it. |
| CODE-8 | ☐ | Docker image for the server side (CUDA 12.8 base, both envs) — the biggest setup-time reducer for external users. |

## 3. Publishing — `PUB-*` (details in PUBLISHING.md)

| id | status | item |
| --- | --- | --- |
| PUB-1 | ☑ | `paper/paper.md` + `paper.bib` drafts. Verify SAM 3 / SAM 3D citations against their official reports; add an ORCID. |
| PUB-2 | ☐ | Tag `v0.2.0` after merging this branch; enable Zenodo GitHub integration; add the DOI badge. |
| PUB-3 | ☑ | `docs/ARCHITECTURE.md`: data flow, per-stage JSON schemas, frame conventions. |
| PUB-4 | ☐ | Evaluation set: 30 room photos with per-photo object lists (consented), a results table (recall, time, failures) in the docs. |
| PUB-5 | ☐ | Changelog + monthly tagged releases for ≥ 6 months (JOSS history requirement). |
| PUB-6 | ☐ | One documented research use (own preprint / course project / external lab). |
| PUB-7 | ☐ | Submit to JOSS (or SoftwareX if the "used in research" bar is not met by then). |

## 4. DCC integrations — `DCC-*` (details in INTEGRATIONS.md)

| id | status | item |
| --- | --- | --- |
| DCC-0 | ☑ | GLB scene export + `/api/scenes/<id>[/glb]`; Blender extension scaffold; Unreal plugin scaffold. |
| DCC-1 | ⛔ | Blender end-to-end test. Blocked by: no Blender install that can reach the server (install Blender on the GPU box, or run the server + ngrok). |
| DCC-2 | ☐ | extensions.blender.org account; `blender --command extension validate`; submit. |
| DCC-3 | ☐ | Metric-scale checkbox and material presets in both bridges (after PIPE-6). |
| DCC-4 | ☐ | Unreal: Editor Utility Widget with progress instead of the blocking call. |
| DCC-5 | ⛔ | Fab publisher onboarding + Win64 build for UE 5.4/5.5. Blocked by: no Windows/UE machine. |
| DCC-6 | ☐ | Vertex-colour material + decimation option shipped with the UE plugin. |
| DCC-7 | ☐ | Streaming import (objects appear as they finish). Needs a per-object job endpoint. |

## 5. RoomBuilder — `RB-*` (details in ROOMBUILDER.md and ../RoomBuilder)

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

## 6. Known nuances worth remembering

* **Units.** Everything from MoGe/SAM 3D is scale-invariant. Within a scene
  it is consistent, across scenes it is not. `layout.estimated_metric_scale`
  is a prior (camera at 1.5 m) — good enough for ceilings, wrong for a
  photo taken sitting down. PIPE-6 fixes this properly.
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
* **GPU sharing on the dev box.** A Docker vLLM container (`routing-vllm`)
  auto-restarts and takes 29 GB; `docker stop routing-vllm` before runs.
