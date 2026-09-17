# Code audit — September 2026

Scope: everything committed on `MigrateSAM` as of `a47ffff` (the demo-day
pipeline) plus the 54 scene directories left in `outputs*/` from those demos.
The goal was to find band-aids that were reasonable under demo pressure but
wrong for a project that wants to be published, plugged into DCC tools and
advertised as "fully local".

Each finding lists the evidence, the fix that landed on the
`audit-and-roadmap` branch, and how it was verified. Items marked **open**
are tracked in [ROADMAP.md](ROADMAP.md).

---

## 0. Method

* Read every source file (≈3 000 lines).
* Mined the existing outputs: for every scene, compared the VLM's object
  list with what SAM 3 actually produced masks for.
* Read the upstream SAM 3 / SAM 3D code the pipeline calls, to check
  assumptions the wrapper code makes about them.
* Ran the unmodified pipeline on three photos (baseline) and the modified
  pipeline on the same photos (see [VALIDATION.md](VALIDATION.md)).
* Added a CPU-only test suite (`tests/`, 39 tests) that pins the geometry
  and every heuristic that used to live only in prose.

Detection-to-segmentation loss across the historical outputs, before any
change (a "lost type" is a label the VLM emitted that produced zero masks):

| detector          | scenes | object types | lost types |
| ----------------- | -----: | -----------: | ---------: |
| Gemini (`_g`)     |     20 |          326 |   65 (20%) |
| local Qwen (`_q`) |     19 |          628 |  285 (45%) |
| older prompts     |     15 |          232 |   80 (34%) |

The Qwen number is dominated by three runaway scenes (see §1.1); excluding
them the local detector loses about the same fraction as Gemini, which says
the hand-off to SAM 3 (§2), not the VLM, is the larger problem.

---

## 1. Stage 1 — object detection

### 1.1 The local VLM loops and nobody stops it  (**fixed**)

`outputs_og2/ucsd-basement-prototype-lab_q_41beef/detected_objects.json`
holds 116 entries, 107 of them the label `tool` with an identical
description. `factory6_q` has 132 entries, `kitchen_q` 115, `factory4_q`
100. Qwen3-VL-8B under greedy decoding with `max_new_tokens=4096` and no
repetition penalty falls into a loop; the server happily parsed it and the
pipeline then ran SAM 3 once per duplicate (107 × the same prompt, each
returning 15 masks) before cross-label NMS threw the duplicates away.

Fix: `repetition_penalty`, `max_new_tokens=2048`, server-side label
dedupe (`vlm_server/server.py: dedupe_objects`), and a second,
detector-agnostic dedupe/normalise pass in `src/detection_postprocess.py`
that also strips counts baked into labels (`"glass storage jars (2)"`) and
caps the list. Verified by `tests/test_postprocess.py::test_runaway_vlm_output_is_collapsed`
and `tests/test_vlm_parsing.py`.

### 1.2 Two prompts that had drifted apart  (**fixed**)

The Gemini prompt and the Qwen prompt were separate copies. One said
"indoor scene", the other did not; one listed "structures" and
"playground" as examples. `forest_q`, `playground_q` and `maze_q` all
returned **zero** objects with the local prompt while Gemini found things
in the same photos. Both backends now import `src/prompts.py`; the prompt
explicitly includes outdoor objects and asks for a `bbox_2d` per type.

### 1.3 Gemini parsing was string surgery  (**fixed**)

Fence stripping by `split('```')[1]`. Gemini supports
`response_mime_type="application/json"`; the detector now uses it (with the
old stripping kept as a fallback) and a configurable model id.

### 1.4 `expected_instances` is unreliable from the local model  (**fixed downstream**)

Qwen marks nearly everything `single` (bathroom1_q: 11 of 12). Stage 2 used
that as a hard truncation to one mask. See §2.3.

### 1.5 People filter matched substrings  (**fixed**)

`'man' in label` dropped "ottoman", "manual", "mantel". Token-based now
(`is_person`).

### 1.6 Importing the Gemini detector pulled in `google-genai` even for local runs  (**fixed**)

`main.py` imported it at module top. A "fully local" install without the
Google SDK could not start. Detectors are now imported lazily by kind.

---

## 2. Stage 2 — segmentation (the VLM → SAM 3 hand-off)

### 2.1 A dead confidence threshold  (**fixed, highest-impact single bug**)

`Sam3Processor` (upstream) filters candidates at its own
`confidence_threshold`, default **0.5**, inside `_forward_grounding`, before
returning anything. `segment_objects` then applied `score_threshold=0.35`
to what was left. Nothing between 0.35 and 0.5 ever existed, so the
"retry with label only when max score < threshold" branch and the "no
masks above threshold, using best" branch were mostly unreachable. Objects
SAM 3 was moderately sure about were silently dropped.

Fix: the processor's threshold is set to ours (default 0.3) before each
scene; NMS handles the extra candidates. This is the change most likely to
recover the "toilet paper holder / hand towel / waste bin / 3D printer"
class of losses seen in `bathroom1` and `ucsd-basement-prototype-lab_g`.

### 2.2 Prompting SAM 3 with a sentence  (**fixed**)

The primary SAM 3 prompt was the VLM's *description*
("Green metal storage cabinet with multiple drawers and wheels"); the label
was only a fallback. SAM 3's text encoder is trained on short noun
phrases. Now both the label and a shortened description are queried and
the union is kept (one extra decoder pass per type on an already-encoded
image; ≈0.1 s on the 5090).

### 2.3 `single` was a hard cap  (**fixed**)

Now a soft prior: additional instances survive if their score is within
85 % of the best (`apply_single_prior`). A room with three identical
chairs no longer gets one.

### 2.4 No geometric fallback  (**fixed**)

SAM 3 accepts box prompts (`add_geometric_prompt`). When both text prompts
fail and the VLM supplied `bbox_2d`, the box is used. Gemini boxes are
approximate; Qwen3-VL boxes are usually tight.

### 2.5 Mask hygiene  (**fixed**)

Masks with tiny disconnected specks are trimmed to their dominant
component; masks covering >60 % of the frame (a "floor" that slipped past
the prompt) are rejected. `tests/test_segmentation_helpers.py`.

### 2.6 Cross-label NMS was O(n²) over full-resolution masks in Python  (**kept, refactored**)

Fine at n≈30; it is now a single reusable `nms_masks` used within-label and
across labels. A bitmask/IoU-on-bboxes prefilter is an easy speed-up if
scenes with 100+ instances become common (**open**).

---

## 3. Stage 3 — reconstruction

### 3.1 MoGe ran once per object  (**fixed**)

`InferencePipelinePointMap.run` computes a full-image MoGe point map on
every call and exposes `pointmap=` to skip it. The wrapper never used it,
so a 27-object scene ran the ViT-L depth model 27 times and threw the
result away each time. `SAM3DReconstructor.compute_scene_pointmap` now
runs it once and passes it to every object (fallback to per-object on any
error, opt-out via `WORLDBUILDER_SHARED_POINTMAP=0`). The same point map
feeds the new layout stage (§4).

### 3.2 Three copies of the pose transform, all on CUDA  (**fixed**)

`_bake_world_space_plys`, `_save_combined_scene` and
`room_generator._transform_verts` each rebuilt
`compose_transform(scale, quaternion_to_matrix(q), t)` on the GPU through
pytorch3d, for a few thousand vertices. One numpy function
(`src/geometry.py: world_vertices`) replaces them and is checked against
pytorch3d to 1e-9 in `tests/test_geometry.py`. The assembly stage no
longer needs a GPU, pytorch3d or the sam3d_objects package.

### 3.3 The notebook wrapper requires an activated conda shell  (**worked around**)

`sam3d_objects_repo/notebook/inference.py` does
`os.environ["CUDA_HOME"] = os.environ["CONDA_PREFIX"]` at import. Running
via the env's interpreter directly (cron, an IDE, `nohup`) crashed with
`KeyError: 'CONDA_PREFIX'`. `reconstruction_3d.py` now derives it from
`sys.executable` when missing.

### 3.4 Failures vanished  (**fixed**)

A failed object was `continue`d past; the results file and the timing CSV
could not tell "not detected" from "reconstruction crashed". Failures are
recorded with `status` and shown greyed out in the viewer.

### 3.5 Dead code  (**removed**)

`save_colored_mesh` (Poisson from Gaussians; never called), a 40-line
commented-out `_save_scene_with_room`, commented-out
`stretch_vertical_structural` wiring with its keyword list also commented
out so the live function referenced an undefined name.

---

## 4. Stage 4 — scene assembly

### 4.1 Substring keyword matching forced the wrong things to the floor  (**fixed**)

`FLOOR_SUPPORTED_KEYWORDS` was matched with `k in label`:
`'table'` → "table lamp", "tablet"; `'bed'` → "bedside lamp", "bedding";
`'oven'` → "microwave oven"; `'desk'` → "desk lamp", "desktop computer";
`'planter'/'potted plant'` → plants on a vanity. Every one of those was
slammed to the floor. Matching is now on the head noun with an explicit
veto list (`tests/test_room_generator.py::test_floor_keyword_matching_is_token_based`).

### 4.2 The floor was "whichever object was lowest"  (**fixed**)

One object with a bad pose below everything else redefined the floor and
dragged every snapped object down to it. The floor now comes from the
point map (RANSAC plane, §5) and, without a point map, from the median
bottom of floor-category objects.

### 4.3 Collision resolution shoved stacked objects off their supports  (**fixed**)

`resolve_xz_collisions` looked only at XZ overlap. A vase on a table
overlaps the table's footprint completely, so it was pushed sideways off
the table; a chair tucked under a table was pushed out. There is now a
support graph (`find_supports`) — footprint containment plus a
bottom-near-top test — used to (a) snap objects onto what they rest on,
(b) skip supporter/supported pairs, (c) move riders with their supporter.
Collisions also require Y overlap and ≥70 % footprint overlap of the
smaller box, and pushes are capped at half the object's extent.

### 4.4 Camera tilt was never corrected  (**fixed**)

`upright_correct` assumed world +Y is up. SAM 3D poses are in the camera
frame; a photo taken pitched 15° down produces a room and objects tilted
15°. The layout stage measures the floor normal and rotates the scene
(objects included, via `rotate_pose`) so that gravity is −Y before any
snapping. Also aligns yaw to the dominant wall direction.

### 4.5 `is_already_processed = lambda x: False` rebinding a module global  (**fixed**)

`--force` in batch mode worked by monkey-patching a function at module
scope inside `__main__`. Replaced with a `force` argument.

---

## 5. Structural elements (new)

`src/room_layout.py` derives floor, gravity, Manhattan yaw, walls and
ceiling from the shared MoGe point map, optionally restricted by SAM 3
"floor / wall / ceiling" masks (which `main.py` now requests in stage 2 at
negligible cost), and textures the six room planes by projecting the photo
through the recovered intrinsics with a surface-consistency depth test.
Regions the photo never saw are filled from the nearest textured sample
blended with the plane's median colour. The box room from before remains
as the fallback. Verified by synthetic ray-cast rooms in
`tests/test_room_layout.py` (recovered floor/ceiling/walls to <5 cm, wall
angle to <0.1°, and >92 % of textured vertices carry the true surface
colour) and on a real photo on CPU.

---

## 6. Web app

* Path-traversal check used `str.startswith`; `outputs_og/` was reachable
  as a "child" of `outputs/`. Now `Path.relative_to`.  (**fixed**)
* Stage 4 was never reported (`_derive_stage` returned 5 twice).  (**fixed**)
* `datetime.utcnow()` (deprecated) → timezone-aware.  (**fixed**)
* Jobs dict grew forever; now pruned after 6 h.  (**fixed**)
* Upload size unbounded; now `MAX_CONTENT_LENGTH` (40 MB default).  (**fixed**)
* New: `GET /api/scenes/<id>` manifest and `GET /api/scenes/<id>/glb`
  (the seam the Blender/Unreal bridges use).
* The React component is still named `HindsightDashboard` and the logo
  alt text says "Hindsight" — leftover from a template.  (**open, cosmetic**)

---

## 7. Repository hygiene

* No license file. Blocks JOSS and any plugin listing. **Added MIT** —
  confirm this is the license you want (see PUBLISHING.md §2 for the
  interaction with the SAM licenses).
* No tests, no CI, no `pyproject.toml`. Added all three.
* `regen_viewer.py` at the root → `scripts/regen_viewer.py`.
* `torch_versions_backup.txt` is fine but belongs under `docs/`.
* Three stray files in `sam3d_objects_repo/` named `0.7.0`, `=2.3.2`,
  `=2.6.0` are pip-redirect artefacts from an unquoted `pip install
  'x>=2.3.2'`. Harmless; delete them.
* `.env` in the working tree contains two commented-out older API keys.
  Rotate them if they were ever live.

---

## 8. Things I looked at and left alone

* The VRAM park/wake dance. It is ugly but correct for a 32 GB card and
  the README explains it well. A cleaner design (one long-lived worker
  per model, or quantised SAM 3D) is on the roadmap.
* The viewer pulling Three.js from a CDN. Fine for now; vendoring it is
  a one-line change when offline viewing matters.
* `open3d` is still a dependency of `requirements.txt` although the
  pipeline no longer calls it (the Poisson path was dead). SAM 3D's own
  plane estimation uses it, so it stays installed via that package.
