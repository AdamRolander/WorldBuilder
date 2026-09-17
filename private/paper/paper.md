---
title: 'WorldBuilder: reconstructing a navigable 3D room from a single photograph with off-the-shelf foundation models'
tags:
  - Python
  - 3D reconstruction
  - single-image
  - scene understanding
  - virtual reality
  - Segment Anything
authors:
  - name: Adam Rolander
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: University of California, San Diego, United States
    index: 1
date: 17 September 2026
bibliography: paper.bib
---

# Summary

WorldBuilder turns one ordinary photograph of a room into a navigable,
editable 3D scene. A vision-language model lists the object types in the
photo; SAM 3 [@sam3] segments every instance of each type from a text
prompt; SAM 3D Objects [@sam3dobjects] lifts each mask into a posed,
textured mesh; a monocular geometry model (MoGe [@moge]) supplies a scene
point map from which WorldBuilder recovers the floor plane, gravity
direction, dominant wall orientation, wall and ceiling positions and
photo-projected textures for the room shell; and a small set of
physically motivated rules (support graph, floor and stacking snap,
interpenetration resolution) assembles the objects into one coherent
scene. The result is exported as per-object meshes, a single glTF binary,
and a self-contained WebXR viewer, and can be pulled into Blender or Unreal
Engine through thin client integrations. Every stage can run on one
consumer GPU with no network access when the local VLM backend is used.

# Statement of need

Researchers in virtual reality, human–computer interaction, embodied AI and
interior-design computing frequently need *plausible 3D replicas of real
rooms* — for user studies, simulation environments, or as editable
starting points — but photogrammetry requires dozens of photographs and a
capture protocol, and commercial room-scanning apps produce closed formats
on proprietary hardware. Recent single-image methods produce excellent
per-object meshes (SAM 3D Objects, TRELLIS [@trellis], Hunyuan3D
[@hunyuan3d]) or room layouts as boxes (LayoutNet [@layoutnet],
HorizonNet [@horizonnet]), but none composes detection, segmentation,
per-object reconstruction, gravity-aligned layout and architecture into a
single reproducible tool with an open pipeline and documented failure
modes.

WorldBuilder fills that gap. Its contribution is the *composition* and the
engineering around it rather than any one model: a robust hand-off between
open-vocabulary detection and text-prompted segmentation (threshold
handling, multi-prompt union, box-prompt fallback, duplicate suppression),
a scene-level point map shared across all objects, a layout stage that
turns that point map into gravity, walls, ceiling and textured room
geometry, and an assembly stage whose heuristics are unit-tested against
synthetic rooms. The software is a Python package with a command-line
interface, a web application with a job API, a local-VLM server for fully
offline use, and a CPU-only test suite; GPU behaviour is documented with
before/after measurements on real photographs.

# State of the field

Single-image 3D has moved quickly: object-centric generators
(TRELLIS, Hunyuan3D 2, SAM 3D Objects) produce high-quality meshes from a
masked object, monocular geometry models (MoGe, Depth Anything
[@depthanything]) give near-metric point maps, and promptable segmenters
(SAM 2 [@sam2], SAM 3) make open-vocabulary instance masks routine. Scene-
level systems that combine these remain either closed (commercial apps) or
research prototypes tied to a dataset. Room-layout estimation from
panoramas is mature but rarely joined to object reconstruction, and
"generative room" approaches (e.g. Holodeck [@holodeck]) synthesise
plausible rooms rather than reconstructing a specific one. WorldBuilder
sits between these: it reconstructs *this* room from *one* photo, keeping
each component swappable so the pipeline improves as the models do.

# Acknowledgements

WorldBuilder builds on SAM 3 and SAM 3D Objects (Meta AI), MoGe (Microsoft),
PyTorch3D, trimesh and three.js. It began as a class/demo project at UC San
Diego.

# References
