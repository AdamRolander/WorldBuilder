# Publishing WorldBuilder as a software paper

Target: the **Journal of Open Source Software (JOSS)**, with a fallback plan.
This document records what JOSS requires, how the repository measures up
today, what is done, and what remains (ticketed in [ROADMAP.md](ROADMAP.md)
under `PUB-*`).

## 1. What JOSS asks for

From the [submission guidelines](https://joss.readthedocs.io/en/latest/submitting.html)
(fetched 2026-09-17):

| requirement | status |
| --- | --- |
| OSI-approved license in the repo | **done** — `LICENSE` (MIT). Confirm the choice (§2). |
| Public, cloneable repo with an issue tracker | done — GitHub |
| "Obvious research application" | plausible: single-image scene reconstruction for VR/robotics/HCI studies; needs to be *argued* in `paper.md` and ideally *shown* (§3) |
| Substantial scholarly effort (rule of thumb: ≥ 3 months, ≥ 1 000 lines, not a thin wrapper) | code is ≈ 4 000 lines of original glue, heuristics and layout logic on top of SAM 3 / SAM 3D / MoGe. Reviewers will ask what is *ours*; the answer is the hand-off logic, the layout stage, the scene assembly and the tooling — say so explicitly. |
| Public development history > 6 months, iterative | repo dates to Jan 2026 but has 6 commits. **Gap.** From now on: small commits, tagged releases, a changelog. Earliest sensible submission ≈ March 2027 unless history is imported. |
| Evidence of use in research (citations, other groups, a workflow) | **Gap, and the hardest one.** JOSS explicitly rejects "software not yet used in research". See §3. |
| Automated tests | done — 34 CPU tests + CI (`.github/workflows/ci.yml`). GPU validation is documented, not automated. |
| Documentation: install, usage, API, contributing | README covers install/usage; `CONTRIBUTING.md` added; an API/architecture page is thin (**PUB-3**). |
| `paper.md` (≈ 250–1 000 words: summary, statement of need, state of the field, key references) + `paper.bib` | draft in `paper/` (**PUB-1**) |
| Authorship agreed; CoI disclosed | single author; fine |
| Not out of scope: not a notebook, not a "minor utility", not "half-baked", not mainly a web tool | The web app is a thin front end over a library; keep the CLI/library primary in the paper. |

## 2. Licensing nuance (read before submitting anywhere)

WorldBuilder's code can be MIT, but it *runs on* SAM 3 and SAM 3D Objects,
which Meta distributes under the **SAM License** (a custom licence with an
acceptable-use policy, attribution requirements and restrictions, not
OSI-approved). MoGe is MIT, PyTorch3D BSD, Qwen3-VL Apache-2.0.

Consequences:

* JOSS only requires *our* code to be OSI-licensed. It is. State clearly in
  README and paper that model weights are obtained separately under Meta's
  terms (already the case: they are gated HF downloads, never vendored).
* Do **not** vendor or redistribute the SAM repos or weights in a release
  archive, a Blender extension, or a Fab listing. Ship a client that talks
  to a locally running WorldBuilder server instead (that is how the
  `integrations/` scaffolds are built).
* Blender extensions on extensions.blender.org must be GPL-compatible;
  MIT is. The add-on itself contains no SAM code.
* If you ever want a permissive-only stack (e.g. for a commercial
  RoomBuilder backend), SAM 3D would need replacing (TRELLIS, Hunyuan3D 2.x,
  or SPAR3D are the usual candidates) — tracked as `PIPE-10`.

## 3. Closing the "used in research" gap

JOSS wants evidence, not intent. Cheapest credible routes, in order:

1. **Use it yourself in a study and write it up.** A short workshop paper or
   arXiv preprint (e.g. "single-photo VR scene reconstruction: a user study
   of spatial fidelity", or a robotics sim-from-photo experiment) that
   *cites the software* is exactly what reviewers look for. A CS course
   project or capstone at UCSD counts if it is public.
2. **Get one external group to use it.** The Blender add-on is the best
   on-ramp: 3D artists and HCI labs try add-ons. Announce on the SAM 3D
   Objects GitHub discussions, r/computervision, Blender Artists.
3. **Benchmarks as a research contribution.** `scripts/bench_detectors.py`
   plus a small labelled set of room photos (even 30) becomes a reusable
   evaluation, which is itself citable research use.

Plan on submitting no earlier than **Q1–Q2 2027**, with (1) or (2) in hand.

## 4. Fallbacks if JOSS is a poor fit

* **SoftwareX (Elsevier)** — peer-reviewed software articles, no "already
  used" requirement as strict as JOSS, longer format (≈ 3 000 words),
  requires a *permanent* code archive (Zenodo DOI). Good second choice.
* **Journal of Open Research Software (JORS)** — similar to SoftwareX,
  metapaper format.
* **arXiv (cs.CV) technical report + Zenodo DOI** — no review, but it is a
  citable artefact today and helps route (1) above. Do this first
  regardless.
* Workshop venues that welcome systems papers: CVPR/ICCV workshops on 3D
  scene understanding, IEEE VR workshops, UIST demos.

## 5. Concrete checklist (mirrors ROADMAP `PUB-*`)

- [x] LICENSE, CITATION.cff, CONTRIBUTING.md, pyproject, tests, CI
- [x] `paper/paper.md` and `paper/paper.bib` first drafts
- [ ] Create a Zenodo record on the first tagged release (`v0.2.0`) for a DOI
- [ ] `docs/ARCHITECTURE.md` with a data-flow diagram and the per-stage JSON schemas
- [ ] A 30-photo evaluation set with per-photo object lists (can be
      crowd-sourced from friends' rooms with consent) + a results table
- [ ] Regular commits for ≥ 6 months; tag `v0.3`, `v0.4`, ...
- [ ] One documented external use or one preprint that cites the software
- [ ] Add the JOSS author checklist items to the PR template
- [ ] Submit; expect 2–6 weeks of reviewer back-and-forth

## 6. Paper skeleton

See `paper/paper.md`. The statement of need should be framed around
*reproducible, local, single-photo scene reconstruction as a tool for other
researchers* (VR/HCI studies, embodied-AI simulation, interior-design
research), and the "state of the field" paragraph should position it
against (a) commercial photogrammetry (many photos, cloud), (b) research
single-image methods that stop at per-object meshes (SAM 3D Objects, TRELLIS,
Hunyuan3D), and (c) layout estimators that stop at boxes (LayoutNet,
HorizonNet, RoomFormer). WorldBuilder's contribution is the *composition*:
detection → segmentation → per-object 3D → gravity/wall-aligned layout →
navigable scene, with a fully local option.
