# RoomBuilder — concept, market notes and plan

A consumer app that lets someone pick (or describe, or photograph) their
room, furnish it with real products in 3D, and buy through affiliate links.
First wedge: **college dorms**, where every year a few million people face
the same small, fixed rooms with the same constraints and the same shopping
list.

This document is the working brief. Decisions taken so far are marked
**decided**; open questions are marked **open**. The technical plan and the
first prototype live in the sister repository (`../RoomBuilder`, see §6).

---

## 1. Why this might work

* **Recurring, concentrated demand.** ~20 M US college students; ~2.5 M move
  into a dorm each August; the National Retail Federation puts back-to-college
  spend around $80–90 B/yr, a large share on dorm furnishing. Everyone shops
  in the same 6-week window.
* **Fixed, knowable rooms.** Dorm layouts are finite and published: housing
  offices post floor plans, several with dimensions (Georgia College, USI,
  Boston College, Cal Poly Pomona, ECU; UCSD posts plans without dimensions).
  Once a room type is modelled it serves every occupant, every year.
* **Real constraints people care about:** will a 5-drawer dresser fit under
  the lofted bed, does a mini-fridge block the closet, can two roommates
  fit a futon. Exactly what a 3D planner answers and a 2D shopping site can't.
* **Natural monetisation with no sales team:** affiliate programmes
  (Amazon Associates 3–4 % on furniture/home, Wayfair/Target/IKEA via
  networks like Impact/CJ/Rakuten at 3–8 %, Dormify/Bed Bath-type
  dorm specialists often 8–10 %) and school-specific "what to bring"
  bundles.

## 2. Who else is doing it

| product | what it is | gap we could exploit |
| --- | --- | --- |
| **Dormscape** (dormscape.us) | free dorm planner: 1 600+ layouts from official housing docs (Michigan, Ohio State, Penn State, NYU, Rutgers, UT Austin, …), 2D + a paid "3D Room Studio", nine style presets, Amazon affiliate links; tiers Free / $4.99 / $14.99 one-time (fetched 2026-09-17) | **This is essentially the MVP already shipped.** Do not compete on "dorm planner with dimensions". Compete on what they lack: (1) photo capture of *your actual room with your stuff* (WorldBuilder), (2) true fit/clearance checks and stacking physics rather than style presets, (3) live roommate co-editing, (4) arbitrary rooms (apartments, first homes) — the market after the dorm. Check whether UCSD is covered; if not, it is a free beachhead. |
| Roomstyler / Planner 5D / HomeByMe / IKEA Kreativ | general 3D room planners | none are dorm-specific; no school catalogue; IKEA is single-retailer |
| Dormify, OCM, "college packing list" sites | shopping, not planning | no spatial model |
| Housing office PDFs | static 2D plans | the *data* source, not a product |
| WorldBuilder | photo → 3D scene | the differentiator for arbitrary (non-dorm) rooms later |

Positioning (revised after the Dormscape check): *"point your phone at
your room and rearrange it"*. Dimensions-from-housing-docs is table stakes
that a competitor already has; the moat has to be the photo pipeline plus
constraint-checking, with the school catalogue as the on-ramp.

## 3. Product scope

### MVP (decided)

1. Pick school → residence hall → room type; the room appears in 3D with
   the *provided* furniture already placed (bed, desk, wardrobe — housing
   lists these and their sizes).
2. Add items from a catalogue (parametric boxes with real dimensions
   from retailer listings; product photo on the box; GLB when the retailer
   provides one). Drag on the floor, rotate in 90° steps, snap to walls,
   stack allowed items (fridge under desk, shelf on desk).
3. Fit warnings: overlap, blocked door/closet swing, exceeds loft clearance.
4. Share a layout by link; roommate sees the same room; both edit.
5. "Buy" buttons with affiliate links; a "your list" page.

### Not in MVP

* Photo-based room capture (WorldBuilder integration) — phase 3.
* Text-described arbitrary rooms ("12x14 with a window on the north wall")
  — phase 2; cheap to add once the room model is parametric.
* Native mobile apps — the web app must work on phones first.

## 4. Data: getting rooms

Sources, in order of effort:

1. **Housing office publications.** Floor-plan PDFs/GIFs (UCSD has them
   for every neighbourhood: Revelle, Muir, ERC, Sixth, Warren, Seventh,
   Eighth, Marshall, Pepper Canyon, Rita Atkinson, Pangea) and, at some
   schools, explicit dimensions. Plan: a per-room YAML (`rooms/ucsd/<hall>/<type>.yaml`)
   with wall polygon, door/window segments, fixed furniture. Where only an
   image exists, dimensions come from (a) the housing page's square-footage
   figure, (b) known furniture sizes visible in the plan (a twin XL bed is
   80×38 in), (c) student measurements crowd-sourced through the app
   ("tape-measure mode": three numbers).
2. **Public floor-plan datasets** for the *general-room* phase and for
   training a plan-image → polygon extractor: CubiCasa5K (5 k Finnish
   plans, vector, furniture-level; research licence), RPLAN (80 k, vector;
   academic use), Zillow Indoor Dataset (2.5 k homes with panoramas;
   research licence). None are usable as-is for a commercial product —
   they inform the parser, they are not the catalogue. **open**: confirm
   licences before any model trained on them ships.
3. **Photos** via WorldBuilder's layout stage (`src/room_layout.py`
   already yields floor plane, walls, ceiling height and a scale
   estimate from one photo). This is the phase-3 differentiator.

Legal note (**open**): floor-plan images are copyrighted by the
university/architect. Re-drawing dimensions as our own vector data is the
standard approach (facts aren't copyrightable; the drawing is). Get a
written OK from UCSD HDH for the first campus anyway — they may also want
to link to the tool.

## 5. Business model sketch

* Affiliate revenue per placed-and-bought item. A conservative model:
  10 k active planners at one school-year, 30 % buy ≥1 item through the
  app, average basket $150, blended 5 % commission → ≈ $22 k/yr per 10 k
  users. It scales with schools, not with engineering.
* Optional: schools pay for a white-label embed on the housing site
  (they already field "will my X fit" emails).
* Optional later: retailers pay for placement/"fits your room" badges.

Costs: static hosting + a small API (layout sharing) ≈ $20–50/mo; the 3D
runs in the browser. Affiliate approval usually requires a live site with
content, so launch the free planner first, add links second.

## 6. Repository strategy — **decided: sister repo**

`RoomBuilder` lives next to `WorldBuilder`, not inside it:

* different product, licence posture (may become closed/commercial),
  release cadence and stack (browser app + small API vs. GPU pipeline);
* WorldBuilder stays a clean open-source research tool for JOSS;
* the only coupling is an HTTP contract: RoomBuilder can call a
  WorldBuilder server for photo capture (`POST /api/upload`,
  `GET /api/scenes/<id>`) and consume `layout.json` + GLB. That contract is
  documented in `docs/INTEGRATIONS.md` and is the same one Blender/Unreal use.

A submodule would couple release histories and force WorldBuilder's
heavyweight setup on every RoomBuilder contributor.

The scaffold created on 2026-09-17 (`../RoomBuilder`) contains a working
browser prototype: parametric room from measurements, a starter catalogue,
drag/rotate/snap/stack, overlap warnings, save/load/share via URL, and a
UCSD room-type stub to be filled from the housing PDFs.

## 7. Phases

| phase | outcome | depends on |
| --- | --- | --- |
| 0 | prototype (done): measurements → 3D room, catalogue boxes, layout JSON | — |
| 1 | UCSD catalogue: 10–20 most common room types with fixed furniture; share links; deploy | HDH data, a weekend of measuring |
| 2 | affiliate links + product import tool (paste a retailer URL → dimensions/photo); text-described rooms | phase 1 live, affiliate approvals |
| 3 | photo capture via WorldBuilder (room shell + existing furniture) | WorldBuilder metric scale (PIPE-6) |
| 4 | second/third campus; school partnerships; mobile polish | traction at UCSD |

Ticket ids `RB-*` in ROADMAP.md.
