"""Stage 4a: scene assembly — placing SAM 3D's objects into one coherent room.

SAM 3D Objects estimates each object's pose independently from a single
image, so the raw poses disagree with each other: floors do not line up,
some objects float, some interpenetrate, and camera tilt leans everything.
This module applies a small set of physically-motivated corrections:

    upright → support graph → floor / stacking snap → collision resolve

Everything is numpy on the CPU (``src/geometry.py`` supplies the pose math;
the previous version re-implemented it on the GPU through pytorch3d three
times). Mesh vertices are loaded once per object and cached, because the
poses change during assembly but the geometry does not.

Changes versus the demo-era code (details and measurements in
docs/AUDIT.md):

* **Keyword matching is token-based.** ``'table' in 'table lamp'`` used to
  force table lamps, tablets and bedside lamps onto the floor.
* **Support-aware placement.** An object whose footprint sits inside another
  object's footprint and whose bottom is near that object's top is *on* it
  (lamp on table, plant on vanity, monitor on desk). It snaps to the
  supporting surface instead of the floor, and collision resolution never
  pushes it off its support.
* **Collision resolution only fires on real interpenetration** (overlap in
  Y as well as XZ, and a large fraction of the smaller object's footprint),
  and displacement is capped, so chairs tucked under tables stay put.
* **The floor comes from the layout stage** (``src/room_layout.py``) when
  available, rather than from whichever object happened to be lowest.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import trimesh

from src.geometry import world_vertices

# ---------------------------------------------------------------------------
# Category knowledge
# ---------------------------------------------------------------------------

# Head nouns (last token) of things that stand on the floor. Token match on
# the *last* word so "coffee table" matches and "table lamp" does not.
FLOOR_SUPPORTED_HEADS = frozenset({
    'chair', 'armchair', 'desk', 'table', 'sofa', 'couch', 'bed', 'stool',
    'bench', 'cabinet', 'dresser', 'nightstand', 'bookshelf', 'bookcase',
    'shelf', 'shelving', 'wardrobe', 'closet', 'trash', 'bin', 'wastebasket',
    'planter', 'rug', 'carpet', 'piano', 'refrigerator', 'fridge', 'oven',
    'stove', 'range', 'dishwasher', 'washer', 'dryer', 'ottoman', 'crib',
    'workbench', 'toolbox', 'cart', 'trolley', 'pallet', 'crate', 'barrel',
    'drum', 'locker', 'safe', 'treadmill', 'bicycle', 'motorcycle', 'car',
    'forklift', 'tractor', 'machine', 'printer3d', 'kiosk', 'podium',
    'lectern', 'easel', 'ladder', 'hamper', 'suitcase', 'tree', 'bush',
    'hedge', 'slide', 'swing', 'seesaw', 'sandbox', 'fountain', 'statue',
})
# Full labels that stand on the floor even though the head noun is ambiguous.
FLOOR_SUPPORTED_LABELS = frozenset({
    'floor lamp', 'potted plant', 'floor plant', 'standing lamp', 'floor fan',
    'floor mirror', 'standing mirror', 'coat rack', 'hat stand', 'tv stand',
    'space heater', 'water cooler', 'vending machine', 'arcade machine',
    'washing machine', 'sewing machine', 'coffee machine stand', 'punching bag',
    '3d printer stand', 'shopping cart', 'wheelchair', 'stroller',
})
# Explicit vetoes: these contain a floor head noun but are never on the floor.
NOT_FLOOR_LABELS = frozenset({
    'table lamp', 'desk lamp', 'bedside lamp', 'tablet', 'table top', 'tabletop',
    'table runner', 'tablecloth', 'table cloth', 'bedding', 'bedspread',
    'bed sheet', 'bed frame headboard', 'headboard', 'microwave oven',
    'toaster oven', 'desk organizer', 'desktop computer', 'desk mat',
    'chair cushion', 'seat cushion', 'wall cabinet', 'medicine cabinet',
    'upper cabinet', 'wall shelf', 'floating shelf', 'wall shelving',
    'bar stool cushion', 'range hood', 'stove hood', 'oven mitt',
    'vanity mirror', 'vanity light', 'bench cushion', 'car seat',
})

VERTICAL_STRUCTURAL_KEYWORDS = (
    'pillar', 'column', 'support column', 'support post', 'i-beam', 'i beam',
    'steel beam', 'structural beam', 'truss', 'girder', 'stanchion',
)


HANGING_KEYWORDS = frozenset({
    'pendant', 'chandelier', 'hanging', 'ceiling', 'suspended', 'overhead',
    'skylight', 'curtain', 'drape', 'drapes', 'blind', 'blinds', 'valance',
})


def _tokens(label: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(label).lower())


def is_hanging(label: str) -> bool:
    """Ceiling-mounted or hanging things: never floor-snapped, never
    'supported' by whatever happens to be under them."""
    toks = _tokens(label)
    return any(t in HANGING_KEYWORDS or t.rstrip('s') in HANGING_KEYWORDS for t in toks)


def is_floor_supported(label: str) -> bool:
    s = " ".join(_tokens(label))
    if s in NOT_FLOOR_LABELS or s in FLOOR_SUPPORTED_LABELS:
        return s in FLOOR_SUPPORTED_LABELS
    toks = _tokens(label)
    if not toks:
        return False
    if any(t in ('wall', 'ceiling', 'hanging', 'mounted', 'pendant', 'overhead') for t in toks):
        return False
    return toks[-1] in FLOOR_SUPPORTED_HEADS


# Kept for callers that imported the old name.
_is_floor_supported = is_floor_supported


def is_vertical_structural(label: str) -> bool:
    s = " ".join(_tokens(label))
    return any(k in s for k in VERTICAL_STRUCTURAL_KEYWORDS)


# ---------------------------------------------------------------------------
# Geometry cache
# ---------------------------------------------------------------------------

_VERT_CACHE: Dict[str, np.ndarray] = {}


def model_vertices(result: Dict) -> Optional[np.ndarray]:
    """Model-space vertices of the object's PLY (cached per path)."""
    p = result.get('model_path') or result.get('ply_path')
    if not p:
        return None
    if p in _VERT_CACHE:
        return _VERT_CACHE[p]
    if not Path(p).exists():
        return None
    mesh = trimesh.load(p, process=False)
    v = np.asarray(mesh.vertices, dtype=np.float64)
    _VERT_CACHE[p] = v
    return v


def clear_vertex_cache():
    _VERT_CACHE.clear()


def _transform_verts(result: Dict) -> np.ndarray:
    """World-space vertices for a result (kept under the old name)."""
    v = model_vertices(result)
    return world_vertices(v, result)


def compute_per_object_aabb(results: List[Dict]) -> Dict[int, Dict]:
    """World-space AABB per object id."""
    aabbs = {}
    for r in results:
        v = model_vertices(r)
        if v is None:
            continue
        w = world_vertices(v, r)
        aabbs[r['id']] = {'min': w.min(axis=0).tolist(), 'max': w.max(axis=0).tolist()}
    return aabbs


def compute_scene_bounds(aabbs: Dict[int, Dict], robust_percentile: float = 1.0) -> Optional[Dict]:
    """Overall scene AABB. Horizontal axes trim outliers when there are
    enough objects; the vertical axis always uses raw min/max so tall
    fixtures stay under the ceiling."""
    if not aabbs:
        return None
    mins = np.array([b['min'] for b in aabbs.values()])
    maxs = np.array([b['max'] for b in aabbs.values()])
    if robust_percentile > 0 and len(aabbs) >= 4:
        lo = np.percentile(mins, robust_percentile, axis=0)
        hi = np.percentile(maxs, 100 - robust_percentile, axis=0)
        lo[1] = float(mins[:, 1].min())
        hi[1] = float(maxs[:, 1].max())
    else:
        lo, hi = mins.min(axis=0), maxs.max(axis=0)
    return {'min': lo.tolist(), 'max': hi.tolist(), 'size': (hi - lo).tolist()}


# ---------------------------------------------------------------------------
# Upright correction
# ---------------------------------------------------------------------------

def upright_correct(results: List[Dict], only_snapped: bool = False,
                    upright_threshold_deg: float = 45.0) -> int:
    """Snap near-upright objects to an exactly upright pose, preserving yaw.

    In SAM 3D's canonical frame (after the GLB→Gaussian swap) local +Z is
    "up"; an upright object has R = R_yaw(α)·R_x(-90°). If local +Z lands
    within ``upright_threshold_deg`` of world +Y we rebuild the rotation from
    the extracted yaw; anything more tilted is left alone (a genuinely
    leaning object is better than a wrong flip). Run this *after* gravity
    alignment so "world +Y" is the real vertical.
    """
    cos_thresh = float(np.cos(np.radians(upright_threshold_deg)))
    cp_h = float(np.sqrt(2) / 2)
    sp_h = float(-np.sqrt(2) / 2)
    corrected = 0
    for r in results:
        if only_snapped and not r.get('snapped', False):
            continue
        w, x, y, z = r['rotation_quaternion']
        up_y = 2.0 * (y * z - w * x)          # world-Y component of local +Z
        r['tilt_deg'] = float(np.degrees(np.arccos(np.clip(up_y, -1, 1))))
        if up_y < cos_thresh:
            r['upright_corrected'] = False
            continue
        cos_a = 1.0 - 2.0 * (y * y + z * z)
        sin_a = 2.0 * (w * z - x * y)
        yaw = float(np.arctan2(sin_a, cos_a))
        cy_h, sy_h = float(np.cos(yaw / 2.0)), float(np.sin(yaw / 2.0))
        r['rotation_quaternion'] = [cy_h * cp_h, cy_h * sp_h, sy_h * cp_h, -sy_h * sp_h]
        r['upright_corrected'] = True
        corrected += 1
    return corrected


# ---------------------------------------------------------------------------
# Support graph and snapping
# ---------------------------------------------------------------------------

def _footprint_containment(inner: Dict, outer: Dict, margin: float) -> float:
    """Fraction of ``inner``'s XZ footprint that lies inside ``outer``'s
    footprint grown by ``margin``."""
    ix0, ix1 = inner['min'][0], inner['max'][0]
    iz0, iz1 = inner['min'][2], inner['max'][2]
    ox0, ox1 = outer['min'][0] - margin, outer['max'][0] + margin
    oz0, oz1 = outer['min'][2] - margin, outer['max'][2] + margin
    ax = max(0.0, min(ix1, ox1) - max(ix0, ox0))
    az = max(0.0, min(iz1, oz1) - max(iz0, oz0))
    area = max(1e-9, (ix1 - ix0) * (iz1 - iz0))
    return (ax * az) / area


def find_supports(results: List[Dict], aabbs: Dict[int, Dict], scene_h: float,
                  min_containment: float = 0.6, gap_frac: float = 0.35) -> Dict[int, int]:
    """Map object id -> id of the object it rests on (if any).

    A rests on B when ≥ ``min_containment`` of A's footprint is inside B's,
    B is not taller-than-A's-top (so A is on B, not inside it), and A's
    bottom is within ``gap_frac`` of A's height (or 8 % of scene height)
    from B's top. The lowest candidate top that still satisfies this wins,
    which handles lamp-on-book-on-table stacks bottom-up.
    """
    supports: Dict[int, int] = {}
    ids = [r['id'] for r in results if r['id'] in aabbs]
    labels = {r['id']: r['label'] for r in results}
    for a in ids:
        if is_hanging(labels[a]):
            continue
        A = aabbs[a]
        ah = max(1e-6, A['max'][1] - A['min'][1])
        tol = max(gap_frac * ah, 0.08 * scene_h)
        best, best_score = None, float('inf')
        for b in ids:
            if b == a:
                continue
            B = aabbs[b]
            bh = B['max'][1] - B['min'][1]
            if bh < 0.5 * ah and (B['max'][0] - B['min'][0]) * (B['max'][2] - B['min'][2]) < \
                    (A['max'][0] - A['min'][0]) * (A['max'][2] - A['min'][2]):
                continue                     # B is small relative to A: not a support
            if _footprint_containment(A, B, margin=0.05 * scene_h) < min_containment:
                continue
            gap = A['min'][1] - B['max'][1]  # positive: A floats above B's top
            if gap < -0.5 * ah or gap > tol:
                continue                     # A is buried in B, or far above it
            if B['max'][1] > A['max'][1]:
                continue                     # B extends above A: A is inside/in front of B
            # Prefer the supporter whose top is closest to A's bottom; sinking
            # into a candidate counts double against it (lamp over a chair back
            # vs. the table 3 cm under it -> table).
            score = gap if gap >= 0 else -2.0 * gap
            if best is None or score < best_score:
                best, best_score = b, score
        if best is not None:
            supports[a] = best
    # Break cycles (A on B and B on A can happen with noisy boxes): keep the
    # relation where the supporter's top is lower.
    for a, b in list(supports.items()):
        if supports.get(b) == a:
            if aabbs[a]['max'][1] >= aabbs[b]['max'][1]:
                supports.pop(a, None)
            else:
                supports.pop(b, None)
    return supports


def find_contained(results: List[Dict], aabbs: Dict[int, Dict], supports: Optional[Dict[int, int]] = None,
                   min_footprint: float = 0.6, max_volume_ratio: float = 0.25) -> Dict[int, int]:
    """Map small object id -> id of the larger object whose box contains it.

    Pillows on a sofa, books on a shelf, a mug on a desk under a hutch: their
    AABBs lie inside a bigger object's AABB, whose top is *above* them, so
    the support test (bottom near the supporter's top) cannot see them.
    Treating them as contained means: leave their height alone, never push
    them out of the container, move them with it. Requires ≥``min_footprint``
    of A's footprint inside B, A's vertical range inside B's, and A's box
    volume ≤ ``max_volume_ratio`` of B's.
    """
    supports = supports or {}
    out: Dict[int, int] = {}
    ids = [r['id'] for r in results if r['id'] in aabbs]
    labels = {r['id']: r['label'] for r in results}

    def _vol(b):
        return max(1e-9, (b['max'][0] - b['min'][0]) * (b['max'][1] - b['min'][1]) * (b['max'][2] - b['min'][2]))

    for a in ids:
        if a in supports or is_hanging(labels[a]):
            continue
        A = aabbs[a]
        ah = A['max'][1] - A['min'][1]
        best, best_vol = None, float('inf')
        for b in ids:
            if b == a:
                continue
            B = aabbs[b]
            if _vol(A) > max_volume_ratio * _vol(B):
                continue
            if _footprint_containment(A, B, margin=0.0) < min_footprint:
                continue
            if A['min'][1] < B['min'][1] - 0.25 * ah or A['max'][1] > B['max'][1] + 0.25 * ah:
                continue
            if _vol(B) < best_vol:
                best, best_vol = b, _vol(B)
        if best is not None:
            out[a] = best
    return out


def robust_floor_y(results: List[Dict], aabbs: Dict[int, Dict]) -> float:
    """Floor height without the layout stage: median bottom of floor-standing
    objects (≥2 of them), else the lowest bottom overall."""
    bottoms = [aabbs[r['id']]['min'][1] for r in results
               if r['id'] in aabbs and is_floor_supported(r['label'])]
    if len(bottoms) >= 2:
        return float(np.median(bottoms))
    return float(min(b['min'][1] for b in aabbs.values()))


def snap_ground_objects(results: List[Dict], aabbs: Dict[int, Dict], bounds: Dict,
                        threshold_frac: float = 0.15, floor_y: Optional[float] = None,
                        supports: Optional[Dict[int, int]] = None,
                        contained: Optional[Dict[int, int]] = None) -> int:
    """Place objects on the floor or on the object supporting them.

    * supported objects snap their bottom to the supporter's top (processed
      bottom-up so stacks propagate);
    * floor-category objects, and anything whose bottom is already within
      ``threshold_frac`` of scene height from the floor, snap to ``floor_y``;
    * everything else (wall art, lights, mirrors) is left where SAM 3D put it.
    Mutates translations *and* the AABB dict. Returns the number snapped.
    """
    if floor_y is None:
        floor_y = robust_floor_y(results, aabbs) if aabbs else bounds['min'][1]
    scene_h = max(1e-6, bounds['max'][1] - floor_y)
    tol = threshold_frac * scene_h
    supports = supports or {}
    by_id = {r['id']: r for r in results}

    def _shift(rid: int, dy: float):
        r = by_id[rid]
        r['translation'] = [r['translation'][0], r['translation'][1] + dy, r['translation'][2]]
        aabbs[rid]['min'][1] += dy
        aabbs[rid]['max'][1] += dy

    snapped = 0
    contained = contained or {}
    order = sorted((r for r in results if r['id'] in aabbs), key=lambda r: aabbs[r['id']]['min'][1])
    for r in order:
        rid = r['id']
        ab = aabbs[rid]
        if rid in contained:
            r.update(snapped=False, supported_by=None, contained_in=contained[rid])
            continue
        if rid in supports and supports[rid] in aabbs:
            target = aabbs[supports[rid]]['max'][1]
            _shift(rid, target - ab['min'][1])
            r.update(snapped=True, snap_forced=False, supported_by=supports[rid])
            snapped += 1
            continue
        dy = ab['min'][1] - floor_y
        if is_hanging(r['label']):
            r.update(snapped=False, supported_by=None)
            continue
        force = is_floor_supported(r['label'])
        if abs(dy) <= tol or force:
            _shift(rid, -dy)
            r.update(snapped=True, snap_forced=bool(force and abs(dy) > tol), supported_by=None)
            snapped += 1
        else:
            r.update(snapped=False, supported_by=None)
    return snapped


# ---------------------------------------------------------------------------
# Collision resolution
# ---------------------------------------------------------------------------

def resolve_xz_collisions(results: List[Dict], aabbs: Dict[int, Dict], iterations: int = 8,
                          padding: float = 0.02, supports: Optional[Dict[int, int]] = None,
                          min_overlap_frac: float = 0.7, max_push_frac: float = 0.5,
                          contained: Optional[Dict[int, int]] = None) -> int:
    """Separate genuinely interpenetrating objects in the XZ plane.

    A pair is only resolved when the boxes overlap in Y *and* the XZ overlap
    covers ≥ ``min_overlap_frac`` of the smaller footprint (0.7: a chair
    pushed two-thirds under a table is normal; two boxes sharing 80 % of a
    footprint are the same object twice or a genuinely bad pose). Pairs in a
    support relation are skipped. Each push is split by confidence (the less
    confident object moves more) and capped at ``max_push_frac`` of the
    moved object's extent along that axis, so nothing gets flung across the
    room to satisfy a bad box. Returns the number of pair pushes.
    """
    if not results or not aabbs:
        return 0
    supports = supports or {}
    contained = contained or {}
    riders = dict(supports)
    riders.update(contained)            # both ride along with their host
    by_id = {r['id']: r for r in results}
    moved_total = 0

    def _overlap(a, b, axis):
        return min(a['max'][axis], b['max'][axis]) - max(a['min'][axis], b['min'][axis])

    for _ in range(iterations):
        any_move = False
        ids = [i for i in aabbs if i in by_id]
        for i, id_a in enumerate(ids):
            for id_b in ids[i + 1:]:
                if riders.get(id_a) == id_b or riders.get(id_b) == id_a:
                    continue
                if id_a in contained or id_b in contained:
                    continue            # things inside a container are the container's business
                a, b = aabbs[id_a], aabbs[id_b]
                ox, oz, oy = _overlap(a, b, 0), _overlap(a, b, 2), _overlap(a, b, 1)
                if ox <= 0 or oz <= 0 or oy <= 0:
                    continue
                area_a = (a['max'][0] - a['min'][0]) * (a['max'][2] - a['min'][2])
                area_b = (b['max'][0] - b['min'][0]) * (b['max'][2] - b['min'][2])
                if ox * oz < min_overlap_frac * max(1e-9, min(area_a, area_b)):
                    continue
                ra, rb = by_id[id_a], by_id[id_b]
                ca = float(ra.get('confidence', 0.5)) or 0.5
                cb = float(rb.get('confidence', 0.5)) or 0.5
                fa, fb = cb / (ca + cb), ca / (ca + cb)
                axis = 0 if ox < oz else 2
                push = (ox if axis == 0 else oz) + padding
                ext_a = a['max'][axis] - a['min'][axis]
                ext_b = b['max'][axis] - b['min'][axis]
                da = min(push * fa, max_push_frac * ext_a)
                db = min(push * fb, max_push_frac * ext_b)
                c_a = 0.5 * (a['min'][axis] + a['max'][axis])
                c_b = 0.5 * (b['min'][axis] + b['max'][axis])
                sign_a = -1.0 if c_a < c_b else 1.0
                for rid, box, d in ((id_a, a, sign_a * da), (id_b, b, -sign_a * db)):
                    by_id[rid]['translation'][axis] += d
                    box['min'][axis] += d
                    box['max'][axis] += d
                    # anything resting on this object rides along
                    for kid, sup in riders.items():
                        if sup == rid and kid in aabbs and kid in by_id:
                            by_id[kid]['translation'][axis] += d
                            aabbs[kid]['min'][axis] += d
                            aabbs[kid]['max'][axis] += d
                any_move = True
                moved_total += 1
        if not any_move:
            break
    return moved_total


# ---------------------------------------------------------------------------
# Keep objects inside detected walls
# ---------------------------------------------------------------------------

def clamp_to_room(results: List[Dict], aabbs: Dict[int, Dict], bounds_min, bounds_max,
                  wall_sources: Dict[str, str], max_push_frac: float = 0.35) -> int:
    """Nudge objects that poke through a *detected* wall back inside.

    SAM 3D's per-object depth is the noisiest part of its pose; a sofa that
    ends up 30 cm inside the back wall is more likely misplaced than the
    wall (which had thousands of point-map votes). Only sides whose
    ``wall_sources`` entry starts with "wall" are enforced, and an object
    is moved at most ``max_push_frac`` of its own extent — beyond that we
    leave it (and its overshoot) alone rather than teleport it.
    Pass ``{"y_min": "floor"}`` in ``wall_sources`` to enforce a fitted floor
    the same way (objects below it are raised, capped).
    Mutates translations and AABBs. Returns the number of objects moved.
    """
    by_id = {r['id']: r for r in results}
    moved = 0
    for rid, ab in aabbs.items():
        r = by_id.get(rid)
        if r is None:
            continue
        for side, src in wall_sources.items():
            # "y_min": "floor" is passed when the floor plane was fitted, so
            # objects sunk below the real floor are raised the same way.
            if not (str(src).startswith("wall") or str(src).startswith("floor")):
                continue
            axis = 0 if side.startswith("x") else (1 if side.startswith("y") else 2)
            ext = ab['max'][axis] - ab['min'][axis]
            if side.endswith("min"):
                over = bounds_min[axis] - ab['min'][axis]
            else:
                over = ab['max'][axis] - bounds_max[axis]
            if over <= 1e-6:
                continue
            d = min(over, max_push_frac * max(ext, 1e-6))
            if side.endswith("max"):
                d = -d
            r['translation'][axis] += d
            ab['min'][axis] += d
            ab['max'][axis] += d
            r['wall_clamped'] = True
            moved += 1
    return moved


# ---------------------------------------------------------------------------
# Optional: stretch pillars/beams to full height (opt-in)
# ---------------------------------------------------------------------------

def stretch_vertical_structural(results: List[Dict], aabbs: Dict[int, Dict], bounds: Dict,
                                min_height_frac: float = 0.6) -> int:
    """Scale floor-to-ceiling structural elements to fill the scene height.
    Opt-in (``WORLDBUILDER_STRETCH_STRUCTURAL=1``): single-image 3D returns
    tall thin things too small, but this also mangles anything mislabelled
    as a pillar. Scales local Z (which upright objects map to world Y)."""
    scene_h = bounds['max'][1] - bounds['min'][1]
    if scene_h <= 0:
        return 0
    target_h = scene_h * 0.95
    stretched = 0
    for r in results:
        if not is_vertical_structural(r['label']):
            continue
        ab = aabbs.get(r['id'])
        if ab is None:
            continue
        cur_h = ab['max'][1] - ab['min'][1]
        if cur_h <= 0 or cur_h >= scene_h * min_height_frac:
            continue
        factor = target_h / cur_h
        r['scale'] = [r['scale'][0], r['scale'][1], r['scale'][2] * factor]
        r['vertical_stretched'] = True
        stretched += 1
    return stretched


# ---------------------------------------------------------------------------
# Fallback room box (used when the layout stage has no point map)
# ---------------------------------------------------------------------------

def make_box_room(bounds: Dict, padding_xz: float = 0.15, padding_y: float = 0.10,
                  wall_thickness: float = 0.02, floor_color=(180, 170, 150, 255),
                  wall_color=(235, 230, 220, 255), ceiling_color=(245, 245, 245, 255),
                  include_ceiling: bool = True, include_front_wall: bool = True) -> trimesh.Trimesh:
    """Axis-aligned floor, walls and ceiling as one vertex-coloured trimesh."""
    mn = np.array(bounds['min']); mx = np.array(bounds['max'])
    min_x, max_x = mn[0] - padding_xz, mx[0] + padding_xz
    min_z, max_z = mn[2] - padding_xz, mx[2] + padding_xz
    floor_y, ceiling_y = mn[1], mx[1] + padding_y
    sx, sz, h = max_x - min_x, max_z - min_z, ceiling_y - floor_y
    cx, cz, cy = (min_x + max_x) / 2, (min_z + max_z) / 2, (floor_y + ceiling_y) / 2

    def plane(extents, center, color):
        b = trimesh.creation.box(extents=extents)
        b.apply_translation(center)
        rgba = np.array(color, dtype=np.uint8)
        if rgba.size == 3:
            rgba = np.concatenate([rgba, [255]])
        return b, rgba

    parts = [
        plane([sx, wall_thickness, sz], [cx, floor_y - wall_thickness / 2, cz], floor_color),
        plane([wall_thickness, h, sz], [min_x - wall_thickness / 2, cy, cz], wall_color),
        plane([wall_thickness, h, sz], [max_x + wall_thickness / 2, cy, cz], wall_color),
        plane([sx, h, wall_thickness], [cx, cy, max_z + wall_thickness / 2], wall_color),
    ]
    if include_front_wall:
        parts.append(plane([sx, h, wall_thickness], [cx, cy, min_z - wall_thickness / 2], wall_color))
    if include_ceiling:
        parts.append(plane([sx, wall_thickness, sz], [cx, ceiling_y + wall_thickness / 2, cz], ceiling_color))

    all_v, all_f, all_c, offset = [], [], [], 0
    for mesh, color in parts:
        v = np.asarray(mesh.vertices); f = np.asarray(mesh.faces) + offset
        all_v.append(v); all_f.append(f); all_c.append(np.tile(color, (len(v), 1)).astype(np.uint8))
        offset += len(v)
    return trimesh.Trimesh(vertices=np.vstack(all_v), faces=np.vstack(all_f),
                           vertex_colors=np.vstack(all_c), process=False)


# ---------------------------------------------------------------------------
# One call that does the whole assembly
# ---------------------------------------------------------------------------

def assemble_scene(results: List[Dict], floor_y: Optional[float] = None,
                   stretch_structural: bool = False, verbose: bool = True) -> Dict:
    """Upright → supports → snap → collisions → re-snap. Mutates ``results``
    in place and returns a diagnostics dict (also useful for tests)."""
    log = print if verbose else (lambda *a, **k: None)
    diag: Dict = {}
    if not results:
        return diag
    corrected = upright_correct(results, only_snapped=False)
    diag['upright_corrected'] = corrected
    log(f"  upright-corrected {corrected}/{len(results)} objects")

    aabbs = compute_per_object_aabb(results)
    bounds = compute_scene_bounds(aabbs)
    if bounds is None:
        return diag
    if stretch_structural:
        n = stretch_vertical_structural(results, aabbs, bounds)
        if n:
            log(f"  vertical-stretched {n} structural element(s)")
            aabbs = compute_per_object_aabb(results)
            bounds = compute_scene_bounds(aabbs)

    scene_h = bounds['max'][1] - (floor_y if floor_y is not None else bounds['min'][1])
    supports = find_supports(results, aabbs, max(scene_h, 1e-6))
    diag['supports'] = {int(k): int(v) for k, v in supports.items()}
    if supports:
        names = {r['id']: r['label'] for r in results}
        log("  support relations: " + ", ".join(f"{names[a]}→{names[b]}" for a, b in supports.items()))

    contained = find_contained(results, aabbs, supports)
    diag['contained'] = {int(k): int(v) for k, v in contained.items()}
    if contained:
        names = {r['id']: r['label'] for r in results}
        log("  contained in: " + ", ".join(f"{names[a]}⊂{names[b]}" for a, b in contained.items()))

    snapped = snap_ground_objects(results, aabbs, bounds, floor_y=floor_y, supports=supports,
                                  contained=contained)
    forced = sum(1 for r in results if r.get('snap_forced'))
    diag['snapped'] = snapped; diag['snap_forced'] = forced
    log(f"  snapped {snapped}/{len(results)} ({forced} via floor-category rule, "
        f"{len(supports)} onto supports, {len(contained)} left in their container)")

    pushes = resolve_xz_collisions(results, aabbs, supports=supports, contained=contained)
    diag['collision_pushes'] = pushes
    log(f"  collision resolution: {pushes} pair-pushes")

    # Collision pushes keep Y, but re-snap so stacks stay attached after
    # their supporter moved and rounding did not open gaps.
    aabbs = compute_per_object_aabb(results)
    bounds = compute_scene_bounds(aabbs)
    snap_ground_objects(results, aabbs, bounds, floor_y=floor_y, supports=supports, contained=contained)
    diag['bounds'] = compute_scene_bounds(compute_per_object_aabb(results))
    return diag
