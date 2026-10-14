"""Silhouette-driven pose search: refinement and instance sharing.

Both use one score, an IoU between the placed mesh's silhouette and the
object's mask that understands occlusion and depth:

* a silhouette pixel where the photo shows something *nearer* than the mesh
  (a desk in front of a chair) is not evidence against the pose and is
  ignored — otherwise every partly hidden object would be "improved" by
  shrinking it to its visible sliver;
* a mask pixel only counts as matched if the mesh is at roughly the depth
  the point map observed there — otherwise pushing an object far away and
  blowing it up would score perfectly.

``refine_pose`` nudges an object that fits poorly (scale, image-plane
shift, depth) by coordinate descent on that score. ``share_instances``
finds objects that are the same model photographed several times (same
label, similar colour) and replaces each with the best-observed one,
re-posed to fit that instance's mask at the exemplar's true size. A chair
seen as a seat back behind a desk becomes a whole chair standing on the
floor, and a row of identical chairs comes out identical.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.placement import PlacedObject, SceneEvidence, mask_box, mask_iou, splat, splat_box

_BOX_CACHE: Dict[int, Tuple] = {}


def _mask_box(mask: np.ndarray):
    """Bounding box of a mask, remembered per array (the searches score one
    mask against thousands of poses)."""
    key = id(mask)
    hit = _BOX_CACHE.get(key)
    if hit is None or hit[0] is not mask:
        if len(_BOX_CACHE) > 4096:
            _BOX_CACHE.clear()
        hit = (mask, mask_box(mask))
        _BOX_CACHE[key] = hit
    return hit[1]


def fit_score(ev: SceneEvidence, verts: np.ndarray, mask: np.ndarray,
              depth_tol: float = 0.15, occl_margin: float = 0.05) -> float:
    """Occlusion- and depth-aware silhouette IoU in [0, 1]."""
    sil, zbuf, sb = splat_box(ev, verts)
    if sb is None:
        return 0.0
    mb = _mask_box(mask)
    if mb is not None:                           # everything below is zero outside the two boxes
        sb = (min(sb[0], mb[0]), max(sb[1], mb[1]), min(sb[2], mb[2]), max(sb[3], mb[3]))
    win = (slice(sb[0], sb[1]), slice(sb[2], sb[3]))
    sil, zbuf, mask = sil[win], zbuf[win], mask[win]
    valid, depth = ev.valid[win], ev.depth[win]
    with np.errstate(invalid="ignore"):
        nearer = valid & (depth < zbuf * (1.0 - occl_margin))
        agree = ~valid | (np.abs(zbuf - depth) <= depth_tol * depth)
    hidden = sil & nearer & ~mask
    visible = sil & ~hidden
    matched = sil & mask & agree
    # Hidden pixels are excused, not free: at a tenth of their weight a pose cannot
    # win by burying most of the object inside a wall.
    union = float((visible | mask).sum()) + 0.1 * float(hidden.sum())
    return float(matched.sum()) / union if union else 0.0


def hidden_fraction(ev: SceneEvidence, verts: np.ndarray, mask: np.ndarray) -> float:
    """Share of the silhouette that the photo shows covered by something nearer."""
    sil, zbuf = splat(ev, verts)
    if not sil.any():
        return 1.0
    with np.errstate(invalid="ignore"):
        nearer = ev.valid & (ev.depth < zbuf * 0.95) & ~mask
    return float((sil & nearer).sum()) / float(sil.sum())


def touches_border(mask: np.ndarray, margin: int = 2) -> Dict[str, bool]:
    return {"top": bool(mask[:margin].any()), "bottom": bool(mask[-margin:].any()),
            "left": bool(mask[:, :margin].any()), "right": bool(mask[:, -margin:].any())}


# ---------------------------------------------------------------------------
# Refinement of a single object
# ---------------------------------------------------------------------------

def refine_pose(ev: SceneEvidence, obj: PlacedObject, min_gain: float = 0.05, rounds: int = 2) -> Optional[float]:
    """Coordinate descent over size and sideways/vertical position (the
    caller refits depth afterwards). Returns the score gain if the pose was
    changed, else None."""
    mask = ev.masks.get(obj.id)
    if mask is None or not mask.any():
        return None
    start = obj.snapshot()
    base = fit_score(ev, obj.fast_verts, mask)
    plain0 = mask_iou(splat(ev, obj.fast_verts)[0], mask)
    # Whatever leaves the frame is invisible to the score, so growing out of
    # the picture would be free. Do not let a search move more of the object
    # outside than was outside to begin with.
    out_max = _outside_fraction(ev, obj.fast_verts) + 0.05
    best = base
    size = float(np.linalg.norm(obj.hi - obj.lo))
    moves = (
        ("scale", (0.7, 0.8, 0.9, 1.1, 1.25, 1.4)),
        ("x", (-0.2, -0.1, -0.05, 0.05, 0.1, 0.2)),
        ("y", (-0.2, -0.1, -0.05, 0.05, 0.1, 0.2)),
    )       # depth is not searched here: sliding along the rays leaves the silhouette unchanged
    for _ in range(rounds):
        improved = False
        for kind, values in moves:
            keep = obj.snapshot()
            cand_best, cand_snap = best, None
            for v in values:
                obj.restore(keep)
                if kind == "scale":
                    obj.scale_about_centre(v)
                elif kind == "x":
                    obj.translate([v * size, 0, 0])
                else:
                    obj.translate([0, v * size, 0])
                if _outside_fraction(ev, obj.fast_verts) > out_max:
                    continue
                s = fit_score(ev, obj.fast_verts, mask)
                if s > cand_best + 1e-4:
                    cand_best, cand_snap = s, obj.snapshot()
            obj.restore(cand_snap if cand_snap is not None else keep)
            if cand_snap is not None:
                best, improved = cand_best, True
        if not improved:
            break
    if best < base + min_gain or mask_iou(splat(ev, obj.fast_verts)[0], mask) < plain0 - 0.02:
        obj.restore(start)
        return None
    obj.r['pose_refined'] = round(best - base, 3)
    return best - base


# ---------------------------------------------------------------------------
# Instance sharing
# ---------------------------------------------------------------------------

def _mean_color(image_small: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return image_small[mask].reshape(-1, 3).astype(np.float64).mean(axis=0) if mask.any() else np.zeros(3)


def _crop_signature(image_small: np.ndarray, mask: np.ndarray, n: int = 16) -> Optional[np.ndarray]:
    """What the object looks like, as a zero-mean unit vector: its bounding
    box in the photo, grey, at ``n``×``n``. Two views of one chair from the
    same side correlate; two different pictures in identical frames do not."""
    import cv2
    ys, xs = np.nonzero(mask)
    if len(ys) < 16:
        return None
    crop = image_small[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32).mean(axis=2)
    v = cv2.resize(crop, (n, n), interpolation=cv2.INTER_AREA).ravel()
    v = v - v.mean()
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 1e-6 else None


def _chroma_hist(image_small: np.ndarray, mask: np.ndarray, bins: int = 8) -> Optional[np.ndarray]:
    """Normalised histogram of rg-chromaticity inside a mask: the colours an
    object is made of, independent of how brightly it is lit and of how
    much of it is visible."""
    px = image_small[mask].reshape(-1, 3).astype(np.float64)
    if len(px) < 16:
        return None
    s = np.maximum(px.sum(axis=1), 1e-6)
    r, g = px[:, 0] / s, px[:, 1] / s
    h, _, _ = np.histogram2d(r, g, bins=bins, range=[[0, 1], [0, 1]])
    return h / max(h.sum(), 1e-9)


def _hist_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """Bhattacharyya coefficient (1 = same colour make-up)."""
    if a is None or b is None:
        return 0.0
    return float(np.sum(np.sqrt(a * b)))


def _chroma(rgb: np.ndarray) -> np.ndarray:
    """Brightness-free colour: the same chair in shade and in sun agrees."""
    return rgb / max(float(rgb.sum()), 1e-6)


def _outside_fraction(ev: SceneEvidence, verts: np.ndarray) -> float:
    h, w = ev.hw
    u, v, z = ev.project(verts)
    return float(np.mean((z <= 0) | (u < 0) | (u >= w) | (v < 0) | (v >= h)))


def _yaw(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


_INWARD = {"x_min": np.array([1.0, 0.0]), "x_max": np.array([-1.0, 0.0]),
           "z_min": np.array([0.0, 1.0]), "z_max": np.array([0.0, -1.0])}


def _fit_copy(ev: SceneEvidence, ex_verts: np.ndarray, mask: np.ndarray, centre_xz: np.ndarray,
              bottom_y: float, wall_side: Optional[str] = None) -> Tuple[float, float, np.ndarray]:
    """Best (score, yaw, xz) for a copy of the exemplar standing at
    ``bottom_y`` near ``centre_xz``. ``ex_verts`` are the exemplar's world
    vertices (subsampled).

    A silhouette cannot tell a bookshelf facing the room from one facing
    the wall (``yaw`` and ``yaw + 180°`` give the same outline against a
    wall), so when the instance stands against an observed wall
    (``wall_side``), headings that turn the exemplar's photographed side —
    the side that faced the camera — toward that wall are not tried.
    """
    lo, hi = ex_verts.min(0), ex_verts.max(0)
    c = np.array([0.5 * (lo[0] + hi[0]), lo[1], 0.5 * (lo[2] + hi[2])])
    local = ex_verts - c                                   # origin at the bottom centre
    foot = float(max(hi[0] - lo[0], hi[2] - lo[2]))
    front = -np.array([c[0], c[2]])                        # from the exemplar toward the camera (origin)
    front = front / max(np.linalg.norm(front), 1e-9)
    inward = _INWARD.get(wall_side)

    def faces_room(yaw):
        if inward is None:
            return True
        cy, sy = np.cos(yaw), np.sin(yaw)
        f = np.array([cy * front[0] + sy * front[1], -sy * front[0] + cy * front[1]])   # _yaw(yaw) @ front, XZ
        return float(f @ inward) > 0.3

    def score(yaw, xz):
        v = local @ _yaw(yaw).T + np.array([xz[0], bottom_y, xz[1]])
        return fit_score(ev, v, mask)

    best = (-1.0, 0.0, np.asarray(centre_xz, float))
    offsets = np.array([-0.4, -0.2, 0.0, 0.2, 0.4]) * foot
    for yaw in np.arange(0, 2 * np.pi, np.pi / 8):
        if not faces_room(yaw):
            continue
        for dx in offsets:
            for dz in offsets:
                xz = centre_xz + np.array([dx, dz])
                s = score(yaw, xz)
                if s > best[0]:
                    best = (s, float(yaw), xz)
    for yaw in best[1] + np.radians([-11, 0, 11]):
        for dx in (-0.1 * foot, 0.0, 0.1 * foot):
            for dz in (-0.1 * foot, 0.0, 0.1 * foot):
                xz = best[2] + np.array([dx, dz])
                s = score(yaw, xz)
                if s > best[0]:
                    best = (s, float(yaw), xz)
    return best


def share_instances(ev: SceneEvidence, objs: List[PlacedObject], image_small: np.ndarray,
                    min_group: int = 2, max_color_dist: float = 0.25, max_score_loss: float = 0.05,
                    min_exemplar_iou: float = 0.5, max_shape_diff: float = 0.2,
                    min_gain: float = 0.05) -> List[Dict]:
    """Replace repeated objects by re-posed copies of the best-observed one.

    A group is ``min_group`` or more objects with the same label (two is
    enough: the exemplar tests below are what protect against a bad
    swap, not the group size); within it
    the exemplar is the instance that is largest in the photo, fits its
    mask, is not cut by the frame and is barely occluded. Another instance
    takes the exemplar's mesh when (a) its mask has a similar mean colour
    (red chairs do not become grey ones) and (b) the exemplar, placed at its
    own size on the instance's support and searched over heading and
    position, explains the instance's mask about as well as the instance's
    own mesh did. Two cases pass (b): the instance already has the
    exemplar's proportions (same sorted extents within ``max_shape_diff``),
    so it is the same model at a slightly different size; or its own mesh
    is a *fragment* — much shorter than the exemplar while the exemplar
    standing there would be partly hidden or cut by the frame, which is
    exactly why SAM 3D only saw a piece. Objects that merely share a label
    (books of different sizes) pass neither.
    ``image_small`` is the photo at point-map resolution. Returns one dict
    per replacement.
    """
    out: List[Dict] = []
    undo: Dict[int, Tuple] = {}
    groups: Dict[str, List[PlacedObject]] = {}
    for o in objs:
        if o.id in ev.masks:
            groups.setdefault(o.r['label'], []).append(o)
    for label, members in groups.items():
        if len(members) < min_group:
            continue
        # exemplar
        ranked = []
        for o in members:
            m = ev.masks[o.id]
            iou = mask_iou(splat(ev, o.fast_verts)[0], m)
            if iou < min_exemplar_iou or any(touches_border(m).values()):
                continue
            if hidden_fraction(ev, o.fast_verts, m) > 0.25:
                continue
            ranked.append((iou * float(np.sqrt(m.sum())), o))
        if not ranked:
            continue
        ex = max(ranked, key=lambda t: t[0])[1]
        ex_color = _chroma(_mean_color(image_small, ev.masks[ex.id]))
        ex_hist = _chroma_hist(image_small, ev.masks[ex.id])
        ex_h = ex.height
        ex_sig = _crop_signature(image_small, ev.masks[ex.id])
        ex_dims = (ex.hi - ex.lo).copy()
        # A flat thing (a picture, a sign, a screen) is identified by what is
        # on its face; frames of one size hold different pictures.
        ex_flat = float(np.sort(ex_dims)[0]) < 0.2 * float(np.sort(ex_dims)[2])
        ex_fast = ex.fast_verts.copy()
        ex_full_lo = ex.lo.copy()
        ex_pose = ex.snapshot()
        for o in members:
            if o is ex:
                continue
            m = ev.masks[o.id]
            if float(np.linalg.norm(_chroma(_mean_color(image_small, m)) - ex_color)) > max_color_dist:
                o.r['share_rejected'] = "colour"
                continue
            own = fit_score(ev, o.fast_verts, m)
            dims_o = np.sort(o.hi - o.lo)
            dims_e = np.sort(ex_dims)
            same_shape = bool(np.all(np.abs(dims_o / np.maximum(dims_e, 1e-9) - 1.0) <= max_shape_diff))
            short = o.height < 0.7 * ex_h
            if not (same_shape or short):
                o.r['share_rejected'] = "shape"
                continue
            if not short or ex_flat:
                # a complete instance must also *look* like the exemplar
                sig = _crop_signature(image_small, m)
                if sig is None or ex_sig is None or float(sig @ ex_sig) < (0.6 if ex_flat else 0.4):
                    o.r['share_rejected'] = "looks different"
                    continue
            # a fragment is judged by its colours: the visible part of the same
            # chair is made of the same colours, a different object is not
            if _hist_similarity(_chroma_hist(image_small, m), ex_hist) < 0.6:
                o.r['share_rejected'] = "colour make-up"
                continue
            # where does it stand: on the floor if the exemplar does, else at its own bottom
            bottom = ex_full_lo[1] if ex.r.get('support') == 'floor' and ev.floor_y is not None else float(o.lo[1])
            centre = np.array([0.5 * (o.lo[0] + o.hi[0]), 0.5 * (o.lo[2] + o.hi[2])])
            score, yaw, xz = _fit_copy(ev, ex_fast, m, centre, bottom, wall_side=o.r.get('against_wall'))
            fragment = False
            if short:
                lo_e, hi_e = ex_fast.min(0), ex_fast.max(0)
                c_e = np.array([0.5 * (lo_e[0] + hi_e[0]), lo_e[1], 0.5 * (lo_e[2] + hi_e[2])])
                copy = (ex_fast - c_e) @ _yaw(yaw).T + np.array([xz[0], bottom, xz[1]])
                fragment = hidden_fraction(ev, copy, m) > 0.15 or any(touches_border(m).values())
                if not fragment:
                    o.r['share_rejected'] = "short but not hidden"
                    continue
            # A fragment only has to be explained nearly as well as its own
            # sliver explained it; a complete instance must be explained
            # *better*, otherwise swapping it gains nothing and risks a lot.
            if score < (0.8 * own if fragment else own + min_gain) or score < 0.15:
                o.r['share_rejected'] = f"fit {score:.2f} vs own {own:.2f}"
                continue
            # The occlusion-aware score excuses pixels hidden behind nearer
            # things, so a copy that is far too big can pass by burying its
            # excess behind its neighbours (a portrait copied onto a small
            # frame's mask). The plain silhouette IoU does not excuse that.
            # A fragment is exempt: a chair behind a table *is* mostly hidden,
            # and what keeps a wrong fragment copy out is the colour make-up
            # test above and, for flat things, their face.
            if not fragment:
                plain_own = mask_iou(splat(ev, o.fast_verts)[0], m)
                lo_e, hi_e = ex_fast.min(0), ex_fast.max(0)
                c_e = np.array([0.5 * (lo_e[0] + hi_e[0]), lo_e[1], 0.5 * (lo_e[2] + hi_e[2])])
                copy_v = (ex_fast - c_e) @ _yaw(yaw).T + np.array([xz[0], bottom, xz[1]])
                if mask_iou(splat(ev, copy_v)[0], m) < max(0.7 * plain_own, 0.12):
                    o.r['share_rejected'] = "plain silhouette"
                    continue
            undo[o.id] = (o.model, o.snapshot(), dict(o.r))
            # adopt the exemplar's mesh and pose, then turn and move it
            o.model = ex.model
            o._idx = None
            o.restore(ex_pose)
            o.r['model_path'] = ex.r.get('model_path')
            o.rotate_about_vertical(yaw)
            c = np.array([0.5 * (o.lo[0] + o.hi[0]), o.lo[1], 0.5 * (o.lo[2] + o.hi[2])])
            o.translate(np.array([xz[0], bottom, xz[1]]) - c)
            o.r.update(instance_of=ex.id, instance_score=round(score, 3), own_score=round(own, 3),
                       support=ex.r.get('support') if bottom == ex_full_lo[1] else o.r.get('support'),
                       support_source="instance", snapped=True)
            out.append({"id": o.id, "label": label, "exemplar": ex.id, "score": round(score, 3),
                        "own_score": round(own, 3), "fragment": bool(fragment)})
        # Two copies fitted into the same place means at least one fit is
        # wrong (the search latched onto a neighbour's pixels): undo both
        # rather than let the duplicate filter delete a real object.
        replaced = [o for o in members if o.id in undo and o.r.get('instance_of') == ex.id]
        clash = set()
        for o in replaced:
            for q in members:
                if q is o:
                    continue
                lo, hi = np.maximum(o.lo, q.lo), np.minimum(o.hi, q.hi)
                inter = float(np.prod(np.clip(hi - lo, 0, None)))
                vmin = min(float(np.prod(np.maximum(o.hi - o.lo, 1e-9))), float(np.prod(np.maximum(q.hi - q.lo, 1e-9))))
                if inter > 0.4 * vmin:
                    clash.add(o.id)
        for o in replaced:
            if o.id in clash:
                model, pose, fields = undo[o.id]
                o.model, o._idx = model, None
                o.r.clear(); o.r.update(fields)
                o.restore(pose)
        out[:] = [d for d in out if d["id"] not in clash]
    return out
