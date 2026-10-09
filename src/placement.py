"""Evidence-based object placement (stage 4b).

SAM 3D poses every object on its own, so the raw poses need reconciling.
The previous placement did that from the objects' bounding boxes and label
lists ("anything called *table* stands on the floor", "a box whose footprint
is inside another box rests on it"). Auditing the result against the photo
(``scripts/audit_scene.py``) showed that those rules moved more objects away
from where the photo says they are than they fixed: a shower tray's box
"supported" the sink, the rug and a side table and lifted all three.

This module asks the photo instead. Per object it uses three pieces of
evidence the pipeline already has, none of them label-specific:

``depth``    the scene point map under the object's mask. If the placed
             mesh's visible surface is nearer or farther than the surface the
             point map observed, the object is slid along its viewing rays
             (position and size scaled together), which fixes depth without
             changing what the object looks like from the photo's viewpoint.
``contact``  the point map *just below* the mask's bottom edge. If those
             pixels are an upward-facing surface at the object's own depth
             and height, that surface carries the object: it is the floor, or
             the object whose mask owns those pixels, or an unmodelled
             surface (a countertop). The object's bottom is moved onto it.
             A wall below the mask (mounted things) or a nearer occluder
             (contact hidden) yields no contact, and the object stays where
             SAM 3D put it.
``reprojection`` the mask itself. Every correction is scored by the IoU of
             the mesh silhouette with the mask and reverted if it made the
             object fit the photo worse.

Same-label objects that occupy the same volume are reported as duplicates
and the worse-fitting one is dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src import room_layout as rl
from src.geometry import matrix_to_quat, quat_to_matrix, rotation_between, world_vertices

FLOOR = "floor"
SURFACE = "surface"


# ---------------------------------------------------------------------------
# Scene evidence
# ---------------------------------------------------------------------------

@dataclass
class SceneEvidence:
    """Point map and masks in the aligned frame, at point-map resolution."""
    points: np.ndarray            # (h, w, 3) aligned world
    valid: np.ndarray             # (h, w) bool
    normals: np.ndarray           # (h, w, 3) aligned
    depth: np.ndarray             # (h, w) camera z (nan where invalid)
    K: np.ndarray                 # normalised intrinsics
    R_total: np.ndarray           # camera-frame world -> aligned
    floor_y: Optional[float]
    room_height: float
    masks: Dict[int, np.ndarray] = field(default_factory=dict)   # id -> (h, w) bool
    owner: Optional[np.ndarray] = None                            # (h, w) object id, 0 = none
    floor_mask: Optional[np.ndarray] = None

    @property
    def hw(self) -> Tuple[int, int]:
        return self.valid.shape

    def project(self, verts_aligned: np.ndarray):
        h, w = self.hw
        return rl.project_world_to_pixels(verts_aligned @ self.R_total, self.K, w, h)


def build_evidence(points_world: np.ndarray, valid: np.ndarray, K: np.ndarray, R_total: np.ndarray,
                   floor_y: Optional[float], room_height: float,
                   masks: Optional[Dict[int, np.ndarray]] = None,
                   floor_mask: Optional[np.ndarray] = None) -> SceneEvidence:
    """``points_world`` is the camera-frame point map; masks may be any
    resolution. ``owner`` resolves overlapping masks to the smaller one (a
    mug's pixels belong to the mug, not to the desk mask around it)."""
    h, w = valid.shape
    P = points_world @ R_total.T
    N = rl.compute_normals(points_world, valid) @ R_total.T
    depth = np.where(valid, (points_world @ rl.OPENCV_TO_WORLD.T)[..., 2], np.nan)
    ms = {}
    owner = np.zeros((h, w), np.int32)
    for rid, m in (masks or {}).items():
        ms[rid] = rl._resize_mask(m, (h, w))
    for rid in sorted(ms, key=lambda i: -int(ms[i].sum())):
        owner[ms[rid]] = rid
    fm = rl._resize_mask(floor_mask, (h, w)) if floor_mask is not None else None
    return SceneEvidence(points=P, valid=valid, normals=N, depth=depth, K=np.asarray(K, float),
                         R_total=np.asarray(R_total, float), floor_y=floor_y, room_height=room_height,
                         masks=ms, owner=owner, floor_mask=fm)


# ---------------------------------------------------------------------------
# Silhouettes and depth buffers
# ---------------------------------------------------------------------------

def splat(ev: SceneEvidence, verts_aligned: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(silhouette, z-buffer) of a dense mesh's vertices at point-map
    resolution. Vertex splatting plus a small closing is within a pixel or
    two of rasterising SAM 3D's meshes (10^4-10^5 vertices each)."""
    from scipy import ndimage
    h, w = ev.hw
    u, v, z = ev.project(verts_aligned)
    ok = (z > 1e-6) & np.isfinite(u) & np.isfinite(v)
    ui = np.round(u[ok]).astype(np.int64)
    vi = np.round(v[ok]).astype(np.int64)
    inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    zbuf = np.full(h * w, np.inf)
    np.minimum.at(zbuf, vi[inside] * w + ui[inside], z[ok][inside])
    zbuf = zbuf.reshape(h, w)
    hit = np.isfinite(zbuf)
    if not hit.any():
        return hit, zbuf
    sil = ndimage.binary_fill_holes(ndimage.binary_closing(hit, iterations=2))
    zfill = ndimage.grey_erosion(zbuf, size=(5, 5))          # nearest splat in a 5x5 window
    zbuf = np.where(hit, zbuf, np.where(sil, zfill, np.inf))
    return sil, zbuf


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    u = int((a | b).sum())
    return float((a & b).sum()) / u if u else 0.0


class PlacedObject:
    """One result dict plus cached geometry; all moves go through here so the
    pose, the vertices and the bounds can never disagree."""

    def __init__(self, result: Dict, model_verts: np.ndarray):
        self.r = result
        self.model = model_verts
        self._verts: Optional[np.ndarray] = None
        self._idx: Optional[np.ndarray] = None

    @property
    def id(self) -> int:
        return self.r['id']

    @property
    def verts(self) -> np.ndarray:
        if self._verts is None:
            self._verts = world_vertices(self.model, self.r)
        return self._verts

    @property
    def lo(self) -> np.ndarray:
        return self.verts.min(axis=0)

    @property
    def hi(self) -> np.ndarray:
        return self.verts.max(axis=0)

    @property
    def height(self) -> float:
        return float(self.hi[1] - self.lo[1])

    @property
    def fast_verts(self) -> np.ndarray:
        """A fixed random subset of the world vertices (≤ 30k) for pose
        searches that splat the object hundreds of times."""
        n = len(self.model)
        if n <= 30_000:
            return self.verts
        if self._idx is None or len(self._idx) and self._idx.max() >= n:
            self._idx = np.random.default_rng(0).choice(n, 30_000, replace=False)
        return self.verts[self._idx]

    def scale_about_centre(self, s: float):
        c = 0.5 * (self.lo + self.hi)
        self.r['scale'] = (np.asarray(self.r['scale'], float) * s).tolist()
        self.r['translation'] = (c + (np.asarray(self.r['translation'], float) - c) * s).tolist()
        self._verts = None

    def rotate_about_vertical(self, yaw: float):
        """Turn the object about the vertical axis through its centre."""
        c, s = np.cos(yaw), np.sin(yaw)
        Y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        self.set_rotation(matrix_to_quat(quat_to_matrix(self.r['rotation_quaternion']) @ Y.T))

    def stretch_down(self, factor: float) -> bool:
        """Lengthen the object downward, keeping its top where it is. Only
        possible when one model axis is vertical (true after the upright
        correction); returns False otherwise."""
        R = quat_to_matrix(self.r['rotation_quaternion'])
        i = int(np.argmax(np.abs(R[:, 1])))
        if abs(R[i, 1]) < 0.97:
            return False
        top = float(self.hi[1])
        sc = np.asarray(self.r['scale'], float).copy()
        sc[i] *= factor
        self.r['scale'] = sc.tolist()
        self._verts = None
        self.translate([0, top - float(self.hi[1]), 0])
        return True

    def translate(self, d):
        self.r['translation'] = (np.asarray(self.r['translation'], float) + np.asarray(d, float)).tolist()
        if self._verts is not None:
            self._verts = self._verts + np.asarray(d, float)

    def slide_along_rays(self, k: float):
        """Scale position and size about the camera (the origin of the
        aligned frame): the image of the object does not change."""
        self.r['translation'] = (np.asarray(self.r['translation'], float) * k).tolist()
        self.r['scale'] = (np.asarray(self.r['scale'], float) * k).tolist()
        if self._verts is not None:
            self._verts = self._verts * k

    def set_rotation(self, quat):
        """Rotate about the object's own centre (not the model origin)."""
        c = 0.5 * (self.lo + self.hi)
        self.r['rotation_quaternion'] = [float(x) for x in quat]
        self._verts = None
        c2 = 0.5 * (self.lo + self.hi)
        self.translate(c - c2)

    def snapshot(self) -> Dict:
        return {k: list(self.r[k]) for k in ('translation', 'rotation_quaternion', 'scale')}

    def restore(self, snap: Dict):
        self.r.update({k: list(v) for k, v in snap.items()})
        self._verts = None


# ---------------------------------------------------------------------------
# Individual corrections
# ---------------------------------------------------------------------------

def fit_iou(ev: SceneEvidence, obj: PlacedObject) -> Optional[float]:
    m = ev.masks.get(obj.id)
    if m is None or not m.any():
        return None
    sil, _ = splat(ev, obj.verts)
    return mask_iou(sil, m)


def upright_rotation(R_obj: np.ndarray) -> Tuple[np.ndarray, float]:
    """Smallest rotation that makes the object's most-vertical model axis
    exactly vertical, and the tilt it removes (degrees).

    SAM 3D's meshes are axis-aligned in their canonical frame but which
    axis is "up" is not fixed across objects, so rather than assuming one we
    take whichever of the six ±axes is closest to world up. The correction
    is the minimal rotation, which leaves the heading untouched.
    ``R_obj`` is the row-vector pose rotation (``world = v @ R_obj``).
    """
    i = int(np.argmax(np.abs(R_obj[:, 1])))
    axis = R_obj[i] * np.sign(R_obj[i, 1])
    axis = axis / np.linalg.norm(axis)
    tilt = float(np.degrees(np.arccos(np.clip(axis[1], -1, 1))))
    fix = rotation_between(axis, np.array([0.0, 1.0, 0.0]))        # column-vector
    return R_obj @ fix.T, tilt


def correct_upright(ev: SceneEvidence, obj: PlacedObject, max_tilt_deg: float = 25.0,
                    max_iou_loss: float = 0.05) -> bool:
    """Remove a small tilt if doing so does not make the object fit its mask
    worse. Small tilts are pose noise; a large one is usually real (a leaning
    ladder, a fruit on its side) and is kept."""
    R_new, tilt = upright_rotation(quat_to_matrix(obj.r['rotation_quaternion']))
    obj.r['tilt_deg'] = round(tilt, 2)
    obj.r['upright_corrected'] = False
    if tilt < 0.5 or tilt > max_tilt_deg:
        return False
    before = fit_iou(ev, obj)
    snap = obj.snapshot()
    obj.set_rotation(matrix_to_quat(R_new))
    after = fit_iou(ev, obj)
    if before is not None and after is not None and after < before - max_iou_loss:
        obj.restore(snap)
        return False
    obj.r['upright_corrected'] = True
    return True


def refit_depth(ev: SceneEvidence, obj: PlacedObject, min_pixels: int = 40,
                deadband: float = 0.03, k_range: Tuple[float, float] = (0.5, 2.0)) -> Optional[float]:
    """Slide the object along its rays so its visible surface sits at the
    depth the point map observed under its mask. Returns the factor applied
    (None if there was not enough overlap to tell)."""
    m = ev.masks.get(obj.id)
    if m is None:
        return None
    sil, zbuf = splat(ev, obj.verts)
    both = sil & m & ev.valid & np.isfinite(zbuf)
    if int(both.sum()) < min_pixels:
        return None
    ratio = ev.depth[both] / zbuf[both]
    k = float(np.median(ratio))
    obj.r['depth_ratio_raw'] = round(1.0 / k, 3)
    if not (k_range[0] <= k <= k_range[1]) or abs(k - 1.0) < deadband:
        return 1.0
    obj.slide_along_rays(k)
    obj.r['depth_refit'] = round(k, 3)
    return k


@dataclass
class Contact:
    y: float                      # height of the supporting surface
    kind: str                     # FLOOR | SURFACE | "object"
    supporter: Optional[int]      # object id when kind == "object"
    pixels: int
    xz: np.ndarray                # median XZ of the supporting points


def find_contact(ev: SceneEvidence, obj: PlacedObject, band_frac: float = 0.02,
                 min_pixels: int = 12) -> Optional[Contact]:
    """Look just below the mask's bottom edge for the surface carrying the
    object. See the module docstring for the conditions."""
    m = ev.masks.get(obj.id)
    if m is None or not m.any():
        return None
    h, w = ev.hw
    band = max(3, int(round(band_frac * h)))
    cols = np.flatnonzero(m.any(axis=0))
    bottom = (h - 1) - np.argmax(m[::-1, cols], axis=0)        # lowest mask row per column
    # Only the lower part of the outline is a candidate for contact: the
    # underside of a table top between its legs is not where it stands.
    rows = np.flatnonzero(m.any(axis=1))
    y0, y1 = rows[0], rows[-1]
    low = bottom >= y1 - 0.25 * max(1, y1 - y0)
    if not low.any():
        return None
    cols, bottom = cols[low], bottom[low]
    rr = (bottom[:, None] + np.arange(1, band + 1)[None, :]).ravel()
    cc = np.repeat(cols, band)
    keep = rr < h
    rr, cc = rr[keep], cc[keep]
    keep = ev.valid[rr, cc] & ~m[rr, cc]
    rr, cc = rr[keep], cc[keep]
    if len(rr) < min_pixels:
        return None
    up = ev.normals[rr, cc, 1] > 0.8
    # At the object's own depth: not an occluder in front of it, not a
    # surface far behind it seen past its edge.
    z = ev.depth[rr, cc]
    own = ev.depth[m & ev.valid]
    if len(own) < 5:
        return None
    z_lo, z_hi = np.percentile(own, [2, 98])
    span = max(z_hi - z_lo, 0.05 * z_hi)
    near = (z > z_lo - 0.5 * span) & (z < z_hi + 0.5 * span)
    sel = up & near
    if int(sel.sum()) < max(min_pixels, 0.25 * len(rr)):
        return None
    pts = ev.points[rr[sel], cc[sel]]
    ys = pts[:, 1]
    y = float(np.median(ys))
    if float(np.percentile(ys, 84) - np.percentile(ys, 16)) > 0.06 * ev.room_height:
        return None                                   # not one surface
    owners = ev.owner[rr[sel], cc[sel]]
    kind, supporter = SURFACE, None
    ids, counts = np.unique(owners[owners > 0], return_counts=True)
    if len(ids) and counts.max() >= 0.5 * len(owners):
        kind, supporter = "object", int(ids[np.argmax(counts)])
    elif ev.floor_y is not None and abs(y - ev.floor_y) < 0.04 * ev.room_height:
        kind = FLOOR
    elif ev.floor_mask is not None and ev.floor_mask[rr[sel], cc[sel]].mean() > 0.5 \
            and ev.floor_y is not None and abs(y - ev.floor_y) < 0.08 * ev.room_height:
        kind = FLOOR
    return Contact(y=y, kind=kind, supporter=supporter, pixels=int(sel.sum()),
                   xz=np.median(pts[:, [0, 2]], axis=0))


def local_top(supporter: PlacedObject, lo: np.ndarray, hi: np.ndarray, y_max: float) -> Optional[float]:
    """Height of the supporter's geometry under a footprint, ignoring
    anything above ``y_max`` (a sofa's back above its seat, the shelf above
    the one the book stands on)."""
    V = supporter.verts
    sel = (V[:, 0] >= lo[0]) & (V[:, 0] <= hi[0]) & (V[:, 2] >= lo[2]) & (V[:, 2] <= hi[2]) & (V[:, 1] <= y_max)
    if int(sel.sum()) < 10:
        return None
    return float(np.percentile(V[sel, 1], 99))


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------

def structural_duplicates(masks: Dict[int, np.ndarray], structural_masks: Optional[Dict[str, np.ndarray]],
                          min_cover: float = 0.5, min_inside: float = 0.7) -> Dict[int, str]:
    """Objects that *are* the floor, a wall or the ceiling.

    Detectors are told not to list architecture but sometimes name it anyway
    ("carpet", "tiled wall"); stage 3 then turns it into a slab that
    competes with the room. An object is structural when its mask lies
    mostly inside a structural mask and covers most of it. A rug covers a
    small part of the floor mask and is kept. Returns ``{id: "floor"|...}``.
    """
    out: Dict[int, str] = {}
    for name, sm in (structural_masks or {}).items():
        if sm is None or not sm.any():
            continue
        for rid, m in masks.items():
            sm_r = rl._resize_mask(sm, m.shape)
            inter = float((m & sm_r).sum())
            if inter >= min_cover * sm_r.sum() and inter >= min_inside * m.sum():
                out[rid] = name
    return out


def contained_parts(masks: Dict[int, np.ndarray], labels: Dict[int, str], min_inside: float = 0.85,
                    group_cover: float = 0.7) -> Dict[int, int]:
    """Same-label masks that are a part of, or a group of, other masks.

    SAM 3 sometimes returns both an object and a piece of it under one
    prompt (a desk and its top, a screen and its picture area), and
    sometimes one mask spanning several instances next to the individual
    ones. IoU-based NMS keeps all of them because the IoU is low; stage 3
    then builds each as a full object in the same place. A mask that is
    ≥ ``min_inside`` inside a larger same-label mask is a part and is
    dropped — unless the contained masks together cover ≥ ``group_cover`` of
    the larger one, in which case the larger one is the group and *it* is
    dropped. Returns ``{dropped_id: id_it_duplicates}``.
    """
    ids = sorted(masks, key=lambda i: -int(masks[i].sum()))
    inside: Dict[int, List[int]] = {}
    for i, big in enumerate(ids):
        for small in ids[i + 1:]:
            if labels.get(big) != labels.get(small):
                continue
            a = float(masks[small].sum())
            if a > 0 and float((masks[small] & masks[big]).sum()) >= min_inside * a:
                inside.setdefault(big, []).append(small)
    out: Dict[int, int] = {}
    for big, parts in inside.items():
        if big in out:
            continue
        union = np.zeros_like(masks[big])
        for q in parts:
            union |= masks[q]
        if len(parts) >= 2 and float((union & masks[big]).sum()) >= group_cover * float(masks[big].sum()):
            out[big] = parts[0]
        else:
            for q in parts:
                out.setdefault(q, big)
    return out


def find_duplicates(objs: List[PlacedObject], ev: Optional[SceneEvidence] = None,
                    min_overlap: float = 0.45, min_size_ratio: float = 0.25) -> List[Tuple[int, int, float]]:
    """Pairs ``(keep, drop, overlap)`` of same-label objects whose boxes share
    most of the smaller one's volume and are of comparable size. Two masks
    of one desk (whole desk + its visible half) each become a full desk in
    stage 3 and land on top of each other. The better-fitting one is kept."""
    out = []
    dropped = set()
    order = sorted(objs, key=lambda o: -_quality(o, ev))
    for i, a in enumerate(order):
        if a.id in dropped:
            continue
        for b in order[i + 1:]:
            if b.id in dropped or a.r['label'] != b.r['label']:
                continue
            lo = np.maximum(a.lo, b.lo)
            hi = np.minimum(a.hi, b.hi)
            inter = float(np.prod(np.clip(hi - lo, 0, None)))
            va = float(np.prod(np.maximum(a.hi - a.lo, 1e-9)))
            vb = float(np.prod(np.maximum(b.hi - b.lo, 1e-9)))
            if min(va, vb) / max(va, vb) < min_size_ratio:
                continue
            ov = inter / min(va, vb)
            if ov >= min_overlap:
                out.append((a.id, b.id, round(ov, 3)))
                dropped.add(b.id)
    return out


def _quality(o: PlacedObject, ev: Optional[SceneEvidence]) -> float:
    q = float(o.r.get('confidence') or 0.5)
    if ev is not None:
        iou = o.r.get('fit_iou')
        if iou is not None:
            q *= 0.25 + iou
        m = ev.masks.get(o.id)
        if m is not None:
            q *= 1.0 + min(1.0, float(m.mean()) * 20)      # larger masks carry more evidence
    return q


# ---------------------------------------------------------------------------
# Settling, walls, visibility
# ---------------------------------------------------------------------------

def attachment(ev: SceneEvidence, obj: "PlacedObject", ring_frac: float = 0.02) -> Optional[str]:
    """How the photo shows an object to be held, if not from below.

    ``"mounted"``: the background just beside and above the object is at the
    object's own depth, i.e. it sits against a surface (a towel on a ring, a
    whiteboard, a wall cabinet). Around a free-standing object the
    background is far behind.
    ``"hanging"``: the surface directly above its top edge faces down at its
    depth: it touches the ceiling (a pendant with its cord, a ceiling fan).
    Returns None for neither.
    """
    from scipy import ndimage
    m = ev.masks.get(obj.id)
    if m is None or not m.any():
        return None
    h, w = ev.hw
    own = ev.depth[m & ev.valid]
    if len(own) < 5:
        return None
    z_obj = float(np.median(own))
    rows = np.flatnonzero(m.any(axis=1))
    y0, y1 = int(rows[0]), int(rows[-1])
    r = max(2, int(round(ring_frac * h)))
    ring = ndimage.binary_dilation(m, iterations=r) & ~m & ev.valid
    ring[int(y1 - 0.25 * max(1, y1 - y0)):] = False          # ignore what it may be standing on
    if ev.owner is not None:
        # Things smaller than the object cannot be what holds it (a tap on a
        # kitchen island); the background or a larger object can (a towel on
        # an oven door).
        ids, counts = np.unique(ev.owner[ring], return_counts=True)
        area = int(m.sum())
        small = [int(i) for i in ids if i > 0 and i != obj.id
                 and i in ev.masks and int(ev.masks[i].sum()) < area]
        if small:
            ring &= ~np.isin(ev.owner, small)
    if int(ring.sum()) >= 12:
        z = ev.depth[ring]
        if float(np.mean(np.abs(z - z_obj) <= 0.07 * z_obj)) >= 0.5:
            return "mounted"
    above = np.zeros_like(m)
    cols = np.flatnonzero(m.any(axis=0))
    top = np.argmax(m[:, cols], axis=0)
    near_top = top <= y0 + 0.25 * max(1, y1 - y0)
    for c, t in zip(cols[near_top], top[near_top], strict=True):
        above[max(0, t - r):t, c] = True
    above &= ev.valid & ~m
    if int(above.sum()) >= 8:
        down = ev.normals[above][:, 1] < -0.8
        close = np.abs(ev.depth[above] - z_obj) <= 0.15 * z_obj
        if float(np.mean(down & close)) >= 0.3:
            return "hanging"
    return None


def settle_unsupported(ev: SceneEvidence, objs: List[PlacedObject], room: Optional[Dict] = None,
                       max_iou_loss: float = 0.25) -> Dict[int, str]:
    """Give every free-standing object something to stand on.

    After the contact pass some objects still float: their bottom edge is
    hidden behind something or cut off by the frame, so the photo offers no
    contact. Physics still applies: an object that is not hanging from the
    ceiling and not mounted on a wall rests on whatever is under it. For
    each such object the target is the floor or the top of a supported
    object under its footprint, and three ways of reaching it are tried:

    * ``shift``    move straight down;
    * ``slide``    slide along the viewing rays (farther and larger);
    * ``stretch``  lengthen the object downward with its top fixed — right
                   when the mesh is short because its lower part was never
                   seen (allowed up to +35 %, up to +100 % when the mask is
                   cut by the bottom of the frame).

    The one that keeps the best silhouette match wins. Left alone: objects
    the photo shows mounted on a surface or hanging from the ceiling
    (``attachment``), and objects whose gap is large compared with their
    own size — a pendant lamp is not a floor lamp.
    """
    out: Dict[int, str] = {}
    if ev.floor_y is None:
        return out
    H = ev.room_height
    ceiling = room.get("ceiling_y") if room else None
    for o in sorted(objs, key=lambda o: o.lo[1]):
        if o.r.get('support') is not None:
            continue
        lo, hi = o.lo, o.hi
        h = max(float(hi[1] - lo[1]), 1e-6)
        if ceiling is not None and hi[1] > ceiling - 0.05 * H:
            continue                                   # reaches the ceiling: hanging
        target, sup_id = ev.floor_y, None
        area = max(1e-9, (hi[0] - lo[0]) * (hi[2] - lo[2]))
        for b in objs:
            if b is o or b.r.get('support') is None:
                continue
            blo, bhi = b.lo, b.hi
            ox = min(hi[0], bhi[0]) - max(lo[0], blo[0])
            oz = min(hi[2], bhi[2]) - max(lo[2], blo[2])
            if ox <= 0 or oz <= 0 or ox * oz < 0.5 * area:
                continue
            top = local_top(b, lo, hi, lo[1] + 0.05 * H)
            if top is not None and top > target:
                target, sup_id = top, b.id
        gap = float(lo[1] - target)
        mask = ev.masks.get(o.id)
        cut = mask is not None and bool(mask[-2:].any())      # bottom of the object is outside the photo
        if gap <= 1e-6 or gap > max((1.0 if cut else 0.5) * h, 0.10 * H):
            continue
        held = attachment(ev, o)
        if held is not None:
            o.r['attachment'] = held                   # the photo shows what holds it; not gravity's business
            continue
        if room and gap > 0.3 * h and _near_detected_wall(lo, hi, room, 0.03 * H):
            continue                                   # small thing high on a wall: mounted
        snap = o.snapshot()
        before = fit_iou(ev, o)
        options = []
        o.translate([0, -gap, 0])
        options.append((fit_iou(ev, o), "shift", o.snapshot()))
        o.restore(snap)
        if abs(lo[1]) > 0.2 * H and target * lo[1] > 0 and 0.8 <= target / lo[1] <= 1.25:
            o.slide_along_rays(target / lo[1])
            if not _beyond_detected_wall(o, room, 0.02 * H):     # farther is only an option inside the room
                options.append((fit_iou(ev, o), "slide", o.snapshot()))
            o.restore(snap)
        if gap <= (1.0 if cut else 0.35) * h and o.stretch_down((h + gap) / h):
            options.append((fit_iou(ev, o), "stretch", o.snapshot()))
        o.restore(snap)
        best = max(options, key=lambda t: -1.0 if t[0] is None else t[0])
        if before is not None and best[0] is not None and best[0] < before - max_iou_loss:
            continue
        o.restore(best[2])
        o.r.update(snapped=True, support=FLOOR if sup_id is None else "object", supported_by=sup_id,
                   support_source=f"settle-{best[1]}", support_y=round(float(target), 4))
        out[o.id] = best[1]
    return out


def _beyond_detected_wall(o: "PlacedObject", room: Optional[Dict], tol: float) -> bool:
    """True if the object sticks out through an observed wall by more than ``tol``."""
    if not room:
        return False
    lo, hi = o.lo, o.hi
    for side, src in room.get("sources", {}).items():
        if not str(src).startswith("wall"):
            continue
        axis = 0 if side.startswith("x") else 2
        over = (room["min"][axis] - lo[axis]) if side.endswith("min") else (hi[axis] - room["max"][axis])
        if over > tol:
            return True
    return False


def _near_detected_wall(lo, hi, room: Dict, tol: float) -> bool:
    for side, src in room.get("sources", {}).items():
        if not str(src).startswith("wall"):
            continue
        axis = 0 if side.startswith("x") else 2
        if side.endswith("min") and lo[axis] - room["min"][axis] < tol:
            return True
        if side.endswith("max") and room["max"][axis] - hi[axis] < tol:
            return True
    return False


def keep_inside_walls(ev: SceneEvidence, objs: List[PlacedObject], room: Dict,
                      min_slide: float = 0.8, max_push_frac: float = 0.35) -> int:
    """Bring objects that poke through an *observed* wall back inside.

    A wall backed by thousands of point-map votes beats one object's depth,
    so the object is wrong. Its position in the image is right, though, so
    it is slid along its viewing rays (nearer and proportionally smaller)
    until it fits, then put back on its support. Only if that needs more
    than a 20 % slide does it fall back to a capped sideways push.
    """
    moved = 0
    for o in objs:
        for side, src in room.get("sources", {}).items():
            if not str(src).startswith("wall"):
                continue
            axis = 0 if side.startswith("x") else 2
            lo, hi = o.lo, o.hi
            wall = room["min"][axis] if side.endswith("min") else room["max"][axis]
            extreme = lo[axis] if side.endswith("min") else hi[axis]
            over = (wall - extreme) if side.endswith("min") else (extreme - wall)
            if over <= 1e-6:
                continue
            bottom = float(lo[1])
            snap = o.snapshot()
            options = []
            k = wall / extreme if extreme * wall > 0 else 0.0
            if min_slide <= k < 1.0:
                o.slide_along_rays(k * 0.999)
                if o.r.get('support') is not None:       # stay on the support it was put on
                    o.translate([0, bottom - float(o.lo[1]), 0])
                options.append((fit_iou(ev, o), o.snapshot()))
                o.restore(snap)
            d = min(over, max_push_frac * max(float(hi[axis] - lo[axis]), 1e-6))
            d = d if side.endswith("min") else -d
            o.translate([d if axis == 0 else 0, 0, d if axis == 2 else 0])
            options.append((fit_iou(ev, o), o.snapshot()))
            best = max(options, key=lambda t: -1.0 if t[0] is None else t[0])
            o.restore(best[1])
            o.r['wall_clamped'] = True
            moved += 1
    return moved


def enforce_visibility(ev: SceneEvidence, objs: List[PlacedObject], min_hidden: float = 0.3) -> Dict[int, float]:
    """An object the photo shows must not end up behind another object.

    For every object, look at the pixels of its mask: there the photo saw
    *it*, so it has to be the nearest thing along those rays. If a larger
    object's mesh is in front on more than ``min_hidden`` of them (pillows
    swallowed by the sofa they lie on), the smaller object is slid toward
    the camera until it clears. Returns ``{id: factor}``.
    """
    h, w = ev.hw
    z1 = np.full((h, w), np.inf); id1 = np.zeros((h, w), np.int32)
    z2 = np.full((h, w), np.inf)
    own: Dict[int, np.ndarray] = {}
    for o in objs:
        sil, zb = splat(ev, o.fast_verts)
        zb = np.where(sil, zb, np.inf)
        own[o.id] = zb
        # Glass does not hide what is behind it. If, over an object's own
        # mask, the photo mostly saw something *farther* than its mesh, the
        # object is transparent (or misplaced) and cannot count as a blocker.
        m = ev.masks.get(o.id)
        if m is not None:
            px = m & sil & ev.valid
            if int(px.sum()) >= 20:
                with np.errstate(invalid="ignore"):
                    through = float(np.mean(ev.depth[px] > zb[px] * 1.15))
                if through > 0.4:
                    o.r['seen_through'] = round(through, 2)
                    continue
        nearer = zb < z1
        z2 = np.where(nearer, z1, np.minimum(z2, zb))
        id1 = np.where(nearer, o.id, id1)
        z1 = np.where(nearer, zb, z1)
    size = {o.id: float(np.prod(np.maximum(o.hi - o.lo, 1e-9))) for o in objs}
    out: Dict[int, float] = {}
    for o in sorted(objs, key=lambda o: size[o.id]):
        m = ev.masks.get(o.id)
        if m is None:
            continue
        zo = own[o.id]
        px = m & np.isfinite(zo)
        if int(px.sum()) < 20:
            continue
        others = np.where(id1 == o.id, z2, z1)
        hidden = px & (others < zo * 0.98)
        if hidden.sum() < min_hidden * px.sum():
            continue
        blockers = id1[hidden]
        ids, counts = np.unique(blockers[blockers != o.id], return_counts=True)
        if not len(ids) or size.get(int(ids[np.argmax(counts)]), 0.0) < size[o.id]:
            continue                                   # the bigger object does not yield
        if o.r.get('support') in (FLOOR, SURFACE):
            continue        # pinned by an observed contact; sliding would lift it off or shift its image
        k = float(np.median(others[hidden] / zo[hidden])) * 0.97
        if not (0.8 <= k < 1.0):
            continue        # a larger disagreement means one of the two meshes is wrong, not just deep
        o.slide_along_rays(k)
        o.r['moved_in_front'] = round(k, 3)
        out[o.id] = round(k, 3)
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def place_objects(results: List[Dict], model_verts: Dict[int, np.ndarray], ev: SceneEvidence,
                  verbose: bool = True, min_supporter_iou: float = 0.3, max_iou_loss: float = 0.12,
                  min_fit_iou: float = 0.08, room: Optional[Dict] = None,
                  image_small: Optional[np.ndarray] = None, share: bool = True) -> Dict:
    """Place every object using the evidence in ``ev``. ``results`` are in
    the aligned frame already (gravity + yaw applied); they are mutated in
    place and duplicates are removed from the list. Returns diagnostics.

    ``room`` (``{"min", "max", "sources", "ceiling_y"}`` from the layout)
    enables the wall and ceiling reasoning; ``image_small`` (the photo at
    point-map resolution) enables instance sharing."""
    log = print if verbose else (lambda *a, **k: None)
    objs = [PlacedObject(r, model_verts[r['id']]) for r in results if r['id'] in model_verts]
    by_id = {o.id: o for o in objs}
    diag: Dict = {"objects": len(objs)}
    H = ev.room_height

    # 1. small tilts, 2. depth
    n_up = sum(correct_upright(ev, o) for o in objs)
    refits = {}
    for o in objs:
        o.r['fit_iou_raw'] = _round(fit_iou(ev, o))
        k = refit_depth(ev, o)
        if k not in (None, 1.0):
            refits[o.id] = round(k, 3)
    diag['upright_corrected'] = n_up
    diag['depth_refit'] = refits
    log(f"  upright-corrected {n_up}/{len(objs)}; depth refit {len(refits)}"
        + (": " + ", ".join(f"{by_id[i].r['label']}×{k}" for i, k in refits.items()) if refits else ""))

    # 2b. objects that still fit their mask poorly: search size/position/depth
    from src import pose_fit
    refined = {}
    for o in objs:
        m = ev.masks.get(o.id)
        if m is None or pose_fit.fit_score(ev, o.fast_verts, m) >= 0.5:
            continue
        gain = pose_fit.refine_pose(ev, o)
        if gain is not None:
            refined[o.id] = round(gain, 3)
            refit_depth(ev, o)               # size or position changed: put the surface back at its depth
    diag['pose_refined'] = refined
    if refined:
        log("  pose refined by silhouette search: " + ", ".join(
            f"{by_id[i].r['label']} +{g}" for i, g in refined.items()))

    # 3. contacts
    contacts: Dict[int, Contact] = {}
    for o in objs:
        c = find_contact(ev, o)
        if c is not None and c.kind == "object" and (
                c.supporter not in by_id or (by_id[c.supporter].r.get('fit_iou_raw') or 0) < min_supporter_iou):
            # the supporter's mesh is missing or does not match the photo:
            # trust the observed surface height, not that mesh
            c = Contact(c.y, SURFACE, None, c.pixels, c.xz)
        if c is not None:
            tol = max(0.35 * o.height, 0.05 * H)
            if abs(o.lo[1] - c.y) <= tol:
                contacts[o.id] = c
    # break support cycles (A on B on A): drop the edge with fewer pixels
    for a, c in list(contacts.items()):
        if c.kind == "object":
            cb = contacts.get(c.supporter)
            if cb is not None and cb.kind == "object" and cb.supporter == a:
                weaker = a if c.pixels <= cb.pixels else c.supporter
                cw = contacts[weaker]
                contacts[weaker] = Contact(cw.y, SURFACE, None, cw.pixels, cw.xz)

    # 4. snap, supporters first
    done: set = set()

    def _snap(o: PlacedObject, depth: int = 0):
        if o.id in done:
            return
        done.add(o.id)
        c = contacts.get(o.id)
        o.r.update(snapped=False, supported_by=None, support=None)
        if c is None:
            # Contact not visible. Only the floor is a safe default, and only
            # when the object is already almost on it.
            if ev.floor_y is not None and abs(o.lo[1] - ev.floor_y) <= 0.05 * H:
                o.translate([0, ev.floor_y - o.lo[1], 0])
                o.r.update(snapped=True, support=FLOOR, support_source="near-floor")
            return
        target = c.y
        if c.kind == FLOOR and ev.floor_y is not None:
            target = ev.floor_y
        elif c.kind == "object":
            sup = by_id[c.supporter]
            if depth < 8:
                _snap(sup, depth + 1)
            top = local_top(sup, o.lo, o.hi, c.y + max(0.25 * o.height, 0.04 * H))
            if top is not None and abs(top - c.y) <= max(0.35 * o.height, 0.05 * H):
                target = top
            o.r['supported_by'] = c.supporter
        # Two ways to put the bottom on the surface: move straight up/down, or
        # slide along the viewing rays (changes depth and size, not the
        # image). Keep whichever fits the mask better; neither if both make
        # the fit clearly worse.
        snap = o.snapshot()
        before = o.r.get('fit_iou_raw')
        options = []
        o.translate([0, target - o.lo[1], 0])
        options.append((fit_iou(ev, o), "shift", o.snapshot()))
        o.restore(snap)
        bottom = o.lo[1]
        if abs(bottom) > 0.2 * H and target * bottom > 0 and 0.8 <= target / bottom <= 1.25:
            o.slide_along_rays(target / bottom)
            if not _beyond_detected_wall(o, room, 0.02 * H):
                options.append((fit_iou(ev, o), "slide", o.snapshot()))
            o.restore(snap)
        best = max(options, key=lambda t: -1.0 if t[0] is None else t[0])
        if before is not None and best[0] is not None and best[0] < before - max_iou_loss:
            o.r.update(supported_by=None, support_source="rejected")
            return
        o.restore(best[2])
        o.r.update(snapped=True, support=c.kind, support_source=f"contact-{best[1]}", support_y=round(target, 4))

    for o in sorted(objs, key=lambda o: o.lo[1]):
        _snap(o)
    kinds = [o.r.get('support') for o in objs]
    diag['support'] = {k: kinds.count(k) for k in (FLOOR, "object", SURFACE)}
    diag['supports'] = {o.id: o.r['supported_by'] for o in objs if o.r.get('supported_by')}
    diag['unsupported'] = kinds.count(None)
    log(f"  contact: {diag['support'][FLOOR]} on the floor, {diag['support']['object']} on another object, "
        f"{diag['support'][SURFACE]} on an unmodelled surface, {diag['unsupported']} left as posed")

    # 4b. free-standing objects with no visible contact still rest on something
    settled = settle_unsupported(ev, objs, room)
    diag['settled'] = settled
    if settled:
        log("  settled: " + ", ".join(f"{by_id[i].r['label']} ({how})" for i, how in settled.items()))

    # 4c. the same model photographed several times: reuse the best-observed one
    diag['instances'] = []
    if share and image_small is not None:
        diag['instances'] = pose_fit.share_instances(ev, objs, image_small)
        if diag['instances']:
            log("  instances shared: " + ", ".join(
                f"{d['label']} #{d['id']}←#{d['exemplar']} ({d['own_score']}→{d['score']})" for d in diag['instances']))

    # 5. nothing below a fitted floor
    raised = 0
    if ev.floor_y is not None:
        for o in objs:
            under = ev.floor_y - o.lo[1]
            if under > 1e-6:
                o.translate([0, min(under, 0.5 * o.height), 0])
                raised += 1
    diag['raised_to_floor'] = raised

    # 5b. inside observed walls; 5c. what the photo shows is not behind something else
    if room:
        diag['wall_clamped'] = keep_inside_walls(ev, objs, room)
        if diag['wall_clamped']:
            log(f"  brought {diag['wall_clamped']} object(s) back inside observed walls")
    diag['moved_in_front'] = enforce_visibility(ev, objs)
    if diag['moved_in_front']:
        log("  moved in front of what hid them: " + ", ".join(
            f"{by_id[i].r['label']}×{k}" for i, k in diag['moved_in_front'].items()))

    # 6. reprojection check
    for o in objs:
        o.r['fit_iou'] = _round(fit_iou(ev, o))

    # 7. reconstructions that do not look like their mask at all
    removed: Dict[int, str] = {}
    for o in objs:
        iou = o.r.get('fit_iou')
        if iou is not None and iou < min_fit_iou and not o.r.get('instance_of'):
            removed[o.id] = f"mesh does not match its mask (IoU {iou:.2f})"
    # 8. duplicates
    dups = find_duplicates([o for o in objs if o.id not in removed], ev)
    diag['duplicates'] = [{"kept": k, "dropped": d, "overlap": ov} for k, d, ov in dups]
    for k, d, _ in dups:
        removed[d] = f"duplicate of #{k}"
    if removed:
        log("  removed: " + ", ".join(f"{by_id[i].r['label']} #{i} ({why})" for i, why in removed.items()))
        results[:] = [r for r in results if r['id'] not in removed]
        for r in results:                    # nothing may rest on a removed object
            if r.get('supported_by') in removed:
                r['supported_by'] = None
                r['support'] = SURFACE
    diag['removed'] = {i: {"label": by_id[i].r['label'], "reason": why} for i, why in removed.items()}
    kept = [o for o in objs if o.id not in removed] or objs
    lo = np.min([o.lo for o in kept], axis=0)
    hi = np.max([o.hi for o in kept], axis=0)
    diag['bounds'] = {'min': lo.tolist(), 'max': hi.tolist(), 'size': (hi - lo).tolist()}
    ious = [r['fit_iou'] for r in results if r.get('fit_iou') is not None]
    if ious:
        diag['mean_fit_iou'] = round(float(np.mean(ious)), 3)
        log(f"  mean silhouette/mask IoU after placement: {diag['mean_fit_iou']}")
    return diag


def _round(x, n=3):
    return None if x is None else round(float(x), n)
