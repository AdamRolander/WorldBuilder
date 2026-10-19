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
    return splat_box(ev, verts_aligned)[:2]


def splat_box(ev: SceneEvidence, verts_aligned: np.ndarray):
    """(silhouette, z-buffer, (y0, y1, x0, x1)) of a dense mesh's vertices at point-map
    resolution. Vertex splatting plus a small closing is within a pixel or
    two of rasterising SAM 3D's meshes (10^4-10^5 vertices each).

    The pose searches call this thousands of times per scene, so the
    morphology runs on the object's bounding box, not on the whole frame
    (identical output; ``tests/test_placement.py`` checks that). The box
    bounds everything that is set; it is ``None`` for an empty silhouette."""
    from scipy import ndimage
    h, w = ev.hw
    u, v, z = ev.project(verts_aligned)
    ok = (z > 1e-6) & np.isfinite(u) & np.isfinite(v)
    ui = np.round(u[ok]).astype(np.int64)
    vi = np.round(v[ok]).astype(np.int64)
    inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    sil = np.zeros((h, w), bool)
    zbuf = np.full((h, w), np.inf)
    if not inside.any():
        return sil, zbuf, None
    ui, vi, zz = ui[inside], vi[inside], z[ok][inside]
    pad = 6                                   # > closing radius + erosion window, so the crop's edge never matters
    y0, y1 = max(0, int(vi.min()) - pad), min(h, int(vi.max()) + pad + 1)
    x0, x1 = max(0, int(ui.min()) - pad), min(w, int(ui.max()) + pad + 1)
    ch, cw = y1 - y0, x1 - x0
    zc = np.full(ch * cw, np.inf)
    np.minimum.at(zc, (vi - y0) * cw + (ui - x0), zz)
    zc = zc.reshape(ch, cw)
    hit = np.isfinite(zc)
    # Sides of the crop that are the frame's own edge keep the frame's
    # behaviour (closing erodes against the outside); interior sides are
    # padded with empty pixels, which is what the frame has there.
    top, left = pad if y0 > 0 else 0, pad if x0 > 0 else 0
    bottom, right = pad if y1 < h else 0, pad if x1 < w else 0
    hp = np.pad(hit, ((top, bottom), (left, right)))
    sp = ndimage.binary_fill_holes(ndimage.binary_closing(hp, iterations=2))
    zp = ndimage.grey_erosion(np.pad(zc, ((top, bottom), (left, right)), constant_values=np.inf), size=(5, 5))
    cut = (slice(top, top + ch), slice(left, left + cw))
    sc, zf = sp[cut], zp[cut]
    sil[y0:y1, x0:x1] = sc
    zbuf[y0:y1, x0:x1] = np.where(hit, zc, np.where(sc, zf, np.inf))
    return sil, zbuf, (y0, y1, x0, x1)


def mask_box(mask: np.ndarray):
    """(y0, y1, x0, x1) of a mask's set pixels, or None."""
    rows = np.flatnonzero(mask.any(1))
    if not len(rows):
        return None
    cols = np.flatnonzero(mask.any(0))
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def _splat_reference(ev: SceneEvidence, verts_aligned: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Whole-frame version of ``splat`` (slow); kept as the definition the
    fast one is tested against."""
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

    def stretch_along(self, axis: int, factor: float, toward: str) -> bool:
        """Lengthen the object along a world axis, keeping the face that
        looks away from ``toward`` (``"x_min"`` … the wall side) where it
        is. Needs a model axis aligned with that world axis; returns False
        otherwise."""
        R = quat_to_matrix(self.r['rotation_quaternion'])
        i = int(np.argmax(np.abs(R[:, axis])))
        if abs(R[i, axis]) < 0.97:
            return False
        fixed = float(self.hi[axis]) if toward.endswith("min") else float(self.lo[axis])
        sc = np.asarray(self.r['scale'], float).copy()
        sc[i] *= factor
        self.r['scale'] = sc.tolist()
        self._verts = None
        now = float(self.hi[axis]) if toward.endswith("min") else float(self.lo[axis])
        d = np.zeros(3); d[axis] = fixed - now
        self.translate(d)
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
    # The surface must *continue* under the object's edge: the band pixel is
    # compared with the mask's own bottom pixel in that column. A supporting
    # surface is at the same height as that edge point and at nearly the
    # same depth (a little nearer, since it runs on toward the camera). An
    # occluder in front is much nearer; the floor seen past a hanging
    # object's edge is lower and farther. Comparing with the whole mask's
    # depth range instead (the old test) rejected most objects standing on
    # counters and sofas, because the surface a few pixels in front of a
    # small object's base is already nearer than its front face.
    edge_ok = ev.valid[bottom, cols]
    cols, bottom = cols[edge_ok], bottom[edge_ok]
    if len(cols) < 3:
        return None
    y_edge = ev.points[bottom, cols, 1]
    z_edge = ev.depth[bottom, cols]
    rr = (bottom[:, None] + np.arange(1, band + 1)[None, :]).ravel()
    cc = np.repeat(cols, band)
    ye = np.repeat(y_edge, band)
    ze = np.repeat(z_edge, band)
    keep = rr < h
    rr, cc, ye, ze = rr[keep], cc[keep], ye[keep], ze[keep]
    keep = ev.valid[rr, cc] & ~m[rr, cc]
    rr, cc, ye, ze = rr[keep], cc[keep], ye[keep], ze[keep]
    if len(rr) < min_pixels:
        return None
    up = ev.normals[rr, cc, 1] > 0.8
    z = ev.depth[rr, cc]
    y = ev.points[rr, cc, 1]
    with np.errstate(invalid="ignore"):
        near = (z > 0.80 * ze) & (z < 1.08 * ze) & (np.abs(y - ye) < 0.04 * ev.room_height)
    sel = up & near
    if int(sel.sum()) < max(min_pixels, 0.25 * len(rr)):
        # No horizontal surface, but another object right under the edge at
        # a continuous depth (a plant in its pot, a pillow on a seat cushion
        # seen from the front): that object carries this one; its top under
        # the footprint is found by the caller.
        if ev.owner is not None:
            owners = ev.owner[rr, cc]
            cand = near & (owners > 0)
            ids, counts = np.unique(owners[cand], return_counts=True)
            if len(ids) and counts.max() >= max(min_pixels, 0.5 * len(rr)):
                sup = int(ids[np.argmax(counts)])
                pts = ev.points[rr[cand], cc[cand]]
                return Contact(y=float(np.median(ye[cand])), kind="object", supporter=sup,
                               pixels=int(counts.max()), xz=np.median(pts[:, [0, 2]], axis=0))
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


def ground_masks(masks: Dict[int, np.ndarray], points_aligned: np.ndarray, normals_aligned: np.ndarray,
                 valid: np.ndarray, floor_y: float, camera_height: float, min_frame: float = 0.03,
                 min_ground: float = 0.6) -> Dict[int, str]:
    """Masks that are the ground itself, for open scenes without a floor mask.

    A detector asked for "objects" in a forest lists the path and the lawn.
    SAM 3D turns those into slabs; they are terrain. A mask is ground when
    it is large and most of its pixels are an upward-facing surface near the
    fitted ground level (within the camera's height of it, which allows for
    slopes). Small flat things (a mat, a manhole cover) stay objects.
    """
    out: Dict[int, str] = {}
    h, w = valid.shape
    up = valid & (normals_aligned[..., 1] > 0.75) & (np.abs(points_aligned[..., 1] - floor_y) < camera_height)
    for rid, m in masks.items():
        if m.shape != (h, w):
            from src.room_layout import _resize_mask
            m = _resize_mask(m, (h, w))
        n = int((m & valid).sum())
        if n < min_frame * h * w:
            continue
        if float((m & up).sum()) / n >= min_ground:
            out[rid] = "ground"
    return out


def contained_parts(masks: Dict[int, np.ndarray], labels: Dict[int, str], min_inside: float = 0.85,
                    group_cover: float = 0.4) -> Dict[int, int]:
    """Same-label masks that are a part of, or a group of, other masks.

    SAM 3 sometimes returns both an object and a piece of it under one
    prompt (a desk and its top, a screen and its picture area), and
    sometimes one mask spanning several instances next to the individual
    ones. IoU-based NMS keeps all of them because the IoU is low; stage 3
    then builds each as a full object in the same place. A mask that is
    ≥ ``min_inside`` inside a larger same-label mask is a part and is
    dropped — unless two or more of them together cover ≥ ``group_cover`` of
    the larger one, in which case the larger one is the group and *it* is
    dropped. (A row of chairs reconstructed from its group mask is one long
    bar; the chairs in it are the objects, even if a few were not
    segmented on their own.) Returns ``{dropped_id: id_it_duplicates}``.
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
        if gap <= 1e-6:
            continue
        if gap > max((1.0 if cut else 0.5) * h, 0.10 * H):
            # Too far to fall. But if the photo shows the object's own outline
            # reaching down to that surface (a counter whose mesh came out as
            # just its top), the mesh is short, not the object high: lengthen
            # it down, when that clearly improves the match with the photo.
            c = find_contact(ev, o)
            if c is not None and abs(c.y - target) <= 0.05 * H:
                before = fit_iou(ev, o)
                snap = o.snapshot()
                if before is not None and o.stretch_down((h + gap) / h):
                    after = fit_iou(ev, o)
                    if after is not None and after >= before + 0.08:
                        o.r.update(snapped=True, support=FLOOR if sup_id is None else "object", supported_by=sup_id,
                                   support_source="settle-reach", support_y=round(float(target), 4))
                        out[o.id] = "reach"
                        continue
                o.restore(snap)
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
        if abs(lo[1]) > 1e-6 and target * lo[1] > 0 and 0.8 <= target / lo[1] <= 1.25:
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


def restack(ev: SceneEvidence, objs: List["PlacedObject"], before: Dict[int, np.ndarray],
            max_iou_loss: float = 0.12) -> Dict[int, float]:
    """Keep what stands on an object on it after the object moved.

    Contacts are resolved early; later passes then move supporters (a
    cabinet pushed back to its wall, a table slid inside the room, a sofa
    stood upright), and whatever was put on them stays where it was: the
    plant hovers over the gap behind the vanity. ``before`` holds every
    object's bottom-centre when the contacts were made. A dependent moves
    sideways by as much as its supporter did (if it did not move itself)
    and is put back down on the supporter's top under its footprint.
    Supporters are handled before what stands on them. Returns
    ``{id: vertical correction}``.
    """
    by_id = {o.id: o for o in objs}
    H = ev.room_height
    out: Dict[int, float] = {}
    done: set = set()

    def bottom_centre(o):
        lo, hi = o.lo, o.hi
        return np.array([0.5 * (lo[0] + hi[0]), lo[1], 0.5 * (lo[2] + hi[2])])

    def visit(o, depth=0):
        if o.id in done or depth > 8:
            return
        done.add(o.id)
        sid = o.r.get('supported_by')
        if o.r.get('support') != "object" or sid not in by_id:
            return
        sup = by_id[sid]
        visit(sup, depth + 1)
        snap = o.snapshot()
        iou0 = fit_iou(ev, o)
        if sid in before and o.id in before:
            moved_sup = bottom_centre(sup) - before[sid]
            moved_own = bottom_centre(o) - before[o.id]
            d = np.array([moved_sup[0], 0.0, moved_sup[2]])
            if np.linalg.norm(d) > 0.005 * H and np.linalg.norm(moved_own[[0, 2]]) < 0.5 * np.linalg.norm(d):
                o.translate(d)
        top = local_top(sup, o.lo, o.hi, float(o.lo[1]) + max(0.25 * o.height, 0.04 * H))
        if top is None:
            o.restore(snap)
            return
        dy = top - float(o.lo[1])
        if abs(dy) > 0.005 * H:
            o.translate([0, dy, 0])
        iou1 = fit_iou(ev, o)
        if iou0 is not None and iou1 is not None and iou1 < iou0 - max_iou_loss:
            o.restore(snap)
            return
        if abs(dy) > 0.005 * H:
            out[o.id] = round(float(dy), 4)
            o.r['restacked'] = round(float(dy), 4)

    for o in objs:
        visit(o)
    return out


def _box_samples(mn: np.ndarray, mx: np.ndarray, n: int = 24) -> np.ndarray:
    """Dense points on a box's surface (for splatting a proxy like a mesh)."""
    a = np.linspace(0.0, 1.0, n)
    g1, g2 = (g.ravel() for g in np.meshgrid(a, a))
    faces = []
    for axis in range(3):
        for val in (0.0, 1.0):
            p = np.zeros((len(g1), 3))
            p[:, axis] = val
            p[:, (axis + 1) % 3] = g1
            p[:, (axis + 2) % 3] = g2
            faces.append(p)
    return mn + np.vstack(faces) * (mx - mn)


def make_proxy(ev: SceneEvidence, o: "PlacedObject", room: Optional[Dict]) -> bool:
    """Replace a failed reconstruction by a box fitted to what the photo saw.

    When SAM 3D's mesh misses its mask entirely (a wall shelf generated as
    something else), the object used to be dropped, and whatever stood on it
    then floated. If other objects rest on it, a box stands in: the 5th to
    95th percentile of the observed points under its own mask, at least a
    few centimetres thick, reaching back to the wall the photo shows it
    against. It is crude, but it carries the bottles. Returns True if done.
    """
    m = ev.masks.get(o.id)
    if m is None:
        return False
    own = m & ev.valid
    if ev.owner is not None:
        own &= ev.owner == o.id
    if int(own.sum()) < 20:
        return False
    P = ev.points[own]
    mn, mx = np.percentile(P, 5, axis=0), np.percentile(P, 95, axis=0)
    t = 0.02 * ev.room_height
    for a in range(3):
        if mx[a] - mn[a] < t:
            c = 0.5 * (mn[a] + mx[a])
            mn[a], mx[a] = c - 0.5 * t, c + 0.5 * t
    side = wall_contact(ev, o, room) if room else None
    if side is not None:
        axis = 0 if side.startswith("x") else 2
        if side.endswith("min"):
            mn[axis] = min(mn[axis], room["min"][axis])
        else:
            mx[axis] = max(mx[axis], room["max"][axis])
    world = _box_samples(mn, mx)
    o.model = np.stack([world[:, 0], world[:, 2], -world[:, 1]], 1)      # inverse of the GLB frame change
    o._idx = None
    o.r.update(translation=[0.0, 0.0, 0.0], rotation_quaternion=[1.0, 0.0, 0.0, 0.0], scale=[1.0, 1.0, 1.0])
    o._verts = None
    o.r['proxy'] = "box from the photo (reconstruction did not match its mask)"
    return True


def upright_resting(ev: SceneEvidence, objs: List["PlacedObject"], max_tilt_deg: float = 15.0,
                    max_iou_loss: float = 0.10) -> Dict[int, float]:
    """Anything resting on a horizontal surface stands upright.

    ``correct_upright`` runs before anything is known about support and
    keeps a tilt whenever removing it costs a little silhouette match, which
    leaves furniture standing on one edge (a vanity 5° off the floor). Once
    an object is known to rest on the floor or on another surface, a small
    tilt is pose noise by physics: it is removed, the bottom put back on
    the support and the object re-centred sideways so its outline stays on
    the mask. Large tilts (a leaning ladder) are kept. Returns ``{id: tilt}``.
    """
    out: Dict[int, float] = {}
    for o in objs:
        if o.r.get('support') is None or o.r.get('upright_corrected'):
            continue
        R_new, tilt = upright_rotation(quat_to_matrix(o.r['rotation_quaternion']))
        if tilt < 0.5 or tilt > max_tilt_deg:
            continue
        before = fit_iou(ev, o)
        snap = o.snapshot()
        bottom = float(o.lo[1])
        o.set_rotation(matrix_to_quat(R_new))
        o.translate([0, bottom - float(o.lo[1]), 0])
        size = float(np.linalg.norm(o.hi - o.lo))
        upright = o.snapshot()
        best = (fit_iou(ev, o), upright)
        for dx in (-0.04, -0.02, 0.02, 0.04):
            o.restore(upright)
            o.translate([dx * size, 0, 0])
            sc = fit_iou(ev, o)
            if sc is not None and (best[0] is None or sc > best[0]):
                best = (sc, o.snapshot())
        if before is not None and best[0] is not None and best[0] < before - max_iou_loss:
            o.restore(snap)
            continue
        o.restore(best[1])
        o.r['upright_corrected'] = True
        o.r['upright_resting'] = round(tilt, 2)
        out[o.id] = round(tilt, 2)
    return out


def wall_contact(ev: SceneEvidence, obj: "PlacedObject", room: Dict, band_frac: float = 0.02,
                 min_pixels: int = 12) -> Optional[str]:
    """The observed wall an object stands against, if the photo shows one.

    Just above the mask's top edge the photo shows what is directly behind
    the object's top. If that is a vertical surface at the same depth as the
    top-edge pixels themselves (the sofa's back rail, the cabinet's top) and
    it lies on one of the room's *observed* walls, the object's back is at
    that wall. Pixels of other objects (books on a shelf, a mirror over a
    vanity) are ignored; nothing is decided when the mask reaches the top of
    the frame. Returns the wall side (``"z_max"`` …) or None.
    """
    m = ev.masks.get(obj.id)
    if m is None or not m.any() or not room:
        return None
    h, w = ev.hw
    band = max(3, int(round(band_frac * h)))
    cols = np.flatnonzero(m.any(axis=0))
    top = np.argmax(m[:, cols], axis=0)
    rows = np.flatnonzero(m.any(axis=1))
    y0, y1 = rows[0], rows[-1]
    if y0 <= 1:
        return None
    high = top <= y0 + 0.25 * max(1, y1 - y0)           # only the upper part of the outline
    cols, top = cols[high], top[high]
    edge_ok = ev.valid[top, cols]
    cols, top = cols[edge_ok], top[edge_ok]
    if len(cols) < 3:
        return None
    # the object's own depth a few pixels inside its outline: the edge pixel
    # itself is often a blend with what is behind it (the wall's depth)
    inner = np.minimum(top + 3, h - 1)
    inner_ok = m[inner, cols] & ev.valid[inner, cols]
    z_edge = np.where(inner_ok, ev.depth[inner, cols], ev.depth[top, cols])
    # the row touching the edge has a normal smeared across the depth step
    rr = (top[:, None] - np.arange(2, band + 2)[None, :]).ravel()
    cc = np.repeat(cols, band)
    ze = np.repeat(z_edge, band)
    keep = rr >= 0
    rr, cc, ze = rr[keep], cc[keep], ze[keep]
    keep = ev.valid[rr, cc] & ~m[rr, cc]
    if ev.owner is not None:
        keep &= (ev.owner[rr, cc] == 0) | (ev.owner[rr, cc] == obj.id)
    rr, cc, ze = rr[keep], cc[keep], ze[keep]
    n = ev.normals[rr, cc]
    z = ev.depth[rr, cc]
    with np.errstate(invalid="ignore"):
        vertical = np.abs(n[:, 1]) < 0.35
        same_depth = np.abs(z / ze - 1.0) <= 0.08
    sel = vertical & same_depth if len(rr) >= min_pixels else np.zeros(len(rr), bool)
    # Part of the band is often the object's own top surface seen from above
    # (a counter top behind its front lip), so a third of it being wall at
    # the right depth is enough — as long as the wall pixels agree among
    # themselves on the depth.
    if len(rr) and int(sel.sum()) >= max(min_pixels, 0.3 * len(rr)) and not (int(vertical.sum()) and sel.sum() < 0.6 * vertical.sum()):
        side = _wall_side(ev.points[rr[sel], cc[sel]], n[sel], room)
        if side is not None:
            return side
    # The mask may stop at the front of the object's top (a vanity whose
    # marble top was not segmented, a desk seen from above): then the pixels
    # above the edge are that top, an upward-facing surface running back
    # continuously from the edge. Follow it up the image; where it meets a
    # vertical surface at the depth it reached, that surface is behind the
    # object. Things standing on the top are stepped over.
    pts, nrm = _climb_top_surface(ev, obj, m, cols, top, max_rows=int(0.25 * h))
    if len(pts) >= max(3, 0.3 * len(cols)):
        return _wall_side(pts, nrm, room)
    return None


def _climb_top_surface(ev: SceneEvidence, obj: "PlacedObject", m: np.ndarray, cols: np.ndarray, top: np.ndarray,
                       max_rows: int, min_up_rows: int = 2):
    """Per column, walk up from the mask's top edge over an upward-facing
    surface (normal y > 0.6) whose depth grows smoothly, and return the first
    vertical surface point reached at a continuous depth (one per column)."""
    h, w = ev.hw
    own_area = int(m.sum())
    smaller = None
    if ev.owner is not None:
        ids = [i for i, mm in ev.masks.items() if i != obj.id and int(mm.sum()) < own_area]
        smaller = np.isin(ev.owner, ids) if ids else np.zeros((h, w), bool)
    pts, nrm = [], []
    for c, t in zip(cols, top, strict=True):
        z_prev = ev.depth[t, c]
        if not np.isfinite(z_prev):
            continue
        n_up = 0
        y_top = 0.0
        r = t - 1
        stop = max(0, t - max_rows)
        lip = max(4, int(0.03 * h))               # the top's own front face, a few percent of the frame
        n_skip = 0
        while r >= stop:
            if m[r, c] or (smaller is not None and smaller[r, c]):
                if n_up and not m[r, c]:
                    n_skip += 1
                    if n_skip > 3 * n_up + 5:
                        break                     # more "things on it" than top: something else behind
                r -= 1
                continue                          # its own pixels, or a thing standing on the top
            if not ev.valid[r, c] or (ev.owner is not None and ev.owner[r, c] not in (0, obj.id)):
                break                             # a larger object in the way: nothing to conclude
            z = ev.depth[r, c]
            ny = ev.normals[r, c, 1]
            front = n_up == 0 and t - r <= lip    # still on the edge / front lip of the top
            hi_ratio = 1.25 if n_up else 1.06     # a top seen at a grazing angle recedes fast
            if not (z_prev * 0.97 <= z <= z_prev * hi_ratio):
                break                             # depth jumps: not one continuous surface
            if ny > 0.6:
                y = ev.points[r, c, 1]
                if n_up == 0:
                    y_top = y
                elif abs(y - y_top) > 0.03 * ev.room_height:
                    break                         # not the same flat top (the floor behind a sofa)
                n_up += 1
                z_prev = z
                r -= 1
                continue
            if abs(ny) < 0.35 and n_up >= min_up_rows:
                # the corner's normals are smeared; take the point a few rows up
                r2 = max(0, r - 3)
                if ev.valid[r2, c] and abs(ev.normals[r2, c, 1]) < 0.35 and \
                        abs(ev.depth[r2, c] / z_prev - 1.0) <= 0.08:
                    pts.append(ev.points[r2, c])
                    nrm.append(ev.normals[r2, c])
                break
            if front:
                z_prev = z
                r -= 1
                continue
            break
    return np.asarray(pts).reshape(-1, 3), np.asarray(nrm).reshape(-1, 3)


def _wall_side(pts: np.ndarray, nrm: np.ndarray, room: Dict) -> Optional[str]:
    """Which observed wall a set of vertical surface points lies on, if any."""
    if len(pts) == 0:
        return None
    nm = nrm.mean(axis=0)
    axis = 0 if abs(nm[0]) > abs(nm[2]) else 2
    if abs(nm[axis]) < 0.6:
        return None
    # the inward normal of an x_max wall points -x: a +x normal means x_min
    side = f"{'x' if axis == 0 else 'z'}_{'min' if nm[axis] > 0 else 'max'}"
    if not str(room.get("sources", {}).get(side, "")).startswith("wall") or side in room.get("disputed", []):
        return None
    wall = room["min"][axis] if side.endswith("min") else room["max"][axis]
    span = float(max(room["max"][0] - room["min"][0], room["max"][2] - room["min"][2]))
    coord = np.median(pts[:, axis])
    if abs(coord - wall) > 0.04 * span:
        return None                                     # a cabinet front or partition, not the wall
    return side


def _heading_off(o: "PlacedObject", axis: int) -> Optional[float]:
    """Signed angle (radians) about the vertical between the object's model
    axis nearest the world ``axis`` and that axis; None if the object is not
    upright (no model axis is vertical)."""
    R = quat_to_matrix(o.r['rotation_quaternion'])          # rows: model axes in world
    up = int(np.argmax(np.abs(R[:, 1])))
    if abs(R[up, 1]) < 0.97:
        return None
    horiz = [i for i in range(3) if i != up]
    best = max(horiz, key=lambda i: abs(R[i, axis]) / max(np.linalg.norm(R[i, [0, 2]]), 1e-9))
    d = R[best, [0, 2]] * np.sign(R[best, axis] or 1.0)
    target = np.array([1.0, 0.0]) if axis == 0 else np.array([0.0, 1.0])
    return float(np.arctan2(d[0] * target[1] - d[1] * target[0], d @ target))


def square_to_wall(ev: SceneEvidence, o: "PlacedObject", side: str, max_deg: float = 20.0,
                   max_iou_loss: float = 0.08) -> Optional[float]:
    """Turn an object that stands against a wall parallel to it.

    Furniture against a wall is built to stand flush with it; SAM 3D's
    heading for it is off by a few degrees more often than not, and the
    back corner then pokes through the wall while the other end stands
    away from it. The turn is about the object's centre and is kept unless
    the silhouette match drops. Returns the turn in degrees, or None.
    """
    axis = 0 if side.startswith("x") else 2
    off = _heading_off(o, axis)
    if off is None or abs(np.degrees(off)) < 1.0 or abs(np.degrees(off)) > max_deg:
        return None
    before = fit_iou(ev, o)
    snap = o.snapshot()
    best = None
    for yaw in (off, -off):
        o.restore(snap)
        o.rotate_about_vertical(yaw)
        resid = _heading_off(o, axis)
        if resid is not None and abs(resid) < 0.25 * abs(off):
            best = yaw
            break
    if best is None:
        o.restore(snap)
        return None
    after = fit_iou(ev, o)
    if before is not None and after is not None and after < before - max_iou_loss:
        o.restore(snap)
        return None
    o.r['squared_to_wall'] = round(float(np.degrees(best)), 2)
    return float(np.degrees(best))


def square_long_objects(ev: SceneEvidence, objs: List["PlacedObject"], room: Optional[Dict],
                        min_aspect: float = 2.0, max_deg: float = 12.0, max_iou_loss: float = 0.05) -> Dict[int, float]:
    """Long furniture runs along the room's axes.

    The layout stage already assumes the room is a box; a counter, a bench,
    a sofa or a long table in it is almost always parallel to its walls,
    and a few degrees off is what SAM 3D's heading noise looks like. An
    object whose footprint is at least ``min_aspect`` times as long as wide
    and within ``max_deg`` of a room axis is turned onto it, if the
    silhouette match holds. Returns ``{id: degrees turned}``.
    """
    out: Dict[int, float] = {}
    span = float(max(room["max"][0] - room["min"][0], room["max"][2] - room["min"][2])) if room else None
    for o in objs:
        if o.r.get('squared_to_wall') is not None:
            continue
        R = quat_to_matrix(o.r['rotation_quaternion'])
        up = int(np.argmax(np.abs(R[:, 1])))
        if abs(R[up, 1]) < 0.97:
            continue
        sc = np.abs(np.asarray(o.r['scale'], float))
        horiz = [i for i in range(3) if i != up]
        ext = [float(np.ptp(o.model[:, i]) * sc[i]) for i in horiz]
        if min(ext) <= 0 or max(ext) / min(ext) < min_aspect:
            continue
        if span is not None and max(ext) < 0.15 * span:
            continue
        offs = [a for a in (_heading_off(o, 0), _heading_off(o, 2)) if a is not None]
        if not offs:
            continue
        off = min(offs, key=abs)
        deg = abs(np.degrees(off))
        if deg < 1.0 or deg > max_deg:
            continue
        before = fit_iou(ev, o)
        snap = o.snapshot()
        done = None
        for yaw in (off, -off):
            o.restore(snap)
            o.rotate_about_vertical(yaw)
            rest = [a for a in (_heading_off(o, 0), _heading_off(o, 2)) if a is not None]
            if rest and min(abs(a) for a in rest) < 0.25 * abs(off):
                done = yaw
                break
        if done is None:
            o.restore(snap)
            continue
        after = fit_iou(ev, o)
        if before is not None and after is not None and after < before - max_iou_loss:
            o.restore(snap)
            continue
        o.r['squared_to_room'] = round(float(np.degrees(done)), 2)
        out[o.id] = round(float(np.degrees(done)), 2)
    return out


def depth_agreement(ev: SceneEvidence, obj: "PlacedObject", min_pixels: int = 40) -> Optional[float]:
    """Median ratio of observed depth to the placed mesh's visible depth
    over the object's mask (1.0 = the mesh's front is where the photo saw
    it). Moving an object along its viewing direction barely changes its
    silhouette, so moves in depth are checked with this instead."""
    m = ev.masks.get(obj.id)
    if m is None:
        return None
    sil, zbuf = splat(ev, obj.fast_verts)
    both = sil & m & ev.valid & np.isfinite(zbuf)
    if int(both.sum()) < min_pixels:
        return None
    return float(np.median(ev.depth[both] / zbuf[both]))


def push_to_walls(ev: SceneEvidence, objs: List[PlacedObject], room: Dict, max_stretch: float = 1.5,
                  max_iou_loss: float = 0.10) -> Dict[int, Dict]:
    """Close the gap between an object and the wall the photo shows it
    against (``wall_contact``).

    SAM 3D's meshes are generated from one side, and their depth is a
    guess: a sofa or a cabinet seen from the front comes out too shallow
    more often than not. After the depth refit the *front* of such an
    object is where the photo saw it, and the gap is at the back, against
    the wall — where cushions, pictures and bottles then land. Two ways to
    close it are tried: lengthen the object toward the wall with its front
    fixed (a mesh that is too shallow; allowed up to ``max_stretch``), or
    move it back as a whole. The silhouette picks, and neither is kept if
    the match with the photo drops. Returns ``{id: {"side", "how", "gap"}}``.
    """
    out: Dict[int, Dict] = {}
    H = ev.room_height
    for o in objs:
        side = wall_contact(ev, o, room)
        if side is None:
            continue
        o.r['against_wall'] = side
        square_to_wall(ev, o, side)
        axis = 0 if side.startswith("x") else 2
        wall = room["min"][axis] if side.endswith("min") else room["max"][axis]
        lo, hi = o.lo, o.hi
        gap = (lo[axis] - wall) if side.endswith("min") else (wall - hi[axis])
        depth = max(float(hi[axis] - lo[axis]), 1e-6)
        if gap <= 0.01 * H or gap > max(2.0 * depth, 0.5 * H):
            continue                      # a thin picture may hang far from where it was generated
        snap = o.snapshot()
        before = fit_iou(ev, o)
        options = []
        d = np.zeros(3); d[axis] = -gap if side.endswith("min") else gap
        o.translate(d)
        k = depth_agreement(ev, o)
        if k is None or abs(k - 1.0) <= 0.06:      # its front must stay where the photo saw it
            options.append((fit_iou(ev, o), "move", o.snapshot()))
        o.restore(snap)
        # Lengthening is for a mesh that came out too shallow: it never makes
        # the object deeper than it is wide (a chair is not a sofa).
        lateral = float(hi[2 - axis] - lo[2 - axis])
        factor = min(max_stretch, (depth + gap) / depth, max(1.0, lateral / depth))
        if factor > 1.02 and o.stretch_along(axis, factor, toward=side):
            if (depth + gap) / depth > factor:          # stretch as far as allowed, move the rest
                lo2, hi2 = o.lo, o.hi
                rest = (lo2[axis] - wall) if side.endswith("min") else (wall - hi2[axis])
                d = np.zeros(3); d[axis] = -rest if side.endswith("min") else rest
                o.translate(d)
            options.append((fit_iou(ev, o), "stretch", o.snapshot()))
            o.restore(snap)
        if not options:
            continue
        best = max(options, key=lambda t: -1.0 if t[0] is None else t[0])
        if before is not None and best[0] is not None and best[0] < before - max_iou_loss:
            continue
        o.restore(best[2])
        o.r['wall_gap_closed'] = best[1]
        out[o.id] = {"side": side, "how": best[1], "gap": round(float(gap), 4)}
    return out


def clamp_below_ceiling(ev: SceneEvidence, objs: List[PlacedObject], room: Dict, max_iou_loss: float = 0.10) -> int:
    """Nothing pokes through an observed ceiling. A hanging lamp whose mesh
    is too tall is slid nearer (and smaller) or moved down, whichever keeps
    its silhouette; a prior-placed ceiling is not evidence and is left to
    grow instead."""
    ceiling = room.get("ceiling_y") if room else None
    if ceiling is None:
        return 0
    H = ev.room_height
    moved = 0
    for o in objs:
        over = float(o.hi[1]) - ceiling
        if over <= 0.01 * H:
            continue
        snap = o.snapshot()
        before = fit_iou(ev, o)
        options = []
        o.translate([0, -over, 0])
        options.append((fit_iou(ev, o), o.snapshot()))
        o.restore(snap)
        top = float(o.hi[1])
        if top > 1e-6 and 0.7 <= ceiling / top < 1.0:
            o.slide_along_rays(ceiling / top * 0.999)
            options.append((fit_iou(ev, o), o.snapshot()))
            o.restore(snap)
        best = max(options, key=lambda t: -1.0 if t[0] is None else t[0])
        if before is not None and best[0] is not None and best[0] < before - max_iou_loss:
            continue
        o.restore(best[1])
        o.r['ceiling_clamped'] = True
        moved += 1
    return moved


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
    # A plane that a quarter of the scene's objects stand beyond is not the
    # room's wall (it is a cabinet front, a partition, a glass door). Such a
    # side is disputed: nothing is moved for it and the caller lets the room
    # grow there instead.
    span = float(max(room["max"][0] - room["min"][0], room["max"][2] - room["min"][2]))
    disputed = room.setdefault("disputed", [])
    for side, src in room.get("sources", {}).items():
        if not str(src).startswith("wall"):
            continue
        axis = 0 if side.startswith("x") else 2
        wall = room["min"][axis] if side.endswith("min") else room["max"][axis]
        n_out = sum(1 for o in objs
                    if ((wall - o.lo[axis]) if side.endswith("min") else (o.hi[axis] - wall)) > 0.02 * span)
        if n_out > max(3, 0.25 * len(objs)) and side not in disputed:
            disputed.append(side)
    for o in objs:
        for side, src in room.get("sources", {}).items():
            if not str(src).startswith("wall") or side in disputed:
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
            before = fit_iou(ev, o)
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
            if before is not None and best[0] is not None and best[0] < before - 0.10:
                # Coming inside would cost the object its match with the photo:
                # more likely the "wall" is a cabinet front or a partition and
                # the object really is beyond it. Leave it where the photo has it.
                o.restore(snap)
                o.r['beyond_wall'] = side
                continue
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
        if (o.r.get('fit_iou_raw') or 0.0) < min_fit_iou:
            continue        # a mesh that misses its mask entirely is a failed reconstruction, not a pose to tune
        gain = pose_fit.refine_pose(ev, o)
        if gain is not None:
            refined[o.id] = round(gain, 3)
            refit_depth(ev, o)               # size or position changed: put the surface back at its depth
    diag['pose_refined'] = refined
    if refined:
        log("  pose refined by silhouette search: " + ", ".join(
            f"{by_id[i].r['label']} +{g}" for i, g in refined.items()))

    # 3. contacts
    raw_contacts = {o.id: find_contact(ev, o) for o in objs}
    # a failed reconstruction that carries other objects becomes a box
    carriers = {c.supporter for c in raw_contacts.values() if c is not None and c.kind == "object"}
    proxies = []
    for o in objs:
        if o.id in carriers and (o.r.get('fit_iou_raw') or 0.0) < min_fit_iou and make_proxy(ev, o, room):
            o.r['fit_iou_raw'] = _round(fit_iou(ev, o))
            proxies.append(o.id)
    diag['proxies'] = proxies
    if proxies:
        log("  replaced by a box from the photo (mesh missed its mask, things rest on it): " + ", ".join(
            f"{by_id[i].r['label']} #{i}" for i in proxies))
    contacts: Dict[int, Contact] = {}
    for o in objs:
        c = raw_contacts[o.id]
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
        before = fit_iou(ev, o)            # the pose as it is now, after refit and refinement
        options = []
        o.translate([0, target - o.lo[1], 0])
        options.append((fit_iou(ev, o), "shift", o.snapshot()))
        o.restore(snap)
        bottom = o.lo[1]
        # Sliding along the rays is well conditioned whenever the factor is
        # moderate; an object at camera height (|bottom| small) simply gets a
        # factor outside the range and is shifted instead.
        if abs(bottom) > 1e-6 and target * bottom > 0 and 0.8 <= target / bottom <= 1.25:
            o.slide_along_rays(target / bottom)
            if not _beyond_detected_wall(o, room, 0.02 * H):
                options.append((fit_iou(ev, o), "slide", o.snapshot()))
            o.restore(snap)
        best = max(options, key=lambda t: -1.0 if t[0] is None else t[0])
        if before is not None and best[0] is not None and best[0] < before - max_iou_loss:
            o.r.update(supported_by=None, support_source="rejected",
                       contact_rejected={"y": round(float(target), 4), "kind": c.kind, "iou_before": _round(before),
                                         "options": {how: _round(sc) for sc, how, _ in options}})
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

    stacked_at = {o.id: np.array([0.5 * (o.lo[0] + o.hi[0]), o.lo[1], 0.5 * (o.lo[2] + o.hi[2])]) for o in objs}

    # 4b''. long furniture runs along the room's axes
    if room:
        diag['squared_to_room'] = square_long_objects(ev, objs, room)
        if diag['squared_to_room']:
            log("  squared to the room: " + ", ".join(
                f"{by_id[i].r['label']} ({d}°)" for i, d in diag['squared_to_room'].items()))

    # 4b'. resting objects stand upright
    diag['upright_resting'] = upright_resting(ev, objs)
    if diag['upright_resting']:
        log("  stood upright on their support: " + ", ".join(
            f"{by_id[i].r['label']} ({t}°)" for i, t in diag['upright_resting'].items()))

    # 4c. which objects the photo shows standing against an observed wall
    # (their copies below must keep their backs to it)
    if room:
        for o in objs:
            side = wall_contact(ev, o, room)
            if side is not None:
                o.r['against_wall'] = side
    # 4d. the same model photographed several times: reuse the best-observed one
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
        diag['disputed_walls'] = list(room.get("disputed", []))
        if diag['wall_clamped']:
            log(f"  brought {diag['wall_clamped']} object(s) back inside observed walls")
        if diag['disputed_walls']:
            log(f"  not a wall after all (many objects beyond it): {', '.join(diag['disputed_walls'])}")
        # 5c. backs against the wall the photo shows behind them; nothing above an observed ceiling
        diag['against_wall'] = push_to_walls(ev, objs, room)
        if diag['against_wall']:
            log("  pushed against the wall: " + ", ".join(
                f"{by_id[i].r['label']} ({d['how']} {d['side']})" for i, d in diag['against_wall'].items()))
        diag['ceiling_clamped'] = clamp_below_ceiling(ev, objs, room)
        if diag['ceiling_clamped']:
            log(f"  brought {diag['ceiling_clamped']} object(s) down below the observed ceiling")
    # 5d. what stands on something that moved goes with it
    diag['restacked'] = restack(ev, objs, stacked_at)
    if diag['restacked']:
        log("  put back on their (moved) supporters: " + ", ".join(
            f"{by_id[i].r['label']}" for i in diag['restacked']))
    diag['moved_in_front'] = enforce_visibility(ev, objs)
    if diag['moved_in_front']:
        log("  moved in front of what hid them: " + ", ".join(
            f"{by_id[i].r['label']}×{k}" for i, k in diag['moved_in_front'].items()))

    # 6. reprojection check
    for o in objs:
        o.r['fit_iou'] = _round(fit_iou(ev, o))
        if o.r.get('proxy'):
            o.r['proxy_box'] = {"min": o.lo.tolist(), "max": o.hi.tolist()}

    # 7. reconstructions that do not look like their mask at all
    removed: Dict[int, str] = {}
    for o in objs:
        iou = o.r.get('fit_iou')
        if iou is not None and iou < min_fit_iou and not o.r.get('instance_of') and not o.r.get('proxy'):
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
