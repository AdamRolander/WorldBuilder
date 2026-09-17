"""Structural layout: floor, walls and ceiling from the scene point map.

SAM 3D Objects runs a monocular geometry model (MoGe) on the full image to
get a point map it conditions on. Until now that point map was computed
once *per object* and thrown away. This module consumes it once for the
whole scene and derives what SAM 3D itself is bad at — architecture:

1. **Floor plane** by RANSAC, either restricted to a SAM 3 "floor" mask
   (preferred) or, without masks, from points in the lower image whose
   normals point roughly up.
2. **Gravity alignment.** Photos are rarely taken level. Objects come back
   in the camera frame, so a 15° downward pitch tilts every object and the
   whole room by 15°. Rotating the scene so the floor normal is +Y fixes
   the room and lets ``upright_correct`` in room_generator work on a true
   vertical.
3. **Manhattan yaw** from the dominant horizontal normal direction, so the
   room box is aligned with the real walls rather than the camera.
4. **Walls / ceiling** as vertical planes fitted to wall-mask points (or to
   horizontal-normal points), snapped to the four box sides; sides with no
   evidence fall back to object bounds plus padding.
5. **Textures by projection.** Each plane is a dense vertex-coloured grid;
   every vertex is projected back into the photo through the recovered
   intrinsics and coloured by the pixel it lands on if it is not occluded
   (depth test against the point map). Hidden regions are filled from the
   nearest visible sample blended with the plane's median colour — a stand-in
   for real inpainting (see docs/ROADMAP.md).

Everything here is numpy/scipy/trimesh on the CPU; ``tests/test_room_layout.py``
exercises it on synthetic rooms and, when the MoGe weights are cached, on a
real photo.

Frames
------
``world`` is SAM 3D's frame: OpenCV camera with X and Y negated (X left, Y up,
Z into the scene), camera at the origin. MoGe emits OpenCV (X right, Y down,
Z forward). ``OPENCV_TO_WORLD`` converts; it is its own inverse.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.geometry import matrix_to_quat, quat_to_matrix, rotation_between

OPENCV_TO_WORLD = np.diag([-1.0, -1.0, 1.0])
UP = np.array([0.0, 1.0, 0.0])


# --------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------

@dataclass
class Plane:
    normal: List[float]          # unit, in whatever frame the caller is in
    offset: float                # n·p + offset = 0
    inliers: int = 0
    inlier_fraction: float = 0.0
    source: str = "none"         # "mask" | "geometric" | "fallback"

    def distance(self, pts: np.ndarray) -> np.ndarray:
        return pts @ np.asarray(self.normal) + self.offset


@dataclass
class RoomLayout:
    gravity_rotation: List[List[float]]      # 3x3, applied to camera-frame world
    yaw_deg: float
    floor: Plane
    floor_y: float
    ceiling_y: float
    ceiling_source: str
    bounds_min: List[float]                  # aligned frame, room interior
    bounds_max: List[float]
    wall_sources: Dict[str, str] = field(default_factory=dict)  # side -> evidence
    camera_height: float = 0.0
    estimated_metric_scale: float = 1.0      # multiply scene units by this for metres (prior: camera at 1.5 m)
    notes: List[str] = field(default_factory=list)

    def to_json(self) -> Dict:
        d = asdict(self)
        return d

    @property
    def R_total(self) -> np.ndarray:
        """Column-vector rotation taking camera-frame world to aligned world."""
        G = np.asarray(self.gravity_rotation)
        y = math.radians(self.yaw_deg)
        Y = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
        return Y @ G


# --------------------------------------------------------------------------
# Point-map utilities
# --------------------------------------------------------------------------

def moge_to_world(points_opencv: np.ndarray) -> np.ndarray:
    """(H,W,3) or (N,3) OpenCV camera points -> SAM 3D world frame."""
    return np.asarray(points_opencv, dtype=np.float64) @ OPENCV_TO_WORLD.T


def project_world_to_pixels(points_world: np.ndarray, K_norm: np.ndarray,
                            width: int, height: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points through normalised intrinsics (MoGe convention:
    cx, cy ≈ 0.5, focal in units of image size). Returns (u, v, depth)."""
    cam = np.asarray(points_world, dtype=np.float64) @ OPENCV_TO_WORLD.T  # back to OpenCV
    z = cam[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (K_norm[0, 0] * cam[..., 0] / z + K_norm[0, 2]) * width
        v = (K_norm[1, 1] * cam[..., 1] / z + K_norm[1, 2]) * height
    return u, v, z


def intrinsics_from_pointmap(points_world: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Recover normalised pinhole intrinsics (MoGe convention) from a point map
    by least squares: u = fx·(x/z) + cx, v = fy·(y/z) + cy with u, v in [0,1].
    Needed because the upstream SAM 3D helper only stores intrinsics when the
    depth model did *not* provide them (an upstream quirk)."""
    H, W = valid.shape
    cam = np.asarray(points_world, dtype=np.float64) @ OPENCV_TO_WORLD.T
    ii, jj = np.nonzero(valid & (cam[..., 2] > 1e-6))
    if len(ii) < 100:
        return np.array([[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1.0]])
    if len(ii) > 50_000:
        sel = np.random.default_rng(0).choice(len(ii), 50_000, replace=False)
        ii, jj = ii[sel], jj[sel]
    x, y, z = cam[ii, jj, 0], cam[ii, jj, 1], cam[ii, jj, 2]
    u, v = (jj + 0.5) / W, (ii + 0.5) / H
    A = np.stack([x / z, np.ones_like(x)], 1)
    fx, cx = np.linalg.lstsq(A, u, rcond=None)[0]
    B = np.stack([y / z, np.ones_like(y)], 1)
    fy, cy = np.linalg.lstsq(B, v, rcond=None)[0]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def compute_normals(points: np.ndarray, valid: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-pixel normals of an (H,W,3) point map via central differences.
    Oriented to face the camera (origin)."""
    P = np.asarray(points, dtype=np.float64)
    dx = np.zeros_like(P)
    dy = np.zeros_like(P)
    dx[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dy[1:-1, :] = P[2:, :] - P[:-2, :]
    n = np.cross(dx, dy)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = np.where(norm > 1e-9, n / norm, 0.0)
    # face the camera: normal · (camera - point) > 0
    flip = np.sum(n * (-P), axis=-1) < 0
    n[flip] *= -1
    if valid is not None:
        n[~valid] = 0.0
    return n


def ransac_plane(pts: np.ndarray, threshold: float, iterations: int = 500,
                 rng: Optional[np.random.Generator] = None,
                 normal_prior: Optional[np.ndarray] = None,
                 max_angle_deg: float = 90.0) -> Tuple[Optional[np.ndarray], float, np.ndarray]:
    """Plain RANSAC. Returns (unit normal, offset, inlier mask). If
    ``normal_prior`` is given, hypotheses further than ``max_angle_deg`` from
    it are skipped (cheap way to say "the floor points roughly up")."""
    pts = np.asarray(pts, dtype=np.float64)
    n_pts = len(pts)
    if n_pts < 3:
        return None, 0.0, np.zeros(n_pts, bool)
    rng = rng or np.random.default_rng(0)
    cos_max = math.cos(math.radians(max_angle_deg))
    best_n, best_in = None, np.zeros(n_pts, bool)
    best_count = 0
    for _ in range(iterations):
        idx = rng.choice(n_pts, 3, replace=False)
        a, b, c = pts[idx]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n /= nn
        if normal_prior is not None:
            if abs(float(n @ normal_prior)) < cos_max:
                continue
            if float(n @ normal_prior) < 0:
                n = -n
        d = -float(n @ a)
        dist = np.abs(pts @ n + d)
        inl = dist < threshold
        cnt = int(inl.sum())
        if cnt > best_count:
            best_count, best_n, best_in = cnt, n, inl
    if best_n is None:
        return None, 0.0, best_in
    # Refine with least squares on inliers (SVD).
    P = pts[best_in]
    cen = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - cen, full_matrices=False)
    n = vt[-1]
    if normal_prior is not None and float(n @ normal_prior) < 0:
        n = -n
    elif normal_prior is None and float(n @ best_n) < 0:
        n = -n
    d = -float(n @ cen)
    inl = np.abs(pts @ n + d) < threshold
    return n, d, inl


def _scene_scale(points: np.ndarray, valid: np.ndarray) -> float:
    """Median distance from camera — the unit everything else is relative to."""
    P = points[valid]
    return float(np.median(np.linalg.norm(P, axis=1))) if len(P) else 1.0


# --------------------------------------------------------------------------
# Floor
# --------------------------------------------------------------------------

def estimate_floor(points: np.ndarray, valid: np.ndarray,
                   floor_mask: Optional[np.ndarray] = None,
                   normals: Optional[np.ndarray] = None,
                   max_tilt_deg: float = 40.0,
                   rng: Optional[np.random.Generator] = None) -> Optional[Plane]:
    """Fit the floor plane in the (camera-frame) world.

    ``points`` (H,W,3) world frame; ``valid`` (H,W) bool; ``floor_mask`` (H,W)
    bool from SAM 3 if available (any resolution — it is resized to the map).
    """
    H, W = valid.shape
    scale = _scene_scale(points, valid)
    thr = 0.015 * scale
    if normals is None:
        normals = compute_normals(points, valid)

    cand = None
    source = "none"
    if floor_mask is not None and floor_mask.any():
        fm = _resize_mask(floor_mask, (H, W))
        cand = fm & valid
        source = "mask"
        if cand.sum() < 200:
            cand = None
    if cand is None:
        # Geometric guess: lower 65 % of the frame, normal within max_tilt of +Y.
        rows = np.arange(H)[:, None] >= int(0.35 * H)
        up_like = normals @ UP > math.cos(math.radians(max_tilt_deg))
        cand = valid & rows & up_like
        source = "geometric"
        if cand.sum() < 100:
            cand = valid & rows
            source = "geometric-loose"
    if cand.sum() < 50:
        return None

    pts = points[cand]
    n, d, inl = ransac_plane(pts, thr, iterations=600, rng=rng,
                             normal_prior=UP, max_angle_deg=max_tilt_deg)
    if n is None:
        return None
    # The camera must be above the floor: signed distance of the origin > 0.
    if d < 0:
        n, d = -n, -d
    frac = float(inl.mean())
    if frac < 0.25:
        return None
    return Plane(normal=n.tolist(), offset=float(d), inliers=int(inl.sum()),
                 inlier_fraction=frac, source=source)


def estimate_ceiling_plane(points: np.ndarray, valid: np.ndarray,
                           ceiling_mask: Optional[np.ndarray] = None,
                           normals: Optional[np.ndarray] = None,
                           max_tilt_deg: float = 40.0,
                           rng: Optional[np.random.Generator] = None) -> Optional[Plane]:
    """Mirror of estimate_floor for the ceiling (normal within max_tilt of -Y,
    upper part of the frame). Used for gravity when no floor is visible."""
    H, W = valid.shape
    scale = _scene_scale(points, valid)
    thr = 0.015 * scale
    if normals is None:
        normals = compute_normals(points, valid)
    cand = None
    source = "none"
    if ceiling_mask is not None and ceiling_mask.any():
        cand = _resize_mask(ceiling_mask, (H, W)) & valid
        source = "mask"
        if cand.sum() < 200:
            cand = None
    if cand is None:
        rows = np.arange(H)[:, None] <= int(0.65 * H)
        down_like = normals @ (-UP) > math.cos(math.radians(max_tilt_deg))
        cand = valid & rows & down_like
        source = "geometric"
    if cand.sum() < 100:
        return None
    n, d, inl = ransac_plane(points[cand], thr, iterations=600, rng=rng,
                             normal_prior=-UP, max_angle_deg=max_tilt_deg)
    if n is None or inl.mean() < 0.25:
        return None
    return Plane(normal=n.tolist(), offset=float(d), inliers=int(inl.sum()),
                 inlier_fraction=float(inl.mean()), source=source)


def _resize_mask(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    if mask.shape == tuple(shape):
        return mask
    from PIL import Image
    im = Image.fromarray(mask.astype(np.uint8) * 255)
    im = im.resize((shape[1], shape[0]), Image.NEAREST)
    return np.asarray(im) > 127


# --------------------------------------------------------------------------
# Gravity + yaw
# --------------------------------------------------------------------------

def gravity_rotation(floor_normal: Sequence[float]) -> np.ndarray:
    """Column-vector rotation G with G @ floor_normal = +Y."""
    return rotation_between(np.asarray(floor_normal, float), UP)


def rotate_pose(pose: Dict, R_world: np.ndarray) -> Dict:
    """Apply a world rotation (column-vector matrix) to a SAM 3D pose dict
    so that ``world_vertices(v, new_pose) == world_vertices(v, pose) @ R.T``.
    Returns a new dict; other keys are copied through."""
    R_obj = quat_to_matrix(pose["rotation_quaternion"])
    R_new = R_obj @ R_world.T
    t_new = R_world @ np.asarray(pose["translation"], dtype=np.float64)
    new = dict(pose)
    new["rotation_quaternion"] = matrix_to_quat(R_new).tolist()
    new["translation"] = t_new.tolist()
    return new


def dominant_yaw_deg(normals_aligned: np.ndarray, weights: Optional[np.ndarray] = None,
                     horizontal_tol: float = 0.25) -> Tuple[float, float]:
    """Manhattan-world yaw from horizontal normals: the angle (mod 90°) that
    most wall normals share. Returns (yaw_deg, support fraction). Applying
    ``yaw_matrix(yaw)`` to the scene aligns walls with the X/Z axes."""
    n = normals_aligned.reshape(-1, 3)
    w = np.ones(len(n)) if weights is None else weights.reshape(-1)
    horiz = (np.abs(n[:, 1]) < horizontal_tol) & (np.linalg.norm(n, axis=1) > 0.5)
    if horiz.sum() < 50:
        return 0.0, 0.0
    ang = np.arctan2(n[horiz, 2], n[horiz, 0])          # angle in XZ plane
    ang4 = np.mod(ang * 4, 2 * math.pi)                  # fold 90° symmetry
    c = np.average(np.cos(ang4), weights=w[horiz])
    s = np.average(np.sin(ang4), weights=w[horiz])
    support = float(math.hypot(c, s))
    yaw = math.atan2(s, c) / 4.0                          # radians, in (-45°, 45°]
    return math.degrees(yaw), support


def yaw_matrix(yaw_deg: float) -> np.ndarray:
    y = math.radians(yaw_deg)
    return np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])


# --------------------------------------------------------------------------
# Walls & ceiling (in the gravity- and yaw-aligned frame)
# --------------------------------------------------------------------------

def estimate_ceiling(points_aligned: np.ndarray, normals_aligned: np.ndarray, valid: np.ndarray,
                     floor_y: float, camera_y: float = 0.0) -> Tuple[Optional[float], str]:
    """Height of a downward-facing plane above the camera, if visible."""
    down = (normals_aligned @ (-UP) > math.cos(math.radians(30)))
    above = points_aligned[..., 1] > camera_y + 0.15 * abs(camera_y - floor_y)
    cand = valid & down & above
    if cand.sum() < 150:
        return None, "none"
    ys = points_aligned[cand][:, 1]
    y = float(np.median(ys))
    spread = float(np.percentile(ys, 84) - np.percentile(ys, 16))
    if spread > 0.15 * (y - floor_y):
        return None, "noisy"
    return y, "geometric"


def wall_positions(points_aligned: np.ndarray, normals_aligned: np.ndarray, valid: np.ndarray,
                   floor_y: float, ceiling_y: float, wall_mask: Optional[np.ndarray] = None,
                   min_support: int = 150) -> Dict[str, Tuple[float, int]]:
    """Find axis-aligned vertical planes. Returns {side: (coordinate, support)}
    for sides in {"x_min","x_max","z_min","z_max"} that have evidence.

    Candidate points: on the wall mask if given, else points with a
    near-horizontal normal that are between floor and ceiling and not near
    the floor (furniture fronts also have horizontal normals, hence the
    later "outermost" requirement).
    """
    H, W = valid.shape
    n = normals_aligned
    horiz = np.abs(n[..., 1]) < 0.3
    y = points_aligned[..., 1]
    band = (y > floor_y + 0.05 * (ceiling_y - floor_y)) & (y < ceiling_y - 0.02 * (ceiling_y - floor_y))
    cand = valid & horiz & band
    if wall_mask is not None and wall_mask.any():
        # The mask adds evidence; it must not restrict it (SAM 3's "wall"
        # mask often covers only the most obvious wall).
        cand |= valid & band & _resize_mask(wall_mask, (H, W))
    out: Dict[str, Tuple[float, int]] = {}
    if cand.sum() < min_support:
        return out
    P = points_aligned[cand]
    N = n[cand]
    room_w = max(np.ptp(points_aligned[valid][:, 0]), 1e-6)
    room_d = max(np.ptp(points_aligned[valid][:, 2]), 1e-6)
    bin_w = 0.02 * max(room_w, room_d)

    def _side(axis: int, sign: int, name: str):
        # Points whose normal faces the interior along this axis.
        facing = N[:, axis] * sign < -0.8       # x_min wall's normal points +x
        if facing.sum() < min_support:
            return
        coord = P[facing, axis]
        # Histogram; take the outermost bin cluster with enough support.
        lo, hi = coord.min(), coord.max()
        nb = max(4, int((hi - lo) / bin_w) + 1)
        hist, edges = np.histogram(coord, bins=nb)
        order = range(nb) if sign < 0 else range(nb - 1, -1, -1)
        peak = int(hist.max())
        for b in order:
            if hist[b] >= max(min_support * 0.5, 0.35 * peak):
                sel = (coord >= edges[b] - bin_w) & (coord <= edges[b + 1] + bin_w)
                out[name] = (float(np.median(coord[sel])), int(sel.sum()))
                return

    _side(0, -1, "x_min")
    _side(0, +1, "x_max")
    _side(2, -1, "z_min")
    _side(2, +1, "z_max")
    return out


# --------------------------------------------------------------------------
# Textured planes
# --------------------------------------------------------------------------

def _sample_image(image_rgb: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Nearest-pixel sample; u,v already inside the image."""
    H, W = image_rgb.shape[:2]
    ui = np.clip(np.round(u).astype(int), 0, W - 1)
    vi = np.clip(np.round(v).astype(int), 0, H - 1)
    return image_rgb[vi, ui]


def textured_plane(origin: np.ndarray, axis_u: np.ndarray, axis_v: np.ndarray,
                   size_u: float, size_v: float, normal: np.ndarray,
                   R_total: np.ndarray, K_norm: np.ndarray, image_rgb: np.ndarray,
                   depth_map: np.ndarray, fallback_rgb: Sequence[int],
                   cells: int = 96, depth_tol: float = 0.06):
    """Vertex-coloured grid for one plane in the aligned frame.

    ``R_total`` maps camera-frame world -> aligned frame; we invert it to
    project. ``depth_map`` is the OpenCV z of the point map at full image
    resolution (used for occlusion). Returns (vertices, faces, colors,
    visible_fraction)."""
    Hi, Wi = image_rgb.shape[:2]
    nu = max(2, int(round(cells * size_u / max(size_u, size_v))) + 1)
    nv = max(2, int(round(cells * size_v / max(size_u, size_v))) + 1)
    su = np.linspace(0, size_u, nu)
    sv = np.linspace(0, size_v, nv)
    gu, gv = np.meshgrid(su, sv, indexing="xy")
    verts = origin[None, None, :] + gu[..., None] * axis_u + gv[..., None] * axis_v
    V = verts.reshape(-1, 3)

    cam_world = V @ R_total            # inverse rotation (R_total orthonormal): p_cam = R^T p
    u, v, z = project_world_to_pixels(cam_world, K_norm, Wi, Hi)
    inside = (z > 1e-6) & (u >= 0) & (u < Wi) & (v >= 0) & (v < Hi)
    colors = np.tile(np.asarray(fallback_rgb, np.uint8), (len(V), 1))
    visible = np.zeros(len(V), bool)
    if inside.any():
        ui = np.clip(np.round(u[inside]).astype(int), 0, Wi - 1)
        vi = np.clip(np.round(v[inside]).astype(int), 0, Hi - 1)
        z_obs = depth_map[vi, ui]
        # Surface-consistency test, not just an occlusion test: a vertex is
        # textured only if the point map actually observed a surface there.
        # A prior-placed ceiling floating in front of the far wall must NOT
        # inherit the wall's pixels.
        ok = np.isfinite(z_obs) & (np.abs(z[inside] - z_obs) <= depth_tol * z_obs)
        idx = np.flatnonzero(inside)[ok]
        visible[idx] = True
        colors[idx] = _sample_image(image_rgb, u[idx], v[idx])

    vis_frac = float(visible.mean())
    if 0 < vis_frac < 1:
        # Fill hidden vertices from nearest visible sample, softened toward the median.
        from scipy import ndimage
        vis2d = visible.reshape(nv, nu)
        _, (iy, ix) = ndimage.distance_transform_edt(~vis2d, return_indices=True)
        nearest = colors.reshape(nv, nu, 3)[iy, ix].reshape(-1, 3).astype(float)
        med = np.median(colors[visible], axis=0)
        fill = 0.5 * nearest + 0.5 * med
        colors[~visible] = np.clip(fill[~visible], 0, 255).astype(np.uint8)
    elif vis_frac == 0:
        colors[:] = np.asarray(fallback_rgb, np.uint8)

    # Faces, wound so the normal matches ``normal``.
    idx = np.arange(nu * nv).reshape(nv, nu)
    a = idx[:-1, :-1].ravel(); b = idx[:-1, 1:].ravel()
    c = idx[1:, 1:].ravel();  d = idx[1:, :-1].ravel()
    faces = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    if np.dot(np.cross(axis_u, axis_v), normal) < 0:
        faces = faces[:, ::-1]
    textured_plane.last_grid_shape = (nv, nu)
    return V, faces, colors, vis_frac


def build_textured_room(layout: RoomLayout, K_norm: np.ndarray, image_rgb: np.ndarray,
                        depth_map: np.ndarray, cells: int = 96,
                        include_ceiling: bool = True, include_front_wall: bool = True):
    """Six textured planes -> one trimesh with vertex colours + per-plane stats."""
    import trimesh
    mn = np.asarray(layout.bounds_min, float)
    mx = np.asarray(layout.bounds_max, float)
    R = layout.R_total
    ex, ey, ez = np.eye(3)
    sx, sy, sz = mx - mn
    planes = {
        "floor":   (np.array([mn[0], mn[1], mn[2]]), ex, ez, sx, sz, +ey, (180, 170, 150)),
        "ceiling": (np.array([mn[0], mx[1], mn[2]]), ex, ez, sx, sz, -ey, (245, 245, 245)),
        "x_min":   (np.array([mn[0], mn[1], mn[2]]), ez, ey, sz, sy, +ex, (235, 230, 220)),
        "x_max":   (np.array([mx[0], mn[1], mn[2]]), ez, ey, sz, sy, -ex, (235, 230, 220)),
        "z_max":   (np.array([mn[0], mn[1], mx[2]]), ex, ey, sx, sy, -ez, (235, 230, 220)),
        "z_min":   (np.array([mn[0], mn[1], mn[2]]), ex, ey, sx, sy, +ez, (235, 230, 220)),
    }
    if not include_ceiling:
        planes.pop("ceiling")
    if not include_front_wall:
        planes.pop("z_min")   # the wall behind the camera
    V_all, F_all, C_all = [], [], []
    stats = {}
    off = 0
    for name, (o, au, av, su, sv, n, fb) in planes.items():
        V, F, C, vis = textured_plane(o, au, av, su, sv, n, R, K_norm, image_rgb,
                                      depth_map, fb, cells=cells)
        V_all.append(V); F_all.append(F + off); C_all.append(C)
        off += len(V)
        stats[name] = {"visible_fraction": round(vis, 3), "vertices": int(len(V))}
    mesh = trimesh.Trimesh(vertices=np.vstack(V_all), faces=np.vstack(F_all),
                           vertex_colors=np.vstack(C_all), process=False)
    return mesh, stats


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def estimate_layout(points_world: np.ndarray, valid: np.ndarray,
                    object_aabbs_world: Sequence[Tuple[np.ndarray, np.ndarray]] = (),
                    structural_masks: Optional[Dict[str, np.ndarray]] = None,
                    padding_frac: float = 0.05,
                    camera_height_prior_m: float = 1.5,
                    rng: Optional[np.random.Generator] = None) -> RoomLayout:
    """Fit floor, gravity, yaw, walls and ceiling.

    ``points_world`` (H,W,3) in SAM 3D's camera-frame world; ``valid`` (H,W).
    ``object_aabbs_world`` are (min, max) pairs in the same frame, used to
    make sure the room contains every object. ``structural_masks`` may hold
    "floor", "wall", "ceiling" boolean masks from SAM 3.
    """
    structural_masks = structural_masks or {}
    notes: List[str] = []
    normals = compute_normals(points_world, valid)

    floor = estimate_floor(points_world, valid, structural_masks.get("floor"), normals, rng=rng)
    if floor is None:
        # Upward-looking photo: no floor in view. Try the ceiling for gravity.
        ceil_plane = estimate_ceiling_plane(points_world, valid, structural_masks.get("ceiling"), normals, rng=rng)
        if ceil_plane is not None:
            n_up = -np.asarray(ceil_plane.normal)
            floor = Plane(normal=n_up.tolist(), offset=0.0, inliers=ceil_plane.inliers,
                          inlier_fraction=ceil_plane.inlier_fraction, source="fallback-ceiling")
            G = gravity_rotation(n_up)
            notes.append(f"no floor visible; gravity from ceiling ({ceil_plane.source}, "
                         f"inliers {ceil_plane.inlier_fraction:.0%}); floor height from objects")
        else:
            notes.append("floor fit failed; using +Y and lowest object as floor")
            floor = Plane(normal=UP.tolist(), offset=0.0, source="fallback")
            G = np.eye(3)
    else:
        G = gravity_rotation(floor.normal)
        tilt = math.degrees(math.acos(min(1.0, max(-1.0, float(np.asarray(floor.normal) @ UP)))))
        notes.append(f"floor from {floor.source}: tilt {tilt:.1f}°, inliers {floor.inlier_fraction:.0%}")

    # Gravity-align, then yaw.
    N_g = normals @ G.T
    yaw, support = dominant_yaw_deg(N_g[valid])
    if support < 0.15:
        notes.append(f"weak Manhattan support ({support:.2f}); yaw left at 0")
        yaw = 0.0
    else:
        notes.append(f"Manhattan yaw {yaw:.1f}° (support {support:.2f})")
    # yaw_matrix(a) subtracts ``a`` from every direction's XZ angle, so
    # applying the measured wall angle itself brings the walls onto the axes.
    Y = yaw_matrix(yaw)
    R_total = Y @ G
    P_a = points_world @ R_total.T
    N_a = normals @ R_total.T

    # Floor height in aligned frame: plane passes through camera at distance offset.
    if floor.source.startswith("fallback"):
        if object_aabbs_world:
            floor_y = float(min((np.asarray(mn) @ R_total.T)[1] for mn, _ in object_aabbs_world))
        else:
            floor_y = float(np.percentile(P_a[valid][:, 1], 2))
    else:
        floor_y = -float(floor.offset)          # camera at origin, normal is +Y after G
    camera_height = -floor_y
    metric_scale = camera_height_prior_m / camera_height if camera_height > 1e-6 else 1.0

    # Object bounds in the aligned frame.
    obj_min = np.full(3, np.inf); obj_max = np.full(3, -np.inf)
    for mn, mx in object_aabbs_world:
        corners = np.array([[x, y, z] for x in (mn[0], mx[0]) for y in (mn[1], mx[1]) for z in (mn[2], mx[2])])
        ca = corners @ R_total.T
        obj_min = np.minimum(obj_min, ca.min(0)); obj_max = np.maximum(obj_max, ca.max(0))
    have_objects = np.all(np.isfinite(obj_min))

    # Ceiling.
    ceil_y, ceil_src = estimate_ceiling(P_a, N_a, valid, floor_y, camera_y=0.0)
    if ceil_y is None:
        default_h = 2.6 / metric_scale            # 2.6 m in scene units
        top = (obj_max[1] if have_objects else floor_y) + 0.15 * camera_height
        ceil_y = max(floor_y + default_h, top)
        ceil_src = "prior"
    notes.append(f"ceiling from {ceil_src}: height {(ceil_y - floor_y) * metric_scale:.2f} m (est.)")

    # Walls.
    walls = wall_positions(P_a, N_a, valid, floor_y, ceil_y, structural_masks.get("wall"))
    scene_pts = P_a[valid]
    span = np.percentile(scene_pts, [0.5, 99.5], axis=0)   # trims flying points only
    raw_lo, raw_hi = scene_pts.min(0), scene_pts.max(0)
    width = np.maximum(span[1] - span[0], 1e-6)
    # Keep the true extreme when it is close to the percentile bound (a
    # genuine edge, not a stray point far away).
    span[0] = np.where(span[0] - raw_lo < 0.15 * width, raw_lo, span[0])
    span[1] = np.where(raw_hi - span[1] < 0.15 * width, raw_hi, span[1])
    pad = padding_frac * max(span[1, 0] - span[0, 0], span[1, 2] - span[0, 2])
    bmin = np.array([span[0, 0], floor_y, span[0, 2]])
    bmax = np.array([span[1, 0], ceil_y, span[1, 2]])
    if have_objects:
        bmin[[0, 2]] = np.minimum(bmin[[0, 2]], obj_min[[0, 2]] - pad)
        bmax[[0, 2]] = np.maximum(bmax[[0, 2]], obj_max[[0, 2]] + pad)
    sources = {"x_min": "extent", "x_max": "extent", "z_min": "extent", "z_max": "extent"}
    n_valid = int(valid.sum())
    for side, (coord, sup) in walls.items():
        axis = 0 if side.startswith("x") else 2
        strong = sup >= 0.01 * n_valid          # ≥1 % of the frame votes for this plane
        if side.endswith("min"):
            # Objects with a bad depth poke through walls; a strong wall wins
            # over object bounds (the assembly stage nudges objects back in).
            ok = coord <= span[0, axis] + pad * 3 and (strong or not have_objects or coord <= obj_min[axis] + pad)
            if ok:
                bmin[axis] = coord; sources[side] = f"wall({sup})"
        else:
            ok = coord >= span[1, axis] - pad * 3 and (strong or not have_objects or coord >= obj_max[axis] - pad)
            if ok:
                bmax[axis] = coord; sources[side] = f"wall({sup})"
    # The camera is inside the room: make sure the box contains the origin
    # with a little slack behind it, unless a wall was actually observed there.
    if sources["z_min"] == "extent":
        bmin[2] = min(bmin[2], -0.05 * camera_height)
    layout = RoomLayout(
        gravity_rotation=G.tolist(), yaw_deg=float(yaw), floor=floor, floor_y=floor_y,
        ceiling_y=float(ceil_y), ceiling_source=ceil_src,
        bounds_min=bmin.tolist(), bounds_max=bmax.tolist(), wall_sources=sources,
        camera_height=float(camera_height), estimated_metric_scale=float(metric_scale), notes=notes,
    )
    return layout


def save_layout(layout: RoomLayout, path: Path, extra: Optional[Dict] = None):
    d = layout.to_json()
    if extra:
        d.update(extra)
    Path(path).write_text(json.dumps(d, indent=2))
