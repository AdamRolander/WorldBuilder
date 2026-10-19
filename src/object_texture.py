"""Paint the photo back onto the objects (stage 4d).

SAM 3D gives every object a complete but soft texture: it is generated,
at a fixed budget, for all sides at once. The photo holds the real pixels
of the side that faces the camera, usually at several times that
resolution. After placement each mesh sits where the photo shows the
object, so those pixels can be put back.

A vertex takes its colour from the photo only where the photo demonstrably
shows *that point of that object*:

* it projects inside the object's own mask (and that pixel is not owned by
  a smaller object in front),
* it is on the mesh's front surface as seen from the camera, and not seen
  at a grazing angle,
* the point map's depth at that pixel agrees with the vertex's depth, so
  the mesh surface really is where the photo's surface is. A mesh that is
  off would otherwise get another part's pixels (the wall seen between a
  chair's legs painted onto its seat).

SAM 3D's own colour is deliberately *not* consulted: where it is wrong (a
black cabinet generated grey) is where the photo matters most.

The tests give a weight in [0, 1] per vertex, smoothed over the mesh so
there is no speckle, and the result is a blend. Everything the photo did
not see keeps SAM 3D's colour untouched. (Matching the generated colours'
exposure to the photo's was tried and removed: on an object with two
materials, a white counter on black cabinets, a per-object gain turns the
white grey.)

Colours stay per-vertex: the meshes have 10^5-10^6 vertices, about one
per photo pixel on the visible side, and ``src/mesh_bake.py`` turns them
into textures for the lite GLB.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from src import room_layout as rl


@dataclass
class PhotoContext:
    """What every object of a scene shares."""
    image: np.ndarray              # (H, W, 3) uint8, full resolution
    K: np.ndarray                  # normalised intrinsics
    R_total: np.ndarray            # camera-frame world -> aligned
    depth: np.ndarray              # (h, w) camera z at point-map resolution, nan where invalid
    owner: np.ndarray              # (h, w) object id per pixel, 0 = none


def make_context(image_rgb: np.ndarray, K: np.ndarray, R_total: np.ndarray, depth: np.ndarray,
                 owner: np.ndarray) -> PhotoContext:
    return PhotoContext(image=image_rgb, K=np.asarray(K, float), R_total=np.asarray(R_total, float),
                        depth=depth, owner=owner)


def _sample(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear sample of an (H, W, 3) image at pixel coordinates (pixel centres at +0.5)."""
    from scipy import ndimage
    out = np.empty((len(u), 3), np.float32)
    coords = np.stack([v - 0.5, u - 0.5])
    for c in range(3):
        out[:, c] = ndimage.map_coordinates(img[..., c], coords, order=1, mode="nearest")
    return out


def _front_surface(u, v, z, hw, tol: float) -> np.ndarray:
    """Vertices on the camera-facing surface: a z-buffer from the vertices
    themselves (the meshes are dense enough that every pixel gets several)."""
    from scipy import ndimage
    h, w = hw
    ui, vi = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h) & (z > 1e-6)
    zbuf = np.full(h * w, np.inf)
    np.minimum.at(zbuf, vi[inside] * w + ui[inside], z[inside])
    zbuf = ndimage.grey_erosion(zbuf.reshape(h, w), size=(3, 3))      # nearest surface in a 3x3 window
    front = np.zeros(len(z), bool)
    front[inside] = z[inside] <= zbuf[vi[inside], ui[inside]] * (1.0 + tol)
    return front


def _smooth_over_mesh(w: np.ndarray, faces: np.ndarray, n: int, rounds: int = 3) -> np.ndarray:
    from scipy import sparse
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    A = sparse.coo_matrix((np.ones(len(e), np.float32), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A = A + A.T
    deg = np.asarray(A.sum(1)).ravel() + 1.0
    for _ in range(rounds):
        w = (A @ w + w) / deg
    return w


def photo_colors(vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray, obj_id: int,
                 ctx: PhotoContext, depth_tol: float = 0.10, max_view_angle_deg: float = 75.0,
                 min_support: int = 200, flat_only: bool = True, min_flatness: float = 0.8) -> Tuple[np.ndarray, Dict]:
    """New vertex colours for one placed object. ``vertices`` are in the
    aligned world frame, ``colors`` (n, 3|4) uint8 are SAM 3D's.

    With ``flat_only`` (the default) only objects whose photographed side
    is one flat surface facing the camera are painted: that is where the
    photo's pixels land on the right geometry (signs, screens, art, cabinet
    and drawer fronts). On free-standing furniture the mesh and the photo
    disagree by a few centimetres, and the right pixels on slightly wrong
    geometry look worse than SAM 3D's soft texture — doubled shelves, a
    streak of background across a seat. Measured over 22 scenes, that was
    the usual outcome.

    Returns ``(colors_uint8 (n, 4), stats)``; the input colours come back
    unchanged when too little of the object passes the tests.
    """
    import trimesh
    n = len(vertices)
    faces = np.asarray(faces)
    base = np.asarray(colors)[:, :3].astype(np.float32)
    out = np.concatenate([base, np.full((n, 1), 255, np.float32)], 1).astype(np.uint8)
    stats = {"painted": 0.0, "seen": 0.0}
    H, W = ctx.image.shape[:2]
    h, w = ctx.depth.shape
    cam = np.asarray(vertices, float) @ ctx.R_total
    u, v, z = rl.project_world_to_pixels(cam, ctx.K, W, H)
    ok = (z > 1e-6) & np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if ok.sum() < min_support:
        return out, stats
    u, v = np.where(ok, u, 0.0), np.where(ok, v, 0.0)
    us, vs = u * (w / W), v * (h / H)                             # point-map pixels
    pi = np.clip(np.floor(vs).astype(np.int64), 0, h - 1)
    pj = np.clip(np.floor(us).astype(np.int64), 0, w - 1)

    own = ok & (ctx.owner[pi, pj] == obj_id)
    front = _front_surface(us, vs, np.where(ok, z, -1.0), (h, w), tol=0.015)
    d_obs = ctx.depth[pi, pj]
    with np.errstate(invalid="ignore", divide="ignore"):
        d_err = np.abs(z - d_obs) / d_obs
    # full weight up to depth_tol, fading out by 1.5x: a mesh a little nearer
    # or farther along the ray still projects onto the same pixels
    depth_w = np.clip((1.5 * depth_tol - np.nan_to_num(d_err, nan=1.0, posinf=1.0)) / (0.5 * depth_tol), 0.0, 1.0)
    seen = own & front & (depth_w > 0)
    stats["seen"] = round(float(seen.mean()), 3)
    if seen.sum() < min_support:
        return out, stats

    # how squarely each vertex faces the camera (which sits at the origin of the camera frame)
    normals = np.asarray(trimesh.Trimesh(vertices=cam, faces=faces, process=False).vertex_normals)
    view = cam / np.maximum(np.linalg.norm(cam, axis=1, keepdims=True), 1e-9)
    cosv = np.abs(np.sum(normals * view, axis=1))
    cos_min = np.cos(np.radians(max_view_angle_deg))
    face_w = np.clip((cosv - cos_min) / (1.0 - cos_min), 0.0, 1.0) ** 0.5
    # Is the photographed side one flat, camera-facing surface? Judged by
    # the mesh's own normals over the vertices the photo saw: their mean
    # direction is long when they agree (a screen, a poster, a cabinet
    # front; 1.0 = all parallel) and short on a sofa, a plant or a chair.
    # (SAM 3D's faces are not consistently wound, so normals are compared up
    # to sign: the top eigenvalue of their scatter matrix, 1.0 = all parallel.)
    nz = normals[seen] / np.maximum(np.linalg.norm(normals[seen], axis=1, keepdims=True), 1e-9)
    stats["flatness"] = round(float(np.linalg.eigvalsh(nz.T @ nz / max(len(nz), 1))[-1]), 3)
    stats["facing"] = round(float(np.median(cosv[seen])), 3)
    # A facade is as good as a flat panel: a bookcase full of books, a
    # dresser, a run of cabinets seen from the front. Its normals disagree
    # (shelves, mouldings, handles) but everything the photo saw lies in a
    # thin layer compared with how wide it is, so the photo's pixels land on
    # the right surfaces. That is also where the photo matters most: the
    # books on a shelf are never generated.
    zs = cam[seen][:, 2] if cam.shape[1] == 3 else z[seen]
    lat = np.ptp(cam[seen][:, :2], axis=0).max() if seen.sum() > 1 else 0.0
    depth_layer = float(np.percentile(np.abs(zs), 90) - np.percentile(np.abs(zs), 10))
    stats["facade"] = round(depth_layer / max(float(lat), 1e-9), 3)
    flat = stats["flatness"] >= min_flatness or stats["facade"] <= 0.2
    if flat_only and not (flat and stats["facing"] >= np.cos(np.radians(45))):
        return out, stats

    idx = np.flatnonzero(seen)
    photo = _sample(ctx.image, u[idx], v[idx])

    wgt = np.zeros(n, np.float32)
    wgt[idx] = depth_w[idx] * face_w[idx]
    wgt = np.minimum(wgt, _smooth_over_mesh(wgt, faces, n))        # shrink, never grow, the painted area
    wgt = _smooth_over_mesh(wgt, faces, n, rounds=2)

    rgb = base.copy()
    rgb[idx] = wgt[idx, None] * photo + (1.0 - wgt[idx, None]) * base[idx]
    out[:, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    stats["painted"] = round(float((wgt > 0.5).mean()), 3)
    return out, stats


def repaint(mesh, obj_id: int, ctx: Optional[PhotoContext]) -> Optional[Dict]:
    """In-place for a trimesh with vertex colours; returns the stats or None."""
    if ctx is None or getattr(mesh.visual, "kind", None) != "vertex":
        return None
    import os
    colors, stats = photo_colors(np.asarray(mesh.vertices), np.asarray(mesh.faces),
                                 np.asarray(mesh.visual.vertex_colors), obj_id, ctx,
                                 flat_only=os.environ.get("WORLDBUILDER_PHOTO_TEXTURE", "1") != "all")
    mesh.visual.vertex_colors = colors
    return stats
