"""Solid built-ins: turn the visible faces of counters and cabinets into boxes.

The photo relief (``room_texture.build_relief``) is a skin: exact from the
photo's viewpoint, but with holes wherever something stood in front and
nothing behind it, so from the side a kitchen's cabinet run is a sheet you
can see through. Built-in furniture has two properties that make it easy
to complete without knowing what it is:

* its faces are parallel to the room's axes (the layout stage has already
  rotated the scene so walls are axis-aligned), and
* it is attached: a counter stands on the floor, a wall cabinet hangs on a
  wall, a soffit hangs from the ceiling. Whatever is *behind* a visible
  face, as seen from the room, is solid until the room's shell.

So: collect background points whose normal is along ±X, ±Y or ±Z, find the
planes they form, cover each plane's connected patches with rectangles, and
extrude every rectangle away from its normal to the shell — a counter top
down to the floor, a cabinet front back to the wall. A box whose interior
the photo saw *through* (points observed inside it) contradicts itself and
is thinned to a slab. The visible face is textured from the photo exactly
like a room plane, hidden parts predicted from the seen part; the other
faces take the face's median colour.

The boxes sit a hair behind the relief, so the photo's own pixels stay on
top where they exist and the boxes show wherever the relief has holes.
They are also what a physics engine collides with.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src import room_layout as rl
from src import room_texture as rt

AXES = np.eye(3)


def find_faces(points: np.ndarray, keep: np.ndarray, layout: rl.RoomLayout,
               min_frame_frac: float = 0.003, max_faces: int = 40) -> List[Dict]:
    """Axis-aligned rectangular faces in the background geometry.

    ``points`` (h,w,3) aligned frame, ``keep`` (h,w) the pixels that are
    neither object nor shell. Returns dicts with ``axis`` (0/1/2), ``sign``
    (+1: the face looks along +axis), ``coord`` (plane position) and
    ``lo``/``hi`` (2-vectors on the other two axes, in axis order).
    """
    from scipy import ndimage
    h, w = keep.shape
    mn, mx = np.asarray(layout.bounds_min, float), np.asarray(layout.bounds_max, float)
    room = float(np.max(mx - mn))
    N = rl.compute_normals(points.astype(np.float64), keep)
    bin_w = 0.012 * room
    cell = 0.015 * room
    min_px = max(60, int(min_frame_frac * h * w))
    faces: List[Dict] = []
    for axis in range(3):
        others = [a for a in range(3) if a != axis]
        for sign in (+1, -1):
            sel = keep & (N[..., axis] * sign > 0.85)
            if int(sel.sum()) < min_px:
                continue
            c = points[..., axis]
            vals = c[sel]
            lo, hi = float(vals.min()), float(vals.max())
            nb = max(2, int((hi - lo) / bin_w) + 1)
            hist, edges = np.histogram(vals, bins=nb, range=(lo, hi + 1e-9))
            sm = np.convolve(hist, [1, 1, 1], mode="same")
            used = np.zeros(nb, bool)
            for b in np.argsort(-sm):
                if sm[b] < min_px or used[max(0, b - 1):b + 2].any():
                    continue
                used[max(0, b - 1):b + 2] = True
                near = sel & (np.abs(c - 0.5 * (edges[b] + edges[b + 1])) < 1.5 * bin_w)
                coord = float(np.median(c[near]))
                lab, n = ndimage.label(ndimage.binary_closing(near, iterations=2))
                for i in range(1, n + 1):
                    comp = (lab == i) & near
                    if int(comp.sum()) < min_px:
                        continue
                    ab = points[comp][:, others]
                    a0 = ab.min(axis=0)
                    gi = np.floor((ab - a0) / cell).astype(int)
                    G = np.zeros(gi.max(axis=0) + 1, bool)
                    G[gi[:, 0], gi[:, 1]] = True
                    G = ndimage.binary_fill_holes(ndimage.binary_closing(G, iterations=1, border_value=0)) | G
                    first = None
                    for _ in range(3):                       # cover an L- or T-shaped patch with rectangles
                        r0, c0, r1, c1 = rt.largest_rectangle(G)
                        area = (r1 - r0) * (c1 - c0)
                        if area < 4 or (first is not None and area < 0.3 * first):
                            break
                        first = first or area
                        faces.append({"axis": axis, "sign": sign, "coord": coord,
                                      "lo": (a0 + np.array([r0, c0]) * cell).tolist(),
                                      "hi": (a0 + np.array([r1, c1]) * cell).tolist(),
                                      "pixels": int(comp.sum())})
                        G[r0:r1, c0:c1] = False
    faces.sort(key=lambda f: -f["pixels"])
    return faces[:max_faces]


def face_to_box(face: Dict, layout: rl.RoomLayout, points: np.ndarray, valid: np.ndarray,
                inset_frac: float = 0.008, slab_frac: float = 0.04) -> Dict:
    """Extrude a face away from its normal to the room's shell.

    Returns ``{"min", "max", "axis", "sign", "thin"}``. The box is thinned
    to a slab when the photo saw into where it would be, or when "to the
    shell" would be most of the room (a free-standing partition is not a
    solid block to the far wall).
    """
    mn, mx = np.asarray(layout.bounds_min, float), np.asarray(layout.bounds_max, float)
    dims = mx - mn
    room = float(np.max(dims))
    axis, sign = face["axis"], face["sign"]
    others = [a for a in range(3) if a != axis]
    front = face["coord"] - sign * inset_frac * room            # a hair behind the observed surface
    back = mn[axis] if sign > 0 else mx[axis]
    lo, hi = np.zeros(3), np.zeros(3)
    lo[others], hi[others] = face["lo"], face["hi"]
    lo[axis], hi[axis] = min(front, back), max(front, back)

    def _slab():
        b = front - sign * slab_frac * room
        l2, h2 = lo.copy(), hi.copy()
        l2[axis], h2[axis] = min(front, b), max(front, b)
        return l2, h2

    thin = False
    depth = abs(front - back)
    if axis != 1 and depth > 0.45 * dims[axis]:
        thin = True
    else:
        # free-space check: observed points strictly inside the box
        m = 0.02 * room
        inside = valid & np.all((points > lo + m) & (points < hi - m), axis=-1)
        if int(inside.sum()) > 0.15 * face["pixels"]:
            thin = True
    if thin:
        lo, hi = _slab()
    return {"min": lo.tolist(), "max": hi.tolist(), "axis": axis, "sign": sign, "thin": thin,
            "pixels": face["pixels"]}


def _contained(a: Dict, b: Dict, tol: float) -> bool:
    return bool(np.all(np.asarray(a["min"]) >= np.asarray(b["min"]) - tol)
                and np.all(np.asarray(a["max"]) <= np.asarray(b["max"]) + tol))


def build_builtins(points_aligned: np.ndarray, keep: np.ndarray, valid: np.ndarray, layout: rl.RoomLayout,
                   K: np.ndarray, image_rgb: np.ndarray, depth: np.ndarray,
                   blocked: Optional[np.ndarray] = None, texel_density: Optional[float] = None
                   ) -> Tuple[Dict, List[Dict]]:
    """Meshes (``room_builtin_<i>_face`` textured, ``room_builtin_<i>_body``
    flat) and the list of boxes (for ``layout.json`` and physics export)."""
    import trimesh
    from PIL import Image
    mn, mx = np.asarray(layout.bounds_min, float), np.asarray(layout.bounds_max, float)
    room = float(np.max(mx - mn))
    faces = find_faces(points_aligned, keep, layout)
    boxes = [face_to_box(f, layout, points_aligned, valid) for f in faces]
    kept: List[Tuple[Dict, Dict]] = []
    for f, b in zip(faces, boxes, strict=True):                 # larger faces first
        if any(_contained(b, kb, 0.01 * room) for _, kb in kept):
            continue
        kept.append((f, b))
    if texel_density is None:
        texel_density = 1024.0 / room
    meshes: Dict = {}
    out_boxes: List[Dict] = []
    R = layout.R_total
    for i, (f, b) in enumerate(kept):
        axis, sign = f["axis"], f["sign"]
        o1, o2 = [a for a in range(3) if a != axis]
        lo, hi = np.asarray(b["min"]), np.asarray(b["max"])
        front = hi[axis] if sign > 0 else lo[axis]
        origin = np.zeros(3)
        origin[axis], origin[o1], origin[o2] = front, lo[o1], lo[o2]
        plane = dict(o=origin, au=AXES[o1], av=AXES[o2], su=float(hi[o1] - lo[o1]), sv=float(hi[o2] - lo[o2]),
                     n=sign * AXES[axis], evidence=False)
        long_side = int(np.clip(max(plane["su"], plane["sv"]) * texel_density, 32, 1024))
        # Sample the photo on the surface that was observed, not on the
        # inset quad: seen at a grazing angle, a centimetre of inset is
        # several centimetres along the ray and would fail the depth test.
        seen_plane = dict(plane, o=origin + (f["coord"] - front) * AXES[axis])
        tex, vis, qual, clean = rt.project_plane_texture(seen_plane, R, K, image_rgb, depth, blocked, long_side,
                                                         keep_flush=True, return_quality=True)
        seen = float(vis.mean())
        if seen < 0.05:
            continue                                            # nothing of this face is actually in the photo
        tex, how = rt.fill_hidden(tex, vis, (128, 128, 128), rows=(axis != 1), quality=qual, source=clean)
        body_rgb = np.median(tex.reshape(-1, 3), axis=0).astype(np.uint8)
        verts = np.array([origin, origin + plane["su"] * plane["au"],
                          origin + plane["su"] * plane["au"] + plane["sv"] * plane["av"],
                          origin + plane["sv"] * plane["av"]])
        tri = np.array([[0, 1, 2], [0, 2, 3]])
        if np.dot(np.cross(plane["au"], plane["av"]), plane["n"]) < 0:
            tri = tri[:, ::-1]
        mat = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(tex), metallicFactor=0.0,
                                                  roughnessFactor=1.0, doubleSided=True, name=f"builtin_{i}_face")
        meshes[f"room_builtin_{i:02d}_face"] = trimesh.Trimesh(
            vertices=verts, faces=tri, process=False,
            visual=trimesh.visual.TextureVisuals(uv=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float), material=mat))
        # the rest of the box, shrunk a little sideways so its sides do not
        # fight with neighbouring faces that lie in the same plane
        eps = 0.004 * room
        blo, bhi = lo.copy(), hi.copy()
        blo[[o1, o2]] += eps
        bhi[[o1, o2]] -= eps
        if sign > 0:
            bhi[axis] -= eps
        else:
            blo[axis] += eps
        if np.all(bhi - blo > 1e-6):
            body = trimesh.creation.box(extents=bhi - blo)
            body.apply_translation(0.5 * (blo + bhi))
            flat = Image.fromarray(np.tile(body_rgb, (2, 2, 1)))
            bmat = trimesh.visual.material.PBRMaterial(baseColorTexture=flat, metallicFactor=0.0,
                                                       roughnessFactor=1.0, doubleSided=True,
                                                       name=f"builtin_{i}_body")
            body.visual = trimesh.visual.TextureVisuals(uv=np.full((len(body.vertices), 2), 0.5), material=bmat)
            meshes[f"room_builtin_{i:02d}_body"] = body
        out_boxes.append({"min": lo.tolist(), "max": hi.tolist(), "axis": axis, "sign": sign,
                          "thin": bool(b["thin"]), "seen_fraction": round(seen, 3), "fill": how,
                          "color": [int(c) for c in body_rgb]})
    return meshes, out_boxes
