"""Textured room shell and background relief (stage 4c).

The first version of the room coloured a 96×96 vertex grid per plane from
the photo. From any viewpoint other than the photo's that reads as a
pixelated, smeared projection: one colour per ~3 cm cell, and hidden areas
filled by dragging the nearest visible colour sideways.

This module builds the same room as ordinary textured geometry:

**Shell.** Each of the floor, walls and ceiling is a quad with UVs and its
own texture image. The texture is rendered by projecting every texel
centre into the photo and sampling bilinearly, so it carries the photo's
full resolution (a rectified view of that surface). Texels the photo did
not see — behind objects, outside the frame — are *synthesised from the
visible part of the same surface*: the largest visible rectangle is tiled
with mirror symmetry (continuous at every seam) and feathered into the
observed texels, so a wooden floor continues as wooden floor and wallpaper
as wallpaper. When the visible part is not texture-like (a wall that is
mostly whiteboard), the fill is the surface's median colour instead.
Pixels belonging to objects are never baked in (no second rug on the
floor); what is seen *through* an evidence-backed wall (a window) is.

**Relief.** Whatever is neither an object nor on a shell plane is built-in
structure the detector is told not to list: counters, cabinet runs, window
recesses, beams. It is meshed directly from the point map (one vertex per
sampled pixel, UV = that pixel, texture = the photo), with triangles that
span depth edges removed and the holes left by small objects closed by
interpolation. This is 2.5D — it has no back side — but it is in the right
place at full photo sharpness, and it gives the things standing on a
countertop something to stand on.

Both are exported as ``room.glb`` (named nodes ``room_floor``,
``room_wall_x_max``, …, ``room_relief``) and the shell additionally as a
vertex-coloured ``room.ply`` for consumers that cannot read textures.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src import room_layout as rl

FALLBACK_RGB = {"floor": (180, 170, 150), "ceiling": (245, 245, 245), "wall": (235, 230, 220)}


def room_planes(layout: rl.RoomLayout, include_ceiling: bool = True,
                include_front_wall: bool = True) -> Dict[str, Dict]:
    """The shell's planes in the aligned frame: origin, in-plane axes, sizes,
    inward normal, and whether the plane's position is backed by evidence."""
    mn = np.asarray(layout.bounds_min, float)
    mx = np.asarray(layout.bounds_max, float)
    ex, ey, ez = np.eye(3)
    sx, sy, sz = mx - mn
    ws = layout.wall_sources
    floor_ok = not str(layout.floor.source).startswith("fallback")
    ceil_ok = layout.ceiling_source == "geometric"
    P = {
        "floor":   dict(o=np.array([mn[0], mn[1], mn[2]]), au=ex, av=ez, su=sx, sv=sz, n=+ey, evidence=floor_ok),
        "ceiling": dict(o=np.array([mn[0], mx[1], mn[2]]), au=ex, av=ez, su=sx, sv=sz, n=-ey, evidence=ceil_ok),
        "wall_x_min": dict(o=np.array([mn[0], mn[1], mn[2]]), au=ez, av=ey, su=sz, sv=sy, n=+ex),
        "wall_x_max": dict(o=np.array([mx[0], mn[1], mn[2]]), au=ez, av=ey, su=sz, sv=sy, n=-ex),
        "wall_z_max": dict(o=np.array([mn[0], mn[1], mx[2]]), au=ex, av=ey, su=sx, sv=sy, n=-ez),
        "wall_z_min": dict(o=np.array([mn[0], mn[1], mn[2]]), au=ex, av=ey, su=sx, sv=sy, n=+ez),
    }
    for k in list(P):
        if k.startswith("wall_"):
            P[k]["evidence"] = str(ws.get(k[5:], "")).startswith("wall")
    if not include_ceiling:
        P.pop("ceiling")
    if not include_front_wall:
        P.pop("wall_z_min")
    return P


# ---------------------------------------------------------------------------
# Plane textures
# ---------------------------------------------------------------------------

def project_plane_texture(plane: Dict, R_total: np.ndarray, K: np.ndarray, image_rgb: np.ndarray,
                          depth: np.ndarray, blocked: Optional[np.ndarray] = None,
                          long_side: int = 1024, depth_tol: float = 0.06) -> Tuple[np.ndarray, np.ndarray]:
    """Rectified view of one plane. Returns ``(texture (th,tw,3) uint8,
    visible (th,tw) bool)``; row 0 is the plane's far edge along ``av`` so
    that trimesh/OpenGL UVs (v up) map ``v = t / sv``.

    ``depth`` is the camera-z map (any resolution, nan = unknown) and
    ``blocked`` a same-size mask of pixels that belong to objects.
    """
    import cv2
    Hi, Wi = image_rgb.shape[:2]
    su, sv = float(plane["su"]), float(plane["sv"])
    tw = max(8, int(round(long_side * su / max(su, sv))))
    th = max(8, int(round(long_side * sv / max(su, sv))))
    s = (np.arange(tw) + 0.5) / tw * su
    t = (1.0 - (np.arange(th) + 0.5) / th) * sv
    pts = plane["o"][None, None, :] + s[None, :, None] * plane["au"] + t[:, None, None] * plane["av"]
    u, v, z = rl.project_world_to_pixels(pts.reshape(-1, 3) @ R_total, K, Wi, Hi)
    u, v, z = u.reshape(th, tw), v.reshape(th, tw), z.reshape(th, tw)
    inside = (z > 1e-6) & (u >= 0) & (u < Wi) & (v >= 0) & (v < Hi)
    Hd, Wd = depth.shape
    ui = np.clip((np.nan_to_num(u) * Wd / Wi).astype(int), 0, Wd - 1)
    vi = np.clip((np.nan_to_num(v) * Hd / Hi).astype(int), 0, Hd - 1)
    z_obs = depth[vi, ui]
    with np.errstate(invalid="ignore"):
        on_surface = np.abs(z - z_obs) <= depth_tol * z_obs
        # Seen through the plane (a window, a doorway): only meaningful when
        # the plane was actually observed, not placed by a prior.
        beyond = (z_obs > z * (1 + depth_tol)) if plane.get("evidence") else np.zeros_like(on_surface)
    visible = inside & np.isfinite(z_obs) & (on_surface | beyond)
    if blocked is not None:
        visible &= ~blocked[vi, ui]
    tex = cv2.remap(image_rgb, (u - 0.5).astype(np.float32), (v - 0.5).astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return tex, visible


def largest_rectangle(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Largest axis-aligned all-True rectangle: (row0, col0, row1, col1),
    half-open. Histogram-stack algorithm, O(rows × cols)."""
    h, w = mask.shape
    heights = np.zeros(w, int)
    best = (0, 0, 0, 0)
    best_area = 0
    for r in range(h):
        heights = np.where(mask[r], heights + 1, 0)
        stack: List[int] = []
        for c in range(w + 1):
            cur = heights[c] if c < w else 0
            while stack and heights[stack[-1]] >= cur:
                top = stack.pop()
                left = stack[-1] + 1 if stack else 0
                area = heights[top] * (c - left)
                if area > best_area:
                    best_area = area
                    best = (r - heights[top] + 1, left, r + 1, c)
            stack.append(c)
    return best


def fill_hidden(tex: np.ndarray, visible: np.ndarray, fallback_rgb=(200, 200, 200),
                min_patch_frac: float = 0.02, max_lowfreq_std: float = 22.0) -> Tuple[np.ndarray, str]:
    """Complete a plane texture. Returns ``(texture, how)`` with ``how`` in
    {"complete", "tiled", "median", "fallback"}."""
    import cv2
    from scipy import ndimage
    th, tw = visible.shape
    frac = float(visible.mean())
    if frac > 0.999:
        return tex, "complete"
    if frac < 0.005:
        return np.tile(np.asarray(fallback_rgb, np.uint8), (th, tw, 1)), "fallback"
    median = np.median(tex[visible], axis=0)
    fill = np.tile(median, (th, tw, 1)).astype(np.float32)
    how = "median"

    # Largest visible rectangle, found on a coarse grid for speed.
    k = max(1, int(np.ceil(max(th, tw) / 160)))
    hs, ws = th // k, tw // k
    if hs >= 4 and ws >= 4:
        coarse = visible[:hs * k, :ws * k].reshape(hs, k, ws, k).all(axis=(1, 3))
        r0, c0, r1, c1 = largest_rectangle(coarse)
        r0, c0, r1, c1 = r0 * k, c0 * k, r1 * k, c1 * k
        if (r1 - r0) * (c1 - c0) >= min_patch_frac * th * tw and min(r1 - r0, c1 - c0) >= 8:
            patch = tex[r0:r1, c0:c1]
            # Texture-like means: no large-scale structure. A patch holding a
            # window or half a whiteboard would stamp it across the wall.
            small = cv2.resize(patch, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32)
            if float(small.std(axis=(0, 1)).max()) <= max_lowfreq_std:
                fill = np.pad(patch, ((r0, th - r1), (c0, tw - c1), (0, 0)), mode="symmetric").astype(np.float32)
                how = "tiled"

    # Feather: observed texels fade into the fill over a few percent of the
    # plane, so there is no hard line where the photo's knowledge ends.
    ramp = max(2.0, 0.02 * max(th, tw))
    alpha = np.clip(ndimage.distance_transform_edt(visible) / ramp, 0, 1)[..., None]
    out = alpha * tex.astype(np.float32) + (1 - alpha) * fill
    return np.clip(out, 0, 255).astype(np.uint8), how


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------

def build_shell(layout: rl.RoomLayout, K: np.ndarray, image_rgb: np.ndarray, depth: np.ndarray,
                blocked: Optional[np.ndarray] = None, long_side: Optional[int] = None,
                include_ceiling: bool = True, include_front_wall: bool = True):
    """Textured quads for every shell plane.

    Returns ``(meshes, stats)``: ``meshes`` maps node name → trimesh with
    ``TextureVisuals``; ``stats`` maps plane → visible fraction, fill mode
    and texture size.
    """
    import trimesh
    from PIL import Image
    if long_side is None:
        long_side = int(np.clip(1.5 * max(image_rgb.shape[:2]), 512, 2048))
    R = layout.R_total
    meshes, stats = {}, {}
    planes = room_planes(layout, include_ceiling, include_front_wall)
    projected = {name: project_plane_texture(pl, R, K, image_rgb, depth, blocked, long_side)
                 for name, pl in planes.items()}
    # A wall the photo never saw takes the colour of the walls it did see,
    # which beats any hard-coded default.
    seen_wall = [tex[vis] for name, (tex, vis) in projected.items() if name.startswith("wall") and vis.mean() > 0.02]
    wall_rgb = tuple(np.median(np.concatenate(seen_wall), axis=0).astype(int)) if seen_wall else FALLBACK_RGB["wall"]
    for name, pl in planes.items():
        kind = "wall" if name.startswith("wall") else name
        tex, vis = projected[name]
        tex, how = fill_hidden(tex, vis, wall_rgb if kind == "wall" else FALLBACK_RGB[kind])
        o, au, av, su, sv = pl["o"], pl["au"], pl["av"], pl["su"], pl["sv"]
        verts = np.array([o, o + su * au, o + su * au + sv * av, o + sv * av])
        uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        if np.dot(np.cross(au, av), pl["n"]) < 0:
            faces = faces[:, ::-1]
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(tex), metallicFactor=0.0, roughnessFactor=1.0,
            doubleSided=False, name=f"room_{name}")
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False,
                               visual=trimesh.visual.TextureVisuals(uv=uv, material=material))
        meshes[f"room_{name}"] = mesh
        stats[name] = {"visible_fraction": round(float(vis.mean()), 3), "fill": how,
                       "texture_px": [int(tex.shape[1]), int(tex.shape[0])], "evidence": bool(pl.get("evidence"))}
    return meshes, stats


def shell_vertex_color_mesh(meshes: Dict, cells: int = 128):
    """The shell as one vertex-coloured grid mesh (for PLY consumers),
    sampled from the textures so both representations agree."""
    import trimesh
    V_all, F_all, C_all, off = [], [], [], 0
    for name, m in meshes.items():
        if name == "room_relief":
            continue
        v = np.asarray(m.vertices)
        o, e_u, e_v = v[0], v[1] - v[0], v[3] - v[0]
        su, sv = np.linalg.norm(e_u), np.linalg.norm(e_v)
        nu = max(2, int(round(cells * su / max(su, sv))) + 1)
        nv = max(2, int(round(cells * sv / max(su, sv))) + 1)
        a, b = np.meshgrid(np.linspace(0, 1, nu), np.linspace(0, 1, nv), indexing="xy")
        V = o + a[..., None] * e_u + b[..., None] * e_v
        tex = np.asarray(m.visual.material.baseColorTexture.convert("RGB"))
        th, tw = tex.shape[:2]
        C = tex[np.clip(((1 - b) * th).astype(int), 0, th - 1), np.clip((a * tw).astype(int), 0, tw - 1)]
        idx = np.arange(nu * nv).reshape(nv, nu)
        p, q, r, s = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
        F = np.concatenate([np.stack([p, q, r], 1), np.stack([p, r, s], 1)])
        f0 = np.asarray(m.faces)[0]
        if np.dot(np.cross(e_u, e_v), np.cross(v[f0[1]] - v[f0[0]], v[f0[2]] - v[f0[0]])) < 0:
            F = F[:, ::-1]
        V_all.append(V.reshape(-1, 3)); F_all.append(F + off); C_all.append(C.reshape(-1, 3))
        off += nu * nv
    return trimesh.Trimesh(vertices=np.vstack(V_all), faces=np.vstack(F_all),
                           vertex_colors=np.vstack(C_all), process=False)


# ---------------------------------------------------------------------------
# Relief
# ---------------------------------------------------------------------------

def build_relief(points_aligned: np.ndarray, valid: np.ndarray, owner: np.ndarray,
                 layout: rl.RoomLayout, image_rgb: np.ndarray, max_vertices: int = 200_000,
                 plane_tol_frac: float = 0.03, small_object_frac: float = 0.015,
                 max_view_angle_deg: float = 80.0, min_component_frac: float = 0.003,
                 max_texture_side: int = 2048):
    """Mesh of the built-in structure that is neither object nor shell.

    ``points_aligned`` (h,w,3) is the point map in the aligned frame (camera
    at the origin), ``owner`` (h,w) the object id per pixel (0 = none).
    Returns ``(trimesh | None, stats)``.
    """
    import cv2
    import trimesh
    from PIL import Image
    from scipy import ndimage
    h, w = valid.shape
    P = points_aligned.astype(np.float32).copy()
    bg = valid & (owner == 0)
    img = np.asarray(Image.fromarray(image_rgb).resize((w, h), Image.BILINEAR))

    # Close the holes small objects leave in the surface they stand on, in
    # both geometry and colour, by interpolating from the hole's surroundings.
    ids, counts = np.unique(owner[owner > 0], return_counts=True)
    small = np.isin(owner, ids[counts < small_object_frac * h * w])
    hole = ndimage.binary_dilation(small, iterations=2) & ~(valid & (owner > 0) & ~small)
    filled = 0
    if hole.any():
        known = bg & ~hole
        # only fill where there is known background close by on at least two sides
        near = ndimage.binary_dilation(known, iterations=max(4, int(0.04 * max(h, w))))
        hole &= near
        hm = hole.astype(np.uint8)
        src = np.where(known[..., None], P, 0).astype(np.float32)
        # cv2.inpaint needs the unknown area to be exactly the mask: treat
        # every not-known pixel as unknown, then keep only the hole's result.
        unk = (~known).astype(np.uint8)
        for c in range(3):
            P[..., c] = np.where(hole, cv2.inpaint(src[..., c], unk, 3, cv2.INPAINT_NS), P[..., c])
        img = cv2.inpaint(img, hm, 3, cv2.INPAINT_TELEA)
        bg = bg | hole
        filled = int(hole.sum())
    # Mask edges are a pixel or two off; keep object colours out of the relief.
    big = (owner > 0) & ~small
    bg &= ~ndimage.binary_dilation(big, iterations=2)

    # Drop what the shell already represents, and what lies outside the room.
    mn, mx = np.asarray(layout.bounds_min, float), np.asarray(layout.bounds_max, float)
    dims = mx - mn
    tol = plane_tol_frac * float(max(dims))
    on_shell = np.zeros((h, w), bool)
    planes = room_planes(layout)
    for pl in planes.values():
        d = (P - pl["o"].astype(np.float32)) @ pl["n"].astype(np.float32)     # >0 inside the room
        if pl.get("evidence"):
            on_shell |= d < tol                 # on the plane, or beyond it (painted onto the wall)
        else:
            on_shell |= d < -0.10 * float(max(dims))     # far outside an assumed side
    keep = bg & ~on_shell

    s = max(1, int(np.ceil(np.sqrt(h * w / max_vertices))))
    ii, jj = np.arange(0, h, s), np.arange(0, w, s)
    Pg, Kg = P[ii][:, jj], keep[ii][:, jj]
    gh, gw = Kg.shape
    if gh < 2 or gw < 2 or Kg.sum() < 50:
        return None, {"vertices": 0, "faces": 0}
    cell = Kg[:-1, :-1] & Kg[:-1, 1:] & Kg[1:, :-1] & Kg[1:, 1:]
    idx = np.arange(gh * gw).reshape(gh, gw)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]
    V = Pg.reshape(-1, 3).astype(np.float64)

    def _facing(t):
        p0, p1, p2 = V[t[..., 0]], V[t[..., 1]], V[t[..., 2]]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n, axis=-1)
        cen = (p0 + p1 + p2) / 3.0
        cn = np.linalg.norm(cen, axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.abs(np.sum(n * cen, axis=-1)) / (nn * cn)

    cos_min = np.cos(np.radians(max_view_angle_deg))
    t1 = np.stack([a, b, c], -1)
    t2 = np.stack([a, c, d], -1)
    with np.errstate(invalid="ignore"):
        cell &= (_facing(t1) > cos_min) & (_facing(t2) > cos_min)
    lab, n = ndimage.label(cell)
    if n:
        sizes = ndimage.sum(cell, lab, index=np.arange(1, n + 1))
        cell &= np.isin(lab, 1 + np.flatnonzero(sizes >= max(8, min_component_frac * cell.size)))
    if not cell.any():
        return None, {"vertices": 0, "faces": 0}
    F = np.concatenate([t1[cell], t2[cell]])
    # Front faces must look at the camera (the origin).
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    if float(np.sum(np.cross(p1 - p0, p2 - p0) * (p0 + p1 + p2))) > 0:
        F = F[:, ::-1]
    used = np.unique(F)
    remap = -np.ones(gh * gw, int)
    remap[used] = np.arange(len(used))
    gi, gj = np.divmod(used, gw)
    uv = np.stack([(jj[gj] + 0.5) / w, 1.0 - (ii[gi] + 0.5) / h], 1)
    tex = Image.fromarray(image_rgb)
    # the relief's own (inpainted, point-map-resolution) image where holes
    # were filled, the full photo everywhere else
    if filled:
        big_hole = np.asarray(Image.fromarray(hole.astype(np.uint8) * 255).resize(tex.size, Image.BILINEAR)) > 40
        up = np.asarray(Image.fromarray(img).resize(tex.size, Image.BICUBIC))
        tex = Image.fromarray(np.where(big_hole[..., None], up, np.asarray(tex)))
    if max(tex.size) > max_texture_side:
        tex.thumbnail((max_texture_side, max_texture_side))
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=tex, metallicFactor=0.0,
                                                   roughnessFactor=1.0, doubleSided=True, name="room_relief")
    mesh = trimesh.Trimesh(vertices=V[used], faces=remap[F], process=False,
                           visual=trimesh.visual.TextureVisuals(uv=uv, material=material))
    return mesh, {"vertices": int(len(used)), "faces": int(len(F)), "stride": s,
                  "filled_hole_px": filled, "frame_fraction": round(float(keep.mean()), 3)}


def export_room(meshes: Dict, path) -> None:
    """Write the shell (+ relief) as one GLB with named nodes."""
    import trimesh
    sc = trimesh.Scene()
    for name, m in meshes.items():
        sc.add_geometry(m, node_name=name, geom_name=name)
    sc.export(str(path), file_type="glb")
