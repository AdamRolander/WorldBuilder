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
                          long_side: int = 1024, depth_tol: float = 0.06,
                          keep_flush: bool = False, return_quality: bool = False):
    """Rectified view of one plane. Returns ``(texture (th,tw,3) uint8,
    visible (th,tw) bool)``; row 0 is the plane's far edge along ``av`` so
    that trimesh/OpenGL UVs (v up) map ``v = t / sv``.

    ``depth`` is the camera-z map (any resolution, nan = unknown) and
    ``blocked`` a same-size mask of pixels that belong to objects. With
    ``keep_flush`` an object pixel whose observed point lies *in* the plane
    (within 1.5 % of its depth) is kept: tiles, panelling or a poster that
    the detector listed as objects are the wall's surface, and without them
    the wall would be painted with some other wall's colour. Used for walls
    and ceilings; on the floor a flush object is a rug, which moves.

    With ``return_quality`` two more arrays are returned: per texel, how
    many photo pixels it spans in its worst direction (small = stretched,
    blurry), and ``clean`` — seen and not part of any object, flush or not.
    A flush mirror belongs on the wall where it was seen, but the wall's
    material must be learned from texels that are only wall.
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
    clean = visible.copy()                   # seen *and* not part of any object
    if blocked is not None:
        blk = blocked[vi, ui]
        clean &= ~blk
        if keep_flush:
            with np.errstate(invalid="ignore"):
                blk = blk & ~(np.abs(z - z_obs) <= 0.015 * z_obs)
        visible &= ~blk
    tex = cv2.remap(image_rgb, (u - 0.5).astype(np.float32), (v - 0.5).astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if not return_quality:
        return tex, visible
    # Smallest singular value of d(photo pixel)/d(texel).
    un, vn = np.nan_to_num(u), np.nan_to_num(v)
    a, b = np.gradient(un, axis=1), np.gradient(un, axis=0)
    c, d = np.gradient(vn, axis=1), np.gradient(vn, axis=0)
    S = a * a + b * b + c * c + d * d
    D = a * d - b * c
    quality = np.sqrt(np.maximum(0.0, 0.5 * (S - np.sqrt(np.maximum(0.0, S * S - 4.0 * D * D)))))
    return tex, visible, np.where(visible, quality, 0.0), clean


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


def _coarse_rectangle(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """``largest_rectangle`` on a ≤160-cell grid, in full-resolution indices."""
    th, tw = mask.shape
    k = max(1, int(np.ceil(max(th, tw) / 160)))
    hs, ws = th // k, tw // k
    if hs < 2 or ws < 2:
        return (0, 0, 0, 0)
    coarse = mask[:hs * k, :ws * k].reshape(hs, k, ws, k).all(axis=(1, 3))
    r0, c0, r1, c1 = largest_rectangle(coarse)
    return r0 * k, c0 * k, r1 * k, c1 * k


def _texture_like(patch: np.ndarray, rows: bool, max_residual: float = 18.0) -> bool:
    """No large-scale structure once smooth shading is discounted.

    The patch is reduced to an 8×8 grid (8 columns per row band in ``rows``
    mode, after removing each row's own colour) and a linear ramp is fitted
    and removed: a lighting gradient passes, half a window or a whiteboard
    edge does not, and would otherwise be stamped across the surface.
    """
    import cv2
    p = patch.astype(np.float32)
    if rows:
        p = p - np.median(p, axis=1, keepdims=True)
    small = cv2.resize(p, (8, 8), interpolation=cv2.INTER_AREA).reshape(64, 3)
    yy, xx = np.mgrid[:8, :8]
    A = np.stack([xx.ravel(), yy.ravel(), np.ones(64)], 1).astype(np.float32)
    coef, *_ = np.linalg.lstsq(A, small, rcond=None)
    return float((small - A @ coef).std(axis=0).max()) <= max_residual


def _detrend(patch: np.ndarray, rows: bool) -> np.ndarray:
    """Detail of a patch on a flat base, so that tiling it repeats the
    material but not the lighting (mirrored shadows read as blobs). In
    ``rows`` mode the base keeps one colour per row, which preserves a
    wall's vertical make-up (skirting, panelling, paper)."""
    import cv2
    p = patch.astype(np.float32)
    h, w = p.shape[:2]
    if rows:
        k = max(3, (w // 2) | 1)
        low = cv2.blur(p, (k, 1), borderType=cv2.BORDER_REFLECT)
        base = np.median(p, axis=1, keepdims=True)
    else:
        k = max(3, (min(h, w) // 2) | 1)
        low = cv2.blur(p, (k, k), borderType=cv2.BORDER_REFLECT)
        base = np.median(p.reshape(-1, 3), axis=0)[None, None, :]
    return p - low + base


def synthesize_fill(tex: np.ndarray, visible: np.ndarray, rows: bool = False,
                    max_bands: int = 4) -> Tuple[Optional[np.ndarray], str]:
    """Predict a whole surface from the part of it that was seen.

    ``rows=False`` (floors, ceilings): the material is the same in every
    direction, so the largest visible rectangle is tiled with mirror
    symmetry over the plane.

    ``rows=True`` (walls): a wall is the same *along* its length but layered
    over its height. Each row takes the median colour seen at that height,
    and up to ``max_bands`` visible rectangles of that material add their
    detail, tiled sideways only.

    Returns ``(fill float32 (th,tw,3) or None, how)`` with ``how`` in
    {"tiled", "banded", "none"}.
    """
    th, tw = visible.shape
    if not rows:
        r0, c0, r1, c1 = _coarse_rectangle(visible)
        if (r1 - r0) * (c1 - c0) < 0.01 * th * tw or min(r1 - r0, c1 - c0) < 8:
            return None, "none"
        patch = tex[r0:r1, c0:c1]
        if not _texture_like(patch, rows=False):
            return None, "none"
        fill = np.pad(_detrend(patch, rows=False), ((r0, th - r1), (c0, tw - c1), (0, 0)), mode="symmetric")
        return fill.astype(np.float32), "tiled"

    # Colour per row: the median of what was seen in that row, so the
    # material that covers most of the wall at that height wins (a cream
    # panel beside a window does not repaint the papered wall above it).
    need = max(8, int(0.02 * tw))
    counts = visible.sum(axis=1)
    rows_ok = np.flatnonzero(counts >= need)
    if not len(rows_ok):
        return None, "none"
    base = np.zeros((th, 3), np.float32)
    for r in rows_ok:
        base[r] = np.median(tex[r, visible[r]].astype(np.float32), axis=0)
    missing = np.setdiff1d(np.arange(th), rows_ok)
    if len(missing):
        base[missing] = base[rows_ok[np.argmin(np.abs(rows_ok[None, :] - missing[:, None]), axis=1)]]
    # Detail on top: visible rectangles of that same material, tiled sideways.
    detail = np.zeros((th, tw, 3), np.float32)
    free = np.ones(th, bool)
    textured = False
    # ...looked for only among texels that *are* that row's dominant colour
    major = visible & (np.abs(tex.astype(np.float32) - base[:, None, :]).mean(axis=-1) < 30.0)
    for _ in range(max_bands):
        r0, c0, r1, c1 = _coarse_rectangle(major & free[:, None])
        if (r1 - r0) < max(4, 0.03 * th) or (c1 - c0) < max(8, 0.05 * tw):
            break
        free[r0:r1] = False
        patch = tex[r0:r1, c0:c1]
        row_col = np.median(patch.astype(np.float32), axis=1)
        if float(np.abs(row_col - base[r0:r1]).mean()) > 20.0 or not _texture_like(patch, rows=True):
            continue                          # another material, or not a material at all
        d = _detrend(patch, rows=True) - row_col[:, None, :]
        detail[r0:r1] = np.pad(d, ((0, 0), (c0, tw - c1), (0, 0)), mode="symmetric")
        textured = True
    return base[:, None, :] + detail, "tiled" if textured else "banded"


def fill_hidden(tex: np.ndarray, visible: np.ndarray, fallback_rgb=(200, 200, 200), rows: bool = False,
                quality: Optional[np.ndarray] = None, donor: Optional[np.ndarray] = None,
                return_fill: bool = False, source: Optional[np.ndarray] = None):
    """Complete a plane texture. Returns ``(texture, how)`` (plus the pure
    synthesized fill when ``return_fill``), ``how`` in {"complete",
    "tiled", "banded", "donor", "median", "fallback"}.

    ``quality`` (photo pixels per texel, from ``project_plane_texture``):
    where a surface was seen at a grazing angle the photo holds a fraction
    of a pixel per texel and the rectified texture is a smear. When the
    surface can be synthesized, such texels are treated as unseen and
    predicted from the sharp part instead. ``donor`` is another surface's
    synthesized fill (same size) to use when this one has too little of its
    own to learn from. ``source`` restricts which seen texels the material
    may be learned from (default: all of them).
    """
    from scipy import ndimage
    th, tw = visible.shape
    frac = float(visible.mean())

    def _ret(t, how, fill=None):
        return (t, how, fill) if return_fill else (t, how)

    if frac < 0.005:
        if donor is not None:
            return _ret(np.clip(donor, 0, 255).astype(np.uint8), "donor", donor)
        flat = np.tile(np.asarray(fallback_rgb, np.float32), (th, tw, 1))
        return _ret(flat.astype(np.uint8), "fallback", None)

    sharp = visible
    if quality is not None and visible.any():
        ref = float(np.percentile(quality[visible], 90))
        cand = visible & (quality >= 0.25 * ref)
        if cand.sum() >= 0.25 * visible.sum():
            sharp = cand
    src = visible if source is None else (visible & source)
    fill, how = synthesize_fill(tex, sharp & src, rows=rows)
    if fill is None and sharp is not visible:
        sharp = visible
        fill, how = synthesize_fill(tex, src, rows=rows)
    if fill is None:
        sharp = visible                          # nothing to predict from: keep every seen texel
        if donor is not None:
            fill, how = donor.astype(np.float32), "donor"
        else:
            fill = np.tile(np.median(tex[visible], axis=0).astype(np.float32), (th, tw, 1))
            how = "median"
    pure = fill if how in ("tiled", "banded") else None
    if frac > 0.999 and sharp is visible:
        return _ret(tex, "complete", pure)
    # Feather: observed texels fade into the prediction over a few percent
    # of the plane, so there is no hard line where the photo's knowledge ends.
    ramp = max(2.0, 0.02 * max(th, tw))
    alpha = np.clip(ndimage.distance_transform_edt(sharp) / ramp, 0, 1)[..., None]
    out = alpha * tex.astype(np.float32) + (1 - alpha) * fill
    return _ret(np.clip(out, 0, 255).astype(np.uint8), how, pure)


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
    projected = {name: project_plane_texture(pl, R, K, image_rgb, depth, blocked, long_side,
                                             keep_flush=(name != "floor"), return_quality=True)
                 for name, pl in planes.items()}
    # A wall the photo never saw takes the colour of the walls it did see,
    # which beats any hard-coded default.
    seen_wall = [tex[clean] for name, (tex, vis, _, clean) in projected.items()
                 if name.startswith("wall") and clean.mean() > 0.02]
    wall_rgb = tuple(np.median(np.concatenate(seen_wall), axis=0).astype(int)) if seen_wall else FALLBACK_RGB["wall"]
    # First pass: every plane predicts itself from what was seen of it.
    done = {}
    for name in planes:
        kind = "wall" if name.startswith("wall") else name
        tex, vis, qual, clean = projected[name]
        done[name] = fill_hidden(tex, vis, wall_rgb if kind == "wall" else FALLBACK_RGB[kind],
                                 rows=(kind == "wall"), quality=qual, return_fill=True, source=clean)
    # Second pass: walls of one room are usually finished alike. A wall that
    # had nothing to learn from borrows the best-seen wall's predicted
    # make-up (all walls share the floor-to-ceiling axis, so skirting and
    # panelling line up), tiled along its own length.
    donors = [(projected[n][1].mean(), n) for n in planes
              if n.startswith("wall") and done[n][2] is not None]
    if donors:
        donor_name = max(donors)[1]
        dfill = done[donor_name][2]
        for name in planes:
            if not name.startswith("wall") or done[name][1] not in ("fallback", "median"):
                continue
            tex, vis, qual, clean = projected[name]
            th, tw = vis.shape
            scaled = _resize_rows(dfill, th)
            reps = int(np.ceil(tw / scaled.shape[1]))
            strip = np.concatenate([scaled if i % 2 == 0 else scaled[:, ::-1] for i in range(reps)], axis=1)[:, :tw]
            t2, how2, _ = fill_hidden(tex, vis, wall_rgb, rows=True, quality=qual, donor=strip,
                                      return_fill=True, source=clean)
            done[name] = (t2, f"like {donor_name}" if how2 == "donor" else how2, None)
    for name, pl in planes.items():
        tex, how, _ = done[name]
        vis = projected[name][1]
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


def _resize_rows(img: np.ndarray, th: int) -> np.ndarray:
    """Scale an image to ``th`` rows, keeping its aspect ratio."""
    import cv2
    h, w = img.shape[:2]
    tw = max(2, int(round(w * th / max(h, 1))))
    return cv2.resize(img.astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA if th < h else cv2.INTER_LINEAR)


def shell_vertex_color_mesh(meshes: Dict, cells: int = 128):
    """The shell as one vertex-coloured grid mesh (for PLY consumers),
    sampled from the textures so both representations agree."""
    import trimesh
    V_all, F_all, C_all, off = [], [], [], 0
    for name, m in meshes.items():
        if name == "room_relief" or name.startswith("room_builtin"):
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
    build_relief.last = (P, keep)             # reused by src/builtin_boxes.py

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
