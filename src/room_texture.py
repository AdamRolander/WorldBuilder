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
    if getattr(layout, "scene_type", "room") == "open":      # ground only
        P = {"floor": P["floor"]}
    if not include_ceiling:
        P.pop("ceiling", None)
    if not include_front_wall:
        P.pop("wall_z_min", None)
    return P


# ---------------------------------------------------------------------------
# Plane textures
# ---------------------------------------------------------------------------

def project_plane_texture(plane: Dict, R_total: np.ndarray, K: np.ndarray, image_rgb: np.ndarray,
                          depth: np.ndarray, blocked: Optional[np.ndarray] = None,
                          long_side: int = 1024, depth_tol: float = 0.06,
                          keep_flush: bool = False, return_quality: bool = False,
                          exclude: Optional[np.ndarray] = None):
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
    material must be learned from texels that are only wall. Texels seen
    *through* the plane (a window, a doorway) and texels under ``exclude``
    (same size as ``depth``; the window mask) are not clean either: the
    trees outside a window are painted where the window is, but they are
    not what the wall is made of.
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
    clean = visible & ~beyond                # seen, is the surface itself, not part of any object
    if exclude is not None:
        clean &= ~exclude[vi, ui]
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


def _stationary(patch: np.ndarray, min_similarity: float = 0.8) -> bool:
    """The same material everywhere in the patch, at any scale of pattern.

    ``_texture_like`` judges an 8×8 reduction, which a checkerboard or large
    tiles alias into "structure". Here the patch is cut into blocks along
    its longer side (each holding at least one repeat of a tile pattern)
    and every block must have the same distribution of colours as the
    whole: a checkerboard passes, a patch that is half window does not.
    """
    h, w = patch.shape[:2]
    if min(h, w) < 16:
        return False
    p = patch.reshape(h, w, 3).astype(np.float32)
    lum = p.mean(axis=2)
    lo, hi = np.percentile(lum, [1, 99])
    if hi - lo < 1e-3:
        return True
    q = np.clip(((lum - lo) / (hi - lo) * 12).astype(int), 0, 11)
    n = max(2, min(4, int(round(max(h, w) / max(1, min(h, w))))) + 1)
    blocks = np.array_split(q, n, axis=0 if h >= w else 1)
    whole = np.bincount(q.ravel(), minlength=12).astype(np.float64)
    whole /= whole.sum()
    for b in blocks:
        hb = np.bincount(b.ravel(), minlength=12).astype(np.float64)
        hb /= max(hb.sum(), 1.0)
        if float(np.sum(np.sqrt(hb * whole))) < min_similarity:
            return False
    return True


def _period(patch: np.ndarray, min_lag: int = 4, min_peak: float = 0.4) -> Tuple[Optional[int], Optional[int]]:
    """Repeat length of a pattern along rows and columns (pixels, a whole
    number of periods), from the patch's autocorrelation; None along an
    axis with no clear repeat."""
    g = patch.astype(np.float32).mean(axis=2)
    g = g - g.mean()
    h, w = g.shape
    F = np.fft.rfft2(g, s=(2 * h, 2 * w))
    ac = np.fft.irfft2(F * np.conj(F), s=(2 * h, 2 * w))
    ac = ac / max(float(ac[0, 0]), 1e-9)
    out = []
    for line, n in ((ac[0, :w], w), (ac[:h, 0], h)):
        # unbiased: each lag is an average over the overlap only
        lags = np.arange(n)
        line = line * n / np.maximum(n - lags, 1)
        best = None
        for k in range(min_lag, int(0.7 * n)):
            if line[k] >= min_peak and line[k] >= line[k - 1] and line[k] >= line[k + 1]:
                best = k
                break
        if best is not None:
            # the longest whole number of repeats in the patch, measured at
            # its own peak: an error of a pixel per period does not add up
            m = max(1, int(0.95 * n) // best)
            while m > 1:
                lo_, hi_ = m * best - 3, min(n - 1, m * best + 4)
                if hi_ > lo_ and float(line[lo_:hi_].max()) >= min_peak:
                    best = lo_ + int(np.argmax(line[lo_:hi_]))
                    break
                m -= 1
        out.append(best)
    return out[1], out[0]            # (rows, cols): one or more whole repeats


def _tile_periodic(patch: np.ndarray, r0: int, c0: int, th: int, tw: int) -> Optional[np.ndarray]:
    """Tile whole periods of a regular pattern over (th, tw), in phase with
    where the patch sits, so the fill continues the seen tiles seamlessly."""
    pr, pc = _period(patch)
    if pr is None and pc is None:
        return None
    h, w = patch.shape[:2]

    def index(n_out, start, size, per, near_end=False):
        i = np.arange(n_out) - start
        if per is not None:                    # whole periods, repeated
            L = min(per, size)
            off = size - L if near_end else 0  # rows: the end nearest the camera is the sharpest
            return (i - off) % L + off
        m = i % (2 * size)                     # no clear period: mirror
        return np.where(m < size, m, 2 * size - 1 - m)

    return patch[index(th, r0, h, pr, near_end=True)][:, index(tw, c0, w, pc)].astype(np.float32)


def _detrend(patch: np.ndarray, rows: bool, scale: float = 0.5) -> np.ndarray:
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
        k = max(3, int(min(h, w) * scale) | 1)
        low = cv2.blur(p, (k, k), borderType=cv2.BORDER_REFLECT)
        base = np.median(p.reshape(-1, 3), axis=0)[None, None, :]
    return p - low + base


def synthesize_fill(tex: np.ndarray, visible: np.ndarray, rows: bool = False,
                    max_bands: int = 4, detail_clip: float = 24.0) -> Tuple[Optional[np.ndarray], str]:
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
        if (r1 - r0) * (c1 - c0) < 0.003 * th * tw or min(r1 - r0, c1 - c0) < 12:
            return None, "none"
        patch = tex[r0:r1, c0:c1]
        flat = _detrend(patch, rows=False)
        regular = _stationary(patch)
        if regular:
            # a regular pattern (tiles, a checkerboard, parquet): repeat whole
            # periods in phase with the seen part rather than mirror it
            per = _tile_periodic(flat, r0, c0, th, tw)
            if per is not None:
                return per, "tiled-periodic"
        if _texture_like(patch, rows=False) or regular:
            # Grain is small and evenly spread; a shadow or a crumb in the
            # patch is not, and tiling would stamp it everywhere. A material
            # without a regular pattern keeps only its fine detail, and
            # outliers in that are clipped.
            flat = _detrend(patch, rows=False, scale=1 / 6)
            base = np.median(flat.reshape(-1, 3), axis=0)
            dev = flat - base
            mad = float(np.median(np.abs(dev))) * 1.4826
            flat = base + np.clip(dev, -max(3.0 * mad, 6.0), max(3.0 * mad, 6.0))
            fill = np.pad(flat, ((r0, th - r1), (c0, tw - c1), (0, 0)), mode="symmetric")
            return fill.astype(np.float32), "tiled"
        # Shadows and highlights make a tiled floor fail the test even when
        # it is a material. If the structure is moderate (a shadow, not a
        # hard edge) and the grain left after removing the low frequencies
        # has none, that grain is repeated at reduced contrast, so what does
        # leak through does not stamp itself across the room. A flat median
        # colour (the old fallback) lost every floor pattern and left the
        # seen patch as a blotch.
        if _texture_like(patch, rows=False, max_residual=36.0) and _texture_like(flat, rows=False):
            base = np.median(patch.reshape(-1, 3).astype(np.float32), axis=0)
            soft = base + 0.5 * np.clip(flat - base, -detail_clip, detail_clip)
            fill = np.pad(soft, ((r0, th - r1), (c0, tw - c1), (0, 0)), mode="symmetric")
            return fill.astype(np.float32), "tiled-soft"
        return None, "none"

    # Colour per row, from the wall's *material* only. The material is the
    # colour most of the seen wall has; a row whose seen texels are mostly
    # a window, a door or a picture takes the material colour rather than
    # its own median, otherwise every such row becomes a pale stripe across
    # the whole wall (a window at mid-height repainted the unseen walls as
    # horizontal streaks). The profile is then smoothed over a few percent
    # of the height: row medians jump from row to row as objects hide
    # different parts of the wall, and tiled sideways those jumps read as
    # streaks; skirting and panelling bands are wider than the filter.
    from scipy import ndimage
    need = max(8, int(0.02 * tw))
    seen_px = tex[visible].astype(np.float32)
    if len(seen_px) < need:
        return None, "none"
    material = np.median(seen_px, axis=0)
    # tolerance relative to the material's brightness: on a dark wall a
    # window frame is only 40 levels lighter and must not count
    tol = float(np.clip(0.3 * material.mean(), 12.0, 40.0))
    is_mat = visible & (np.abs(tex.astype(np.float32) - material).mean(axis=-1) < tol)
    counts = is_mat.sum(axis=1)
    rows_ok = np.flatnonzero(counts >= need)
    base = np.tile(material, (th, 1)).astype(np.float32)
    for r in rows_ok:
        base[r] = np.median(tex[r, is_mat[r]].astype(np.float32), axis=0)
    k = max(3, int(0.08 * th) | 1)
    base = ndimage.median_filter(base, size=(k, 1), mode="nearest")
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
        # material grain is small; anything larger is an edge or a shadow
        # that would otherwise repeat along the wall
        d = np.clip(d, -detail_clip, detail_clip)
        detail[r0:r1] = np.pad(d, ((0, 0), (c0, tw - c1), (0, 0)), mode="symmetric")
        textured = True
    return base[:, None, :] + detail, "tiled" if textured else "banded"


def _match_local(tex: np.ndarray, fill: np.ndarray, known: np.ndarray, scale_frac: float = 0.02,
                 reach_frac: float = 0.2, clip: float = 80.0) -> np.ndarray:
    """Carry the seen surface's local colour into the prediction near it.

    A fill learned from the whole surface has the surface's average colour,
    but light falls off across a wall and a floor: the patch hidden behind a
    lamp or a chair, filled with the average, is a silhouette of that object
    in a slightly different colour, visible from every other viewpoint. The
    low-frequency difference between what was seen and the prediction is
    interpolated into the unseen texels (normalised convolution, fine to
    coarse) and faded out with distance from anything seen, so a hole
    surrounded by wall takes that wall's colour and an unseen wall far away
    keeps the predicted material. ``known`` are the texels seen sharply.
    """
    import cv2
    from scipy import ndimage
    th, tw = known.shape
    if int(known.sum()) < 16:
        return fill
    f = max(1, int(np.ceil(max(th, tw) / 256)))
    sh, sw = max(2, th // f), max(2, tw // f)
    r = np.clip(tex.astype(np.float32) - fill.astype(np.float32), -clip, clip) * known[..., None]
    rs = cv2.resize(r, (sw, sh), interpolation=cv2.INTER_AREA)
    ws = cv2.resize(known.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    acc = np.zeros_like(rs)
    accw = np.zeros((sh, sw), np.float32)
    base = max(1.0, scale_frac * max(sh, sw))
    for sigma in (base, 3 * base, 9 * base):
        num = cv2.GaussianBlur(rs, (0, 0), sigma)
        den = cv2.GaussianBlur(ws, (0, 0), sigma)
        est = num / np.maximum(den, 1e-4)[..., None]
        conf = np.clip(den / 0.25, 0, 1) * (1 - accw)
        acc += conf[..., None] * est
        accw += conf
    dist = ndimage.distance_transform_edt(ws < 0.5)
    beta = np.clip(1.0 - dist / (reach_frac * max(sh, sw)), 0, 1)
    corr = cv2.resize(acc * beta[..., None], (tw, th), interpolation=cv2.INTER_LINEAR)
    return fill.astype(np.float32) + corr



def fill_hidden(tex: np.ndarray, visible: np.ndarray, fallback_rgb=(200, 200, 200), rows: bool = False,
                quality: Optional[np.ndarray] = None, donor: Optional[np.ndarray] = None,
                return_fill: bool = False, source: Optional[np.ndarray] = None, min_quality: float = 0.35):
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
        # relative to the best of this surface, and never below a third of
        # a photo pixel per texel: a window seen edge-on would otherwise be
        # the only "seen" part of a wall and be dragged across all of it
        cand = visible & (quality >= max(0.25 * ref, min_quality))
        if cand.sum() >= 0.25 * visible.sum() or cand.sum() >= 0.003 * visible.size:
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
    pure = fill if how in ("tiled", "tiled-soft", "tiled-periodic", "banded") else None
    if frac > 0.999 and sharp is visible:
        return _ret(tex, "complete", pure)
    # the material comes from clean wall only, but the local colour around a
    # hidden patch from everything seen (a lamp in front of a window hides
    # window, not wall)
    fill = _match_local(tex, fill, sharp)
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
                include_ceiling: bool = True, include_front_wall: bool = True,
                exclude: Optional[np.ndarray] = None):
    """Textured quads for every shell plane.

    Returns ``(meshes, stats)``: ``meshes`` maps node name → trimesh with
    ``TextureVisuals``; ``stats`` maps plane → visible fraction, fill mode
    and texture size. ``exclude`` (same size as ``depth``) marks pixels that
    are painted where seen but never used to learn a surface's material
    (windows).
    """
    import trimesh
    from PIL import Image
    if long_side is None:
        long_side = int(np.clip(1.5 * max(image_rgb.shape[:2]), 512, 2048))
    R = layout.R_total
    meshes, stats = {}, {}
    planes = room_planes(layout, include_ceiling, include_front_wall)
    projected = {name: project_plane_texture(pl, R, K, image_rgb, depth, blocked, long_side,
                                             keep_flush=(name != "floor"), return_quality=True,
                                             exclude=exclude)
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
    # The donor is the wall with the most *clean* texels: a wall that is
    # mostly window was seen a lot but shows little of the wall's material.
    donors = [(projected[n][3].mean(), n) for n in planes
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
    # A ceiling the photo never shows: the colour where the seen walls meet
    # it (their top band), not a white lid that reads as a hole in a dark room.
    if "ceiling" in done and done["ceiling"][1] == "fallback":
        tops = []
        for n in planes:
            if not n.startswith("wall"):
                continue
            tex, vis, qual, clean = projected[n]
            band = clean[: max(2, int(0.15 * clean.shape[0]))]
            if band.sum() >= 50:
                tops.append(tex[: band.shape[0]][band])
        if tops:
            rgb = np.median(np.concatenate(tops), axis=0).astype(np.float32)
            th, tw = projected["ceiling"][1].shape
            done["ceiling"] = (np.tile(np.clip(rgb, 0, 255).astype(np.uint8), (th, tw, 1)), "wall top colour", None)
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
        if name.startswith("room_relief") or name.startswith("room_builtin"):
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

def _fill_holes(P: np.ndarray, hole: np.ndarray, known: np.ndarray, rays_ok: np.ndarray,
                plane_rms_frac: float = 0.012) -> Tuple[np.ndarray, np.ndarray]:
    """Geometry behind small objects that were cut out of the background.

    A hole whose surroundings lie on one plane (a bowl on a counter, a
    picture on a wall) is filled with that plane, each pixel placed where
    its own viewing ray meets it. A hole whose surroundings do not (a vase
    in front of both a shelf and the wall behind it) has no single surface
    to continue; interpolating across it draws a sheet from the near surface
    to the far one, which from any other viewpoint is a spike pointing at
    the camera. Those holes are left open.

    Returns ``(P_filled, filled_mask)``.
    """
    from scipy import ndimage
    out = P.copy()
    filled = np.zeros(hole.shape, bool)
    lab, n = ndimage.label(hole)
    h, w = hole.shape
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        pad = max(4, int(0.25 * max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)))
        ys = slice(max(0, sl[0].start - pad), min(h, sl[0].stop + pad))
        xs = slice(max(0, sl[1].start - pad), min(w, sl[1].stop + pad))
        comp = lab[ys, xs] == i
        ring = ndimage.binary_dilation(comp, iterations=3) & known[ys, xs]
        if ring.sum() < 12:
            continue
        B = P[ys, xs][ring].astype(np.float64)
        c = B.mean(0)
        _, sv, vt = np.linalg.svd(B - c, full_matrices=False)
        nrm = vt[2]
        rms = sv[2] / np.sqrt(len(B))
        d_med = float(np.median(np.linalg.norm(B, axis=1)))
        if rms > plane_rms_frac * d_med:
            continue
        sel = comp & rays_ok[ys, xs]
        if not sel.any():
            continue
        Q = P[ys, xs][sel].astype(np.float64)                  # the object's own surface: same rays
        r_obj = np.linalg.norm(Q, axis=1)
        ray = Q / r_obj[:, None]
        den = ray @ nrm
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (c @ nrm) / den
        # the background is behind the object, and near the ring's own depth
        ok = np.isfinite(t) & (t > 0.97 * r_obj) & (t < 1.6 * d_med) & (t > 0.6 * d_med)
        if ok.mean() < 0.8:
            continue
        yy, xx = np.nonzero(sel)
        yy, xx = yy[ok] + ys.start, xx[ok] + xs.start
        out[yy, xx] = (ray[ok] * t[ok, None]).astype(out.dtype)
        filled[yy, xx] = True
    return out, filled


def build_relief(points_aligned: np.ndarray, valid: np.ndarray, owner: np.ndarray,
                 layout: rl.RoomLayout, image_rgb: np.ndarray, max_vertices: int = 200_000,
                 plane_tol_frac: float = 0.03, small_object_frac: float = 0.015,
                 max_view_angle_deg: float = 80.0, min_component_frac: float = 0.003,
                 max_texture_side: int = 2048, thin_steep_deg: float = 62.0, max_pieces: int = 16,
                 terrain: bool = False):
    """Meshes of the built-in structure that is neither object nor shell.

    ``points_aligned`` (h,w,3) is the point map in the aligned frame (camera
    at the origin), ``owner`` (h,w) the object id per pixel (0 = none).
    Returns ``({name: trimesh}, stats)``: one mesh per connected piece of
    background surface (``room_relief_00`` is the largest), so a viewer or
    an importer can hide the cabinet run without losing the window recess.

    With ``terrain`` the ground itself is kept (an open scene has no flat
    floor to stand in for it): the relief is then the terrain, textured by
    the photo, and nothing is dropped for being on or below the floor plane.
    """
    import cv2
    import trimesh
    from PIL import Image
    from scipy import ndimage
    h, w = valid.shape
    P = points_aligned.astype(np.float32).copy()
    bg = valid & (owner == 0)
    img = np.asarray(Image.fromarray(image_rgb).resize((w, h), Image.BILINEAR))

    # Close the holes small objects leave in the surface they stand on.
    ids, counts = np.unique(owner[owner > 0], return_counts=True)
    small = np.isin(owner, ids[counts < small_object_frac * h * w])
    hole = ndimage.binary_dilation(small, iterations=2) & ~(valid & (owner > 0) & ~small)
    filled = 0
    hole_filled = np.zeros((h, w), bool)
    if hole.any():
        known = bg & ~hole
        P, hole_filled = _fill_holes(P, hole, known, valid)
        if hole_filled.any():
            img = cv2.inpaint(img, hole_filled.astype(np.uint8), 3, cv2.INPAINT_TELEA)
            bg = (bg & ~hole) | hole_filled
            filled = int(hole_filled.sum())
        else:
            bg = bg & ~hole
    # Mask edges are a pixel or two off; keep object colours out of the relief.
    big = (owner > 0) & ~small
    bg &= ~ndimage.binary_dilation(big, iterations=2)
    bg &= ~(ndimage.binary_dilation(small, iterations=2) & ~hole_filled)

    # Drop what the shell already represents, and what lies outside the room.
    mn, mx = np.asarray(layout.bounds_min, float), np.asarray(layout.bounds_max, float)
    dims = mx - mn
    tol = plane_tol_frac * float(max(dims))
    on_shell = np.zeros((h, w), bool)
    planes = room_planes(layout)
    for pname, pl in planes.items():
        if terrain and pname == "floor":
            continue
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
    empty = ({}, {"vertices": 0, "faces": 0, "pieces": 0})
    if gh < 2 or gw < 2 or Kg.sum() < 50:
        return empty
    cell = Kg[:-1, :-1] & Kg[:-1, 1:] & Kg[1:, :-1] & Kg[1:, 1:]
    have_points = cell.copy()
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

    t1 = np.stack([a, b, c], -1)
    t2 = np.stack([a, c, d], -1)
    with np.errstate(invalid="ignore"):
        face = np.minimum(_facing(t1), _facing(t2))
    face = np.nan_to_num(face, nan=0.0)
    cell &= face > np.cos(np.radians(max_view_angle_deg))
    # A depth edge in the point map is not a clean step: the pixels along it
    # are blends of the two surfaces and form a narrow ramp between them.
    # Each of its triangles can pass the angle test above, and together they
    # are a sheet along the viewing rays. A real surface seen obliquely (a
    # counter top from standing height) is just as steep but many cells
    # wide, so only steep cells that form a *thin* band are cut.
    steep = cell & (face < np.cos(np.radians(thin_steep_deg)))
    if steep.any():
        wide = ndimage.binary_opening(steep, structure=np.ones((5, 5), bool))
        wide = ndimage.binary_dilation(wide, iterations=2) & steep
        cell &= ~(steep & ~wide)
    # what is left attached by a strip one or two cells wide is a sliver
    cell &= ndimage.binary_opening(cell, structure=np.ones((3, 3), bool))
    if terrain:
        # Ground seen at a shallow angle passes and fails the angle test row by
        # row; close those gaps so the terrain is a surface and not stripes.
        cell |= ndimage.binary_closing(cell, structure=np.ones((3, 3), bool), iterations=2) & have_points
    lab, n = ndimage.label(cell)
    dropped_floating = 0
    if n:
        sizes = ndimage.sum(cell, lab, index=np.arange(1, n + 1))
        good = 1 + np.flatnonzero(sizes >= max(8, min_component_frac * cell.size))
        if not terrain:
            # A small piece that touches neither the floor, a wall nor the
            # ceiling, and is not part of a larger surface, is a fragment of
            # something the detector did not list (the back and the legs of
            # an unsegmented chair, with nothing in between). Built-ins are
            # attached; floating fragments are not, and as 2.5D skins they
            # look worse than nothing. The gap they leave is filled by the
            # shell behind them.
            H = float(mx[1] - mn[1])
            keep_ids = []
            for cid in good:
                sel = lab == cid
                pts = Pg[:-1, :-1][sel]
                ext = float(np.max(pts.max(0) - pts.min(0)))
                if ext >= 0.25 * H:
                    keep_ids.append(cid); continue
                near_floor = float(pts[:, 1].min()) <= mn[1] + 0.03 * H
                near_ceiling = float(pts[:, 1].max()) >= mx[1] - 0.03 * H
                near_wall = False
                for pname, pl in planes.items():
                    if pname in ("floor", "ceiling") or not pl.get("evidence"):
                        continue
                    d = (pts - pl["o"]) @ pl["n"]
                    if float(np.min(d)) <= 0.03 * float(max(dims)):
                        near_wall = True; break
                if near_floor or near_ceiling or near_wall:
                    keep_ids.append(cid)
                else:
                    dropped_floating += 1
            good = np.asarray(keep_ids, int)
        cell &= np.isin(lab, good)
        order = good[np.argsort(-sizes[good - 1])] if len(good) else good
    if not cell.any():
        return empty

    tex_full = np.asarray(Image.fromarray(image_rgb))
    Hf, Wf = tex_full.shape[:2]
    if filled:      # the inpainted image where holes were filled, the photo everywhere else
        m = np.asarray(Image.fromarray(hole_filled.astype(np.uint8) * 255).resize((Wf, Hf), Image.BILINEAR)) > 40
        up = np.asarray(Image.fromarray(img).resize((Wf, Hf), Image.BICUBIC))
        tex_full = np.where(m[..., None], up, tex_full)
    tex_scale = min(1.0, max_texture_side / float(max(Hf, Wf)))

    meshes: Dict = {}
    pieces: List[Dict] = []
    total_v = total_f = 0
    flip = None
    for k, comp_id in enumerate(order):
        # pieces beyond the cap are merged into the last one
        sel = (lab == comp_id) if k < max_pieces - 1 else np.isin(lab, order[k:])
        F = np.concatenate([t1[sel], t2[sel]])
        if flip is None:                         # front faces look at the camera (the origin)
            p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
            flip = float(np.sum(np.cross(p1 - p0, p2 - p0) * (p0 + p1 + p2))) > 0
        if flip:
            F = F[:, ::-1]
        used = np.unique(F)
        remap = -np.ones(gh * gw, int)
        remap[used] = np.arange(len(used))
        gi, gj = np.divmod(used, gw)
        py, px = ii[gi], jj[gj]                                   # point-map pixels
        # crop the photo to this piece (one small texture each, not N copies of the photo)
        y0, y1 = int(py.min() * Hf / h), int(np.ceil((py.max() + 1) * Hf / h))
        x0, x1 = int(px.min() * Wf / w), int(np.ceil((px.max() + 1) * Wf / w))
        y0, x0, y1, x1 = max(0, y0 - 2), max(0, x0 - 2), min(Hf, y1 + 2), min(Wf, x1 + 2)
        crop = Image.fromarray(np.ascontiguousarray(tex_full[y0:y1, x0:x1]))
        if tex_scale < 1.0:
            crop = crop.resize((max(2, int(crop.width * tex_scale)), max(2, int(crop.height * tex_scale))),
                               Image.LANCZOS)
        fx, fy = (px + 0.5) * Wf / w, (py + 0.5) * Hf / h         # photo pixels
        uv = np.stack([(fx - x0) / (x1 - x0), 1.0 - (fy - y0) / (y1 - y0)], 1)
        name = f"room_relief_{k:02d}"
        material = trimesh.visual.material.PBRMaterial(baseColorTexture=crop, metallicFactor=0.0,
                                                       roughnessFactor=1.0, doubleSided=True, name=name)
        meshes[name] = trimesh.Trimesh(vertices=V[used], faces=remap[F], process=False,
                                       visual=trimesh.visual.TextureVisuals(uv=uv, material=material))
        ext = V[used].max(0) - V[used].min(0)
        pieces.append({"name": name, "faces": int(len(F)), "center": np.round(V[used].mean(0), 3).tolist(),
                       "extent": np.round(ext, 3).tolist(),
                       "bbox_px": [int(px.min()), int(py.min()), int(px.max()), int(py.max())]})
        total_v += len(used)
        total_f += len(F)
        if k >= max_pieces - 1:
            break
    return meshes, {"vertices": int(total_v), "faces": int(total_f), "stride": s, "pieces": len(meshes),
                    "piece_info": pieces, "filled_hole_px": filled, "dropped_floating": dropped_floating,
                    "frame_fraction": round(float(keep.mean()), 3)}


def export_room(meshes: Dict, path) -> None:
    """Write the shell (+ relief) as one GLB with named nodes."""
    import trimesh
    sc = trimesh.Scene()
    for name, m in meshes.items():
        sc.add_geometry(m, node_name=name, geom_name=name)
    sc.export(str(path), file_type="glb")
