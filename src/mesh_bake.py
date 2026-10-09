"""Decimate a dense vertex-coloured mesh and bake its colours into a texture.

SAM 3D returns 10^5-10^6 triangles per object with colour per vertex: a
35-object kitchen is 19 million triangles and several hundred MB, which no
game engine, simulator or web viewer wants. Decimating a vertex-coloured
mesh blurs its colours (they live on the vertices being removed), so the
two steps go together:

1. quadric decimation to a triangle budget (``fast_simplification``),
2. a UV atlas for the small mesh (``xatlas``),
3. for every atlas texel, the colour of the nearest point on the *original*
   mesh (KD-tree over its vertices), so the texture keeps the detail the
   geometry lost.

The result is a few thousand triangles plus one PNG per object and looks
the same from a metre away. Both libraries are optional; ``lite_mesh``
degrades to "decimated, mean colour" without xatlas and to the original
mesh without fast_simplification.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def decimate(vertices: np.ndarray, faces: np.ndarray, target_faces: int) -> Tuple[np.ndarray, np.ndarray]:
    """Quadric decimation. Returns the input unchanged if it is already
    small or the library is missing."""
    if len(faces) <= target_faces:
        return np.asarray(vertices), np.asarray(faces)
    try:
        import fast_simplification
    except ImportError:
        return np.asarray(vertices), np.asarray(faces)
    v, f = fast_simplification.simplify(np.asarray(vertices, np.float32), np.asarray(faces, np.int32),
                                        target_count=int(target_faces))
    return np.asarray(v, np.float64), np.asarray(f, np.int64)


def bake_vertex_colors(v: np.ndarray, f: np.ndarray, src_vertices: np.ndarray, src_colors: np.ndarray,
                       tex_size: int = 512, samples_per_texel: float = 2.0):
    """UV-unwrap ``(v, f)`` and bake colours sampled from the source mesh.

    Returns ``(vertices, faces, uv, image)`` with vertices duplicated along
    UV seams (as any textured mesh needs) and ``image`` a PIL RGB image.
    """
    import xatlas
    from PIL import Image
    from scipy import ndimage
    from scipy.spatial import cKDTree

    vmapping, indices, uvs = xatlas.parametrize(np.asarray(v, np.float32), np.asarray(f, np.uint32))
    V = np.asarray(v)[vmapping]
    F = np.asarray(indices, np.int64)
    uv = np.asarray(uvs, np.float64)

    # Sample every face densely enough to cover its texels, in one batch.
    t_uv = uv[F] * tex_size                                   # (nf, 3, 2) in texels
    e1, e2 = t_uv[:, 1] - t_uv[:, 0], t_uv[:, 2] - t_uv[:, 0]
    area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    n = np.maximum(3, np.ceil(area * samples_per_texel).astype(int))
    face_idx = np.repeat(np.arange(len(F)), n)
    rng = np.random.default_rng(0)
    r1, r2 = rng.random(len(face_idx)), rng.random(len(face_idx))
    flip = r1 + r2 > 1
    r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
    bary = np.stack([1 - r1 - r2, r1, r2], 1)
    pos = np.einsum("nk,nkd->nd", bary, V[F[face_idx]])
    tex = np.einsum("nk,nkd->nd", bary, t_uv[face_idx])
    _, nearest = cKDTree(src_vertices).query(pos, workers=-1)
    col = np.asarray(src_colors)[nearest, :3].astype(np.float64)

    px = np.clip(tex[:, 0].astype(int), 0, tex_size - 1)
    py = np.clip(((tex_size - tex[:, 1])).astype(int), 0, tex_size - 1)      # image row 0 = v 1
    flat = py * tex_size + px
    acc = np.zeros((tex_size * tex_size, 3))
    cnt = np.zeros(tex_size * tex_size)
    np.add.at(acc, flat, col)
    np.add.at(cnt, flat, 1)
    filled = cnt > 0
    img = np.zeros((tex_size * tex_size, 3))
    img[filled] = acc[filled] / cnt[filled, None]
    img = img.reshape(tex_size, tex_size, 3)
    filled = filled.reshape(tex_size, tex_size)
    if not filled.all() and filled.any():
        # Bleed chart colours outward so bilinear filtering at chart borders
        # never picks up empty texels.
        _, (iy, ix) = ndimage.distance_transform_edt(~filled, return_indices=True)
        img = img[iy, ix]
    return V, F, uv, Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def lite_mesh(mesh, target_faces: int = 8000, tex_size: int = 512, name: Optional[str] = None):
    """Decimated, texture-baked copy of a vertex-coloured trimesh. Meshes
    without vertex colours are only decimated."""
    import trimesh
    src_v = np.asarray(mesh.vertices, np.float64)
    src_f = np.asarray(mesh.faces)
    v, f = decimate(src_v, src_f, target_faces)
    colors = np.asarray(mesh.visual.vertex_colors) if getattr(mesh.visual, "kind", None) == "vertex" else None
    if colors is None:
        return trimesh.Trimesh(vertices=v, faces=f, process=False)
    try:
        V, F, uv, image = bake_vertex_colors(v, f, src_v, colors, tex_size)
    except ImportError:
        mean = colors[:, :3].mean(axis=0).astype(np.uint8)
        return trimesh.Trimesh(vertices=v, faces=f, process=False,
                               vertex_colors=np.tile(np.append(mean, 255), (len(v), 1)))
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=image, metallicFactor=0.0,
                                                   roughnessFactor=1.0, name=name or "baked")
    return trimesh.Trimesh(vertices=V, faces=F, process=False,
                           visual=trimesh.visual.TextureVisuals(uv=uv, material=material))


def texture_budget(mesh_extent: float, scene_extent: float, lo: int = 256, hi: int = 1024) -> int:
    """Texture side for an object by its share of the scene: a sofa gets
    1024², a mug 256²."""
    frac = float(np.clip(mesh_extent / max(scene_extent, 1e-9), 0.0, 1.0))
    side = lo * 2 ** int(round(np.log2(max(1.0, frac * 8))))
    return int(np.clip(side, lo, hi))
