"""Headless renders of a finished scene (pyrender + EGL; no display needed).

    python scripts/render_scene.py outputs/<scene> [--out renders.png] [--views photo,orbit,top,side]

Writes one contact sheet: the scene from the photo's own camera (compare it
with the photo next to it), a raised three-quarter view, a top-down view and
a side elevation. The side view is the one that shows floating, sunk or
tilted objects, which are invisible from the photo viewpoint by construction.
Reads ``scene.glb`` when present (so textured rooms show their textures),
otherwise the world-space PLYs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _look_at(eye, target, up=(0, 1, 0)):
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    if np.linalg.norm(s) < 1e-6:
        s = np.cross(f, [0, 0, 1.0])
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = s, u, -f, eye
    return M


def load_scene_meshes(scene: Path, include_room: bool = True):
    """[(name, trimesh)] in the aligned world frame."""
    import trimesh
    out = []
    glb = scene / "scene.glb"
    if glb.exists():
        sc = trimesh.load(str(glb), process=False)
        for name, geom in sc.geometry.items():
            nodes = sc.graph.geometry_nodes.get(name, [])
            T = sc.graph.get(nodes[0])[0] if nodes else np.eye(4)
            g = geom.copy()
            g.apply_transform(T)
            if not include_room and name.startswith("room"):
                continue
            out.append((name, g))
        return out
    rec = json.loads((scene / "reconstruction_results.json").read_text())
    for o in rec["objects"]:
        p = Path(o["ply_path"])
        if not p.exists():
            p = scene / "3d_models" / p.name
        if p.exists():
            out.append((f"{o['id']:03d}_{o['label']}", trimesh.load(str(p), process=False)))
    if include_room and (scene / "3d_models" / "room.ply").exists():
        out.append(("room", trimesh.load(str(scene / "3d_models" / "room.ply"), process=False)))
    return out


def render_views(scene: Path, views, size=640, include_room=True):
    import pyrender
    meshes = load_scene_meshes(scene, include_room)
    npz = np.load(scene / "3d_models" / "scene_pointmap.npz")
    K, R = npz["intrinsics"], npz["R_total"].astype(np.float64)
    H, W = (int(x) for x in npz["image_hw"])

    sc = pyrender.Scene(bg_color=[30, 30, 30, 255], ambient_light=[1.0, 1.0, 1.0])
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for name, m in meshes:
        is_room = name.startswith("room")
        is_shell = is_room and not (name.startswith("room_relief") or name.startswith("room_builtin"))
        pm = pyrender.Mesh.from_trimesh(m, smooth=True)
        for prim in pm.primitives:
            # Room faces point inward and are culled from behind, so outside
            # views look through the near walls; objects are never culled.
            prim.material.doubleSided = not is_shell
            prim.material.metallicFactor = 0.0
            prim.material.roughnessFactor = 1.0
        sc.add(pm)
        if not is_room:
            lo, hi = np.minimum(lo, m.bounds[0]), np.maximum(hi, m.bounds[1])
    c = 0.5 * (lo + hi)
    ext = float(np.linalg.norm(hi - lo))

    out = {}
    for v in views:
        if v == "photo":
            h = size
            w = int(round(size * W / H))
            fy = K[1, 1] * h
            cam = pyrender.IntrinsicsCamera(fx=K[0, 0] * w, fy=fy, cx=K[0, 2] * w, cy=K[1, 2] * h,
                                            znear=0.01, zfar=100)
            # camera-frame world -> aligned: p_a = R p_c. World frame is X left,
            # Y up, Z forward; OpenGL camera looks down -Z with X right.
            pose = np.eye(4)
            pose[:3, :3] = R @ np.diag([-1.0, 1.0, -1.0])
        else:
            w = h = size
            cam = pyrender.PerspectiveCamera(yfov=np.radians(45), znear=0.01, zfar=100)
            if v == "top":
                eye = [c[0], c[1] + 1.4 * ext, c[2]]
                pose = _look_at(eye, c, up=(0, 0, 1))
            elif v == "side":
                eye = [c[0] + 1.3 * ext, c[1] + 0.05 * ext, c[2]]
                pose = _look_at(eye, c)
            elif v == "side2":
                eye = [c[0] - 1.3 * ext, c[1] + 0.05 * ext, c[2]]
                pose = _look_at(eye, c)
            else:   # orbit: raised, off to the side of the photo camera
                eye = [c[0] - 0.75 * ext, c[1] + 0.55 * ext, c[2] - 0.75 * ext]
                pose = _look_at(eye, c)
        node = sc.add(cam, pose=pose)
        r = pyrender.OffscreenRenderer(w, h)
        color, _ = r.render(sc)
        r.delete()
        sc.remove_node(node)
        out[v] = color
    return out, (H, W)


def contact_sheet(scene: Path, renders, hw, out: Path, size=640):
    tiles = []
    rec = json.loads((scene / "reconstruction_results.json").read_text())
    src = Path(rec["metadata"]["source_image"])
    if src.exists() and "photo" in renders:
        ph = Image.open(src).convert("RGB").resize((renders["photo"].shape[1], size))
        tiles.append(("photo (input)", np.asarray(ph)))
    for k, v in renders.items():
        tiles.append((k if k != "photo" else "scene from photo camera", v))
    Wt = sum(t.shape[1] for _, t in tiles)
    sheet = Image.new("RGB", (Wt, size), (0, 0, 0))
    x = 0
    d = ImageDraw.Draw(sheet)
    for name, t in tiles:
        sheet.paste(Image.fromarray(t), (x, 0))
        d.text((x + 6, 6), name, fill=(255, 255, 0))
        x += t.shape[1]
    sheet.save(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--views", default="photo,orbit,top,side")
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--no-room", action="store_true")
    a = ap.parse_args()
    renders, hw = render_views(a.scene, a.views.split(","), a.size, not a.no_room)
    out = contact_sheet(a.scene, renders, hw, a.out or a.scene / "renders.png", a.size)
    print(f"✓ {out}")


if __name__ == "__main__":
    main()
