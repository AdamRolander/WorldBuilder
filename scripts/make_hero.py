"""Before/after animation of one scene: the photo next to the scene seen from a moving camera.

    python scripts/make_hero.py outputs/<scene> --photo photo.jpg --out docs/assets/hero.gif [--frames 48] [--height 360]

The camera starts at the photo's own viewpoint, swings to one side of the room
and to the other, and returns, so the first and last frames match the photo.
Writes a GIF small enough for a README (about 6 MB at the defaults).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.render_scene import _look_at, load_scene_meshes  # noqa: E402


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def camera_path(R, target, n, swing_deg=28.0, lift=0.25):
    """Poses orbiting ``target`` from the photo camera at the origin and back."""
    eye0 = np.zeros(3)
    look0 = R @ np.diag([-1.0, 1.0, -1.0])
    poses = []
    for i in range(n):
        t = i / n
        a = np.radians(swing_deg) * np.sin(2 * np.pi * t)          # 0 → +swing → -swing → 0
        up = lift * np.sin(np.pi * t) ** 2                           # rise a little mid-path
        off = eye0 - target
        eye = target + _rot_y(a) @ off + np.array([0, up * np.linalg.norm(off), 0])
        P = _look_at(eye, target)
        w = min(1.0, 4 * min(t, 1 - t))                              # photo orientation at the ends
        if w < 1.0:
            # blend from the photo's own orientation into the look-at one
            from scipy.spatial.transform import Rotation, Slerp
            rots = Rotation.from_matrix(np.stack([look0, P[:3, :3]]))
            P[:3, :3] = Slerp([0, 1], rots)(w).as_matrix()
        poses.append(P)
    return poses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", type=Path)
    ap.add_argument("--photo", type=Path, required=True, help="the photo the scene was made from")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--swing", type=float, default=28.0, help="degrees of orbit to each side")
    args = ap.parse_args()

    import pyrender

    npz = np.load(args.scene / "3d_models" / "scene_pointmap.npz")
    K, R = npz["intrinsics"], npz["R_total"].astype(np.float64)
    H, W = (int(x) for x in npz["image_hw"])
    h = args.height
    w = int(round(h * W / H)) // 2 * 2

    sc = pyrender.Scene(bg_color=[30, 30, 30, 255], ambient_light=[1.0, 1.0, 1.0])
    pts = []
    for name, m in load_scene_meshes(args.scene, True):
        is_shell = name.startswith("room") and not (name.startswith("room_relief") or name.startswith("room_builtin"))
        pm = pyrender.Mesh.from_trimesh(m, smooth=True)
        for prim in pm.primitives:
            prim.material.doubleSided = not is_shell
            prim.material.metallicFactor = 0.0
            prim.material.roughnessFactor = 1.0
        sc.add(pm)
        if not name.startswith("room"):
            pts.append(m.vertices)
    # orbit around the middle of the furniture, at the photo's own distance
    target = np.median(np.concatenate(pts), axis=0)

    cam = pyrender.IntrinsicsCamera(fx=K[0, 0] * w, fy=K[1, 1] * h, cx=K[0, 2] * w, cy=K[1, 2] * h,
                                    znear=0.01, zfar=100)
    node = sc.add(cam, pose=np.eye(4))
    r = pyrender.OffscreenRenderer(w, h)
    photo = Image.open(args.photo).convert("RGB").resize((w, h), Image.LANCZOS)

    gap = 8
    frames = []
    for P in camera_path(R, target, args.frames, args.swing):
        sc.set_pose(node, P)
        color, _ = r.render(sc, flags=pyrender.RenderFlags.FLAT)
        canvas = Image.new("RGB", (2 * w + gap, h), (255, 255, 255))
        canvas.paste(photo, (0, 0))
        canvas.paste(Image.fromarray(color), (w + gap, 0))
        frames.append(canvas)
    r.delete()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pal = [f.quantize(colors=160, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    pal[0].save(args.out, save_all=True, append_images=pal[1:], duration=int(1000 / args.fps), loop=0,
                optimize=True)
    print(f"{args.out}: {args.out.stat().st_size / 1e6:.1f} MB, {len(frames)} frames")


if __name__ == "__main__":
    main()
