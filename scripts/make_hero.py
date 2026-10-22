"""Before/after animation for the README: each photo next to the scene built from it.

    python scripts/make_hero.py --scene outputs/<a> a.jpg --scene outputs/<b> b.jpg \\
        --out docs/assets/hero.webp [--stills docs/assets/examples] [--edits]

For every scene the camera starts at the photo's own viewpoint, swings to one
side of the room and to the other, and returns. Meanwhile the room pops in
piece by piece, then each object one at a time (its mask lights up on the
photo as it appears), a few of the largest objects are switched off and on
again to show they are separate, and everything pops out before the next
scene. ``.webp`` output is several times smaller than ``.gif`` for the same
frames; GitHub shows both.

``--stills`` also writes one image per scene (photo, the scene from the
photo's viewpoint, the scene from the side) for a gallery table.
``--edits`` applies the scene's ``edits.json`` (fixes made in the browser
viewer); without it the pipeline's own output is shown.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.render_scene import _look_at, load_scene_meshes  # noqa: E402

BG = (24, 24, 27)
ACCENT = np.array([255, 176, 46], float)
_ID = re.compile(r"^(\d{3})_")


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def camera_pose(R, target, t, swing_deg=28.0, lift=0.25):
    """Pose at time ``t`` in [0, 1) of an orbit around ``target`` that starts and ends at
    the photo camera (the origin): 0 → +swing → -swing → 0."""
    from scipy.spatial.transform import Rotation, Slerp
    look0 = R @ np.diag([-1.0, 1.0, -1.0])
    a = np.radians(swing_deg) * np.sin(2 * np.pi * t)
    up = lift * np.sin(np.pi * t) ** 2
    off = -target
    eye = target + _rot_y(a) @ off + np.array([0, up * np.linalg.norm(off), 0])
    P = _look_at(eye, target)
    w = min(1.0, 4 * min(t, 1 - t))                  # the photo's own orientation at the ends
    if w < 1.0:
        rots = Rotation.from_matrix(np.stack([look0, P[:3, :3]]))
        P[:3, :3] = Slerp([0, 1], rots)(w).as_matrix()
    return P


def _pop(x):
    """0 → 1 with a small overshoot, for an object scaling into place."""
    x = float(np.clip(x, 0.0, 1.0))
    c = 1.70158
    return 1 + (c + 1) * (x - 1) ** 3 + c * (x - 1) ** 2


def _meshes(scene: Path, edits: bool):
    out = load_scene_meshes(scene, True)
    if not edits:
        return out
    from src import scene_edits
    e = scene_edits.load(scene)
    if e is None:
        return out
    kept = []
    for name, m in out:
        if name in set(e["hidden_nodes"]):
            continue
        mid = _ID.match(name)
        oe = scene_edits.for_object(e, int(mid.group(1))) if mid else None
        if oe is not None:
            if oe["hidden"]:
                continue
            m = m.copy()
            m.vertices = scene_edits.apply(m.vertices, oe)
        kept.append((name, m))
    return kept


def _masks(scene: Path, size):
    """{object id: boolean mask at ``size``} from the scene's segmentation masks."""
    out = {}
    for p in sorted((scene / "masks").glob("[0-9][0-9][0-9]_*.png")):
        m = Image.open(p).convert("L").resize(size, Image.NEAREST)
        out[int(p.name[:3])] = np.asarray(m) > 127
    return out


def _font(px):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", px)
    except OSError:
        return ImageFont.load_default()


def _label(img: Image.Image, text: str, px: int):
    d = ImageDraw.Draw(img, "RGBA")
    f = _font(px)
    x0, y0, x1, y1 = d.textbbox((0, 0), text, font=f)
    pad = f.size // 3
    d.rounded_rectangle((8, 8, 8 + x1 - x0 + 2 * pad, 8 + y1 - y0 + 2 * pad), radius=pad,
                        fill=(0, 0, 0, 150))
    d.text((8 + pad - x0, 8 + pad - y0), text, font=f, fill=(255, 255, 255, 235))


class SceneClip:
    """One scene's renderer: the meshes as pyrender nodes, each scaled about its own centre."""

    def __init__(self, scene: Path, photo: Path, box, edits: bool):
        import pyrender
        npz = np.load(scene / "3d_models" / "scene_pointmap.npz")
        K, self.R = npz["intrinsics"], npz["R_total"].astype(np.float64)
        H, W = (int(x) for x in npz["image_hw"])
        s = min(box[0] / W, box[1] / H)
        self.w, self.h = int(W * s) // 2 * 2, int(H * s) // 2 * 2
        self.sc = pyrender.Scene(bg_color=[*BG, 255], ambient_light=[1.0, 1.0, 1.0])
        self.items = []                      # (kind, name, node, centre)
        pts = []
        for name, m in _meshes(scene, edits):
            shell = name.startswith("room") and not name.startswith(("room_relief", "room_builtin"))
            pm = pyrender.Mesh.from_trimesh(m, smooth=True)
            for prim in pm.primitives:
                prim.material.doubleSided = not shell
                prim.material.metallicFactor = 0.0
                prim.material.roughnessFactor = 1.0
            node = self.sc.add(pm)
            kind = "shell" if shell else ("extra" if name.startswith("room") else "object")
            self.items.append((kind, name, node, m.bounds.mean(axis=0), np.ptp(m.bounds, axis=0)))
            if kind == "object":
                pts.append(m.vertices)
        self.target = np.median(np.concatenate(pts), axis=0)
        cam = pyrender.IntrinsicsCamera(fx=K[0, 0] * self.w, fy=K[1, 1] * self.h,
                                        cx=K[0, 2] * self.w, cy=K[1, 2] * self.h, znear=0.01, zfar=100)
        self.cam = self.sc.add(cam, pose=np.eye(4))
        self.photo = np.asarray(Image.open(photo).convert("RGB").resize((self.w, self.h), Image.LANCZOS),
                                float)
        self.masks = _masks(scene, (self.w, self.h))
        self.r = pyrender.OffscreenRenderer(self.w, self.h)

        # pop order: floor, walls, ceiling, then the photo's relief and built-ins, then the
        # objects from left to right as the photo shows them (sorted by their camera-frame x)
        def u(c):
            return -(np.linalg.inv(self.R) @ c)[0] / max((np.linalg.inv(self.R) @ c)[2], 1e-3)
        shell_rank = {"room_floor": 0, "room_ceiling": 2}
        self.shell = sorted((it for it in self.items if it[0] == "shell"),
                            key=lambda it: shell_rank.get(it[1], 1))
        self.extra = [it for it in self.items if it[0] == "extra"]
        self.objects = sorted((it for it in self.items if it[0] == "object"), key=lambda it: u(it[3]))
        # the objects switched off and on mid-orbit: the three biggest
        self.toggled = sorted(self.objects, key=lambda it: -np.prod(np.maximum(it[4], 1e-3)))[:3]

    def scales(self, t):
        """{node: scale} at time t, and the ids of objects whose mask should glow."""
        out, glow = {}, {}

        def ramp(items, t0, t1, out_t0, out_t1, dur):
            n = max(len(items), 1)
            for k, it in enumerate(items):
                a = t0 + (t1 - t0) * k / n
                b = out_t0 + (out_t1 - out_t0) * (n - 1 - k) / n
                s_in = _pop((t - a) / dur)
                s_out = _pop((b + dur - t) / dur) if t > b else 1.0
                s = min(s_in, s_out) if t >= a else 0.0
                out[it[2]] = s
                mid = _ID.match(it[1])
                if mid and 0 < t - a < 2.5 * dur:
                    glow[int(mid.group(1))] = 1 - (t - a) / (2.5 * dur)
        ramp(self.shell, 0.00, 0.06, 0.95, 0.985, 0.025)
        ramp(self.extra, 0.05, 0.10, 0.93, 0.95, 0.025)
        ramp(self.objects, 0.09, 0.40, 0.84, 0.93, 0.03)
        for k, it in enumerate(self.toggled):            # off and back on, one after another
            a = 0.47 + 0.11 * k
            x = (t - a) / 0.09
            if 0 <= x <= 1:
                if x < 0.3:
                    s = np.cos(np.pi / 2 * x / 0.3) ** 2         # shrink away
                elif x < 0.6:
                    s = 0.0
                else:
                    s = _pop((x - 0.6) / 0.4)                    # and back
                out[it[2]] = min(out.get(it[2], 1.0), s)
                glow[int(_ID.match(it[1]).group(1))] = 1.0
        return out, glow

    def render(self, t, swing, scales=None):
        import pyrender
        scales, glow = self.scales(t) if scales is None else (scales, {})
        for _kind, _name, node, c, _ in self.items:
            s = max(scales.get(node, 1.0), 1e-4)
            M = np.eye(4)
            M[:3, :3] *= s
            M[:3, 3] = c - s * c
            node.matrix = M
        self.sc.set_pose(self.cam, camera_pose(self.R, self.target, t, swing))
        color, _ = self.r.render(self.sc, flags=pyrender.RenderFlags.FLAT)
        photo = self.photo.copy()
        for oid, g in glow.items():
            m = self.masks.get(oid)
            if m is not None and g > 0:
                photo[m] = photo[m] * (1 - 0.55 * g) + ACCENT * 0.55 * g
        return Image.fromarray(photo.astype(np.uint8)), Image.fromarray(color)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene", nargs=2, action="append", required=True, metavar=("DIR", "PHOTO"))
    ap.add_argument("--out", type=Path, required=True, help=".webp (smaller) or .gif")
    ap.add_argument("--frames", type=int, default=84, help="frames per scene")
    ap.add_argument("--fps", type=int, default=14)
    ap.add_argument("--width", type=int, default=1000, help="canvas width")
    ap.add_argument("--height", type=int, default=420, help="canvas height")
    ap.add_argument("--swing", type=float, default=30.0, help="degrees of orbit to each side")
    ap.add_argument("--quality", type=int, default=72, help="WebP quality")
    ap.add_argument("--stills", type=Path, help="folder for one still image per scene")
    ap.add_argument("--edits", action="store_true", help="apply each scene's edits.json")
    args = ap.parse_args()

    gap = 10
    box = ((args.width - 3 * gap) // 2, args.height - 2 * gap)
    px = max(12, args.height // 28)
    frames = []
    for scene, photo in args.scene:
        clip = SceneClip(Path(scene), Path(photo), box, args.edits)
        x0 = (args.width - 2 * clip.w - gap) // 2
        y0 = (args.height - clip.h) // 2
        for i in range(args.frames):
            left, right = clip.render(i / args.frames, args.swing)
            _label(left, "Photo", px)
            _label(right, "3D scene", px)
            canvas = Image.new("RGB", (args.width, args.height), BG)
            canvas.paste(left, (x0, y0))
            canvas.paste(right, (x0 + clip.w + gap, y0))
            frames.append(canvas)
        if args.stills:
            args.stills.mkdir(parents=True, exist_ok=True)
            full = {it[2]: 1.0 for it in clip.items}
            views = [clip.render(0.0, args.swing, full), clip.render(0.25, args.swing * 1.3, full)]
            panels = [views[0][0], views[0][1], views[1][1]]
            for p, text in zip(panels, ("Photo", "3D scene, photo view", "3D scene, from the side"), strict=True):
                _label(p, text, px)
            st = Image.new("RGB", (3 * clip.w + 2 * gap, clip.h), (255, 255, 255))
            for k, p in enumerate(panels):
                st.paste(p, (k * (clip.w + gap), 0))
            dest = args.stills / f"{Path(photo).stem}.jpg"
            st.save(dest, quality=88)
            print(f"{dest}: {dest.stat().st_size / 1e6:.1f} MB")
        clip.r.delete()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    ms = int(1000 / args.fps)
    if args.out.suffix.lower() == ".webp":
        frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=ms, loop=0,
                       quality=args.quality, method=6)
    else:
        pal = [f.quantize(colors=160, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
               for f in frames]
        pal[0].save(args.out, save_all=True, append_images=pal[1:], duration=ms, loop=0, optimize=True)
    print(f"{args.out}: {args.out.stat().st_size / 1e6:.1f} MB, {len(frames)} frames")


if __name__ == "__main__":
    main()
