"""Run only the layout stage on a photo — CPU, no SAM models — for fast tuning.

    python scripts/replay_layout.py demo_day/lr2.webp
    python scripts/replay_layout.py demo_day/lr2.webp --masks outputs/lr2_q/masks --out /tmp/lr2_layout

MoGe (ViT-L) runs on the CPU in a few seconds, then ``room_layout.estimate_layout``
and ``build_textured_room`` are applied exactly as the pipeline does (minus
object bounds). Prints floor/yaw/wall/ceiling decisions and per-plane texture
coverage, writes ``room.ply``, ``layout.json`` and the unwrapped plane
textures as PNGs so you can eyeball them. Re-uses a cached point map per
photo (``<out>/moge.npz``) so repeated runs are instant.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import room_layout as rl  # noqa: E402


def moge_pointmap(image: np.ndarray, cache: Path):
    if cache.exists():
        d = np.load(cache)
        return d["points"], d["mask"].astype(bool), d["intrinsics"]
    import torch
    from moge.model.v1 import MoGeModel
    m = MoGeModel.from_pretrained("Ruicheng/moge-vitl").eval()
    x = torch.from_numpy(image).float().permute(2, 0, 1) / 255
    t = time.time()
    with torch.no_grad():
        out = m.infer(x, force_projection=False)
    print(f"MoGe on CPU: {time.time() - t:.1f}s")
    P, M, K = out["points"].numpy(), out["mask"].numpy(), out["intrinsics"].numpy()
    np.savez_compressed(cache, points=P, mask=M, intrinsics=K)
    return P, M.astype(bool), K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--masks", type=Path, help="directory with structural_{floor,wall,ceiling}.png")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--max-side", type=int, default=1200)
    ap.add_argument("--cells", type=int, default=96)
    a = ap.parse_args()
    out = a.out or Path("outputs") / f"{a.image.stem}_layout_replay"
    out.mkdir(parents=True, exist_ok=True)

    img = Image.open(a.image).convert("RGB")
    img.thumbnail((a.max_side, a.max_side))
    image = np.asarray(img)
    P_cv, valid, K = moge_pointmap(image, out / "moge.npz")
    H, W = valid.shape
    masks = {}
    if a.masks:
        for k in ("floor", "wall", "ceiling"):
            p = a.masks / f"structural_{k}.png"
            if p.exists():
                masks[k] = np.asarray(Image.open(p).convert("L").resize((W, H), Image.NEAREST)) > 127

    Pw = rl.moge_to_world(P_cv)
    t = time.time()
    layout = rl.estimate_layout(Pw, valid, structural_masks=masks)
    print(f"layout in {time.time() - t:.2f}s (masks: {sorted(masks) or 'none'})")
    for n in layout.notes:
        print("  ", n)
    print("  walls:", layout.wall_sources)
    print("  bounds:", np.round(layout.bounds_min, 3).tolist(), np.round(layout.bounds_max, 3).tolist())
    depth = P_cv[..., 2].astype(np.float64)
    mesh, stats = rl.build_textured_room(layout, K, image, depth, cells=a.cells)
    for k, v in stats.items():
        print(f"  {k:8s} textured {v['visible_fraction']:.0%}")
    mesh.export(str(out / "room.ply"), file_type="ply")
    rl.save_layout(layout, out / "layout.json", extra={"plane_texture_stats": stats})

    # Unwrapped plane textures for eyeballing.
    mn = np.asarray(layout.bounds_min); mx = np.asarray(layout.bounds_max)
    ex, ey, ez = np.eye(3); sx, sy, sz = mx - mn
    planes = {"floor": (mn, ex, ez, sx, sz, ey), "ceiling": (np.array([mn[0], mx[1], mn[2]]), ex, ez, sx, sz, -ey),
              "z_max": (np.array([mn[0], mn[1], mx[2]]), ex, ey, sx, sy, -ez),
              "x_min": (mn, ez, ey, sz, sy, ex), "x_max": (np.array([mx[0], mn[1], mn[2]]), ez, ey, sz, sy, -ex)}
    for name, (o, au, av, su, sv, n) in planes.items():
        _, _, C, _ = rl.textured_plane(o, au, av, su, sv, n, layout.R_total, K, image, depth,
                                       (200, 200, 200), cells=a.cells)
        nv, nu = rl.textured_plane.last_grid_shape
        Image.fromarray(C.reshape(nv, nu, 3)[::-1]).resize((nu * 3, nv * 3), Image.NEAREST).save(out / f"tex_{name}.png")
    print(f"✓ wrote {out}/room.ply, layout.json, tex_*.png")
    print(json.dumps({k: v["visible_fraction"] for k, v in stats.items()}))


if __name__ == "__main__":
    main()
