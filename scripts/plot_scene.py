"""Plan-view diagnostic of a finished scene (no GPU, no GL).

    python scripts/plot_scene.py outputs/<scene> [--out plan.png]

Draws, looking down the Y axis: the room rectangle from layout.json (solid
edges = detected walls, dashed = extent fallback), every object's footprint
with its label, support edges (what rests on what), the camera at the origin
with its viewing direction, and a sample of the photo point cloud. It is the
quickest way to sanity-check placement without opening the viewer, and it
renders headless for docs and bug reports.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--no-points", action="store_true")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import trimesh

    rec = json.loads((a.scene / "reconstruction_results.json").read_text())
    objects = rec["objects"]
    layout = None
    lp = a.scene / "3d_models" / "layout.json"
    if lp.exists():
        layout = json.loads(lp.read_text())

    fig, ax = plt.subplots(figsize=(9, 9))
    # Room
    if layout and "bounds_min" in layout:
        mn, mx = layout["bounds_min"], layout["bounds_max"]
        src = layout.get("wall_sources", {})
        sides = {"z_min": ((mn[0], mn[2]), (mx[0], mn[2])), "z_max": ((mn[0], mx[2]), (mx[0], mx[2])),
                 "x_min": ((mn[0], mn[2]), (mn[0], mx[2])), "x_max": ((mx[0], mn[2]), (mx[0], mx[2]))}
        for side, (p, q) in sides.items():
            detected = str(src.get(side, "")).startswith("wall")
            ax.plot([p[0], q[0]], [p[1], q[1]], "k-" if detected else "k--", lw=2.5 if detected else 1.2)
            ax.annotate(f"{side}: {src.get(side, '?')}", ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2), fontsize=7,
                        color="k", ha="center")
        ax.set_title(f"{a.scene.name} — plan view ({'; '.join(layout.get('notes', [])[:2])})", fontsize=9)
    # Points
    npz = a.scene / "3d_models" / "scene_pointmap.npz"
    if npz.exists() and not a.no_points:
        d = np.load(npz)
        P, C = d["points"], d["colors"]
        if len(P) > 40000:
            idx = np.random.default_rng(0).choice(len(P), 40000, replace=False)
            P, C = P[idx], C[idx]
        ax.scatter(P[:, 0], P[:, 2], s=0.3, c=C[:, :3] / 255.0, alpha=0.5, linewidths=0)
    # Objects
    by_id = {}
    for o in objects:
        p = Path(o["ply_path"])
        if not p.is_absolute():
            p = a.scene / "3d_models" / p.name
        if not p.exists():
            continue
        v = np.asarray(trimesh.load(str(p), process=False).vertices)
        mn, mx = v.min(0), v.max(0)
        by_id[o["id"]] = (mn, mx)
        ax.add_patch(plt.Rectangle((mn[0], mn[2]), mx[0] - mn[0], mx[2] - mn[2], fill=True, alpha=0.25,
                                   ec="tab:blue", fc="tab:blue", lw=1))
        ax.annotate(f"{o['id']} {o['label']}", ((mn[0] + mx[0]) / 2, (mn[2] + mx[2]) / 2), fontsize=6, ha="center")
    for o in objects:
        s = o.get("supported_by")
        if s in by_id and o["id"] in by_id:
            a0 = by_id[o["id"]]; b0 = by_id[s]
            ax.annotate("", xy=((b0[0][0] + b0[1][0]) / 2, (b0[0][2] + b0[1][2]) / 2),
                        xytext=((a0[0][0] + a0[1][0]) / 2, (a0[0][2] + a0[1][2]) / 2),
                        arrowprops=dict(arrowstyle="->", color="tab:green", lw=0.8, alpha=0.8))
    # Camera
    ax.plot(0, 0, "r^", ms=10)
    ax.annotate("camera", (0, 0), fontsize=8, color="r", xytext=(4, -12), textcoords="offset points")
    ax.arrow(0, 0, 0, 0.4, head_width=0.06, color="r", length_includes_head=True)
    ax.set_aspect("equal")
    ax.invert_xaxis()   # world +X is camera-left; flip so the plan reads like the photo
    ax.set_xlabel("x (scene units; camera-left is +x, shown flipped)")
    ax.set_ylabel("z (depth from camera)")
    ax.grid(alpha=0.2)
    out = a.out or (a.scene / "plan_view.png")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"✓ {out}")


if __name__ == "__main__":
    main()
