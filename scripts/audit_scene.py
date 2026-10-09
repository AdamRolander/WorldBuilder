"""Audit a finished scene against the photo it came from (no GPU, no GL).

    python scripts/audit_scene.py outputs/<scene> [--out audit.png] [--json audit.json]

Every placed object is projected back through the recovered camera and
compared with two independent pieces of evidence the pipeline already has:

* its SAM 3 mask (``reproj_iou``: silhouette of the placed mesh vs. the mask;
  ``pixel_shift``: distance between the two centroids as a fraction of the
  image diagonal), and
* the scene point map under that mask (``depth_ratio``: object depth over
  observed depth, 1.0 = agrees; ``offset``: 3D distance between the object
  and the observed surface, in units of the object's own size).

It also lists same-label pairs whose boxes overlap heavily (duplicates) and
objects whose bottom floats above or sinks below whatever is under them.
The overlay image draws mask outlines (green) and placed-mesh outlines
(red): where the two disagree, placement moved the object away from where
the photo says it is.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import room_layout as rl  # noqa: E402


def _silhouette(uv: np.ndarray, z: np.ndarray, hw, faces: Optional[np.ndarray] = None) -> np.ndarray:
    """Binary silhouette of projected vertices on an (h, w) grid. Dense
    meshes are splatted per vertex and closed morphologically; that is
    within a pixel or two of a real rasteriser at audit resolution."""
    from scipy import ndimage
    h, w = hw
    out = np.zeros((h, w), bool)
    ok = (z > 1e-6) & np.isfinite(uv).all(1)
    u = np.round(uv[ok, 0]).astype(int)
    v = np.round(uv[ok, 1]).astype(int)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    out[v[inside], u[inside]] = True
    if out.any():
        out = ndimage.binary_closing(out, iterations=3, border_value=0)
        out = ndimage.binary_fill_holes(out)
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def _box_iou3(a_min, a_max, b_min, b_max) -> float:
    lo = np.maximum(a_min, b_min)
    hi = np.minimum(a_max, b_max)
    inter = float(np.prod(np.clip(hi - lo, 0, None)))
    va = float(np.prod(a_max - a_min))
    vb = float(np.prod(b_max - b_min))
    return inter / max(1e-12, va + vb - inter)


def audit_scene(scene: Path, max_side: int = 720) -> Dict:
    import trimesh
    rec = json.loads((scene / "reconstruction_results.json").read_text())
    objects = rec["objects"]
    npz = np.load(scene / "3d_models" / "scene_pointmap.npz")
    K = npz["intrinsics"]
    R = npz["R_total"].astype(np.float64)
    H, W = (int(x) for x in npz["image_hw"])
    s = min(1.0, max_side / max(H, W))
    h, w = int(round(H * s)), int(round(W * s))
    diag = float(np.hypot(h, w))

    # Observed surface per pixel (aligned frame), from the saved point cloud.
    P = npz["points"].astype(np.float64)
    pu, pv, pz = rl.project_world_to_pixels(P @ R, K, w, h)
    pix = np.stack([np.clip(np.round(pv).astype(int), 0, h - 1), np.clip(np.round(pu).astype(int), 0, w - 1)], 1)

    rows: List[Dict] = []
    sil_all, mask_all = {}, {}
    boxes = {}
    for o in objects:
        # Prefer the file inside this scene directory: a copied scene keeps the
        # original's paths in its JSON and must not be audited against them.
        p = scene / "3d_models" / Path(o["ply_path"]).name
        if not p.exists():
            p = Path(o["ply_path"])
        if not p.exists():
            continue
        V = np.asarray(trimesh.load(str(p), process=False).vertices, dtype=np.float64)
        u, v, z = rl.project_world_to_pixels(V @ R, K, w, h)
        sil = _silhouette(np.stack([u, v], 1), z, (h, w))
        mp = scene / "masks" / Path(o.get("mask_path") or "x").name
        if not mp.exists():
            mp = Path(o.get("mask_path") or "")
        mask = None
        if mp.exists():
            mask = np.asarray(Image.open(mp).convert("L").resize((w, h), Image.NEAREST)) > 127
        row = {"id": o["id"], "label": o["label"], "supported_by": o.get("supported_by"),
               "contained_in": o.get("contained_in")}
        mn, mx = V.min(0), V.max(0)
        boxes[o["id"]] = (mn, mx)
        size = float(np.linalg.norm(mx - mn))
        if mask is not None and mask.any():
            row["reproj_iou"] = round(_iou(sil, mask), 3)
            if sil.any():
                cs = np.argwhere(sil).mean(0)
                cm = np.argwhere(mask).mean(0)
                row["pixel_shift"] = round(float(np.linalg.norm(cs - cm)) / diag, 4)
            else:
                row["pixel_shift"] = None        # object projects outside the frame
            sel = mask[pix[:, 0], pix[:, 1]]
            if sel.sum() >= 5:
                obs = P[sel]
                obs_c = np.median(obs, axis=0)
                # Compare the *visible* face of the object with the observed
                # surface: nearest 30 % of vertices by depth.
                near = V[z <= np.percentile(z, 30)]
                row["depth_ratio"] = round(float(np.median(z[z <= np.percentile(z, 30)]) /
                                                 max(1e-9, np.median(pz[sel]))), 3)
                row["offset"] = round(float(np.linalg.norm(np.median(near, axis=0) - obs_c)) / max(size, 1e-9), 3)
                row["offset_y"] = round(float(np.median(near, axis=0)[1] - obs_c[1]) / max(size, 1e-9), 3)
            mask_all[o["id"]] = mask
        sil_all[o["id"]] = sil
        rows.append(row)

    dups = []
    ids = list(boxes)
    labels = {o["id"]: o["label"] for o in objects}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            iou3 = _box_iou3(*boxes[a], *boxes[b])
            if iou3 > 0.3:
                dups.append({"a": a, "b": b, "labels": [labels[a], labels[b]], "box_iou": round(iou3, 3),
                             "mask_iou": round(_iou(mask_all[a], mask_all[b]), 3)
                             if a in mask_all and b in mask_all else None})

    def _mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(float(np.mean(vals)), 3) if vals else None

    summary = {
        "scene": scene.name, "objects": len(rows),
        "mean_reproj_iou": _mean("reproj_iou"),
        "median_reproj_iou": round(float(np.median([r["reproj_iou"] for r in rows if "reproj_iou" in r])), 3)
        if any("reproj_iou" in r for r in rows) else None,
        "low_iou(<0.3)": sum(1 for r in rows if r.get("reproj_iou", 1) < 0.3),
        "mean_pixel_shift": _mean("pixel_shift"),
        "mean_offset": _mean("offset"),
        "overlapping_pairs": len(dups),
        "same_label_overlaps": sum(1 for d in dups if d["labels"][0] == d["labels"][1]),
    }
    return {"summary": summary, "objects": rows, "overlaps": dups,
            "_sil": sil_all, "_mask": mask_all, "_hw": (h, w)}


def draw_overlay(scene: Path, audit: Dict, out: Path):
    from scipy import ndimage
    rec = json.loads((scene / "reconstruction_results.json").read_text())
    src = Path(rec["metadata"]["source_image"])
    h, w = audit["_hw"]
    if src.exists():
        img = np.asarray(Image.open(src).convert("RGB").resize((w, h), Image.BILINEAR)).astype(np.float32)
    else:
        img = np.full((h, w, 3), 60, np.float32)
    img = 0.55 * img + 0.45 * 255

    def _edge(m):
        return m & ~ndimage.binary_erosion(m, iterations=2)

    for m in audit["_mask"].values():
        img[_edge(m)] = (0, 160, 0)
    for sil in audit["_sil"].values():
        img[_edge(sil)] = (220, 0, 0)
    im = Image.fromarray(img.astype(np.uint8))
    from PIL import ImageDraw
    d = ImageDraw.Draw(im)
    for r in audit["objects"]:
        sil = audit["_sil"].get(r["id"])
        if sil is None or not sil.any():
            continue
        cy, cx = np.argwhere(sil).mean(0)
        d.text((cx, cy), str(r["id"]), fill=(140, 0, 0))
    im.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, help="overlay image (single scene only; default <scene>/audit.png)")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--quiet", action="store_true", help="summary line only")
    a = ap.parse_args()
    all_out = []
    for scene in a.scene:
        audit = audit_scene(scene)
        draw_overlay(scene, audit, a.out if (a.out and len(a.scene) == 1) else scene / "audit.png")
        if not a.quiet:
            print(f"{'id':>3} {'label':22s} {'iou':>5} {'shift':>6} {'depth':>6} {'off':>5} {'offY':>6}  rel")
            for r in audit["objects"]:
                rel = (f"on {r['supported_by']}" if r.get("supported_by") else
                       f"in {r['contained_in']}" if r.get("contained_in") else "")
                print(f"{r['id']:>3} {r['label'][:22]:22s} {r.get('reproj_iou', float('nan')):5.2f} "
                      f"{(r.get('pixel_shift') if r.get('pixel_shift') is not None else float('nan')):6.3f} "
                      f"{r.get('depth_ratio', float('nan')):6.2f} {r.get('offset', float('nan')):5.2f} "
                      f"{r.get('offset_y', float('nan')):6.2f}  {rel}")
            for dpl in audit["overlaps"]:
                print(f"  overlap {dpl['a']}({dpl['labels'][0]}) / {dpl['b']}({dpl['labels'][1]}): "
                      f"box IoU {dpl['box_iou']}, mask IoU {dpl['mask_iou']}")
        print(json.dumps(audit["summary"]))
        all_out.append({k: v for k, v in audit.items() if not k.startswith("_")})
    if a.json:
        a.json.write_text(json.dumps(all_out if len(all_out) > 1 else all_out[0], indent=2))


if __name__ == "__main__":
    main()
