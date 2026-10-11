"""One table for a run: how good, how fast, how much memory, how big.

    python scripts/benchmark.py outputs/<run>                       # every scene in a run folder
    python scripts/benchmark.py outputs/<run> --baseline outputs/<earlier run>
    python scripts/benchmark.py outputs/<run>/<scene> --appearance  # + re-render the photo's view

A *run* is a folder of scene directories produced by ``main.py`` (or
replayed by ``scripts/replay_assembly.py``). Per scene this collects

* **placement** (``scripts/audit_scene.py``): mean / median silhouette IoU
  between each placed mesh and its mask, how many objects score below 0.3,
  overlapping pairs;
* **speed** (``metrics.json``, written by the pipeline): seconds per phase,
  time until the first object exists, SAM 3D seconds per object (median and
  90th percentile), stage 4 (CPU) and export time;
* **memory**: peak GPU memory on the device while each phase ran (includes
  the local VLM's process) and the peak this process allocated;
* **size**: ``scene.glb``, ``scene_lite.glb``, ``room.glb`` in MB;
* **appearance** (``--appearance``, needs pyrender + EGL): the scene
  rendered from the photo's own camera and compared with the photo —
  coverage (share of the frame the scene explains), mean colour error, and
  sharpness inside and outside the object masks relative to the photo
  (1.0 = as sharp as the photo; objects were 0.3-0.5 before photo
  re-projection).

It writes ``benchmark.json`` and ``benchmark.md`` into the run folder.
With ``--baseline`` the table shows the change against another run of the
same photos and the exit status is 1 if a guarded metric got worse by more
than its tolerance (``--tolerances``), so it can gate a change.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# metric -> (direction, tolerance). direction +1: higher is better.
# Tolerances are absolute for scores and relative (fraction) for times/sizes.
GUARDS = {
    "mean_iou": (+1, 0.02, "abs"),
    "low_iou": (-1, 2, "abs"),
    "stage4_sec": (-1, 0.5, "rel"),
    "per_object_p50_sec": (-1, 0.25, "rel"),
    "vram_device_peak_gb": (-1, 1.0, "abs"),
    "lite_mb": (-1, 0.5, "rel"),
    "object_sharpness": (+1, 0.05, "abs"),
    "color_error": (-1, 0.01, "abs"),
}

COLUMNS = [  # key, header, format
    ("objects", "objects", "{:d}"),
    ("mean_iou", "IoU mean", "{:.3f}"),
    ("median_iou", "IoU med", "{:.3f}"),
    ("low_iou", "IoU<0.3", "{:d}"),
    ("detect_sec", "detect s", "{:.0f}"),
    ("segment_sec", "segment s", "{:.0f}"),
    ("first_object_sec", "first obj s", "{:.0f}"),
    ("per_object_p50_sec", "obj p50 s", "{:.1f}"),
    ("per_object_p90_sec", "obj p90 s", "{:.1f}"),
    ("reconstruct_sec", "SAM 3D s", "{:.0f}"),
    ("stage4_sec", "stage 4 s", "{:.0f}"),
    ("export_sec", "exports s", "{:.0f}"),
    ("total_sec", "total s", "{:.0f}"),
    ("vram_device_peak_gb", "VRAM peak GB", "{:.1f}"),
    ("vram_process_peak_gb", "this proc GB", "{:.1f}"),
    ("lite_mb", "lite MB", "{:.1f}"),
    ("coverage", "coverage", "{:.2f}"),
    ("color_error", "colour err", "{:.3f}"),
    ("object_sharpness", "obj sharp", "{:.2f}"),
    ("room_sharpness", "room sharp", "{:.2f}"),
]


def find_scenes(paths: List[Path]) -> List[Path]:
    out = []
    for p in paths:
        if (p / "reconstruction_results.json").exists():
            out.append(p)
        else:
            out += sorted(d for d in p.iterdir() if d.is_dir() and (d / "reconstruction_results.json").exists())
    return out


def _grad(gray: np.ndarray) -> np.ndarray:
    from scipy import ndimage
    g = gray.astype(np.float32)
    return np.hypot(ndimage.sobel(g, axis=0), ndimage.sobel(g, axis=1))


def appearance_metrics(scene: Path, size: int = 512) -> Dict:
    """Render the scene from the photo's camera and compare with the photo."""
    from render_scene import render_views
    rec = json.loads((scene / "reconstruction_results.json").read_text())
    src = Path(rec["metadata"]["source_image"])
    if not src.exists():
        return {}
    depth: Dict = {}
    renders, _ = render_views(scene, ["photo"], size=size, depth_out=depth)
    r = renders["photo"][..., :3].astype(np.float32)
    h, w = r.shape[:2]
    photo = np.asarray(Image.open(src).convert("RGB").resize((w, h), Image.LANCZOS)).astype(np.float32)
    covered = depth["photo"] > 0                                   # some geometry along the pixel's ray
    obj = np.zeros((h, w), bool)
    for o in rec["objects"]:
        mp = scene / "masks" / Path(o.get("mask_path") or "x").name
        if mp.exists():
            obj |= np.asarray(Image.open(mp).convert("L").resize((w, h), Image.NEAREST)) > 127
    out = {"coverage": round(float(covered.mean()), 3)}
    if covered.any():
        out["color_error"] = round(float(np.abs(r - photo)[covered].mean() / 255.0), 4)
    gr, gp = _grad(r.mean(-1)), _grad(photo.mean(-1))
    for name, region in (("object_sharpness", obj & covered), ("room_sharpness", ~obj & covered)):
        if region.sum() > 100 and gp[region].mean() > 1e-6:
            out[name] = round(float(gr[region].mean() / gp[region].mean()), 3)
    return out


def scene_row(scene: Path, appearance: bool = False, reaudit: bool = True) -> Dict:
    row: Dict = {"scene": scene.name}
    summary = None
    if reaudit:
        try:
            from audit_scene import audit_scene
            summary = audit_scene(scene)["summary"]
        except Exception as e:                                   # a scene without a point map, etc.
            row["audit_error"] = str(e)[:120]
    if summary is None and (scene / "audit.json").exists():
        summary = json.loads((scene / "audit.json").read_text()).get("summary")
    if summary:
        row.update(objects=summary.get("objects"), mean_iou=summary.get("mean_reproj_iou"),
                   median_iou=summary.get("median_reproj_iou"), low_iou=summary.get("low_iou(<0.3)"),
                   overlapping_pairs=summary.get("overlapping_pairs"))
    mp = scene / "metrics.json"
    if mp.exists():
        m = json.loads(mp.read_text())
        sec = m.get("seconds", {})
        row.update(detect_sec=sec.get("detect"), segment_sec=sec.get("segment"),
                   reconstruct_sec=sec.get("reconstruct"),
                   first_object_sec=m.get("marks_sec", {}).get("first_object"))
        s4 = [v for k, v in sec.items() if k.startswith("assemble.")]
        ex = [v for k, v in sec.items() if k.startswith("finalize.")]
        row["stage4_sec"] = round(sum(s4), 2) if s4 else None
        row["export_sec"] = round(sum(ex), 2) if ex else None
        row["phases_sec"] = sec
        if "reconstruct" in sec:
            row["total_sec"] = m.get("wall_sec")
        po = m.get("seconds_per_object", {})
        row.update(per_object_p50_sec=po.get("p50"), per_object_p90_sec=po.get("p90"))
        row["vram_device_peak_gb"] = m.get("vram_device_peak_gb", {}).get("total")
        pp = m.get("vram_process_peak_gb", {})
        row["vram_process_peak_gb"] = max(pp.values()) if pp else None
        row["vram_by_phase_gb"] = m.get("vram_device_peak_gb")
        row["lite_mb"] = m.get("size_mb", {}).get("scene_lite.glb")
        row["full_mb"] = m.get("size_mb", {}).get("scene.glb")
    else:                                                        # scenes from before metrics.json
        rec = json.loads((scene / "reconstruction_results.json").read_text())
        t = rec.get("metadata", {}).get("timings_sec", {})
        row.update(detect_sec=t.get("detect"), segment_sec=t.get("segment"))
        secs = sorted(o["seconds"] for o in rec["objects"] if o.get("seconds") is not None)
        if secs:
            row.update(per_object_p50_sec=secs[len(secs) // 2], per_object_p90_sec=secs[min(len(secs) - 1, int(0.9 * len(secs)))],
                       reconstruct_sec=round(sum(secs), 1))
        v = rec.get("metadata", {}).get("vram_peak_gb", {})
        row["vram_process_peak_gb"] = max(v.values()) if v else None
        lite = scene / "scene_lite.glb"
        row["lite_mb"] = round(lite.stat().st_size / 1e6, 2) if lite.exists() else None
    if appearance:
        try:
            row.update(appearance_metrics(scene))
        except Exception as e:
            row["appearance_error"] = str(e)[:120]
    return row


def _mean_row(rows: List[Dict]) -> Dict:
    out: Dict = {"scene": f"mean of {len(rows)}"}
    for key, _, _ in COLUMNS:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        if vals:
            out[key] = float(np.mean(vals))
    return out


def _fmt(key: str, fmt: str, v) -> str:
    if v is None:
        return "–"
    if "d" in fmt:
        return fmt.format(int(round(v))) if float(v).is_integer() else f"{v:.1f}"
    return fmt.format(v)


def to_markdown(rows: List[Dict], base: Optional[Dict[str, Dict]] = None) -> str:
    cols = [c for c in COLUMNS if any(r.get(c[0]) is not None for r in rows)]
    lines = ["| scene | " + " | ".join(h for _, h, _ in cols) + " |", "| --- |" + " ---: |" * len(cols)]
    for r in rows:
        cells = []
        b = (base or {}).get(r["scene"])
        for key, _, fmt in cols:
            cell = _fmt(key, fmt, r.get(key))
            if b and isinstance(b.get(key), (int, float)) and isinstance(r.get(key), (int, float)):
                d = r[key] - b[key]
                if abs(d) > 1e-9:
                    cell += f" ({d:+.2f})" if abs(d) < 10 else f" ({d:+.0f})"
            cells.append(cell)
        lines.append(f"| {r['scene']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def regressions(rows: List[Dict], base: Dict[str, Dict], guards=GUARDS) -> List[str]:
    out = []
    for r in rows:
        b = base.get(r["scene"])
        if not b:
            continue
        for key, (direction, tol, kind) in guards.items():
            x, y = r.get(key), b.get(key)
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                continue
            worse = (y - x) * direction                       # > 0 when the new value is worse
            limit = tol if kind == "abs" else tol * abs(y)
            if worse > limit + 1e-9:
                out.append(f"{r['scene']}: {key} {y:.3g} → {x:.3g} (allowed {'±' if kind == 'abs' else ''}"
                           f"{tol:g}{'' if kind == 'abs' else '×'})")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", type=Path, nargs="+", help="run folders and/or scene folders")
    ap.add_argument("--baseline", type=Path, help="another run folder (or its benchmark.json) to compare with")
    ap.add_argument("--appearance", action="store_true", help="render each scene from the photo's camera (pyrender)")
    ap.add_argument("--no-audit", action="store_true", help="use each scene's existing audit.json")
    ap.add_argument("--out", type=Path, help="folder for benchmark.json / benchmark.md (default: first path)")
    ap.add_argument("--tolerances", type=json.loads, default={},
                    help='JSON overriding guard tolerances, e.g. \'{"mean_iou": 0.05}\'')
    a = ap.parse_args(argv)

    scenes = find_scenes(a.paths)
    if not scenes:
        print("no scenes found")
        return 2
    rows = []
    for s in scenes:
        rows.append(scene_row(s, a.appearance, not a.no_audit))
        print(f"  {s.name}: IoU {rows[-1].get('mean_iou')}", flush=True)
    rows.append(_mean_row(rows))

    base = None
    if a.baseline:
        bj = a.baseline if a.baseline.suffix == ".json" else a.baseline / "benchmark.json"
        if bj.exists():
            base_rows = json.loads(bj.read_text())["scenes"]
        else:
            base_rows = [scene_row(s, a.appearance, not a.no_audit) for s in find_scenes([a.baseline])]
            base_rows.append(_mean_row(base_rows))
        base = {r["scene"]: r for r in base_rows}

    md = to_markdown(rows, base)
    print("\n" + md)
    out_dir = a.out or (a.paths[0] if not (a.paths[0] / "reconstruction_results.json").exists() else a.paths[0].parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark.json").write_text(json.dumps({"scenes": rows}, indent=1))
    (out_dir / "benchmark.md").write_text(md + "\n")
    if base:
        guards = {k: (d, a.tolerances.get(k, t), kind) for k, (d, t, kind) in GUARDS.items()}
        bad = regressions(rows, base, guards)
        if bad:
            print("\nWorse than the baseline beyond tolerance:\n  " + "\n  ".join(bad))
            return 1
        print("\nNo guarded metric got worse beyond its tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
