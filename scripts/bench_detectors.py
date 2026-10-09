"""Benchmark the local VLM detector against Gemini reference lists.

There is no ground truth for "which objects are in this photo", so Gemini's
list is used as the reference and the local model is scored by fuzzy label
recall against it. That is enough to (a) catch regressions when the prompt
or model changes and (b) compare candidate local models on the same photos.

Offline (scores existing outputs, no GPU):

    python scripts/bench_detectors.py --offline outputs outputs_og2

Live (asks the running VLM server about every image in a folder and scores
against outputs/<stem>_g/detected_objects.json when that exists):

    python scripts/bench_detectors.py --live demo_day --reference outputs --save bench_qwen3vl8b.json

Matching: two labels match if they share a head noun after singularisation
("office chair" ~ "chair"), or if their token Jaccard ≥ 0.5, or if a small
synonym table says so. Tune the table in SYNONYMS as you learn the models'
vocabularies; do not expect >0.8 recall — Gemini and Qwen legitimately
disagree about what counts as an object.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.detection_postprocess import clean_detections, dedupe_key  # noqa: E402

SYNONYMS = [
    {"sofa", "couch", "settee", "loveseat"}, {"tv", "television", "screen", "monitor"},
    {"fridge", "refrigerator"}, {"bin", "trash", "wastebasket", "garbage"},
    {"lamp", "light", "fixture", "sconce"}, {"picture", "painting", "art", "artwork", "frame", "poster"},
    {"plant", "planter", "pot"}, {"rug", "carpet", "mat"}, {"cabinet", "cupboard", "dresser", "vanity"},
    {"desk", "table", "workbench", "counter"}, {"stool", "chair", "seat"}, {"shelf", "bookshelf", "bookcase", "shelving"},
    {"pillow", "cushion"}, {"blanket", "throw"}, {"curtain", "drape"}, {"toilet", "wc"},
    {"tap", "faucet"}, {"basin", "sink"}, {"mirror", "glass"}, {"box", "crate", "container"},
]
_SYN = {w: i for i, s in enumerate(SYNONYMS) for w in s}


def _toks(label: str) -> List[str]:
    return dedupe_key(label).split()


def labels_match(a: str, b: str) -> bool:
    ta, tb = _toks(a), _toks(b)
    if not ta or not tb:
        return False
    if ta == tb:
        return True
    ha, hb = ta[-1], tb[-1]
    if ha == hb:
        return True
    if ha in _SYN and hb in _SYN and _SYN[ha] == _SYN[hb]:
        return True
    sa, sb = set(ta), set(tb)
    return len(sa & sb) / len(sa | sb) >= 0.5


def score(candidate: Iterable[Dict], reference: Iterable[Dict]) -> Dict:
    cand = [o["label"] for o in clean_detections(candidate, verbose=False)]
    ref = [o["label"] for o in clean_detections(reference, verbose=False)]
    hit_ref: Set[int] = set()
    matched_cand = 0
    for c in cand:
        for i, r in enumerate(ref):
            if i not in hit_ref and labels_match(c, r):
                hit_ref.add(i)
                matched_cand += 1
                break
    recall = len(hit_ref) / len(ref) if ref else 0.0
    precision = matched_cand / len(cand) if cand else 0.0
    return {"n_candidate": len(cand), "n_reference": len(ref), "recall": round(recall, 3),
            "precision": round(precision, 3),
            "missed": [r for i, r in enumerate(ref) if i not in hit_ref],
            "extra": [c for c in cand if not any(labels_match(c, r) for r in ref)]}


def degenerate_stats(raw: List[Dict]) -> Dict:
    labels = [str(o.get("label", "")).lower() for o in raw if isinstance(o, dict)]
    counts: Dict[str, int] = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    top = max(counts.values()) if counts else 0
    return {"n_raw": len(labels), "n_unique": len(counts), "max_repeat": top}


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def offline(dirs: List[Path]) -> None:
    scenes: Dict[str, Dict[str, Path]] = {}
    for base in dirs:
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            m = re.match(r"(.+?)_(g|q)(?:_[0-9a-f]{6})?$", d.name)
            if not m:
                continue
            f = d / "detected_objects_raw.json"
            if not f.exists():
                f = d / "detected_objects.json"
            if f.exists():
                scenes.setdefault(m.group(1), {})[m.group(2)] = f
    rows = []
    for stem, files in sorted(scenes.items()):
        if "g" not in files or "q" not in files:
            continue
        g, q = _load(files["g"]), _load(files["q"])
        if not isinstance(g, list) or not isinstance(q, list):
            continue
        s = score(q, g)
        s.update(stem=stem, **degenerate_stats(q))
        rows.append(s)
        print(f"{stem:<32s} recall={s['recall']:.2f} prec={s['precision']:.2f} "
              f"q={s['n_candidate']:2d}/{s['n_raw']:3d}raw g={s['n_reference']:2d} missed={s['missed'][:5]}")
    if rows:
        mr = sum(r["recall"] for r in rows) / len(rows)
        mp = sum(r["precision"] for r in rows) / len(rows)
        print(f"\n{len(rows)} paired scenes: mean recall {mr:.2f}, mean precision {mp:.2f}, "
              f"{sum(1 for r in rows if r['max_repeat'] >= 5)} scenes with a label repeated ≥5×")
    else:
        print("no paired _g/_q scenes found")


def live(image_dir: Path, reference: Path, server: str, save: Path | None, limit: int) -> None:
    import requests
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in exts)[:limit]
    info = requests.get(f"{server}/health", timeout=5).json()
    print(f"VLM server: {info}")
    out = {"model": info.get("model"), "server": server, "results": []}
    recalls = []
    for img in images:
        t0 = time.time()
        r = requests.post(f"{server}/detect", json={"image_path": str(img.resolve())}, timeout=600)
        dt = time.time() - t0
        payload = r.json()
        objs = payload.get("objects", [])
        row = {"image": img.name, "seconds": round(dt, 1), "tokens": payload.get("tokens"),
               "n": len(objs), "labels": [o.get("label") for o in objs], "error": payload.get("error")}
        ref = None
        for cand in (reference / f"{img.stem}_g" / "detected_objects_raw.json",
                     reference / f"{img.stem}_g" / "detected_objects.json"):
            if cand.exists():
                ref = _load(cand)
                break
        if isinstance(ref, list):
            s = score(objs, ref)
            row.update(recall=s["recall"], precision=s["precision"], missed=s["missed"], extra=s["extra"])
            recalls.append(s["recall"])
            print(f"{img.name:<26s} {dt:5.1f}s n={len(objs):2d} recall={s['recall']:.2f} missed={s['missed'][:6]}")
        else:
            print(f"{img.name:<26s} {dt:5.1f}s n={len(objs):2d} (no Gemini reference) {row['labels'][:8]}")
        out["results"].append(row)
    if recalls:
        print(f"\nmean recall vs Gemini over {len(recalls)} images: {sum(recalls) / len(recalls):.2f}")
    requests.post(f"{server}/unload", timeout=60)
    if save:
        save.write_text(json.dumps(out, indent=2))
        print(f"saved {save}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", nargs="*", type=Path, help="output dirs holding *_g and *_q scenes")
    ap.add_argument("--live", type=Path, help="image folder to run through the VLM server")
    ap.add_argument("--reference", type=Path, default=Path("outputs"))
    ap.add_argument("--server", default="http://127.0.0.1:8765")
    ap.add_argument("--save", type=Path)
    ap.add_argument("--limit", type=int, default=100)
    a = ap.parse_args()
    if a.offline is not None:
        offline(a.offline or [Path("outputs")])
    if a.live:
        live(a.live, a.reference, a.server, a.save, a.limit)
    if a.offline is None and not a.live:
        ap.print_help()
