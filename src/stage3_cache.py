"""What stage 3 leaves behind so stage 4 can be replayed without a GPU.

SAM 3D is the expensive part of a run (~8 s per object); everything after
it is CPU geometry that we want to iterate on in seconds. Until now the
model-space PLYs were overwritten by world-space ones and the raw poses were
mutated in place, so trying a different placement rule meant another
five-minute GPU run. Stage 3 now writes, under ``3d_models/stage3/``:

* ``<id>_<label>.ply``  the mesh exactly as SAM 3D returned it (model space)
* ``raw.json``          the untouched pose per object (camera frame) + failures
* ``pointmap.npz``      the scene point map (camera-frame world), validity
                        mask and normalised intrinsics, at ≤ ``MAX_SIDE`` px

``src/scene_assembly.py`` consumes only this directory, the photo and the
masks; ``scripts/replay_assembly.py`` reruns it on any finished scene.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

DIRNAME = "stage3"
MAX_SIDE = 1024


def cache_dir(models_dir: Path) -> Path:
    return Path(models_dir) / DIRNAME


def save_pointmap(models_dir: Path, points_world: np.ndarray, valid: np.ndarray,
                  intrinsics: np.ndarray, image_hw) -> Path:
    """``points_world`` (H,W,3) in SAM 3D's camera-frame world. Subsampled by
    index (never interpolated: averaging across a depth edge invents points)."""
    H, W = valid.shape
    s = min(1.0, MAX_SIDE / max(H, W))
    h, w = max(1, int(round(H * s))), max(1, int(round(W * s)))
    ii = np.clip(np.round((np.arange(h) + 0.5) / s - 0.5).astype(int), 0, H - 1)
    jj = np.clip(np.round((np.arange(w) + 0.5) / s - 0.5).astype(int), 0, W - 1)
    P = np.asarray(points_world, dtype=np.float32)[ii][:, jj]
    V = np.asarray(valid, dtype=bool)[ii][:, jj]
    P = np.where(V[..., None], P, 0.0).astype(np.float32)
    d = cache_dir(models_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "pointmap.npz"
    np.savez_compressed(path, points=P, valid=V, intrinsics=np.asarray(intrinsics, dtype=np.float64),
                        image_hw=np.asarray(image_hw, dtype=np.int64))
    return path


def load_pointmap(models_dir: Path) -> Optional[Dict]:
    path = cache_dir(models_dir) / "pointmap.npz"
    if not path.exists():
        return None
    z = np.load(path)
    return {"points": z["points"].astype(np.float64), "valid": z["valid"].astype(bool),
            "intrinsics": z["intrinsics"], "image_hw": tuple(int(x) for x in z["image_hw"])}


def save_raw(models_dir: Path, results: List[Dict], failed: List[Dict], source_image: str) -> Path:
    d = cache_dir(models_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "raw.json"
    path.write_text(json.dumps({"source_image": str(source_image), "objects": results, "failed": failed}, indent=2))
    return path


def load_raw(models_dir: Path) -> Optional[Dict]:
    """Raw results with ``model_path`` resolved against the cache directory,
    so a scene directory can be moved or copied and still replay."""
    path = cache_dir(models_dir) / "raw.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    for r in data["objects"]:
        r["model_path"] = str(cache_dir(models_dir) / Path(r["model_path"]).name)
        if r.get("mask_path"):
            local = Path(models_dir).parent / "masks" / Path(r["mask_path"]).name
            if local.exists():
                r["mask_path"] = str(local)
    return data
