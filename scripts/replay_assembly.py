"""Re-run stage 4 (layout, placement, room, exports) on a finished scene. CPU only.

    python scripts/replay_assembly.py outputs/<scene> [more scenes...]
    python scripts/replay_assembly.py outputs/<scene> --placement legacy   # old box/label rules
    python scripts/replay_assembly.py outputs/<scene> --placement raw      # SAM 3D poses untouched

Needs the stage-3 cache (``3d_models/stage3/``) that the pipeline writes
since October 2026; a scene takes a few seconds instead of a GPU run.
Combine with ``scripts/audit_scene.py`` to measure a placement change.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import stage3_cache  # noqa: E402
from src.scene_assembly import assemble_from_cache, finalize_scene, load_structural_masks  # noqa: E402


def replay(scene: Path, placement=None, build_room=True, quiet=False) -> None:
    models = scene / "3d_models"
    raw = stage3_cache.load_raw(models)
    if raw is None:
        raise SystemExit(f"{scene}: no stage-3 cache (3d_models/stage3/raw.json)")
    image = np.asarray(Image.open(raw["source_image"]).convert("RGB"))
    t = time.time()
    results, failed = assemble_from_cache(models, image, load_structural_masks(scene / "masks") or None,
                                          build_room=build_room, verbose=not quiet, placement=placement)
    meta = {}
    old = scene / "reconstruction_results.json"
    if old.exists():
        meta = json.loads(old.read_text()).get("metadata", {})
    meta.update(source_image=raw["source_image"], image_size=[image.shape[1], image.shape[0]],
                total_objects=len(results), replayed=datetime.now().isoformat(timespec="seconds"))
    finalize_scene(scene, results, failed, meta, verbose=not quiet)
    print(f"✓ {scene.name}: {len(results)} objects placed in {time.time() - t:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", type=Path, nargs="+")
    ap.add_argument("--placement", choices=["evidence", "legacy", "raw"])
    ap.add_argument("--no-room", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    for s in a.scene:
        replay(s, a.placement, not a.no_room, a.quiet)


if __name__ == "__main__":
    main()
