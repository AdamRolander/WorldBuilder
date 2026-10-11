"""Light copy of finished scenes for replaying stage 4 without touching the originals.

    python scripts/clone_scene.py outputs/<run>/<scene> [more...] --into outputs/<new run>
    python scripts/replay_assembly.py outputs/<new run>/*

Copies what stage 4 reads (masks, detection and segmentation JSON, the
stage-3 point map, poses and metric scale) and symlinks the stage-3 meshes,
so a clone costs a few MB. Replay the clone with different code or settings
and compare the two runs with ``scripts/benchmark.py --baseline`` or
``scripts/make_gallery.py --before``.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

SMALL = ("detected_objects.json", "detected_objects_raw.json", "segmentation_results.json",
         "reconstruction_results.json", "metrics.json")


def clone(scene: Path, into: Path) -> Path:
    scene = scene.resolve()
    dst = into / scene.name
    src_cache = scene / "3d_models" / "stage3"
    if not (src_cache / "raw.json").exists():
        raise SystemExit(f"{scene}: no stage-3 cache to clone")
    (dst / "3d_models" / "stage3").mkdir(parents=True, exist_ok=True)
    for name in SMALL:
        if (scene / name).exists():
            shutil.copy2(scene / name, dst / name)
    if (scene / "masks").is_dir():
        shutil.copytree(scene / "masks", dst / "masks", dirs_exist_ok=True)
    for f in src_cache.iterdir():
        out = dst / "3d_models" / "stage3" / f.name
        if out.exists() or out.is_symlink():
            out.unlink()
        if f.suffix == ".ply":
            out.symlink_to(f.resolve())
        else:
            shutil.copy2(f, out)
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", type=Path, nargs="+")
    ap.add_argument("--into", type=Path, required=True)
    a = ap.parse_args()
    a.into.mkdir(parents=True, exist_ok=True)
    for s in a.scene:
        print(f"✓ {clone(s, a.into)}")


if __name__ == "__main__":
    main()
