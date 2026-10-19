"""Add structural masks (e.g. ``window``) to scenes segmented before the prompt existed.

    python scripts/add_structural_masks.py outputs/<run>/<scene> [more...] [--prompts window]

Runs SAM 3 only (one image encoding and one decoder pass per prompt), writes
``masks/structural_<prompt>.png`` next to the existing masks and leaves
everything else alone, so ``scripts/replay_assembly.py`` can use the new
mask without a stage-3 rerun. Needs the GPU (SAM 3 does not run on CPU).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def source_image(scene: Path) -> Path:
    for name in ("reconstruction_results.json",):
        p = scene / name
        if p.exists():
            src = json.loads(p.read_text()).get("metadata", {}).get("source_image")
            if src and Path(src).exists():
                return Path(src)
    raw = scene / "3d_models" / "stage3" / "raw.json"
    if raw.exists():
        src = json.loads(raw.read_text()).get("source_image")
        if src and Path(src).exists():
            return Path(src)
    raise FileNotFoundError(f"{scene}: source image not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", type=Path, nargs="+")
    ap.add_argument("--prompts", default="window")
    ap.add_argument("--threshold", type=float, default=0.25)
    a = ap.parse_args()
    import torch

    from src.segmentation import SAM3Segmenter
    seg = SAM3Segmenter()
    seg._to_device("cuda")
    seg.processor.confidence_threshold = a.threshold
    for scene in a.scene:
        img_path = source_image(scene)
        image = Image.open(img_path).convert("RGB")
        with torch.inference_mode():
            state = seg.processor.set_image(image)
        for prompt in a.prompts.split(","):
            masks, scores, _ = seg.query_text(state, prompt)
            out = scene / "masks" / f"structural_{prompt}.png"
            if len(scores) == 0:
                if out.exists():
                    out.unlink()
                print(f"{scene.name}: '{prompt}' none")
                continue
            union = np.any(masks, axis=0)
            Image.fromarray(union.astype(np.uint8) * 255).save(out)
            print(f"{scene.name}: '{prompt}' {len(scores)} mask(s), {100 * union.mean():.1f}% of frame", flush=True)
    seg._to_device("cpu")


if __name__ == "__main__":
    main()
