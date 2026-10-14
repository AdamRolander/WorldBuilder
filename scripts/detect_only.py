"""Stage 1 only, for a batch: write ``detected_objects.json`` per photo.

    python scripts/detect_only.py <photo dir> --output outputs/<run>

The local VLM server must be running. Afterwards ``main.py --input-dir
<photo dir> --output outputs/<run> --resume`` reuses these detections, so the
VLM server can be stopped before SAM 3 / SAM 3D load (the two together need
most of a 64 GB machine's RAM).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from main import build_detector, detector_kind, get_output_folder_name  # noqa: E402
from src.detection_postprocess import clean_detections  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--detector", default=None)
    a = ap.parse_args()
    kind = detector_kind(a.detector)
    det = build_detector(kind)
    for img in sorted(p for p in a.input_dir.iterdir() if p.suffix.lower() in EXTS):
        out = a.output / get_output_folder_name(img, a.detector)
        out.mkdir(parents=True, exist_ok=True)
        if (out / "detected_objects.json").exists():
            print(f"{img.name}: already detected", flush=True)
            continue
        raw = det.detect_objects(str(img))
        with Image.open(img) as im:
            objs = clean_detections(raw, image_size=im.size)
        (out / "detected_objects_raw.json").write_text(json.dumps(raw, indent=2))
        (out / "detected_objects.json").write_text(json.dumps(objs, indent=2))
        print(f"{img.name}: {len(objs)} types", flush=True)


if __name__ == "__main__":
    main()
