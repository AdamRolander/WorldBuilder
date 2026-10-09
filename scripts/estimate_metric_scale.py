"""Metric scale for a scene from MoGe-2 (roadmap PIPE-6).

    python scripts/estimate_metric_scale.py outputs/<scene> [more scenes...]

Everything SAM 3D and MoGe-1 produce is scale-free: consistent within a
scene, in unknown units. Until now the conversion to metres was a guess
(the camera is 1.5 m above the floor), which is wrong for a photo taken
sitting down or from a ladder and made ceilings come out at 4-5 m.

MoGe-2 predicts a *metric* point map. This script runs it on the scene's
photo, compares its depths pixel by pixel with the scale-free point map in
the stage-3 cache, and writes the median ratio to
``3d_models/stage3/metric_scale.json``. Stage 4 picks that file up
(``scripts/replay_assembly.py`` afterwards) and uses it for
``layout.estimated_metric_scale`` and everything derived from it: the
ceiling prior, the Blender add-on's scale, the MuJoCo export.

MoGe-2 needs a newer ``moge`` package than the one SAM 3D pins, so it lives
in its own folder and this script is the only code that imports it:

    pip install --no-deps --target third_party/moge2 "git+https://github.com/microsoft/MoGe.git"

The pipeline calls this script in a subprocess after stage 3 when that
folder exists (``WORLDBUILDER_METRIC=0`` to skip).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "third_party" / "moge2"
MODEL = "Ruicheng/moge-2-vitl-normal"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", type=Path, nargs="+")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda if it has 4 GB free)")
    ap.add_argument("--max-side", type=int, default=1024)
    a = ap.parse_args()
    if not VENDOR.exists():
        print(f"MoGe-2 is not installed in {VENDOR}; see this script's docstring.")
        return 2
    sys.path.insert(0, str(VENDOR))            # must precede site-packages' moge 1.x
    import numpy as np
    import torch
    from moge.model.v2 import MoGeModel
    from PIL import Image

    device = a.device
    if device is None:
        device = "cpu"
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            if free > 4 * 1024 ** 3:
                device = "cuda"
    model = MoGeModel.from_pretrained(MODEL).to(device).eval()
    for scene in a.scene:
        cache = scene / "3d_models" / "stage3"
        raw, pmf = cache / "raw.json", cache / "pointmap.npz"
        if not raw.exists() or not pmf.exists():
            print(f"{scene}: no stage-3 cache, skipped")
            continue
        src = json.loads(raw.read_text())["source_image"]
        img = Image.open(src).convert("RGB")
        img.thumbnail((a.max_side, a.max_side))
        x = torch.from_numpy(np.asarray(img)).float().permute(2, 0, 1).to(device) / 255
        t = time.time()
        with torch.no_grad():
            out = model.infer(x)
        z2 = out["points"][..., 2].float().cpu().numpy()
        m2 = out["mask"].cpu().numpy().astype(bool)
        pm = np.load(pmf)
        z1 = np.abs(pm["points"][..., 2])          # camera-frame world: Z forward
        v1 = pm["valid"].astype(bool)
        h, w = v1.shape
        z2r = np.asarray(Image.fromarray(z2).resize((w, h), Image.NEAREST))
        m2r = np.asarray(Image.fromarray(m2.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 127
        ok = v1 & m2r & (z1 > 1e-6) & np.isfinite(z2r) & (z2r > 1e-6)
        if ok.sum() < 500:
            print(f"{scene}: too few overlapping valid pixels, skipped")
            continue
        ratio = z2r[ok] / z1[ok]
        scale = float(np.median(ratio))
        p16, p84 = np.percentile(ratio, [16, 84])
        result = {"scale": scale, "spread": float((p84 - p16) / (2 * scale)), "model": MODEL,
                  "pixels": int(ok.sum()), "seconds": round(time.time() - t, 2), "device": device}
        (cache / "metric_scale.json").write_text(json.dumps(result, indent=2))
        print(f"{scene.name}: ×{scale:.3f} m per scene unit (±{100 * result['spread']:.0f}% across pixels, "
              f"{result['seconds']} s on {device})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
