"""WorldBuilder CLI: one photo in, a navigable 3D scene out.

    python main.py --image photo.jpg                 # single image
    python main.py --input-dir photos --output out   # batch
    python main.py --image photo.jpg --detector qwen # override .env

Stages (each a module in src/):
    1. detect      Gemini or a local VLM lists object types      -> detected_objects.json
    2. segment     SAM 3 masks every instance (+ floor/wall/ceiling) -> segmentation_results.json, masks/
    3. reconstruct SAM 3D lifts each mask to a posed mesh          -> reconstruction_results.json, 3d_models/
    4. assemble    layout + placement + room + viewer              -> 3d_models/room.ply, layout.json, viewer.html

Only one of the three large models is on the GPU at a time; see README
"The VRAM dance".
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv()

LOCAL_DETECTOR_ALIASES = ("qwen", "vlm", "local")
STRUCTURAL_PROMPTS = ("floor", "wall", "ceiling")


def detector_kind(override: Optional[str] = None) -> str:
    kind = (override or os.environ.get("WORLDBUILDER_DETECTOR", "gemini")).lower()
    return "local" if kind in LOCAL_DETECTOR_ALIASES else "gemini"


def build_detector(kind: str):
    """Import lazily so a Gemini-free install never imports google-genai."""
    if kind == "local":
        from src.local_vlm_detection import LocalVLMObjectDetector
        return LocalVLMObjectDetector()
    from src.object_detection import GeminiObjectDetector
    return GeminiObjectDetector()


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2))
    print(f"✓ Saved to {path}")


def process_image(image_path: str, output_dir: str = "outputs", detector: Optional[str] = None,
                  quality: str = "high", build_room: bool = True,
                  structural: bool = True, resume: bool = False) -> Path:
    """Run the complete pipeline on one image. Returns the output directory.

    With ``resume`` a stage whose output already exists in ``output_dir`` is
    loaded instead of rerun (detection JSON; segmentation JSON + masks), so a
    scene can be re-reconstructed from the same masks."""
    print("=" * 80)
    print(f"Processing: {image_path}")
    print("=" * 80)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timings: Dict[str, float] = {}
    vram_peak_gb: Dict[str, float] = {}
    kind = detector_kind(detector)

    def _mark_vram(stage: str):
        """Peak VRAM this process allocated during a stage (the local VLM
        lives in another process and is not included)."""
        try:
            import torch
            if torch.cuda.is_available():
                vram_peak_gb[stage] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    # ---- 1. detect ---------------------------------------------------------
    t0 = time.time()
    print(f"\n[1/4] Detecting objects with {'local VLM' if kind == 'local' else 'Gemini'}...")
    from PIL import Image
    with Image.open(image_path) as im:
        image_size = im.size
    det_file = output_path / "detected_objects.json"
    if resume and det_file.exists():
        objects = json.loads(det_file.read_text())
        print(f"  ↺ reusing {det_file} ({len(objects)} types)")
    else:
        raw_objects = build_detector(kind).detect_objects(image_path)
        from src.detection_postprocess import clean_detections
        objects = clean_detections(raw_objects, image_size=image_size)
        _write_json(det_file, objects)
        _write_json(output_path / "detected_objects_raw.json", raw_objects)
    timings["detect"] = time.time() - t0
    if not objects:
        raise RuntimeError("No reconstructable objects were detected in the image.")

    # ---- 2. segment --------------------------------------------------------
    t0 = time.time()
    print("\n[2/4] Segmenting objects with SAM 3...")
    seg_file = output_path / "segmentation_results.json"
    structural_masks = None
    if resume and seg_file.exists():
        segments = json.loads(seg_file.read_text())
        for s in segments:      # masks travel with the scene directory
            s["mask_path"] = str(output_path / "masks" / Path(s["mask_path"]).name)
        if structural:
            from src.scene_assembly import load_structural_masks
            structural_masks = load_structural_masks(output_path / "masks") or None
        print(f"  ↺ reusing {seg_file} ({len(segments)} instances)")
    else:
        from src.segmentation import SAM3Segmenter
        segmenter = SAM3Segmenter()
        segmenter._to_device("cuda")
        try:
            segments = segmenter.segment_objects(image_path, objects, output_dir=str(output_path / "masks"))
            if structural:
                structural_masks = segment_structural(segmenter, image_path, output_path / "masks")
        finally:
            segmenter._to_device("cpu")
        _write_json(seg_file, [{k: v for k, v in s.items() if k != "mask_array"} for s in segments])
    timings["segment"] = time.time() - t0
    _mark_vram("segment")
    if not segments:
        raise RuntimeError("SAM 3 produced no masks for the detected objects.")

    # ---- 3. reconstruct + 4. assemble -------------------------------------
    t0 = time.time()
    print("\n[3/4] Generating 3D assets with SAM 3D Objects...")
    from src.reconstruction_3d import SAM3DReconstructor
    reconstructor = SAM3DReconstructor()
    reconstructor._to_device("cuda")
    try:
        all_results = reconstructor.reconstruct_objects(
            image_path, segments, output_dir=str(output_path / "3d_models"),
            quality=quality, structural_masks=structural_masks, build_room=build_room)
    finally:
        reconstructor._to_device("cpu")
    assets_3d = [r for r in all_results if r.get("status") == "ok"]
    failed = [r for r in all_results if r.get("status") != "ok"]
    timings["reconstruct_and_assemble"] = time.time() - t0
    _mark_vram("reconstruct")
    print("\n[4/4] Writing viewer and exports...")
    from src.scene_assembly import finalize_scene
    finalize_scene(output_path, assets_3d, failed, {
        "source_image": str(image_path),
        "image_size": list(image_size),
        "detector": kind,
        "total_objects": len(assets_3d),
        "timings_sec": {k: round(v, 2) for k, v in timings.items()},
        "vram_peak_gb": vram_peak_gb,
        "created": datetime.now().isoformat(timespec="seconds"),
    })

    print("\n" + "=" * 80)
    print("✓ Pipeline complete!")
    print(f"  Objects detected:      {len(objects)}")
    print(f"  Instances segmented:   {len(segments)}")
    print(f"  Objects reconstructed: {len(assets_3d)}" + (f"  ({len(failed)} failed)" if failed else ""))
    print("  Timings: " + ", ".join(f"{k}={v:.0f}s" for k, v in timings.items()))
    if vram_peak_gb:
        print("  Peak VRAM: " + ", ".join(f"{k}={v:.1f} GB" for k, v in vram_peak_gb.items()))
    print(f"  Output directory: {output_path}")
    print("=" * 80)

    # Free per-job tensors; the models themselves stay resident (singletons).
    import gc

    import torch
    del segments, all_results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_path


def segment_structural(segmenter, image_path: str, mask_dir: Path) -> Dict[str, np.ndarray]:
    """Floor / wall / ceiling masks for the layout stage. Cheap: the image is
    already encoded; each prompt is one decoder pass. Saved for debugging."""
    from PIL import Image
    out: Dict[str, np.ndarray] = {}
    image = Image.open(image_path).convert("RGB")
    segmenter.processor.confidence_threshold = 0.25
    import torch
    with torch.inference_mode():
        state = segmenter.processor.set_image(image)
    for prompt in STRUCTURAL_PROMPTS:
        try:
            masks, scores, _ = segmenter.query_text(state, prompt)
        except Exception as e:
            print(f"  structural '{prompt}': error {str(e)[:80]}")
            continue
        if len(scores) == 0:
            print(f"  structural '{prompt}': none")
            continue
        union = np.any(masks, axis=0)
        out[prompt] = union
        Image.fromarray(union.astype(np.uint8) * 255).save(mask_dir / f"structural_{prompt}.png")
        print(f"  structural '{prompt}': {len(scores)} mask(s), {100 * union.mean():.0f}% of frame")
    return out


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def get_output_folder_name(image_path: Path, detector: Optional[str] = None) -> str:
    """<stem>_<g|q>: Gemini and local runs of one photo live side by side."""
    return f"{image_path.stem}_{'q' if detector_kind(detector) == 'local' else 'g'}"


REQUIRED_OUTPUTS = ("detected_objects.json", "segmentation_results.json", "reconstruction_results.json")


def is_already_processed(output_dir: Path) -> bool:
    return all((output_dir / f).exists() for f in REQUIRED_OUTPUTS)


def get_image_info(image_path: Path) -> dict:
    from PIL import Image
    try:
        with Image.open(image_path) as img:
            return {"width": img.width, "height": img.height,
                    "megapixels": round(img.width * img.height / 1e6, 2), "format": img.format,
                    "file_size_mb": round(image_path.stat().st_size / (1024 * 1024), 2)}
    except Exception:
        return {"width": 0, "height": 0, "megapixels": 0, "format": "unknown", "file_size_mb": 0}


def get_result_counts(output_dir: Path) -> dict:
    counts = {"objects_detected": 0, "objects_segmented": 0, "objects_reconstructed": 0, "objects_failed": 0}
    for key, fname in (("objects_detected", "detected_objects.json"),
                       ("objects_segmented", "segmentation_results.json")):
        try:
            counts[key] = len(json.loads((output_dir / fname).read_text()))
        except (OSError, ValueError):
            pass
    try:
        data = json.loads((output_dir / "reconstruction_results.json").read_text())
        counts["objects_reconstructed"] = len(data.get("objects", []))
        counts["objects_failed"] = len(data.get("failed", []))
    except (OSError, ValueError):
        pass
    return counts


TIMING_FIELDS = ["image_name", "status", "total_time_sec", "width", "height", "megapixels", "format",
                 "file_size_mb", "objects_detected", "objects_segmented", "objects_reconstructed",
                 "objects_failed", "time_per_object_sec", "timestamp"]


def save_timing_report(timing_data: List[dict], output_path: Path) -> Optional[Path]:
    if not timing_data:
        return None
    csv_path = output_path / f"timing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TIMING_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(timing_data)
    print(f"\n📊 Timing report saved: {csv_path}")
    ok = [t["total_time_sec"] for t in timing_data if t["status"] == "success"]
    if ok:
        print(f"   {len(ok)} successful: min {min(ok):.1f}s, max {max(ok):.1f}s, "
              f"mean {sum(ok) / len(ok):.1f}s, total {sum(ok):.1f}s")
    return csv_path


def process_all_images(input_dir: str = "test_images", output_base: str = "outputs",
                       force: bool = False, **kwargs) -> List[dict]:
    input_path, output_base_path = Path(input_dir), Path(output_base)
    output_base_path.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        print(f"❌ Input directory not found: {input_path}")
        return []
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
    images = sorted(f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in exts)
    if not images:
        print(f"❌ No images found in {input_path}")
        return []
    print(f"Found {len(images)} images in {input_path}\n" + "=" * 80)

    processed = skipped = failed = 0
    timing_data: List[dict] = []
    batch_start = time.time()
    for i, image_path in enumerate(images, 1):
        output_dir = output_base_path / get_output_folder_name(image_path, kwargs.get("detector"))
        print(f"\n[{i}/{len(images)}] {image_path.name}")
        if output_dir.exists() and is_already_processed(output_dir):
            if force:
                shutil.rmtree(output_dir)
                print(f"  🗑️  Removed existing: {output_dir}")
            else:
                print("  ⏭️  Already processed, skipping")
                skipped += 1
                continue
        info = get_image_info(image_path)
        start = time.time()
        row = {"image_name": image_path.name, **info, "timestamp": datetime.now().isoformat()}
        try:
            process_image(str(image_path), str(output_dir), **kwargs)
            elapsed = time.time() - start
            counts = get_result_counts(output_dir)
            row.update(status="success", total_time_sec=round(elapsed, 2), **counts,
                       time_per_object_sec=round(elapsed / (counts["objects_reconstructed"] or 1), 2))
            processed += 1
            print(f"\n  ⏱️  Completed in {elapsed:.1f}s ({row['time_per_object_sec']:.1f}s per object)")
        except Exception as e:
            elapsed = time.time() - start
            row.update(status=f"failed: {str(e)[:50]}", total_time_sec=round(elapsed, 2),
                       objects_detected=0, objects_segmented=0, objects_reconstructed=0,
                       objects_failed=0, time_per_object_sec=0)
            failed += 1
            print(f"  ❌ Failed after {elapsed:.1f}s: {e}")
        timing_data.append(row)

    total = time.time() - batch_start
    print("\n" + "=" * 80 + f"\nBATCH COMPLETE\n  Processed: {processed}\n  Skipped:   {skipped}\n"
          f"  Failed:    {failed}\n  Total time: {total:.1f}s ({total / 60:.1f}m)\n" + "=" * 80)
    save_timing_report(timing_data, output_base_path)
    return timing_data


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="WorldBuilder: photo → 3D scene")
    parser.add_argument("--image", help="Path to a single input image")
    parser.add_argument("--input-dir", default="test_images", help="Directory of images (batch mode)")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--detector", choices=["gemini", "qwen", "local"],
                        help="Override WORLDBUILDER_DETECTOR from .env")
    parser.add_argument("--quality", choices=["medium", "high"], default="high")
    parser.add_argument("--no-room", action="store_true", help="Skip floor/wall/ceiling generation")
    parser.add_argument("--no-structural", action="store_true",
                        help="Skip SAM 3 floor/wall/ceiling masks (layout falls back to geometry)")
    parser.add_argument("--force", action="store_true", help="Reprocess even if already done")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse detection/segmentation outputs already in the scene directory")
    args = parser.parse_args(argv)

    kwargs = dict(detector=args.detector, quality=args.quality,
                  build_room=not args.no_room, structural=not args.no_structural, resume=args.resume)
    if args.image:
        start = time.time()
        image_path = Path(args.image)
        if not image_path.exists():
            parser.error(f"image not found: {image_path}")
        output_dir = Path(args.output) / get_output_folder_name(image_path, args.detector)
        if output_dir.exists() and args.force:
            shutil.rmtree(output_dir)
            print(f"🗑️  Removed existing: {output_dir}")
        elif output_dir.exists() and is_already_processed(output_dir):
            print(f"⏭️  {output_dir} already processed — use --force to redo")
            return 0
        process_image(str(image_path), str(output_dir), **kwargs)
        print(f"\n⏱️  Total time: {time.time() - start:.1f}s")
        return 0
    process_all_images(args.input_dir, args.output, force=args.force, **kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
