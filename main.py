from dotenv import load_dotenv; load_dotenv()
import os
import json
from pathlib import Path
from src.object_detection import GeminiObjectDetector
from src.segmentation import SAM3Segmenter
from src.reconstruction_3d import SAM3DReconstructor
import time
import csv
from datetime import datetime

def process_image(image_path: str, output_dir: str = "outputs") -> Path:
    """Run complete pipeline on single image. Returns the output directory."""
    
    print("="*80)
    print(f"Processing: {image_path}")
    print("="*80)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Step 1: Detect objects (Gemini API or local VLM, picked via env var)
    detector_kind = os.environ.get("WORLDBUILDER_DETECTOR", "gemini").lower()
    if detector_kind in ("qwen", "vlm", "local"):
        from src.local_vlm_detection import LocalVLMObjectDetector
        print("\n[1/3] Detecting objects with local VLM...")
        detector = LocalVLMObjectDetector()
    else:
        print("\n[1/3] Detecting objects with Gemini...")
        detector = GeminiObjectDetector()
    objects = detector.detect_objects(image_path)

    from src.detection_postprocess import filter_excluded
    objects = filter_excluded(objects)

    objects_file = output_path / "detected_objects.json"
    with open(objects_file, "w") as f:
        json.dump(objects, f, indent=2)
    print(f"✓ Saved to {objects_file}")
    
    # Step 2: Segment objects. Wake SAM 3 (it may be parked on CPU from
    # the previous image), run, then park back to CPU to free VRAM for
    # SAM 3D — the same dance the VLM server does for Qwen.
    print("\n[2/3] Segmenting objects with SAM 3...")
    segmenter = SAM3Segmenter()
    segmenter._to_device('cuda')
    try:
        segments = segmenter.segment_objects(
            image_path,
            objects,
            output_dir=str(output_path / "masks")
        )
    finally:
        segmenter._to_device('cpu')

    segments_file = output_path / "segmentation_results.json"
    with open(segments_file, "w") as f:
        json.dump([{k: v for k, v in s.items() if k != 'mask_array'}
                   for s in segments], f, indent=2)
    print(f"✓ Saved to {segments_file}")

    # Step 3: Generate 3D assets. Same wake/park pattern.
    print("\n[3/3] Generating 3D assets with SAM 3D Objects...")
    reconstructor = SAM3DReconstructor()
    reconstructor._to_device('cuda')
    try:
        assets_3d = reconstructor.reconstruct_objects(
            image_path,
            segments,
            output_dir=str(output_path / "3d_models"),
            quality='high'
        )
    finally:
        reconstructor._to_device('cpu')
    
    assets_file = output_path / "reconstruction_results.json"
    with open(assets_file, "w") as f:
        json.dump({
            'objects': assets_3d,
            'metadata': {
                'source_image': str(image_path),
                'total_objects': len(assets_3d)
            }
        }, f, indent=2)
    print(f"✓ Saved to {assets_file}")
    
    print("\n" + "="*80)
    print("✓ Pipeline complete!")
    print(f"  Objects detected: {len(objects)}")
    print(f"  Objects segmented: {len(segments)}")
    print(f"  Objects reconstructed: {len(assets_3d)}")
    print(f"  Output directory: {output_path}")
    print("="*80)

    try:
        from src.viewer_generator import generate_viewer
        viewer_path = generate_viewer(output_path, assets_3d,
                                      room_file="3d_models/room.ply")
        print(f"✓ Viewer: {viewer_path}  (serve with: python -m http.server -d {output_path})")
    except Exception as e:
        print(f"⚠️  Viewer generation failed: {e}")

    # Free per-job tensors. Models are kept in VRAM (they're singletons),
    # but scratch buffers, masks, point clouds, etc. need to go between
    # webapp runs or the second image OOMs.
    import gc, torch
    del objects, segments, assets_3d
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return output_path


def get_output_folder_name(image_path: Path) -> str:
    """Folder name = <image_stem>_<g|q> so Gemini and Qwen runs of the same
    image live side by side and --force only clobbers the matching detector.
    """
    detector = os.environ.get("WORLDBUILDER_DETECTOR", "gemini").lower()
    suffix = "q" if detector in ("qwen", "vlm", "local") else "g"
    return f"{image_path.stem}_{suffix}"


def is_already_processed(output_dir: Path) -> bool:
    """Check if an image has already been fully processed."""
    required_files = [
        "detected_objects.json",
        "segmentation_results.json",
        "reconstruction_results.json"
    ]
    return all((output_dir / f).exists() for f in required_files)


def get_image_info(image_path: Path) -> dict:
    """Get image metadata."""
    from PIL import Image
    img = Image.open(image_path)
    return {
        'width': img.width,
        'height': img.height,
        'megapixels': round((img.width * img.height) / 1_000_000, 2),
        'format': img.format,
        'file_size_mb': round(image_path.stat().st_size / (1024 * 1024), 2)
    }


def get_result_counts(output_dir: Path) -> dict:
    """Read result counts from output files."""
    counts = {
        'objects_detected': 0,
        'objects_segmented': 0,
        'objects_reconstructed': 0
    }
    
    try:
        with open(output_dir / "detected_objects.json") as f:
            counts['objects_detected'] = len(json.load(f))
    except:
        pass
    
    try:
        with open(output_dir / "segmentation_results.json") as f:
            counts['objects_segmented'] = len(json.load(f))
    except:
        pass
    
    try:
        with open(output_dir / "reconstruction_results.json") as f:
            data = json.load(f)
            counts['objects_reconstructed'] = len(data.get('objects', []))
    except:
        pass
    
    return counts


def save_timing_report(timing_data: list, output_path: Path):
    """Save timing results to CSV."""
    if not timing_data:
        return
    
    csv_path = output_path / f"timing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    fieldnames = [
        'image_name', 'status', 'total_time_sec',
        'width', 'height', 'megapixels', 'format', 'file_size_mb',
        'objects_detected', 'objects_segmented', 'objects_reconstructed',
        'time_per_object_sec', 'timestamp'
    ]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(timing_data)
    
    print(f"\n📊 Timing report saved: {csv_path}")
    
    # Also print summary stats
    successful = [t for t in timing_data if t['status'] == 'success']
    if successful:
        times = [t['total_time_sec'] for t in successful]
        print(f"\n   Summary ({len(successful)} successful runs):")
        print(f"   ├─ Min:     {min(times):.1f}s")
        print(f"   ├─ Max:     {max(times):.1f}s")
        print(f"   ├─ Mean:    {sum(times)/len(times):.1f}s")
        print(f"   └─ Total:   {sum(times):.1f}s")

def process_all_images(input_dir: str = "test_images", output_base: str = "outputs"):
    """Process all images in input directory with timing."""
    input_path = Path(input_dir)
    output_base_path = Path(output_base)
    output_base_path.mkdir(exist_ok=True)
    
    if not input_path.exists():
        print(f"❌ Input directory not found: {input_path}")
        return
    
    # Find all supported images
    supported_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}
    images = [
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]
    
    if not images:
        print(f"❌ No images found in {input_path}")
        return
    
    images.sort()
    print(f"Found {len(images)} images in {input_path}")
    print("="*80)
    
    processed = 0
    skipped = 0
    failed = 0
    timing_data = []
    
    batch_start = time.time()
    
    for i, image_path in enumerate(images, 1):
        folder_name = get_output_folder_name(image_path)
        output_dir = output_base_path / folder_name
        
        print(f"\n[{i}/{len(images)}] {image_path.name}")
        
        if is_already_processed(output_dir):
            print(f"  ⏭️  Already processed, skipping")
            skipped += 1
            continue
        
        # Get image info before processing
        try:
            img_info = get_image_info(image_path)
        except:
            img_info = {'width': 0, 'height': 0, 'megapixels': 0, 'format': 'unknown', 'file_size_mb': 0}
        
        start_time = time.time()
        
        try:
            process_image(str(image_path), str(output_dir))
            elapsed = time.time() - start_time
            processed += 1
            
            # Get result counts
            counts = get_result_counts(output_dir)
            
            # Calculate time per object
            total_objects = counts['objects_reconstructed'] or 1
            time_per_object = elapsed / total_objects
            
            timing_data.append({
                'image_name': image_path.name,
                'status': 'success',
                'total_time_sec': round(elapsed, 2),
                **img_info,
                **counts,
                'time_per_object_sec': round(time_per_object, 2),
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"\n  ⏱️  Completed in {elapsed:.1f}s ({time_per_object:.1f}s per object)")
            
        except Exception as e:
            elapsed = time.time() - start_time
            failed += 1
            
            timing_data.append({
                'image_name': image_path.name,
                'status': f'failed: {str(e)[:50]}',
                'total_time_sec': round(elapsed, 2),
                **img_info,
                'objects_detected': 0,
                'objects_segmented': 0,
                'objects_reconstructed': 0,
                'time_per_object_sec': 0,
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"  ❌ Failed after {elapsed:.1f}s: {e}")
    
    batch_elapsed = time.time() - batch_start
    
    print("\n" + "="*80)
    print("BATCH COMPLETE")
    print(f"  Processed: {processed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Failed:    {failed}")
    print(f"  Total time: {batch_elapsed:.1f}s ({batch_elapsed/60:.1f}m)")
    print("="*80)
    
    # Save timing report
    save_timing_report(timing_data, output_base_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="WorldBuilder SAM3D Pipeline")
    parser.add_argument("--image", help="Path to single input image")
    parser.add_argument("--input-dir", default="test_images", help="Directory of images to process")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--force", action="store_true", help="Reprocess even if already done")
    
    args = parser.parse_args()
    
    if args.image:
        # Single image mode — write into a per-image subfolder so runs don't
        # overwrite each other and batch/single outputs live side by side.
        import shutil
        start = time.time()
        image_path = Path(args.image)
        output_dir = Path(args.output) / get_output_folder_name(image_path)

        if output_dir.exists() and args.force:
            shutil.rmtree(output_dir)
            print(f"🗑️  Removed existing: {output_dir}")
        elif output_dir.exists() and is_already_processed(output_dir):
            print(f"⏭️  {output_dir} already processed — use --force to redo")
            raise SystemExit(0)

        process_image(str(image_path), str(output_dir))
        print(f"\n⏱️  Total time: {time.time() - start:.1f}s")
    else:
        # Batch mode
        if args.force:
            import shutil
            # Wipe any already-processed subfolders so this run is clean
            base = Path(args.output)
            if base.exists():
                for sub in base.iterdir():
                    if sub.is_dir() and is_already_processed(sub):
                        shutil.rmtree(sub)
                        print(f"🗑️  Removed: {sub}")
            is_already_processed = lambda x: False

        process_all_images(args.input_dir, args.output)