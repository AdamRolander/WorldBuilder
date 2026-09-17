"""Debug tool: overlay segmentation masks on original image to verify alignment."""
import numpy as np
from PIL import Image
from pathlib import Path
import json


def overlay_masks(image_path: str, output_dir: str):
    """Create a composite image showing all masks overlaid on the original."""
    image = Image.open(image_path).convert('RGB')
    img_array = np.array(image, dtype=np.float32)
    
    output_path = Path(output_dir)
    seg_file = output_path / "segmentation_results.json"
    
    if not seg_file.exists():
        print(f"No segmentation results found in {output_path}")
        return
    
    with open(seg_file) as f:
        segments = json.load(f)
    
    # Generate distinct colors for each object
    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(len(segments), 3))
    
    overlay = img_array.copy()
    
    for i, seg in enumerate(segments):
        mask_path = seg.get('mask_path')
        if mask_path:
            # Try as-is first, then resolve relative to output dir
            if not Path(mask_path).exists():
                # Try masks subdirectory, then output dir directly
                masks_subdir = output_path / "masks" / Path(mask_path).name
                if masks_subdir.exists():
                    mask_path = str(masks_subdir)
                else:
                    mask_path = str(output_path / Path(mask_path).name)
        if not mask_path or not Path(mask_path).exists():
            print(f"  Mask not found: {mask_path}")
            continue
        
        mask = np.array(Image.open(mask_path).convert('L'))
        
        # Check dimension match
        if mask.shape[:2] != img_array.shape[:2]:
            print(f"  ⚠️  DIMENSION MISMATCH for {seg['label']}:")
            print(f"     Image: {img_array.shape[:2]}, Mask: {mask.shape[:2]}")
            # Resize mask to match
            mask = np.array(
                Image.open(mask_path).convert('L').resize(
                    (image.width, image.height), Image.NEAREST
                )
            )
        
        binary = mask > 128
        pixel_count = binary.sum()
        total_pixels = binary.size
        coverage = pixel_count / total_pixels * 100
        
        print(f"  {seg['id']:3d} {seg['label']:<30s} "
              f"pixels={pixel_count:>8,}  coverage={coverage:.1f}%  "
              f"conf={seg.get('confidence', 0):.2f}")
        
        # Blend color onto masked region
        color = colors[i].astype(np.float32)
        overlay[binary] = overlay[binary] * 0.5 + color * 0.5
    
    # Save overlay
    result = Image.fromarray(overlay.astype(np.uint8))
    debug_path = output_path / "debug_mask_overlay.png"
    result.save(debug_path)
    print(f"\n✓ Overlay saved: {debug_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    overlay_masks(args.image, args.output_dir)