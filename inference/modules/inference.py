"""
Minimal inference pipeline using unprojection for 3D pose estimation.
"""

import torch
import numpy as np
import json
import os
from pathlib import Path

from segmentation import segment_image
from depth_estimation import (
    load_midas_model, 
    estimate_depth_with_midas,
    normalize_depth_to_metric,
    extract_depth_from_mask
)
from size_estimation import estimate_object_sizes
from geometry import unproject_to_3d, calculate_analytical_depth
from rotation_estimation import estimate_rotation_from_mask, estimate_semantic_rotation

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super().default(obj)


def run_minimal_inference(image_path: str, output_dir: str = "."):
    """
    Run minimal 3D pose estimation using unprojection.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save output JSON
    """
    # Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    FOV_DEG = 90.0
    CAMERA_POSITION = np.array([0.0, 0.0, 0.0])
    CAMERA_ROTATION = np.array([0.0, 0.0, 0.0])
    
    print(f"\n=== Minimal 3D Pose Inference ===")
    print(f"Device: {DEVICE}")
    print(f"Image: {image_path}\n")
    
    # Step 1: Segmentation
    print("Step 1: Segmenting image...")
    seg_result = segment_image(image_path, device=DEVICE)
    segments = seg_result['segments']
    img_width = seg_result['img_width']
    img_height = seg_result['img_height']
    
    if len(segments) < 1:
        print("ERROR: No objects found in image")
        return
    
    print(f"Found {len(segments)} objects")

    # # DEBUG: Visualize mask orientations
    # from debug_rotation import visualize_mask_orientation, analyze_rotation_consistency
    # visualize_mask_orientation(image_path, segments, "rotation_debug.png")
    # analyze_rotation_consistency(segments)
    
    # Step 2: Size estimation
    print("\nStep 2: Estimating object sizes...")
    estimated_sizes = estimate_object_sizes(segments, image_path)
    
    if not estimated_sizes:
        print("ERROR: Size estimation failed")
        return
    
    valid_segments = [s for s in segments if s['id'] in estimated_sizes]
    if len(valid_segments) < 1:
        print("ERROR: No valid segments with size estimates")
        return
    
    # Step 3: Depth estimation
    print("\nStep 3: Estimating depth...")
    midas_model, midas_transform = load_midas_model(device=DEVICE)
    depth_map_cache = {}
    
    raw_depth_map = estimate_depth_with_midas(
        image_path, midas_model, midas_transform, depth_map_cache
    )
    
    # Choose reference object (largest by area)
    ref_segment = max(valid_segments, key=lambda x: x['mask_area'])
    ref_size_data = estimated_sizes[ref_segment['id']]

    print(f"Reference object: {ref_segment['label']} (ID: {ref_segment['id']})")

    # Calculate reference depth using pinhole model
    # Match physical dimensions to visual dimensions properly
    ref_dims = ref_size_data['dimensions_meters']  # [length, width, height]
    ref_bbox = ref_segment['bbox']

    ref_horizontal_dim = max(ref_dims[0], ref_dims[1])  # Length or width
    ref_vertical_dim = ref_dims[2]  # Height

    ref_analytical_depth = calculate_analytical_depth(
        ref_vertical_dim, ref_bbox['height'], img_width, FOV_DEG
    )

    print(f"Reference using vertical: {ref_vertical_dim:.2f}m phys, {ref_bbox['height']:.0f}px visual")
    
    # Get reference MiDaS value
    ref_depth_stats = extract_depth_from_mask(raw_depth_map, ref_segment['mask'])
    ref_midas_value = ref_depth_stats['median']
    
    # Normalize depth map to metric scale
    metric_depth_map = normalize_depth_to_metric(
        raw_depth_map, ref_analytical_depth, ref_midas_value
    )
    
    print(f"Reference depth: {ref_analytical_depth:.2f}m")
    
    # Step 4: Calculate 3D poses
    print("\nStep 4: Calculating 3D poses...\n")

    results = []

    for segment in valid_segments:
        seg_id = segment['id']
        label = segment['label']
        bbox = segment['bbox']
        
        # Get depth from metric depth map
        depth_stats = extract_depth_from_mask(metric_depth_map, segment['mask'])
        depth = depth_stats['median']
        
        # Unproject center point to 3D
        world_pos = unproject_to_3d(
            bbox['center_x_norm'] * img_width, 
            bbox['center_y_norm'] * img_height, 
            depth,
            img_width, img_height, FOV_DEG,
            CAMERA_POSITION, CAMERA_ROTATION
        )

        if abs(world_pos[0]) > 100 or abs(world_pos[1]) > 100 or abs(world_pos[2]) > 100:
            print(f"  WARNING: Extreme coordinates for {label}: {world_pos}")
            print(f"    Depth: {depth:.2f}m, Bbox center: ({bbox['center_x']:.1f}, {bbox['center_y']:.1f})")
        
        is_reference = (seg_id == ref_segment['id'])

        seg_data = estimated_sizes[seg_id]
        dimensions = seg_data['dimensions_meters']
        rotation_y = seg_data['rotation_y_degrees']
        rotation_confidence = seg_data['confidence']
        rotation_justification = seg_data.get('justification', '')

        result = {
            'id': int(seg_id),
            'label': label,
            'position_m': world_pos.tolist(),
            'depth_m': float(depth),
            'dimensions_m': dimensions,  # Already a list
            'rotation_y_deg': rotation_y,
            'rotation_confidence': rotation_confidence,
            'is_reference': is_reference
        }
        
        results.append(result)
        
        ref_marker = " (REFERENCE)" if is_reference else ""
        print(f"{label} (ID: {seg_id}){ref_marker}")
        print(f"  Position: ({world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f}) m")
        print(f"  Rotation Y: {rotation_y:.1f}° (confidence: {rotation_confidence:.2f})")
        print(f"  Justification: {rotation_justification}")
        print(f"  Depth: {depth:.3f} m")
        print(f"  Dimensions: {dimensions} m\n")
    
    # Save results
    output_path = Path(__file__).parent / "coords_json" / f"{Path(image_path).stem}_3d_poses.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    # Update this path to your test image
    IMAGE_PATH = "/Users/adamrolander/WorldBuilder/inference/modules/input/room_test.png"
    
    if not os.path.exists(IMAGE_PATH):
        print(f"ERROR: Image not found at {IMAGE_PATH}")
    else:
        run_minimal_inference(IMAGE_PATH)