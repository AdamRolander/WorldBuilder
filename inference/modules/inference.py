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
    extract_depth_from_mask,
    preprocess_bright_image
)
from size_estimation import estimate_object_sizes
from geometry import unproject_to_3d, calculate_analytical_depth

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
    FOV_DEG = 60.0
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
    
    # Step 3: Depth estimation with multi-reference calibration
    print("\nStep 3: Estimating depth with multi-reference calibration...")
    preprocessed_image = preprocess_bright_image(image_path)
    midas_model, midas_transform = load_midas_model(device=DEVICE)
    depth_map_cache = {}
    
    raw_depth_map = estimate_depth_with_midas(
        image_path, midas_model, midas_transform, depth_map_cache
    )
    
    # Calculate calibration constants from ALL valid objects
    from multi_reference_depth import (
        calculate_calibration_constants,
        get_robust_calibration_constant,
        normalize_depth_with_constant
    )
    
    print("\nCalculating calibration from multiple objects:")
    calibration_points = calculate_calibration_constants(
        valid_segments,
        estimated_sizes,
        raw_depth_map,
        img_width,
        FOV_DEG,
        min_confidence=0.5  # Adjust this threshold as needed
    )
    
    if not calibration_points:
        print("ERROR: No valid calibration points found")
        return
    
    # Get robust calibration constant
    # Try different methods: 'weighted_median', 'median', 'confidence_weighted'
    k_value, k_metadata = get_robust_calibration_constant(
        calibration_points,
        method='weighted_median'
    )
    
    if k_value is None:
        print("ERROR: Could not determine calibration constant")
        return
    
    # Normalize depth map using the multi-object calibration
    metric_depth_map = normalize_depth_with_constant(raw_depth_map, k_value)
    
    # print(f"Reference depth: {ref_analytical_depth:.2f}m")
    
    # Step 4: Calculate 3D poses
    print("\nStep 4: Calculating 3D poses...\n")

    from semantic_adjustments import (
        should_use_bottom_center, 
        get_unproject_point,
        estimate_ground_plane_y,
        adjust_position_to_ground_plane,
        extract_depth_from_bottom_region,
        should_use_elevated_depth_adjustment
    )

    results = []
    temp_results = []

    for segment in valid_segments:
        seg_id = segment['id']
        label = segment['label']
        bbox = segment['bbox']
        
        # Get depth from metric depth map
        from semantic_adjustments import extract_depth_from_bottom_region
        
        if should_use_bottom_center(label):
            # For floor objects, try bottom region first
            depth = extract_depth_from_bottom_region(
                metric_depth_map, 
                segment['mask'].cpu().numpy(), 
                bbox, 
                bottom_fraction=0.4
            )
            if depth is None:  # Fallback
                depth_stats = extract_depth_from_mask(metric_depth_map, segment['mask'])
                depth = depth_stats['median']
        else:
            depth_stats = extract_depth_from_mask(metric_depth_map, segment['mask'])
            depth = depth_stats['median']
        
        # Determine which point to unproject from
        use_bottom = should_use_bottom_center(label)
        pixel_x, pixel_y = get_unproject_point(bbox, use_bottom)

        # # Special handling for elevated objects (lights, etc.)
        # if should_use_elevated_depth_adjustment(label):
        #     # Sample depth from the region BELOW the light instead
        #     sample_y = bbox['y_min'] + bbox['height'] + 50  # 50 pixels below
        #     sample_y = min(sample_y, img_height - 1)  # Clamp to image bounds
            
        #     # Create a small sample region
        #     sample_mask = np.zeros_like(metric_depth_map, dtype=bool)
        #     sample_x = int(pixel_x)
        #     sample_y_int = int(sample_y)
        #     # Sample a 10x10 region
        #     sample_mask[max(0, sample_y_int-5):min(img_height, sample_y_int+5),
        #                 max(0, sample_x-5):min(img_width, sample_x+5)] = True
            
        #     depth_below = np.median(metric_depth_map[sample_mask]) if sample_mask.any() else depth
            
        #     print(f"  Elevated object {label}: Using depth from below: {depth_below:.2f}m (was {depth:.2f}m)")
        #     depth = depth_below
        
        # Unproject to 3D
        world_pos = unproject_to_3d(
            pixel_x, pixel_y, depth,
            img_width, img_height, FOV_DEG,
            CAMERA_POSITION, CAMERA_ROTATION
        )

        if abs(world_pos[0]) > 100 or abs(world_pos[1]) > 100 or abs(world_pos[2]) > 100:
            print(f"  WARNING: Extreme coordinates for {label}: {world_pos}")
        
        seg_data = estimated_sizes[seg_id]
        dimensions = seg_data['dimensions_meters']

        temp_results.append({
            'id': int(seg_id),
            'label': label,
            'position_m': world_pos,
            'dimensions_m': dimensions,
            'rotation_y_deg': seg_data['rotation_y_degrees'],
            'rotation_confidence': seg_data['confidence'],
            'justification': seg_data.get('justification', ''),
            'depth_m': float(depth)
        })
    
    # Estimate ground plane from floor objects
    ground_y = estimate_ground_plane_y(temp_results)
    print(f"\nEstimated ground plane Y: {ground_y:.3f}m\n")
    
    # Adjust positions to ground plane and create final results
    results = []
    for temp_result in temp_results:
        adjusted_pos = adjust_position_to_ground_plane(
            temp_result['position_m'],
            temp_result['dimensions_m'],
            ground_y,
            temp_result['label']
        )
        
        result = {
            'id': temp_result['id'],
            'label': temp_result['label'],
            'position_m': adjusted_pos.tolist(),
            'depth_m': temp_result['depth_m'],
            'dimensions_m': temp_result['dimensions_m'],
            'rotation_y_deg': temp_result['rotation_y_deg'],
            'rotation_confidence': temp_result['rotation_confidence']
        }
        
        results.append(result)

    # from pattern_recognition import apply_pattern_recognition
    # results = apply_pattern_recognition(valid_segments, results)
       
    print(f"{result['label']} (ID: {result['id']})")
    print(f"  Position: ({adjusted_pos[0]:.3f}, {adjusted_pos[1]:.3f}, {adjusted_pos[2]:.3f}) m")
    print(f"  Rotation Y: {result['rotation_y_deg']:.1f}° (confidence: {result['rotation_confidence']:.2f})")
    print(f"  Justification: {temp_result['justification']}")
    print(f"  Depth: {result['depth_m']:.3f} m")
    print(f"  Dimensions: {result['dimensions_m']} m\n")
    
    # Save results
    output_path = Path(__file__).parent / "coords_json" / f"{Path(image_path).stem}_3d_poses.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    # Update this path to your test image
    IMAGE_PATH = "/Users/adamrolander/WorldBuilder/inference/modules/input/bathroom.png"
    
    if not os.path.exists(IMAGE_PATH):
        print(f"ERROR: Image not found at {IMAGE_PATH}")
    else:
        run_minimal_inference(IMAGE_PATH)