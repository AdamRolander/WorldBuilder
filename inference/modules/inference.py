"""
Inference pipeline using unprojection for 3D pose estimation with PCA-based rotation.
"""

import torch
import numpy as np
import json
import os
from pathlib import Path

from segmentation_V2 import segment_and_split
from depth_estimation import (
    load_midas_model, 
    estimate_depth_with_midas,
    extract_depth_from_mask,
    preprocess_bright_image
)
from size_estimation import estimate_object_sizes
from geometry import unproject_to_3d, calculate_analytical_depth
from rotation_estimation import estimate_pca_rotations
from multi_reference_depth import (
    calculate_calibration_constants,
    get_robust_calibration_constant,
    normalize_depth_with_constant
)
from semantic_adjustments import (
    should_use_bottom_center, 
    get_unproject_point,
    estimate_ground_plane_y,
    adjust_position_to_ground_plane,
    extract_depth_from_bottom_region
)


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


def run_inference(image_path: str, output_dir: str = "."):
    """
    Run 3D pose estimation with PCA-based rotation.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save output JSON
    """
    # Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    FOV_DEG = 60.0
    CAMERA_POSITION = np.array([0.0, 0.0, 0.0])
    CAMERA_ROTATION = np.array([0.0, 0.0, 0.0])
    
    print(f"\n=== 3D Pose Inference with PCA Rotation ===")
    print(f"Device: {DEVICE}")
    print(f"Image: {image_path}\n")
    
    # Step 1: Segmentation
    print("Step 1: Segmenting image...")
    seg_result = segment_and_split(image_path, device=DEVICE)
    segments = seg_result['segments']
    img_width = seg_result['img_width']
    img_height = seg_result['img_height']

    if len(segments) < 1:
        print("ERROR: No objects found in image")
        return

    print(f"Found {len(segments)} objects")
    
    # Step 2: Depth estimation
    print("\nStep 2: Estimating depth...")
    preprocessed_image = preprocess_bright_image(image_path)
    midas_model, midas_transform = load_midas_model(device=DEVICE)
    depth_map_cache = {}
    
    raw_depth_map = estimate_depth_with_midas(
        preprocessed_image, midas_model, midas_transform, depth_map_cache
    )
    
    # Step 3: PCA-based rotation estimation (NEW!)
    print("\nStep 3: Estimating rotations with depth-aware PCA...")
    segments = estimate_pca_rotations(
        segments,
        raw_depth_map,
        img_width,
        img_height,
        fov_deg=FOV_DEG
    )
    
    # Step 4: Size estimation
    print("\nStep 4: Estimating object sizes...")
    estimated_sizes = estimate_object_sizes(segments, image_path)
    
    if not estimated_sizes:
        print("ERROR: Size estimation failed")
        return
    
    valid_segments = [s for s in segments if s['id'] in estimated_sizes]
    if len(valid_segments) < 1:
        print("ERROR: No valid segments with size estimates")
        return
       
    # Step 5: Multi-reference depth calibration
    print("\nStep 5: Calibrating depth with multiple reference objects...")
    
    calibration_points = calculate_calibration_constants(
        valid_segments,
        estimated_sizes,
        raw_depth_map,
        img_width,
        FOV_DEG,
        min_confidence=0.5
    )
    
    if not calibration_points:
        print("ERROR: No valid calibration points found")
        return
    
    k_value, k_metadata = get_robust_calibration_constant(
        calibration_points,
        method='weighted_median'
    )
    
    if k_value is None:
        print("ERROR: Could not determine calibration constant")
        return
    
    # Normalize depth map using the multi-object calibration
    metric_depth_map = normalize_depth_with_constant(raw_depth_map, k_value)
    
    # Step 6: Calculate 3D poses
    print("\nStep 6: Calculating 3D poses...\n")

    results = []
    temp_results = []

    for segment in valid_segments:
        seg_id = segment['id']
        label = segment['label']
        bbox = segment['bbox']
        
        # Get depth from metric depth map
        if should_use_bottom_center(label):
            # For floor objects, try bottom region first
            depth = extract_depth_from_bottom_region(
                metric_depth_map, 
                segment['mask'],
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

        # Use PCA-based rotation from Step 3
        rotation_y = segment.get('rotation_y_deg', 0.0)
        rotation_confidence = segment.get('rotation_confidence', 0.0)

        temp_results.append({
            'id': int(seg_id),
            'label': label,
            'position_m': world_pos,
            'dimensions_m': dimensions,
            'rotation_y_deg': rotation_y,
            'rotation_confidence': rotation_confidence,
            'depth_m': float(depth),
            'rests_on_id': seg_data.get('rests_on_id', None)
        })

    # Step 6.5: Apply scene-scaled dispersion
    # print("\nStep 6.5: Applying dispersion to reduce clustering...")

    # # Calculate scene depth range
    # depths = [r['depth_m'] for r in temp_results]
    # min_depth = min(depths)
    # max_depth = max(depths)
    # depth_range = max_depth - min_depth

    # # Apply depth scaling to spread objects (5-10% increase)
    # dispersion_factor = 1.15  # 8% increase in distances

    # for temp_result in temp_results:
    #     pos = temp_result['position_m']
    #     # Scale X and Z from origin (camera is at origin)
    #     pos[0] *= dispersion_factor
    #     pos[2] *= dispersion_factor
    #     temp_result['position_m'] = pos

    # print(f"  Depth range: {min_depth:.2f}m - {max_depth:.2f}m (range: {depth_range:.2f}m)")
    # print(f"  Applied dispersion factor: {dispersion_factor:.2f}x")

    print("\nStep 6.5: Applying spatial dispersion...")
    from spatial_dispersion import apply_dispersion

    temp_results = apply_dispersion(temp_results, dispersion_strength=0.35)

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
            'rotation_confidence': temp_result['rotation_confidence'],
            'rests_on_id': temp_result.get('rests_on_id', None)
        }
        
        results.append(result)
    
    # Step 6.7: Apply LLM-detected support relationships
    print("\nStep 6.7: Applying support relationships from LLM...")

    results_by_id = {r['id']: r for r in results}
    support_count = 0

    for result in results:
        rests_on_id = result.get('rests_on_id')
        
        # Not working
        if rests_on_id is not None and rests_on_id in results_by_id:
            supporter = results_by_id[rests_on_id]
            
            # Calculate new Y position
            supporter_top_y = supporter['position_m'][1] + (supporter['dimensions_m'][2] / 2)
            supported_height = result['dimensions_m'][2]
            new_y = supporter_top_y + (supported_height / 2)
            
            old_y = result['position_m'][1]
            result['position_m'][1] = new_y
            
            print(f"  ✓ Placing {result['label']} (ID {result['id']}) on {supporter['label']} (ID {supporter['id']})")
            print(f"    Y: {old_y:.3f}m → {new_y:.3f}m")
            support_count += 1

    print(f"\nApplied {support_count} support relationships\n")

    # NOW print final coordinates
    print("\n" + "="*70)
    print("FINAL COORDINATES")
    print("="*70 + "\n")

    for result in results:
        pos = result['position_m']
        print(f"{result['label']} (ID: {result['id']})")
        print(f"  Position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) m")
        print(f"  Rotation Y: {result['rotation_y_deg']:.1f}°")
        print(f"  Dimensions: {result['dimensions_m']} m")
        if result.get('rests_on_id'):
            print(f"  Rests on: ID {result['rests_on_id']}")
        print()

    # Save results
    output_path = Path(output_dir) / "coords_json" / f"{Path(image_path).stem}_3d_poses.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    # Update this path to your test image
    IMAGE_PATH = "input/room_test.png"
    
    if not os.path.exists(IMAGE_PATH):
        print(f"ERROR: Image not found at {IMAGE_PATH}")
    else:
        run_inference(IMAGE_PATH)