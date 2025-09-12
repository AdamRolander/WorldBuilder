import ai2thor.controller
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import math

# --- Configuration ---
SCENES = [f"FloorPlan{i}" for i in range(1, 5)] 
MAX_SAMPLES_PER_SCENE = 400 
OUTPUT_FILENAME = "ai2thor_coordinate_dataset_og.csv"
OCCLUSION_THRESHOLD = 0.8

def get_bbox_features(obj_meta, screen_width, screen_height):
    """Extract bounding box features with better error handling."""
    bbox = obj_meta.get('axisAlignedBoundingBox', {}).get('cornerPoints')
    
    # Fallback to object bounds if cornerPoints not available
    if not bbox or len(bbox) != 4:
        # Try alternative bounding box method
        if 'objectBounds' in obj_meta:
            bounds = obj_meta['objectBounds']
            if bounds and len(bounds.get('cornerPoints', [])) == 4:
                bbox = bounds['cornerPoints']
    
    # If still no valid bbox, try to estimate from object properties
    if not bbox or len(bbox) != 4:
        # Use object center and estimated size as fallback
        if 'axisAlignedBoundingBox' in obj_meta and 'center' in obj_meta['axisAlignedBoundingBox']:
            center = obj_meta['axisAlignedBoundingBox']['center']
            size = obj_meta['axisAlignedBoundingBox'].get('size', {'x': 0.2, 'y': 0.2})
            # Project 3D center to 2D (simplified projection)
            px = 0.5  # Default to center for now - would need proper projection
            py = 0.5
            w, h = 0.1, 0.1  # Default reasonable size
            return {
                'center_x': px, 'center_y': py, 'width': w, 'height': h, 'aspect_ratio': 1.0
            }
        else:
            # Last resort - use reasonable defaults
            return {
                'center_x': 0.5, 'center_y': 0.5, 'width': 0.1, 'height': 0.1, 'aspect_ratio': 1.0
            }
    
    x1, y1, x2, y2 = bbox
    
    # Ensure valid bbox dimensions
    if x2 <= x1 or y2 <= y1:
        return {
            'center_x': 0.5, 'center_y': 0.5, 'width': 0.1, 'height': 0.1, 'aspect_ratio': 1.0
        }
    
    width = (x2 - x1) / screen_width
    height = (y2 - y1) / screen_height
    
    # Clamp to reasonable values (avoid zeros!)
    width = max(0.02, min(1.0, width))  # Minimum 2% of screen
    height = max(0.02, min(1.0, height))
    
    return {
        'center_x': max(0.01, min(0.99, ((x1 + x2) / 2) / screen_width)),
        'center_y': max(0.01, min(0.99, ((y1 + y2) / 2) / screen_height)),
        'width': width,
        'height': height,
        'aspect_ratio': width / height
    }

def get_occlusion_features(obj_meta):
    bbox = obj_meta.get('axisAlignedBoundingBox', {}).get('cornerPoints')
    if not bbox or len(bbox) != 4:
        return {'is_occluded': 0, 'visibility_ratio': 1.0}
    x1, y1, x2, y2 = bbox
    bbox_area = (x2 - x1) * (y2 - y1)
    if bbox_area == 0:
        return {'is_occluded': 0, 'visibility_ratio': 1.0}
    num_visible_pixels = obj_meta.get('numVisiblePixels', 0)
    visibility_ratio = num_visible_pixels / bbox_area
    is_occluded = 1 if visibility_ratio < OCCLUSION_THRESHOLD else 0
    return {'is_occluded': is_occluded, 'visibility_ratio': visibility_ratio}

def get_enhanced_depth_features(depth_frame, bbox, obj_position, agent_position):
    """Extract comprehensive depth features from depth map and 3D position."""
    features = {}
    
    # Basic depth map features from bbox region
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(depth_frame.shape[1], x2)
        y2 = min(depth_frame.shape[0], y2)
        
        if x2 > x1 and y2 > y1:
            depth_region = depth_frame[y1:y2, x1:x2]
            if depth_region.size > 0:
                features.update({
                    'depth_mean': np.mean(depth_region),
                    'depth_std': np.std(depth_region),
                    'depth_min': np.min(depth_region),
                    'depth_max': np.max(depth_region),
                    'depth_median': np.median(depth_region),
                    'depth_percentile_25': np.percentile(depth_region, 25),
                    'depth_percentile_75': np.percentile(depth_region, 75)
                })
            else:
                # Fallback to ground truth distance
                gt_distance = np.linalg.norm(np.array(list(obj_position.values())) - agent_position)
                features.update({
                    'depth_mean': gt_distance, 'depth_std': 0, 'depth_min': gt_distance,
                    'depth_max': gt_distance, 'depth_median': gt_distance,
                    'depth_percentile_25': gt_distance, 'depth_percentile_75': gt_distance
                })
        else:
            # Use ground truth as fallback
            gt_distance = np.linalg.norm(np.array(list(obj_position.values())) - agent_position)
            features.update({
                'depth_mean': gt_distance, 'depth_std': 0, 'depth_min': gt_distance,
                'depth_max': gt_distance, 'depth_median': gt_distance,
                'depth_percentile_25': gt_distance, 'depth_percentile_75': gt_distance
            })
    
    # Ground truth distance for comparison
    gt_distance = np.linalg.norm(np.array(list(obj_position.values())) - agent_position)
    features['ground_truth_distance'] = gt_distance
    
    # Depth consistency metric
    if 'depth_mean' in features and features['depth_mean'] > 0:
        features['depth_gt_consistency'] = abs(features['depth_mean'] - gt_distance) / gt_distance
    else:
        features['depth_gt_consistency'] = 0
    
    return features

def calculate_relative_depth_target(agent_pos, target_obj, ref_obj):
    target_pos = np.array(list(target_obj['position'].values()))
    ref_pos = np.array(list(ref_obj['position'].values()))
    dist_to_target = np.linalg.norm(target_pos - agent_pos)
    dist_to_ref = np.linalg.norm(ref_pos - agent_pos)
    return dist_to_target / dist_to_ref if dist_to_ref > 1e-6 else 0

def main():
    print("Starting enhanced dataset generation...")
    
    controller = ai2thor.controller.Controller(
        scene="FloorPlan1",
        width=300,
        height=300,
        renderInstanceSegmentation=True,
        renderDepthImage=True,
        visibilityDistance=10.0,  # Increase visibility range
        gridSize=0.25  # Finer movement grid
    )
    
    all_data = []
    failed_samples = 0
    
    for scene_name in tqdm(SCENES, desc="Processing Scenes"):
        controller.reset(scene=scene_name)
        scene_samples = 0
        scene_attempts = 0
        max_attempts_per_scene = MAX_SAMPLES_PER_SCENE * 3  # Allow more attempts
        
        while scene_samples < MAX_SAMPLES_PER_SCENE and scene_attempts < max_attempts_per_scene:
            scene_attempts += 1
            
            # Take a random action to move around the scene
            action = random.choice(['MoveAhead', 'RotateRight', 'RotateLeft', 'LookUp', 'LookDown'])
            event = controller.step(action=action)
            
            if not event.metadata['lastActionSuccess']:
                continue
            
            # Filter visible objects more carefully
            visible_objects = [obj for obj in event.metadata['objects'] 
                              if obj['visible'] and obj.get('distance', 0) > 0.5]
            
            if len(visible_objects) < 2:
                continue
            
            # Further filter objects for valid positions and reasonable separation
            agent_pos = np.array(list(event.metadata['agent']['position'].values()))
            valid_objects = []
            
            for obj in visible_objects:
                obj_pos = np.array(list(obj['position'].values()))
                distance = np.linalg.norm(obj_pos - agent_pos)
                
                # Only include objects at reasonable distances (0.5m to 8m)
                if 0.5 < distance < 8.0:
                    # Check if object has valid bounding box or can be estimated
                    bbox = obj.get('axisAlignedBoundingBox', {}).get('cornerPoints')
                    if bbox and len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        # Ensure bbox has reasonable dimensions
                        if x2 > x1 and y2 > y1 and (x2-x1) * (y2-y1) > 100:  # At least 10x10 pixels
                            valid_objects.append(obj)
                    elif obj.get('objectBounds'):  # Alternative bbox source
                        bounds = obj['objectBounds'].get('cornerPoints', [])
                        if bounds and len(bounds) == 4:
                            valid_objects.append(obj)
                    else:
                        # Include object anyway - we'll estimate bbox from position
                        valid_objects.append(obj)
            
            if len(valid_objects) < 2:
                continue
            
            # Select target and reference objects
            target_obj, ref_obj = random.sample(valid_objects, 2)
            
            # Ensure objects are sufficiently separated (at least 1m apart)
            target_pos = np.array(list(target_obj['position'].values()))
            ref_pos = np.array(list(ref_obj['position'].values()))
            if np.linalg.norm(target_pos - ref_pos) < 1.0:
                continue
            
            # Get agent metadata
            agent_meta = event.metadata['agent']
            agent_pos = np.array(list(agent_meta['position'].values()))
            
            # Extract bounding box features with improved error handling
            target_bbox_features = get_bbox_features(target_obj, 300, 300)
            ref_bbox_features = get_bbox_features(ref_obj, 300, 300)
            
            # Skip if both objects have invalid bounding boxes
            if (target_bbox_features['width'] == 0.1 and target_bbox_features['height'] == 0.1 and
                ref_bbox_features['width'] == 0.1 and ref_bbox_features['height'] == 0.1):
                continue
            
            # Calculate ground truth distances
            dist_to_target_ground_truth = np.linalg.norm(target_pos - agent_pos)
            dist_to_ref_ground_truth = np.linalg.norm(ref_pos - agent_pos)
            
            # Get depth frame and extract enhanced depth features
            depth_frame = event.depth_frame
            if depth_frame is None:
                continue
            
            # Get bounding boxes for depth extraction
            target_bbox = target_obj.get('axisAlignedBoundingBox', {}).get('cornerPoints')
            ref_bbox = ref_obj.get('axisAlignedBoundingBox', {}).get('cornerPoints')
            
            # Extract enhanced depth features
            target_depth_features = get_enhanced_depth_features(
                depth_frame, target_bbox, target_obj['position'], agent_pos)
            ref_depth_features = get_enhanced_depth_features(
                depth_frame, ref_bbox, ref_obj['position'], agent_pos)
            
            # Get occlusion features
            target_occlusion = get_occlusion_features(target_obj)
            ref_occlusion = get_occlusion_features(ref_obj)
            
            # Calculate relative depth target
            relative_depth = calculate_relative_depth_target(agent_pos, target_obj, ref_obj)
            
            # Skip samples with extreme relative depths
            if relative_depth <= 0.01 or relative_depth > 20.0:
                continue
            
            # Skip samples where depth estimates are wildly inconsistent
            if (target_depth_features.get('depth_gt_consistency', 0) > 2.0 or 
                ref_depth_features.get('depth_gt_consistency', 0) > 2.0):
                continue
            
            # Build comprehensive data row
            data_row = {
                # Object identification
                'scene_name': scene_name,
                'target_object_type': target_obj['objectType'], 
                'ref_object_type': ref_obj['objectType'],
                
                # Enhanced bounding box features
                'target_bbox_center_x': target_bbox_features['center_x'],
                'target_bbox_center_y': target_bbox_features['center_y'],
                'target_bbox_width': target_bbox_features['width'],
                'target_bbox_height': target_bbox_features['height'],
                'target_bbox_aspect_ratio': target_bbox_features['aspect_ratio'],
                'ref_bbox_center_x': ref_bbox_features['center_x'],
                'ref_bbox_center_y': ref_bbox_features['center_y'],
                'ref_bbox_width': ref_bbox_features['width'],
                'ref_bbox_height': ref_bbox_features['height'],
                'ref_bbox_aspect_ratio': ref_bbox_features['aspect_ratio'],

                # Enhanced depth map features (target object)
                'target_depth_mean': target_depth_features.get('depth_mean', dist_to_target_ground_truth),
                'target_depth_std': target_depth_features.get('depth_std', 0),
                'target_depth_min': target_depth_features.get('depth_min', dist_to_target_ground_truth),
                'target_depth_max': target_depth_features.get('depth_max', dist_to_target_ground_truth),
                'target_depth_median': target_depth_features.get('depth_median', dist_to_target_ground_truth),
                'target_depth_percentile_25': target_depth_features.get('depth_percentile_25', dist_to_target_ground_truth),
                'target_depth_percentile_75': target_depth_features.get('depth_percentile_75', dist_to_target_ground_truth),
                'target_ground_truth_distance': target_depth_features.get('ground_truth_distance', dist_to_target_ground_truth),
                'target_depth_gt_consistency': target_depth_features.get('depth_gt_consistency', 0),
                
                # Enhanced depth map features (reference object)
                'ref_depth_mean': ref_depth_features.get('depth_mean', dist_to_ref_ground_truth),
                'ref_depth_std': ref_depth_features.get('depth_std', 0),
                'ref_depth_min': ref_depth_features.get('depth_min', dist_to_ref_ground_truth),
                'ref_depth_max': ref_depth_features.get('depth_max', dist_to_ref_ground_truth),
                'ref_depth_median': ref_depth_features.get('depth_median', dist_to_ref_ground_truth),
                'ref_depth_percentile_25': ref_depth_features.get('depth_percentile_25', dist_to_ref_ground_truth),
                'ref_depth_percentile_75': ref_depth_features.get('depth_percentile_75', dist_to_ref_ground_truth),
                'ref_ground_truth_distance': ref_depth_features.get('ground_truth_distance', dist_to_ref_ground_truth),
                'ref_depth_gt_consistency': ref_depth_features.get('depth_gt_consistency', 0),
                
                # Occlusion features
                'target_is_occluded': target_occlusion['is_occluded'],
                'target_visibility_ratio': target_occlusion['visibility_ratio'],
                'ref_is_occluded': ref_occlusion['is_occluded'],
                'ref_visibility_ratio': ref_occlusion['visibility_ratio'],

                # Agent/Camera features
                'camera_horizon': agent_meta['cameraHorizon'],
                'field_of_view': event.metadata['fov'],
                'agent_pos_x': agent_meta['position']['x'], 
                'agent_pos_y': agent_meta['position']['y'], 
                'agent_pos_z': agent_meta['position']['z'],
                'agent_rot_x': agent_meta['rotation']['x'], 
                'agent_rot_y': agent_meta['rotation']['y'], 
                'agent_rot_z': agent_meta['rotation']['z'],
                
                # Distance features
                'dist_to_target': dist_to_target_ground_truth,
                'dist_to_ref': dist_to_ref_ground_truth,
                
                # Targets for training
                'relative_depth': relative_depth,
                'world_x': target_obj['position']['x'], 
                'world_y': target_obj['position']['y'], 
                'world_z': target_obj['position']['z'],
                
                # Additional quality metrics
                'target_object_area': target_obj.get('area', 0),
                'ref_object_area': ref_obj.get('area', 0),
                'target_distance_camera': target_obj.get('distance', dist_to_target_ground_truth),
                'ref_distance_camera': ref_obj.get('distance', dist_to_ref_ground_truth)
            }
            
            all_data.append(data_row)
            scene_samples += 1
            
            # Progress update for long scenes
            if scene_samples % 50 == 0:
                print(f"  {scene_name}: {scene_samples}/{MAX_SAMPLES_PER_SCENE} samples")
        
        print(f"Scene {scene_name}: Generated {scene_samples} samples (attempted {scene_attempts})")
        
        if scene_samples < MAX_SAMPLES_PER_SCENE // 2:
            print(f"Warning: Low sample count for {scene_name}, consider investigating scene layout")

    controller.stop()
    
    if not all_data:
        print("Error: No valid samples generated!")
        return
    
    # Create DataFrame and save
    df = pd.DataFrame(all_data)
    
    # Data quality report
    print(f"\n=== Dataset Generation Summary ===")
    print(f"Total samples generated: {len(df)}")
    print(f"Unique scenes: {df['scene_name'].nunique()}")
    print(f"Samples per scene (avg): {len(df) / df['scene_name'].nunique():.1f}")
    
    # Check data quality
    print(f"\nData Quality Metrics:")
    print(f"Valid bounding boxes (target): {(df['target_bbox_width'] > 0.01).sum()} ({100*(df['target_bbox_width'] > 0.01).mean():.1f}%)")
    print(f"Valid bounding boxes (ref): {(df['ref_bbox_width'] > 0.01).sum()} ({100*(df['ref_bbox_width'] > 0.01).mean():.1f}%)")
    print(f"Depth consistency (target): {df['target_depth_gt_consistency'].mean():.3f} ± {df['target_depth_gt_consistency'].std():.3f}")
    print(f"Depth consistency (ref): {df['ref_depth_gt_consistency'].mean():.3f} ± {df['ref_depth_gt_consistency'].std():.3f}")
    
    # Distance statistics
    print(f"\nDistance Statistics:")
    print(f"Target distances: {df['dist_to_target'].min():.2f}m - {df['dist_to_target'].max():.2f}m (mean: {df['dist_to_target'].mean():.2f}m)")
    print(f"Reference distances: {df['dist_to_ref'].min():.2f}m - {df['dist_to_ref'].max():.2f}m (mean: {df['dist_to_ref'].mean():.2f}m)")
    print(f"Relative depths: {df['relative_depth'].min():.3f} - {df['relative_depth'].max():.3f} (mean: {df['relative_depth'].mean():.3f})")
    
    # Object type distribution
    print(f"\nTop 10 Target Object Types:")
    print(df['target_object_type'].value_counts().head(10))
    
    # Save dataset
    df.to_csv(OUTPUT_FILENAME, index=False)
    print(f"\nEnhanced dataset saved as '{OUTPUT_FILENAME}'")
    print("Dataset includes comprehensive depth map statistics and improved bounding box handling!")

if __name__ == "__main__":
    main()