import ai2thor.controller
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import math

# --- Configuration ---
SCENES = [f"FloorPlan{i}" for i in range(1, 31)] 
MAX_SAMPLES_PER_SCENE = 400 
OUTPUT_FILENAME = "ai2thor_adjusted.csv"
OCCLUSION_THRESHOLD = 0.8
# <<< NEW: Minimum area in pixels for a bounding box to be considered valid
MIN_BBOX_AREA = 25 

# <<< REWRITTEN: This function now uses the modern, correct method for getting bounding boxes.
def get_bbox_features(object_id, detections, screen_width, screen_height):
    """
    Extracts bounding box features from event.instance_detections2D.
    Returns a tuple: (dictionary of features, raw bbox list or None).
    """
    bbox = detections.get(object_id)

    # If no valid bbox is found in the detections dictionary, return default values.
    if not bbox or len(bbox) != 4:
        features = {'center_x': 0.0, 'center_y': 0.0, 'width': 0.0, 'height': 0.0, 'aspect_ratio': 0.0}
        return features, None

    # The format is [x_min, y_min, x_max, y_max]
    x1, y1, x2, y2 = bbox

    # Ensure valid bbox dimensions (lower right > upper left)
    if x2 <= x1 or y2 <= y1:
        features = {'center_x': 0.0, 'center_y': 0.0, 'width': 0.0, 'height': 0.0, 'aspect_ratio': 0.0}
        return features, None

    width_px = x2 - x1
    height_px = y2 - y1

    # Normalize dimensions by screen size to get values between 0 and 1
    width = width_px / screen_width
    height = height_px / screen_height

    features = {
        'center_x': ((x1 + x2) / 2) / screen_width,
        'center_y': ((y1 + y2) / 2) / screen_height,
        'width': width,
        'height': height,
        'aspect_ratio': width / height if height > 0 else 0
    }
    return features, bbox

# <<< REWRITTEN: This function now accepts a bbox, making it more reliable.
def get_occlusion_features(obj_meta, bbox):
    """Calculates occlusion based on a provided bounding box."""
    if not bbox or len(bbox) != 4:
        # Default to fully visible if no bbox is found
        return {'is_occluded': 0, 'visibility_ratio': 1.0}

    x1, y1, x2, y2 = bbox
    bbox_area = (x2 - x1) * (y2 - y1)

    if bbox_area < 1:
        return {'is_occluded': 0, 'visibility_ratio': 1.0}

    # 'numVisiblePixels' from metadata is still the source of truth for visibility
    num_visible_pixels = obj_meta.get('numVisiblePixels', 0)
    visibility_ratio = num_visible_pixels / bbox_area
    is_occluded = 1 if visibility_ratio < OCCLUSION_THRESHOLD else 0
    return {'is_occluded': is_occluded, 'visibility_ratio': visibility_ratio}

def get_enhanced_depth_features(depth_frame, bbox, obj_position, agent_position):
    """Extract comprehensive depth features from depth map and 3D position."""
    features = {}
    
    # Basic depth map features from bbox region
    if bbox and len(bbox) == 4:
        # Bbox coords are floats/ints, convert to int for slicing
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        # Clamp coordinates to be within the frame dimensions
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

    # Fallback to ground truth if depth calculation fails
    gt_distance = np.linalg.norm(np.array(list(obj_position.values())) - agent_position)
    if 'depth_mean' not in features:
        features.update({
            'depth_mean': gt_distance, 'depth_std': 0, 'depth_min': gt_distance,
            'depth_max': gt_distance, 'depth_median': gt_distance,
            'depth_percentile_25': gt_distance, 'depth_percentile_75': gt_distance
        })

    features['ground_truth_distance'] = gt_distance
    
    # Depth consistency metric
    if 'depth_mean' in features and gt_distance > 0:
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

def calculate_base_position(obj, all_objects):
    """Calculate object's base position (floor level or on supporting surface)."""
    obj_pos = np.array(list(obj['position'].values()))
    obj_bounds = obj.get('axisAlignedBoundingBox', {})
    
    if not obj_bounds or 'size' not in obj_bounds:
        return obj_pos[1]  # Return current Y if no bounds
    
    obj_bottom_y = obj_pos[1] - (obj_bounds['size']['y'] / 2)
    
    potential_supports = []
    for other_obj in all_objects:
        if other_obj['objectId'] == obj['objectId']:
            continue
        other_pos = np.array(list(other_obj['position'].values()))
        other_bounds = other_obj.get('axisAlignedBoundingBox', {})
        if not other_bounds or 'size' not in other_bounds:
            continue
            
        other_top_y = other_pos[1] + (other_bounds['size']['y'] / 2)
        if abs(obj_bottom_y - other_top_y) < 0.1:
            x_overlap = (abs(obj_pos[0] - other_pos[0]) < 
                        (obj_bounds['size']['x'] + other_bounds['size']['x']) / 2)
            z_overlap = (abs(obj_pos[2] - other_pos[2]) < 
                        (obj_bounds['size']['z'] + other_bounds['size']['z']) / 2)
            if x_overlap and z_overlap:
                potential_supports.append(other_top_y)
    
    return max(potential_supports) if potential_supports else 0.0

def classify_object_support(obj, all_objects):
    """Classify if object is on ground (0) or on another object (1)."""
    base_y = calculate_base_position(obj, all_objects)
    return 1 if base_y > 0.1 else 0

def main():
    print("Starting enhanced dataset generation...")
    
    controller = ai2thor.controller.Controller(
        scene="FloorPlan1",
        width=300,
        height=300,
        # This setting is required to get event.instance_detections2D
        renderInstanceSegmentation=True,
        renderDepthImage=True,
        visibilityDistance=10.0,
        gridSize=0.25
    )
    
    all_data = []
    
    for scene_name in tqdm(SCENES, desc="Processing Scenes"):
        controller.reset(scene=scene_name)
        scene_samples = 0
        scene_attempts = 0
        max_attempts_per_scene = MAX_SAMPLES_PER_SCENE * 5 # Increase attempts to find valid samples
        
        while scene_samples < MAX_SAMPLES_PER_SCENE and scene_attempts < max_attempts_per_scene:
            scene_attempts += 1
            
            action = random.choice(['MoveAhead', 'RotateRight', 'RotateLeft', 'LookUp', 'LookDown', 'MoveBack'])
            event = controller.step(action=action)
            
            if not event.metadata['lastActionSuccess']:
                continue

            # <<< NEW: Get 2D detections right after the event is created. This is the main fix.
            detections = event.instance_detections2D

            EXCLUDED_OBJECT_TYPES = {'Wall', 'Floor', 'Ceiling', 'Window', 'Door'}
            
            visible_objects = [obj for obj in event.metadata['objects'] 
                  if obj['visible'] 
                  and obj.get('distance', 0) > 0.5
                  and obj['objectType'] not in EXCLUDED_OBJECT_TYPES]
            
            if len(visible_objects) < 2:
                continue
            
            agent_pos = np.array(list(event.metadata['agent']['position'].values()))
            
            # More efficient to shuffle once and pick
            random.shuffle(visible_objects)
            
            # Find a valid pair of objects
            found_pair = False
            for i in range(len(visible_objects)):
                for j in range(i + 1, len(visible_objects)):
                    target_obj = visible_objects[i]
                    ref_obj = visible_objects[j]

                    target_id = target_obj['objectId']
                    ref_id = ref_obj['objectId']

                    # <<< CHANGED: Use the new function to get bbox features and the raw bbox
                    target_bbox_features, target_bbox = get_bbox_features(target_id, detections, 300, 300)
                    ref_bbox_features, ref_bbox = get_bbox_features(ref_id, detections, 300, 300)

                    # <<< NEW: Filter out samples with missing or tiny bounding boxes
                    if target_bbox is None or ref_bbox is None:
                        continue
                    
                    target_area = (target_bbox[2] - target_bbox[0]) * (target_bbox[3] - target_bbox[1])
                    ref_area = (ref_bbox[2] - ref_bbox[0]) * (ref_bbox[3] - ref_bbox[1])
                    if target_area < MIN_BBOX_AREA or ref_area < MIN_BBOX_AREA:
                        continue

                    # Check that objects are not too close to each other in 3D space
                    target_pos = np.array(list(target_obj['position'].values()))
                    ref_pos = np.array(list(ref_obj['position'].values()))
                    if np.linalg.norm(target_pos - ref_pos) < 1.0:
                        continue
                    
                    found_pair = True
                    break
                if found_pair:
                    break
            
            if not found_pair:
                continue

            agent_meta = event.metadata['agent']
            
            dist_to_target_ground_truth = np.linalg.norm(target_pos - agent_pos)
            dist_to_ref_ground_truth = np.linalg.norm(ref_pos - agent_pos)
            
            depth_frame = event.depth_frame
            if depth_frame is None:
                continue
            
            target_depth_features = get_enhanced_depth_features(
                depth_frame, target_bbox, target_obj['position'], agent_pos)
            ref_depth_features = get_enhanced_depth_features(
                depth_frame, ref_bbox, ref_obj['position'], agent_pos)
            
            # <<< CHANGED: Pass the valid bbox to the occlusion function
            target_occlusion = get_occlusion_features(target_obj, target_bbox)
            ref_occlusion = get_occlusion_features(ref_obj, ref_bbox)
            
            relative_depth = calculate_relative_depth_target(agent_pos, target_obj, ref_obj)
            
            # Filter out extreme or invalid values
            if relative_depth <= 0.01 or relative_depth > 20.0:
                continue
            
            if (target_depth_features.get('depth_gt_consistency', 0) > 2.0 or 
                ref_depth_features.get('depth_gt_consistency', 0) > 2.0):
                continue
            
            # Calculate world coordinates as usual
            target_world_pos = np.array([target_obj['position']['x'], 
                                        target_obj['position']['y'], 
                                        target_obj['position']['z']])

            # NEW: Transform to camera-relative coordinates
            from scipy.spatial.transform import Rotation as R

            # Get camera pose
            cam_pos = agent_pos  # Already defined earlier
            cam_rot = R.from_euler('xyz', 
                                [agent_meta['rotation']['x'], 
                                    agent_meta['rotation']['y'], 
                                    agent_meta['rotation']['z']], 
                                degrees=True)

            # Transform world coords to camera frame
            cam_to_world = cam_rot.as_matrix()
            world_to_cam = cam_to_world.T
            target_in_cam_frame = world_to_cam @ (target_world_pos - cam_pos)

            # ALSO transform reference object to camera frame
            ref_world_pos = np.array([ref_obj['position']['x'], 
                                    ref_obj['position']['y'], 
                                    ref_obj['position']['z']])
            ref_in_cam_frame = world_to_cam @ (ref_world_pos - cam_pos)

            # NEW: These camera-frame coords become the "world" coords for training
            # This makes it as if the camera was always at origin looking forward
            data_row = {
                'scene_name': scene_name,
                'target_object_type': target_obj['objectType'], 
                'ref_object_type': ref_obj['objectType'],
                
                # ... all your existing bbox, depth, occlusion features ...
                
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

                'target_depth_mean': target_depth_features.get('depth_mean'),
                'target_depth_std': target_depth_features.get('depth_std'),
                'target_depth_min': target_depth_features.get('depth_min'),
                'target_depth_max': target_depth_features.get('depth_max'),
                'target_depth_median': target_depth_features.get('depth_median'),
                'target_depth_percentile_25': target_depth_features.get('depth_percentile_25'),
                'target_depth_percentile_75': target_depth_features.get('depth_percentile_75'),
                'target_ground_truth_distance': target_depth_features.get('ground_truth_distance'),
                'target_depth_gt_consistency': target_depth_features.get('depth_gt_consistency'),
                
                'ref_depth_mean': ref_depth_features.get('depth_mean'),
                'ref_depth_std': ref_depth_features.get('depth_std'),
                'ref_depth_min': ref_depth_features.get('depth_min'),
                'ref_depth_max': ref_depth_features.get('depth_max'),
                'ref_depth_median': ref_depth_features.get('depth_median'),
                'ref_depth_percentile_25': ref_depth_features.get('depth_percentile_25'),
                'ref_depth_percentile_75': ref_depth_features.get('depth_percentile_75'),
                'ref_ground_truth_distance': ref_depth_features.get('ground_truth_distance'),
                'ref_depth_gt_consistency': ref_depth_features.get('depth_gt_consistency'),
                
                'target_is_occluded': target_occlusion['is_occluded'],
                'target_visibility_ratio': target_occlusion['visibility_ratio'],
                'ref_is_occluded': ref_occlusion['is_occluded'],
                'ref_visibility_ratio': ref_occlusion['visibility_ratio'],

                # CHANGED: Store camera as if at origin
                'camera_horizon': 0.0,  # Normalized to level
                'field_of_view': event.metadata['fov'],
                'agent_pos_x': 0.0,  # Camera at origin
                'agent_pos_y': 0.0, 
                'agent_pos_z': 0.0,
                'agent_rot_x': 0.0,  # Looking straight ahead
                'agent_rot_y': 0.0, 
                'agent_rot_z': 0.0,
                
                'dist_to_target': dist_to_target_ground_truth,
                'dist_to_ref': dist_to_ref_ground_truth,
                
                'relative_depth': relative_depth,
                
                # CHANGED: Use camera-frame coordinates
                'world_x': target_in_cam_frame[0], 
                'world_y': target_in_cam_frame[1], 
                'world_z': target_in_cam_frame[2],

                # ADD: Store reference object camera-frame coordinates too
                'ref_world_x': ref_in_cam_frame[0],
                'ref_world_y': ref_in_cam_frame[1],
                'ref_world_z': ref_in_cam_frame[2],

                # NEW: Transform target rotation to camera frame
                'target_rot_x': 0.0,  # Simplified - could transform if needed
                'target_rot_y': target_obj['rotation']['y'] - agent_meta['rotation']['y'], 
                'target_rot_z': 0.0,
                'ref_rot_x': 0.0,
                'ref_rot_y': ref_obj['rotation']['y'] - agent_meta['rotation']['y'],
                'ref_rot_z': 0.0,
                
                'target_base_y': calculate_base_position(target_obj, event.metadata['objects']),
                'target_on_object': classify_object_support(target_obj, event.metadata['objects']),
            }
            
            all_data.append(data_row)
            scene_samples += 1
            
            if scene_samples % 50 == 0:
                print(f"  {scene_name}: {scene_samples}/{MAX_SAMPLES_PER_SCENE} samples collected.")
        
        print(f"Scene {scene_name}: Generated {scene_samples} valid samples from {scene_attempts} attempts.")

    controller.stop()
    
    if not all_data:
        print("Error: No valid samples were generated! Try increasing MAX_SAMPLES_PER_SCENE * 5.")
        return
    
    df = pd.DataFrame(all_data)
    
    print(f"\n=== Dataset Generation Summary ===")
    print(f"Total samples generated: {len(df)}")
    print(f"Data columns: {list(df.columns)}")
    
    print(f"\nData Quality Check:")
    print(f"BBox widths > 0: {100 * (df['target_bbox_width'] > 0).mean():.1f}%")
    print(f"Depth std > 0: {100 * (df['target_depth_std'] > 0).mean():.1f}% of samples have depth variation.")
    
    df.to_csv(OUTPUT_FILENAME, index=False)
    print(f"\nDataset successfully saved as '{OUTPUT_FILENAME}'")

if __name__ == "__main__":
    main()