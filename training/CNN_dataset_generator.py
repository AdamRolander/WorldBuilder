import ai2thor.controller
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import math
import os
from PIL import Image

# --- Configuration ---
SCENES = [f"FloorPlan{i}" for i in range(1, 31)] 
MAX_SAMPLES_PER_SCENE = 400 
BASE_DIR = "data"
IMAGE_DIR = os.path.join(BASE_DIR, "images")
OUTPUT_CSV = os.path.join(BASE_DIR, "master.csv")
OCCLUSION_THRESHOLD = 0.8

def get_bbox_features(bbox, screen_width, screen_height):
    if not bbox or len(bbox) != 4:
        return {'center_x': 0, 'center_y': 0, 'width': 0, 'height': 0, 'aspect_ratio': 1}
    x1, y1, x2, y2 = bbox
    width = (x2 - x1) / screen_width
    height = (y2 - y1) / screen_height
    return {'center_x': ((x1 + x2) / 2) / screen_width, 'center_y': ((y1 + y2) / 2) / screen_height, 'width': width, 'height': height, 'aspect_ratio': width / height if height > 0 else 1}

def get_occlusion_features(obj_meta):
    bbox_3d_corners = obj_meta.get('axisAlignedBoundingBox', {}).get('cornerPoints')
    if not bbox_3d_corners:
        return {'is_occluded': 0, 'visibility_ratio': 1.0}
    
    # Calculate 2D bounding box area from 3D corner projections
    all_x = [p[0] for p in bbox_3d_corners]
    all_y = [p[1] for p in bbox_3d_corners]
    bbox_area = (max(all_x) - min(all_x)) * (max(all_y) - min(all_y))

    if bbox_area <= 0:
        return {'is_occluded': 0, 'visibility_ratio': 1.0}
        
    num_visible_pixels = obj_meta.get('numVisiblePixels', 0)
    visibility_ratio = num_visible_pixels / bbox_area
    is_occluded = 1 if visibility_ratio < OCCLUSION_THRESHOLD else 0
    return {'is_occluded': is_occluded, 'visibility_ratio': visibility_ratio}

def get_depth_features(depth_frame, bbox):
    if not bbox or len(bbox) != 4:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    depth_slice = depth_frame[y1:y2, x1:x2]
    if depth_slice.size == 0:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
    return {'mean': np.mean(depth_slice), 'std': np.std(depth_slice), 'min': np.min(depth_slice), 'max': np.max(depth_slice)}

def calculate_relative_depth_target(agent_pos, target_obj, ref_obj):
    target_pos = np.array(list(target_obj['position'].values()))
    ref_pos = np.array(list(ref_obj['position'].values()))
    dist_to_target = np.linalg.norm(target_pos - agent_pos)
    dist_to_ref = np.linalg.norm(ref_pos - agent_pos)
    return dist_to_target / dist_to_ref if dist_to_ref > 1e-6 else 0

def main():
    print("Generating dataset with cropped images and all features...")
    os.makedirs(IMAGE_DIR, exist_ok=True)
    controller = ai2thor.controller.Controller(
        scene="FloorPlan1",
        renderInstanceSegmentation=True,
        renderDepthImage=True
    )
    all_data = []
    
    for scene_idx, scene_name in enumerate(tqdm(SCENES, desc="Processing Scenes")):
        controller.reset(scene=scene_name)
        for pose_idx in range(MAX_SAMPLES_PER_SCENE):
            event = controller.step(action=random.choice(['MoveAhead', 'RotateRight', 'RotateLeft', 'LookUp', 'LookDown']))
            
            visible_objects = [obj for obj in event.metadata['objects'] if obj['visible']]
            if len(visible_objects) < 2: continue
            target_obj, ref_obj = random.sample(visible_objects, 2)
            
            # --- FIX: Use reliable 2D instance detections and add safety checks ---
            detections = event.instance_detections2D
            if target_obj['objectId'] not in detections or ref_obj['objectId'] not in detections:
                continue # Skip if the objects don't have a 2D detection
                
            target_bbox = detections[target_obj['objectId']]
            ref_bbox = detections[ref_obj['objectId']]
            
            # Safety check for valid bounding box dimensions
            if target_bbox[2] <= target_bbox[0] or target_bbox[3] <= target_bbox[1]:
                continue
            if ref_bbox[2] <= ref_bbox[0] or ref_bbox[3] <= ref_bbox[1]:
                continue
            
            # --- Continue with image cropping and feature extraction ---
            frame_pil = Image.fromarray(event.frame)
            target_crop = frame_pil.crop(target_bbox)
            ref_crop = frame_pil.crop(ref_bbox)
            
            target_img_name = f"{scene_idx}_{pose_idx}_target.png"
            ref_img_name = f"{scene_idx}_{pose_idx}_ref.png"
            target_img_path = os.path.join(IMAGE_DIR, target_img_name)
            ref_img_path = os.path.join(IMAGE_DIR, ref_img_name)
            target_crop.save(target_img_path)
            ref_crop.save(ref_img_path)

            agent_meta = event.metadata['agent']
            agent_pos = np.array(list(agent_meta['position'].values()))
            
            target_bbox_features = get_bbox_features(target_bbox, 300, 300)
            ref_bbox_features = get_bbox_features(ref_bbox, 300, 300)
            
            occlusion_target = get_occlusion_features(target_obj)
            occlusion_ref = get_occlusion_features(ref_obj)

            depth_frame = event.depth_frame
            target_depth_features = get_depth_features(depth_frame, target_bbox)
            ref_depth_features = get_depth_features(depth_frame, ref_bbox)
            dist_to_ref_ground_truth = np.linalg.norm(np.array(list(ref_obj['position'].values())) - agent_pos)

            if target_depth_features['mean'] == 0:
                target_depth_features['mean'] = np.linalg.norm(np.array(list(target_obj['position'].values())) - agent_pos)
            if ref_depth_features['mean'] == 0:
                ref_depth_features['mean'] = dist_to_ref_ground_truth

            data_row = {
                'target_image_path': target_img_path, 'ref_image_path': ref_img_path,
                'target_object_type': target_obj['objectType'], 'ref_object_type': ref_obj['objectType'],
                'target_bbox_center_x': target_bbox_features['center_x'], 'target_bbox_center_y': target_bbox_features['center_y'],
                'target_bbox_width': target_bbox_features['width'], 'target_bbox_height': target_bbox_features['height'],
                'target_bbox_aspect_ratio': target_bbox_features['aspect_ratio'],
                'ref_bbox_center_x': ref_bbox_features['center_x'], 'ref_bbox_center_y': ref_bbox_features['center_y'],
                'ref_bbox_width': ref_bbox_features['width'], 'ref_bbox_height': ref_bbox_features['height'],
                'ref_bbox_aspect_ratio': ref_bbox_features['aspect_ratio'],
                'target_depth_mean': target_depth_features['mean'], 'target_depth_std': target_depth_features['std'],
                'ref_depth_mean': ref_depth_features['mean'],
                'target_is_occluded': occlusion_target['is_occluded'], 'target_visibility_ratio': occlusion_target['visibility_ratio'],
                'ref_is_occluded': occlusion_ref['is_occluded'], 'ref_visibility_ratio': occlusion_ref['visibility_ratio'],
                'camera_horizon': agent_meta['cameraHorizon'], 'field_of_view': event.metadata['fov'],
                'agent_pos_x': agent_meta['position']['x'], 'agent_pos_y': agent_meta['position']['y'], 'agent_pos_z': agent_meta['position']['z'],
                'agent_rot_x': agent_meta['rotation']['x'], 'agent_rot_y': agent_meta['rotation']['y'], 'agent_rot_z': agent_meta['rotation']['z'],
                'dist_to_ref': dist_to_ref_ground_truth,
                'relative_depth': calculate_relative_depth_target(agent_pos, target_obj, ref_obj),
                'world_x': target_obj['position']['x'], 'world_y': target_obj['position']['y'], 'world_z': target_obj['position']['z'],
            }
            all_data.append(data_row)

    controller.stop()
    df = pd.DataFrame(all_data)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDataset generation complete! Generated {len(df)} samples.")
    print(f"Data saved to {BASE_DIR}/")

if __name__ == "__main__":
    main()