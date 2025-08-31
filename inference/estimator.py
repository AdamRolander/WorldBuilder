import torch
import numpy as np
from PIL import Image
import requests
import json
from sklearn.preprocessing import StandardScaler

# Import your geometric consistency model class
# Assuming it's in the same directory or properly imported
class GeometricConsistencyEstimator(torch.nn.Module):
    """Recreate the model class for loading - must match training definition."""
    
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.4):
        super().__init__()
        
        # First predict relative depth
        self.depth_network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.Dropout(dropout_rate),
            
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.Dropout(dropout_rate),
            
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_dim // 2),
            torch.nn.Dropout(dropout_rate),
            
            torch.nn.Linear(hidden_dim // 2, 1),
            torch.nn.ReLU()
        )
        
        # Then use geometric reasoning for world coordinates
        coord_input_dim = input_dim + 1  # +1 for predicted relative depth
        self.coord_network = torch.nn.Sequential(
            torch.nn.Linear(coord_input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.Dropout(dropout_rate),
            
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(hidden_dim // 2),
            torch.nn.Dropout(dropout_rate),
            
            torch.nn.Linear(hidden_dim // 2, 3)  # x, y, z world coordinates
        )
    
    def forward(self, x):
        # First predict relative depth
        relative_depth = self.depth_network(x).squeeze(-1)
        
        # Then predict world coordinates using the predicted relative depth
        coord_input = torch.cat([x, relative_depth.unsqueeze(1)], dim=1)
        world_coords = self.coord_network(coord_input)
        
        return relative_depth, world_coords


# --- 1. LOAD GEOMETRIC CONSISTENCY MODEL ---
print("Loading geometric consistency model...")
checkpoint = torch.load('world_coordinate_model_geometric.pth', map_location=torch.device('cpu'), weights_only=False)

model = GeometricConsistencyEstimator(input_dim=len(checkpoint['feature_names']))
model.load_state_dict(checkpoint['model_state_dict'])
scaler = checkpoint['scaler']
coord_scaler = checkpoint['coord_scaler']
feature_names = checkpoint['feature_names']
model.eval()
print("Geometric model loaded successfully.")


# --- 2. GATHER INPUT DATA (Same as before) ---
panoptic_segments = [
    {'id': 1, 'label_id': 57, 'bounding_box': [269, 437, 607, 327], 'mask_area': 137615}, # couch
    {'id': 2, 'label_id': 120, 'bounding_box': [954, 533, 449, 238], 'mask_area': 73117}, # cabinet-merged
    {'id': 3, 'label_id': 92, 'bounding_box': [800, 249, 114, 103], 'mask_area': 9880}, # light
    {'id': 7, 'label_id': 121, 'bounding_box': [648, 643, 359, 195], 'mask_area': 29127}, # table-merged
    {'id': 8, 'label_id': 58, 'bounding_box': [107, 328, 196, 398], 'mask_area': 42743}, # potted plant
]

llm_data_string = """
[
  {"box_2d": 1, "label": "couch", "size_in_feet": [6, 3]},
  {"box_2d": 2, "label": "cabinet-merged", "size_in_feet": [4.5, 3.5]},
  {"box_2d": 3, "label": "light", "size_in_feet": [1, 1]},
  {"box_2d": 7, "label": "table-merged", "size_in_feet": [3.5, 2]},
  {"box_2d": 8, "label": "potted plant", "size_in_feet": [1.2, 1.2]}
]
"""
llm_objects = json.loads(llm_data_string)
llm_sizes = {item["box_2d"]: (np.array(item["size_in_feet"]) * 0.3048).tolist() for item in llm_objects}

img_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
img = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
img_width, img_height = img.size


# --- 3. SETUP CAMERA PARAMETERS (Enhanced for Geometric Model) ---
ASSUMED_FOCAL_LENGTH_MM = 50
ASSUMED_SENSOR_WIDTH_MM = 36
fx = (ASSUMED_FOCAL_LENGTH_MM * img_width) / ASSUMED_SENSOR_WIDTH_MM
fy = fx
cx = img_width / 2
cy = img_height / 2

# Create camera intrinsics matrix
camera_intrinsics = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])

# Assume camera pose (identity - camera at origin looking down -Z axis)
# In practice, you'd get this from SLAM, SfM, or manual calibration
camera_pose = np.eye(4)  # Identity matrix = camera at world origin


# --- 4. ENHANCED FEATURE EXTRACTION FOR GEOMETRIC MODEL ---
def extract_features_for_geometric_inference(target_segment, ref_segment, target_size, ref_size, 
                                           camera_intrinsics, camera_pose, image_dims):
    """
    Extract features for the geometric consistency model.
    This model needs: [target_features, ref_features, comparative_features, camera_features]
    """
    img_w, img_h = image_dims
    
    def extract_object_features(segment, size_data):
        """Extract features for a single object."""
        bbox_xywh = segment['bounding_box']
        bbox_width = bbox_xywh[2]
        bbox_height = bbox_xywh[3]
        pixel_count = segment['mask_area']

        bbox_area = bbox_width * bbox_height
        bbox_aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 1.0
        pixel_density = pixel_count / bbox_area if bbox_area > 0 else 0.0

        longest_3d_dim = max(size_data) if size_data else 1.0
        longest_2d_dim_px = max(bbox_width, bbox_height)
        analytical_depth = (longest_3d_dim * fx) / longest_2d_dim_px if longest_2d_dim_px > 0 else 10.0

        center_x = bbox_xywh[0] + bbox_width / 2
        center_y = bbox_xywh[1] + bbox_height / 2
        norm_pixel_x = center_x / img_w
        norm_pixel_y = center_y / img_h
        norm_dist_from_center = np.sqrt((norm_pixel_x - 0.5)**2 + (norm_pixel_y - 0.5)**2) * np.sqrt(2)

        return np.array([
            pixel_count, longest_3d_dim, bbox_width, bbox_height, 
            bbox_aspect_ratio, pixel_density, analytical_depth, 
            norm_pixel_x, norm_pixel_y, norm_dist_from_center
        ])
    
    def extract_comparative_features(target_seg, ref_seg, target_sz, ref_sz):
        """Extract comparative features between target and reference objects."""
        target_bbox = target_seg['bounding_box']
        ref_bbox = ref_seg['bounding_box']
        
        # Size ratios
        bbox_width_ratio = target_bbox[2] / ref_bbox[2] if ref_bbox[2] > 0 else 1.0
        bbox_height_ratio = target_bbox[3] / ref_bbox[3] if ref_bbox[3] > 0 else 1.0
        pixel_count_ratio = target_seg['mask_area'] / ref_seg['mask_area'] if ref_seg['mask_area'] > 0 else 1.0
        
        # True size ratio
        target_true_size = max(target_sz) if target_sz else 1.0
        ref_true_size = max(ref_sz) if ref_sz else 1.0
        true_size_ratio = target_true_size / ref_true_size if ref_true_size > 0 else 1.0
        
        # Spatial relationship
        target_center = np.array([target_bbox[0] + target_bbox[2]/2, target_bbox[1] + target_bbox[3]/2])
        ref_center = np.array([ref_bbox[0] + ref_bbox[2]/2, ref_bbox[1] + ref_bbox[3]/2])
        pixel_distance = np.linalg.norm(target_center - ref_center)
        
        # Relative position (normalized)
        rel_x = (target_center[0] - ref_center[0]) / img_w
        rel_y = (target_center[1] - ref_center[1]) / img_h
        
        return np.array([
            bbox_width_ratio, bbox_height_ratio, pixel_count_ratio, 
            true_size_ratio, pixel_distance, rel_x, rel_y
        ])
    
    def extract_camera_features(cam_intrinsics, cam_pose):
        """Extract camera-specific features."""
        # Camera position (translation)
        camera_position = cam_pose[:3, 3]
        
        # Camera orientation (rotation matrix elements)
        rotation_matrix = cam_pose[:3, :3]
        orientation_features = rotation_matrix.flatten()[:6]  # First 6 elements
        
        # Intrinsic parameters
        fx_val, fy_val = cam_intrinsics[0, 0], cam_intrinsics[1, 1]
        cx_val, cy_val = cam_intrinsics[0, 2], cam_intrinsics[1, 2]
        
        return np.concatenate([
            camera_position,      # 3 features
            orientation_features, # 6 features
            [fx_val, fy_val, cx_val, cy_val]  # 4 features
        ])
    
    # Extract all feature components
    target_features = extract_object_features(target_segment, target_size)
    ref_features = extract_object_features(ref_segment, ref_size)
    comparative_features = extract_comparative_features(target_segment, ref_segment, target_size, ref_size)
    camera_features = extract_camera_features(camera_intrinsics, camera_pose)
    
    # Combine all features
    combined_features = np.concatenate([
        target_features, ref_features, comparative_features, camera_features
    ])
    
    return combined_features


# --- 5. INFERENCE WITH GEOMETRIC MODEL ---
def perform_geometric_inference(segments, sizes, camera_intrinsics, camera_pose, image_dims):
    """
    Perform inference using the geometric consistency model.
    Uses largest object as reference (similar to training approach).
    """
    # Filter valid segments
    valid_segments = [(seg, sizes[seg['id']]) for seg in segments if seg['id'] in sizes]
    
    if len(valid_segments) < 2:
        print("Need at least 2 valid objects for relative depth estimation.")
        return []
    
    # Choose reference object (largest bbox area)
    ref_segment, ref_size = max(valid_segments, key=lambda x: x[0]['bounding_box'][2] * x[0]['bounding_box'][3])
    
    print(f"Using object ID {ref_segment['id']} as reference object.")
    
    results = []
    for target_segment, target_size in valid_segments:
        if target_segment['id'] == ref_segment['id']:
            # For the reference object, we can still predict but relative depth will be ~1.0
            continue
        
        # Extract features for this target-reference pair
        features = extract_features_for_geometric_inference(
            target_segment, ref_segment, target_size, ref_size,
            camera_intrinsics, camera_pose, image_dims
        )
        
        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Predict with geometric model
        with torch.no_grad():
            relative_depth, world_coords_scaled = model(torch.FloatTensor(features_scaled))
            
            # Unscale world coordinates
            world_coords = coord_scaler.inverse_transform(world_coords_scaled.numpy())
            
            relative_depth_val = relative_depth.item()
            world_x, world_y, world_z = world_coords[0]
        
        # Also calculate absolute depth estimate for comparison
        # (This is approximate - you'd need reference object's true depth)
        ref_bbox = ref_segment['bounding_box']
        ref_longest_3d = max(ref_size)
        ref_longest_2d = max(ref_bbox[2], ref_bbox[3])
        ref_estimated_depth = (ref_longest_3d * fx) / ref_longest_2d
        
        estimated_absolute_depth = relative_depth_val * ref_estimated_depth
        
        results.append({
            'id': target_segment['id'],
            'relative_depth': relative_depth_val,
            'estimated_absolute_depth': estimated_absolute_depth,
            'world_coordinates': [world_x, world_y, world_z],
            'reference_object_id': ref_segment['id']
        })
    
    # Add reference object with relative depth = 1.0
    ref_bbox = ref_segment['bounding_box']
    ref_longest_3d = max(ref_size)
    ref_longest_2d = max(ref_bbox[2], ref_bbox[3])
    ref_depth = (ref_longest_3d * fx) / ref_longest_2d
    
    # For reference object, world coordinates would be approximately at camera forward direction
    ref_center_x = ref_bbox[0] + ref_bbox[2] / 2
    ref_center_y = ref_bbox[1] + ref_bbox[3] / 2
    ref_world_x = (ref_center_x - cx) * ref_depth / fx
    ref_world_y = -(ref_center_y - cy) * ref_depth / fy
    ref_world_z = ref_depth  # Positive Z in our coordinate system
    
    results.append({
        'id': ref_segment['id'],
        'relative_depth': 1.0,  # Reference object
        'estimated_absolute_depth': ref_depth,
        'world_coordinates': [ref_world_x, ref_world_y, ref_world_z],
        'reference_object_id': ref_segment['id'],
        'is_reference': True
    })
    
    return results


# --- 6. RUN INFERENCE ---
print("\n--- Running Geometric Consistency Inference ---")
reconstruction_results = perform_geometric_inference(
    panoptic_segments, llm_sizes, camera_intrinsics, camera_pose, (img_width, img_height)
)

print("\n--- 3D Reconstruction Results (Geometric Model) ---")
for result in reconstruction_results:
    coords = result['world_coordinates']
    ref_status = " (REFERENCE)" if result.get('is_reference', False) else ""
    print(f"Object ID {result['id']}{ref_status}:")
    print(f"  > Relative Depth: {result['relative_depth']:.3f}")
    print(f"  > Estimated Absolute Depth: {result['estimated_absolute_depth']:.2f} meters")
    print(f"  > World Coordinates (X,Y,Z): ({coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f}) meters")
    if not result.get('is_reference', False):
        print(f"  > Relative to Object ID: {result['reference_object_id']}")
    print()

print("--- Summary ---")
print(f"Successfully reconstructed {len(reconstruction_results)} objects using geometric consistency model.")
print("The model predicts both relative depth relationships AND absolute world coordinates.")
print("World coordinates are in meters, assuming the camera is at the origin.")