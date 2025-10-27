import torch
import torch.nn as nn
from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import json
import os
import math
from scipy.spatial.transform import Rotation as R
import random
import pandas as pd
from typing import List, Dict, Any
from dotenv import load_dotenv
import cv2

load_dotenv()

def normalize_midas_to_metric_depth(midas_depth_map, reference_object_depth_estimate, reference_midas_median):
    """
    Convert MiDaS inverse depth (disparity) to metric depth using a reference object.
    
    MiDaS outputs inverse depth where:
    - Higher values = closer objects
    - Scale is arbitrary and image-dependent
    
    We calibrate using a reference object with known/estimated depth.
    
    Args:
        midas_depth_map: Raw MiDaS output (NxM array, inverse depth)
        reference_object_depth_estimate: Estimated metric depth of reference (meters)
        reference_midas_median: Median MiDaS value for reference object bbox
        
    Returns:
        Metric depth map in meters (NxM array)
    """
    # Avoid division by zero
    safe_midas_map = np.clip(midas_depth_map, 0.1, None)
    
    # Calibration constant: depth * inverse_depth = k
    k = reference_object_depth_estimate * reference_midas_median
    
    # Convert: metric_depth = k / inverse_depth
    metric_depth_map = k / safe_midas_map
    
    return metric_depth_map

def estimate_depth_with_midas(image_path: str, bbox_data: dict, img_width: int, img_height: int, 
                              midas_model, midas_transform, depth_map_cache: dict = None) -> float:
    """
    Estimate depth for a bounding box using MiDaS depth estimation.
    
    Args:
        image_path: Path to the input image
        bbox_data: Dictionary with bbox coordinates (x_min, y_min, width, height)
        img_width: Image width
        img_height: Image height
        midas_model: Loaded MiDaS model
        midas_transform: MiDaS transform function
        depth_map_cache: Optional cache for depth maps
        
    Returns:
        Estimated depth in meters (relative scale)
    """
    print(f"[DEBUG] estimate_depth_with_midas called for {image_path}")
    print(f"[DEBUG] Cache exists: {depth_map_cache is not None}")
    print(f"[DEBUG] Image in cache: {image_path in depth_map_cache if depth_map_cache else False}")

    if depth_map_cache is not None and image_path in depth_map_cache:
        depth_map = depth_map_cache[image_path]
        print(f"[DEBUG] Using cached depth map, shape: {depth_map.shape}")
    else:
        print(f"[DEBUG] Generating new depth map...")
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        print(f"[DEBUG] Image loaded, shape: {img.shape}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply MiDaS transform - this returns a tensor ready for the model
        input_batch = midas_transform(img)
        
        # Move to device and ensure correct shape [1, 3, H, W]
        if input_batch.dim() == 3:
            input_batch = input_batch.unsqueeze(0)
        
        device = next(midas_model.parameters()).device
        input_batch = input_batch.to(device)
        
        # Predict depth
        with torch.no_grad():
            prediction = midas_model(input_batch)
            
            # Interpolate to original image size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()

        # Cache it
        if depth_map_cache is not None:
            depth_map_cache[image_path] = depth_map
    
    # Extract depth values within the bounding box
    x_min = int(bbox_data['x_min'])
    y_min = int(bbox_data['y_min'])
    x_max = int(x_min + bbox_data['width'])
    y_max = int(y_min + bbox_data['height'])
    
    # Clamp to image bounds
    x_min = max(0, min(x_min, img_width - 1))
    x_max = max(0, min(x_max, img_width))
    y_min = max(0, min(y_min, img_height - 1))
    y_max = max(0, min(y_max, img_height))
    
    # Ensure valid bbox
    if x_max <= x_min or y_max <= y_min:
        print(f"Warning: Invalid bbox bounds after clamping: ({x_min}, {y_min}, {x_max}, {y_max})")
        return 3.0  # Return default depth
    
    # Get median depth in bbox (more robust than mean)
    bbox_depth_values = depth_map[y_min:y_max, x_min:x_max]
    if bbox_depth_values.size == 0:
        print(f"Warning: Empty bbox region")
        return 3.0
    
    median_depth = np.median(bbox_depth_values)
    
    return float(median_depth)

# --- 1. RECREATE NECESSARY CLASSES FROM enhanced_attention.py ---

# NOTE: These classes are re-defined here exactly as they were in the training script
# to ensure the torch.load() function can successfully load the state_dict.

class MultiScaleAttentionBlock(nn.Module):
    """Multi-scale attention that processes features at different granularities."""
    def __init__(self, input_dim, num_heads=8, dropout_rate=0.1):
        super().__init__()
    
        self.input_dim = input_dim
        self.feat_dim = 64
        
        possible_heads = [1, 2, 4, 8, 16]
        self.heads_per_group = max([h for h in possible_heads if h <= num_heads and self.feat_dim % h == 0])
        
        # print(f"MultiScaleAttention: input_dim={input_dim}, feat_dim={self.feat_dim}, heads_per_group={self.heads_per_group}")
        
        # Group features by type for specialized attention
        self.geometric_attention = nn.MultiheadAttention(
            embed_dim=self.feat_dim, num_heads=self.heads_per_group, 
            dropout=dropout_rate, batch_first=True
        )
        self.visual_attention = nn.MultiheadAttention(
            embed_dim=self.feat_dim, num_heads=self.heads_per_group, 
            dropout=dropout_rate, batch_first=True
        )
        self.camera_attention = nn.MultiheadAttention(
            embed_dim=self.feat_dim, num_heads=self.heads_per_group, 
            dropout=dropout_rate, batch_first=True
        )
        
        # Cross-attention to combine different feature types
        combined_dim = self.feat_dim * 3
        cross_heads = min(num_heads, combined_dim)
        while combined_dim % cross_heads != 0 and cross_heads > 1:
            cross_heads -= 1
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=combined_dim, num_heads=cross_heads, 
            dropout=dropout_rate, batch_first=True
        )
        
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.feat_dim) for _ in range(3)])
        self.final_norm = nn.LayerNorm(combined_dim)
        
        # Feature type projections
        self.geometric_proj = nn.Linear(input_dim, self.feat_dim)
        self.visual_proj = nn.Linear(input_dim, self.feat_dim)
        self.camera_proj = nn.Linear(input_dim, self.feat_dim)
        self.combine_proj = nn.Linear(combined_dim, input_dim)
        
    def forward(self, x):
        # Project to different feature spaces
        geo_feat = self.geometric_proj(x).unsqueeze(1)
        vis_feat = self.visual_proj(x).unsqueeze(1)
        cam_feat = self.camera_proj(x).unsqueeze(1)
        
        # Apply specialized attention
        geo_attn, _ = self.geometric_attention(geo_feat, geo_feat, geo_feat)
        vis_attn, _ = self.visual_attention(vis_feat, vis_feat, vis_feat)
        cam_attn, _ = self.camera_attention(cam_feat, cam_feat, cam_feat)
        
        # Layer normalization with residuals
        geo_out = self.layer_norms[0](geo_feat.squeeze(1) + geo_attn.squeeze(1))
        vis_out = self.layer_norms[1](vis_feat.squeeze(1) + vis_attn.squeeze(1))
        cam_out = self.layer_norms[2](cam_feat.squeeze(1) + cam_attn.squeeze(1))
        
        # Combine all features
        combined = torch.cat([geo_out, vis_out, cam_out], dim=1)
        combined_proj = combined.unsqueeze(1)
        
        # Final cross-attention
        final_attn, _ = self.cross_attention(combined_proj, combined_proj, combined_proj)
        output = self.final_norm(combined_proj.squeeze(1) + final_attn.squeeze(1))
        
        # Project back to input dimension
        output = self.combine_proj(output)
        
        return output


class EnhancedGeometricEstimator(nn.Module):
    """Enhanced model with multi-scale attention and uncertainty estimation including angle prediction."""
    def __init__(self, input_dim, hidden_dim=512, dropout_rate=0.2, num_heads=8, use_uncertainty=True):
        super().__init__()
        
        self.use_uncertainty = use_uncertainty
        
        # Multi-scale attention block
        self.multi_attention = MultiScaleAttentionBlock(input_dim, num_heads, dropout_rate)
        
        # Shared encoder with residual connections
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ) for i in range(3)
        ])
        
        # Head definitions (simplified for brevity, matching training script structure)
        if use_uncertainty:
            self.depth_mean_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))
            self.depth_var_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1), nn.Softplus())
            self.coord_mean_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 3))
            self.coord_var_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 3), nn.Softplus())
            self.angle_mean_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))
            self.angle_var_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1), nn.Softplus())
        else:
            self.depth_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))
            self.coord_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 3))
            self.angle_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))
        
    def forward(self, x, return_uncertainty=None):
        if return_uncertainty is None:
            return_uncertainty = self.use_uncertainty
            
        x_att = self.multi_attention(x)
        
        features = x_att
        for layer in self.encoder_layers:
            new_features = layer(features)
            if new_features.shape == features.shape:
                features = features + new_features
            else:
                features = new_features
        
        if self.use_uncertainty and return_uncertainty:
            # Uncertainty predictions
            depth_mean = self.depth_mean_head(features).squeeze(-1)
            coord_mean = self.coord_mean_head(features)
            angle_mean = self.angle_mean_head(features).squeeze(-1)
            depth_var = self.depth_var_head(features).squeeze(-1)
            coord_var = self.coord_var_head(features)
            angle_var = self.angle_var_head(features).squeeze(-1)
            return depth_mean, coord_mean, angle_mean, depth_var, coord_var, angle_var
        else:
            if self.use_uncertainty:
                depth_pred = self.depth_mean_head(features).squeeze(-1)
                coord_pred = self.coord_mean_head(features)
                angle_pred = self.angle_mean_head(features).squeeze(-1)
            else:
                depth_pred = self.depth_head(features).squeeze(-1)
                coord_pred = self.coord_head(features)
                angle_pred = self.angle_head(features).squeeze(-1)
            return depth_pred, coord_pred, angle_pred

# --- 2. ADAPTED FEATURE ENGINEERING FUNCTIONS ---

def calculate_unprojection(row, img_width=300, img_height=300):
    """Calculates an initial 3D world coordinate guess using the unprojection formula."""
    
    # Get Camera Intrinsics
    fov_rad = row['field_of_view'] * (math.pi / 180.0)
    focal_length = (img_width / 2.0) / math.tan(fov_rad / 2.0)
    cx = img_width / 2.0
    cy = img_height / 2.0
    
    # K_inv calculation
    K_inv = np.linalg.inv(np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ]))

    # Get 2D Pixel Coordinates and an Estimated Depth
    px = row['target_bbox_center_x'] * img_width
    py = row['target_bbox_center_y'] * img_height

    if 'target_depth_median' in row:
        depth = row['target_depth_median']
    else:
        depth = row.get('dist_to_target', row.get('dist_to_ref', 3.0))
    
    print(f"    [Unprojection] Using depth: {row.get('dist_to_ref', 'MISSING')} for {row.get('target_object_id', 'unknown')}")

    # Unproject from 2D to 3D (in Camera's coordinate system)
    pixel_coords = np.array([px, py, 1])
    camera_coords = K_inv @ pixel_coords * depth
    
    # Get Camera Extrinsics (Position and Rotation - ASSUMED)
    rotation = R.from_euler('xyz', [row['agent_rot_x'], row['agent_rot_y'], row['agent_rot_z']], degrees=True)
    rotation_matrix = rotation.as_matrix()
    translation_vector = np.array([row['agent_pos_x'], row['agent_pos_y'], row['agent_pos_z']])

    # Transform from Camera Coordinates to World Coordinates
    world_coords = rotation_matrix @ camera_coords + translation_vector
    
    return world_coords[0], world_coords[1], world_coords[2]


def calculate_comprehensive_geometric_features(row, img_width=300, img_height=300):
    """
    Calculate comprehensive features for coordinate prediction.
    Unprojection is now a FEATURE (input), not a target to correct.
    """
    features = {}
    
    # 1. Unprojection coordinates as FEATURES (not targets)
    proj_x, proj_y, proj_z = calculate_unprojection(row, img_width, img_height)
    features.update({
        'unproj_x': proj_x,
        'unproj_y': proj_y,
        'unproj_z': proj_z
    })
    
    # 2. Bounding box geometry features (CRITICAL for bbox quality assessment)
    bbox_center_x = max(0.001, min(0.999, row.get('target_bbox_center_x', 0.5)))
    bbox_center_y = max(0.001, min(0.999, row.get('target_bbox_center_y', 0.5)))
    bbox_width = row.get('target_bbox_width', 0.1)
    bbox_height = row.get('target_bbox_height', 0.1)
    
    ref_bbox_center_x = max(0.001, min(0.999, row.get('ref_bbox_center_x', 0.5)))
    ref_bbox_center_y = max(0.001, min(0.999, row.get('ref_bbox_center_y', 0.5)))
    ref_bbox_width = row.get('ref_bbox_width', 0.1)
    ref_bbox_height = row.get('ref_bbox_height', 0.1)
    
    features.update({
        # Target bbox features
        'target_bbox_center_x': bbox_center_x,
        'target_bbox_center_y': bbox_center_y,
        'target_bbox_width': bbox_width,
        'target_bbox_height': bbox_height,
        'target_bbox_area': bbox_width * bbox_height,
        'target_bbox_aspect_ratio': bbox_width / max(bbox_height, 1e-6),
        
        # Reference bbox features
        'ref_bbox_center_x': ref_bbox_center_x,
        'ref_bbox_center_y': ref_bbox_center_y,
        'ref_bbox_width': ref_bbox_width,
        'ref_bbox_height': ref_bbox_height,
        'ref_bbox_area': ref_bbox_width * ref_bbox_height,
        'ref_bbox_aspect_ratio': ref_bbox_width / max(ref_bbox_height, 1e-6),
        
        # Relative bbox features
        'bbox_area_ratio': (bbox_width * bbox_height) / max(ref_bbox_width * ref_bbox_height, 1e-6),
        'bbox_width_ratio': bbox_width / max(ref_bbox_width, 1e-6),
        'bbox_height_ratio': bbox_height / max(ref_bbox_height, 1e-6),
        
        # Geometric relationships
        'center_offset_x': bbox_center_x - 0.5,
        'center_offset_y': bbox_center_y - 0.5,
        'center_distance': np.sqrt((bbox_center_x - 0.5)**2 + (bbox_center_y - 0.5)**2),
        'angle_from_center': np.arctan2(bbox_center_y - 0.5, bbox_center_x - 0.5),
        
        'ref_center_offset_x': ref_bbox_center_x - 0.5,
        'ref_center_offset_y': ref_bbox_center_y - 0.5,
    })
    
    # 3. Enhanced depth features
    depth_cols = ['depth_mean', 'depth_std', 'depth_min', 'depth_max', 
                'depth_median', 'depth_percentile_25', 'depth_percentile_75']

    # INFERENCE: Use actual MiDaS depth statistics from MASK region
    if 'normalized_metric_depth_map' in row and row['normalized_metric_depth_map'] is not None:
        
        full_depth_map = row['normalized_metric_depth_map']
        segmentation_map = row['segmentation_map'] # The full segmentation map tensor

        # --- TARGET MASK-BASED DEPTH EXTRACTION ---
        target_mask = (segmentation_map.cpu().numpy() == row['target_segment_id'])
        target_depth_region = full_depth_map[target_mask]
        
        # --- REFERENCE MASK-BASED DEPTH EXTRACTION ---
        ref_mask = (segmentation_map.cpu().numpy() == row['ref_segment_id'])
        ref_depth_region = full_depth_map[ref_mask]

        if target_depth_region.size > 0 and ref_depth_region.size > 0:
            # ✅ FILTER OUTLIERS from both target and reference depth regions
            def filter_depth_outliers(depth_values):
                """Remove extreme outliers using IQR method."""
                if depth_values.size < 10:
                    return depth_values
                    
                q1 = np.percentile(depth_values, 25)
                q3 = np.percentile(depth_values, 75)
                iqr = q3 - q1
                
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                filtered = depth_values[
                    (depth_values >= lower_bound) & 
                    (depth_values <= upper_bound)
                ]
                
                # Only use filtered if we retain enough data
                return filtered if filtered.size > max(10, depth_values.size * 0.3) else depth_values
            
            # Apply filtering
            target_depth_region = filter_depth_outliers(target_depth_region)
            ref_depth_region = filter_depth_outliers(ref_depth_region)
            
            # Compute actual statistics from MASKED MiDaS depth map
            target_depth_stats = {
                'depth_mean': np.mean(target_depth_region),
                'depth_std': np.std(target_depth_region),
                'depth_min': np.min(target_depth_region),
                'depth_max': np.max(target_depth_region),
                'depth_median': np.median(target_depth_region),
                'depth_percentile_25': np.percentile(target_depth_region, 25),
                'depth_percentile_75': np.percentile(target_depth_region, 75)
            }
            
            ref_depth_stats = {
                'depth_mean': np.mean(ref_depth_region),
                'depth_std': np.std(ref_depth_region),
                'depth_min': np.min(ref_depth_region),
                'depth_max': np.max(ref_depth_region),
                'depth_median': np.median(ref_depth_region),
                'depth_percentile_25': np.percentile(ref_depth_region, 25),
                'depth_percentile_75': np.percentile(ref_depth_region, 75)
            }
            
            # ✅ ADD DEBUG OUTPUT TO VERIFY FILTERING
            print(f"    [Feature Calc] Target mask pixels: {target_depth_region.size}")
            print(f"    [Feature Calc] Target depth median: {target_depth_stats['depth_median']:.3f}m")
            print(f"    [Feature Calc] Ref mask pixels: {ref_depth_region.size}")
        else:
            # Fallback for empty mask (shouldn't happen)
            target_depth_stats = ref_depth_stats = {
                'depth_mean': row.get('dist_to_ref', 3.0), 'depth_std': 0.1, 
                'depth_min': 0.1, 'depth_max': 5.0, 'depth_median': row.get('dist_to_ref', 3.0),
                'depth_percentile_25': row.get('dist_to_ref', 3.0) - 0.2, 
                'depth_percentile_75': row.get('dist_to_ref', 3.0) + 0.2
            }

    for col in depth_cols:
        features[f'target_{col}'] = target_depth_stats[col]
        features[f'ref_{col}'] = ref_depth_stats[col]
        
        # Use median for ratios instead of raw values (more stable)
        if col == 'depth_mean' or col == 'depth_median':
            # For critical depth features, use median-based calculations
            ratio_numerator = target_depth_stats['depth_median']
            ratio_denominator = max(1e-6, ref_depth_stats['depth_median'])
        else:
            ratio_numerator = target_depth_stats[col]
            ratio_denominator = max(1e-6, ref_depth_stats[col])
        
        features[f'{col}_ratio'] = ratio_numerator / ratio_denominator
        features[f'{col}_diff'] = target_depth_stats[col] - ref_depth_stats[col]

    # Ground truth consistency (set to neutral values)
    features['target_gt_distance'] = 0.0
    features['ref_gt_distance'] = 0.0
    features['target_depth_consistency'] = 0.0
    features['ref_depth_consistency'] = 0.0
    
    # 4. Ground truth consistency features (if available)
    if 'target_ground_truth_distance' in row:
        features['target_gt_distance'] = row['target_ground_truth_distance']
        features['ref_gt_distance'] = row['ref_ground_truth_distance']
        features['target_depth_consistency'] = row.get('target_depth_gt_consistency', 0)
        features['ref_depth_consistency'] = row.get('ref_depth_gt_consistency', 0)
    
    # 5. Camera features (simplified - no rotation since camera is at origin)
    fov_rad = row['field_of_view'] * (np.pi / 180.0)
    features.update({
        'fov_rad': fov_rad,
        'focal_length_normalized': 1.0 / np.tan(fov_rad / 2.0)
    })
    
    # REMOVED: horizon_sin, horizon_cos, rot_y_sin, rot_y_cos
    # These were confusing since camera is always at origin in training
    
    return features


def calculate_bounding_box_from_mask(mask_tensor):
    """Calculates the bounding box [x, y, width, height] from a boolean mask tensor."""
    if not mask_tensor.any():
        return None
        
    coords = torch.nonzero(mask_tensor)
    y_min, x_min = coords.min(dim=0).values
    y_max, x_max = coords.max(dim=0).values
    
    width = (x_max - x_min + 1).item()
    height = (y_max - y_min + 1).item()
    
    # Return normalized coordinates (center_x, center_y, width, height)
    # Required by the feature extractor in the training script
    
    return {
        'x_min': x_min.item(), 'y_min': y_min.item(), 'width': width, 'height': height,
        'center_x': (x_min.item() + width / 2),
        'center_y': (y_min.item() + height / 2)
    }

# --- 3. LLM INTEGRATION FOR SIZE ESTIMATION (REQUIRES GOOGLE-GENAI) ---

def call_llm_for_size_estimation(segmented_objects: List[Dict[str, Any]], image_path: str) -> Dict[int, List[float]]:
    """
    Calls the Gemini API to estimate the real-world size/dimensions of each object.
    
    Args:
        segmented_objects: List of dicts, each containing 'id', 'label', and 'normalized_bbox'.
        image_path: Path to the input image for context.
        
    Returns:
        A dictionary mapping object ID to a list of [length, width, height] in meters.
    """
    try:
        from google import genai
        from google.genai.errors import APIError
    except ImportError:
        print("\nERROR: google-genai library not found.")
        print("Please install it: pip install google-genai")
        return {}
        
    # Check for API Key
    if 'GEMINI_API_KEY' not in os.environ:
        print("\nERROR: GEMINI_API_KEY environment variable not set.")
        print("Please set your API key to proceed with LLM size estimation.")
        return {}

    print("Connecting to Gemini API for size estimation...")
    client = genai.Client()
    
    # 1. Format the segmented objects data
    object_list_str = "\n".join([
        f"- ID {obj['id']}: {obj['label']} (normalized bbox: {obj['normalized_bbox']})"
        for obj in segmented_objects
    ])
    
    # 2. Define the LLM Prompt
    prompt = f"""
    You are an expert AI for 3D world reconstruction. Your task is to estimate the real-world dimensions (length, width, height) of objects shown in the accompanying image and described below. 
    
    The dimensions must be in **meters**. Provide the estimate in a JSON list format.
    
    Objects in the image:
    {object_list_str}
    
    Your JSON output must be a list of objects, one for each ID provided above, and must strictly follow this format:
    [
      {{
        "id": [OBJECT_ID], 
        "label": "[OBJECT_LABEL]",
        "dimensions_meters": [[LENGTH], [WIDTH], [HEIGHT]], 
        "justification": "[BRIEF REASONING FOR THE DIMENSIONS]"
      }},
      ...
    ]
    
    The order of the dimensions should correspond to the object's typical longest, second-longest, and vertical axes, or just a generic L, W, H. Be conservative with your estimates, as they are for a geometric model.
    """
    
    # 3. Prepare content for multimodal request
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"ERROR: Image file not found at {image_path}")
        return {}
        
    contents = [prompt, img]
    
    # 4. Call the model
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents
        )
        
        # 5. Extract and parse the JSON response
        # The model might include some text before or after the JSON block.
        print("LLM response received. Parsing JSON...")
        
        # Find the JSON block (often enclosed in triple backticks)
        json_start = response.text.find('[')
        json_end = response.text.rfind(']') + 1
        
        if json_start == -1 or json_end == -1:
            print("ERROR: Could not find valid JSON list in LLM response.")
            print("Raw response:", response.text)
            return {}
            
        json_string = response.text[json_start:json_end]
        llm_output = json.loads(json_string)
        
        # 6. Format the output into the required dictionary
        estimated_sizes = {}
        for item in llm_output:
            if 'id' in item and 'dimensions_meters' in item:
                # Use the longest dimension as a key feature for the geometric model
                estimated_sizes[item['id']] = item['dimensions_meters']
                
        print(f"Successfully estimated sizes for {len(estimated_sizes)} objects.")
        return estimated_sizes

    except APIError as e:
        print(f"Gemini API Error: {e}")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred during LLM call: {e}")
        return {}

# --- 4. MAIN INFERENCE PIPELINE ---

def run_enhanced_inference_pipeline(
    image_path: str, 
    model_checkpoint_path: str = 'best_enhanced_ai2thor_model_with_angles.pth'
):
    
    # --- CONFIGURATION ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n--- Enhanced 6D Pose Inference Pipeline ---")
    print(f"Using device: {DEVICE}")
    print(f"Input Image: {image_path}")
    
    # Assumed Camera Parameters (MUST align with features expected by the model)
    # These mimic the static features used in the AI2-THOR dataset.
    ASSUMED_FOV_DEG = 90.0
    ASSUMED_CAMERA_HORIZON_DEG = 0.0 # Straight ahead
    ASSUMED_AGENT_ROT_Y_DEG = 0.0   # Looking straight
    ASSUMED_AGENT_POS_XYZ = [0.0, 0.0, 0.0] # Camera at world origin
    ASSUMED_AGENT_ROT_XYZ = [0.0, ASSUMED_AGENT_ROT_Y_DEG, 0.0]
    
    # --- 1. LOAD MODEL CHECKPOINT ---
    print("\n1. Loading Geometric Estimator Model...")
    try:
        checkpoint = torch.load(model_checkpoint_path, map_location=torch.device(DEVICE), weights_only=False)

        # Get necessary info from checkpoint
        input_dim = checkpoint['input_dim']
        feature_scaler = checkpoint['feature_scaler']
        coord_scaler = checkpoint['coord_scaler']
        feature_names = checkpoint['feature_names']
        use_uncertainty = checkpoint['use_uncertainty']

        print("\n[DEBUG] Feature Scaler Info:")
        print(f"Feature means: {feature_scaler.mean_[:10]}")  # First 10 features
        print(f"Feature stds: {feature_scaler.scale_[:10]}")
        
        # Initialize and load model
        model = EnhancedGeometricEstimator(
            input_dim=input_dim,
            use_uncertainty=use_uncertainty
        ).to(DEVICE)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✓ Model loaded. Features: {input_dim}, Uncertainty: {use_uncertainty}")
        
    except Exception as e:
        print(f"✗ ERROR: Failed to load model from {model_checkpoint_path}. Ensure it's in the directory and the class definitions match the training script.")
        print(f"Details: {e}")
        return
        
    # --- 2. PERFORM PANOPTIC SEGMENTATION ---
    print("\n2. Performing Panoptic Segmentation...")
    try:
        img = Image.open(image_path).convert("RGB")
        img_width, img_height = img.size
    except FileNotFoundError:
        print(f"✗ ERROR: Image file not found at {image_path}")
        return
        
    model_name = "facebook/mask2former-swin-large-coco-panoptic"
    processor = AutoImageProcessor.from_pretrained(model_name)
    
    # Load and run segmentation model (can be memory-intensive)
    seg_model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name).to(DEVICE)
    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt").to(DEVICE)
        outputs = seg_model(**inputs)
        panoptic_result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[img.size[::-1]])[0]

    segmentation_map = panoptic_result["segmentation"]
    segments_info = panoptic_result["segments_info"]
    
    panoptic_segments = []
    # Define labels to exclude (common labels for structural elements in COCO-Panoptic/similar datasets)
    EXCLUDED_LABELS = {
        'wall', 'floor', 'ceiling', 'window', 'door', 'sky', 'ground',
        'wall-other-merged', # Explicitly exclude the 'wall-other-merged' you saw in your traceback
        'building' 
    }
    
    for segment in segments_info:
        segment_id = segment['id']
        label_id = segment['label_id']
        segment_label = seg_model.config.id2label[label_id].lower() # Convert to lowercase for checking
        
        # 💡 FILTER STEP: Skip structural elements
        if any(excluded_word in segment_label for excluded_word in EXCLUDED_LABELS):
            print(f"   --> Skipping structural element: ID {segment_id}, Label: {segment_label}")
            continue
            
        mask_tensor = (segmentation_map == segment_id).cpu()
        bbox_data = calculate_bounding_box_from_mask(mask_tensor)
        mask_area = mask_tensor.sum().item()

        if bbox_data is not None and mask_area > 0:
            # Calculate normalized bbox for LLM prompt
            normalized_bbox = [
                bbox_data['x_min'] / img_width, bbox_data['y_min'] / img_height,
                bbox_data['width'] / img_width, bbox_data['height'] / img_height
            ]
            
            panoptic_segments.append({
                'id': segment_id,
                'label': seg_model.config.id2label[label_id],
                'normalized_bbox': [round(v, 4) for v in normalized_bbox],
                'bbox_data': bbox_data, # pixel values
                'mask_area': mask_area
            })
            
    if len(panoptic_segments) < 2:
        print("✗ ERROR: Found less than 2 objects. Cannot perform relative depth estimation.")
        return
        
    print(f"✓ Segmentation complete. Found {len(panoptic_segments)} objects.")
    
    print("\n2.5. Loading MiDaS depth estimation model...")
    try:
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        midas.to(DEVICE)
        midas.eval()
        
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        midas_transform = midas_transforms.small_transform
        
        print("✓ MiDaS model loaded successfully")
        print(f"[DEBUG] MiDaS model device: {next(midas.parameters()).device}")
        print(f"[DEBUG] MiDaS model type: {type(midas)}")
    except Exception as e:
        print(f"✗ ERROR: Failed to load MiDaS model: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 3. LLM API CALL FOR SIZE ESTIMATION ---
    print("\n3. Calling LLM for Real-World Size Estimation...")
    estimated_sizes_m = call_llm_for_size_estimation(panoptic_segments, image_path)
    
    if not estimated_sizes_m:
        print("✗ ERROR: Size estimation failed or returned no results.")
        return
        
    valid_segments = [seg for seg in panoptic_segments if seg['id'] in estimated_sizes_m]
    if len(valid_segments) < 2:
        print("✗ ERROR: Less than 2 objects have estimated sizes for relative depth calculation.")
        return
        
    # --- 4. PREPARE FEATURE DATAFRAME AND REFERENCE ---
        
    # Choose reference object: the one with the largest pixel area
    ref_segment = max(valid_segments, key=lambda x: x['mask_area'])
    ref_size_m = estimated_sizes_m[ref_segment['id']]

    print(f"\n4. Starting Geometric Feature Extraction...")
    print(f"Using **{ref_segment['label']} (ID: {ref_segment['id']})** as reference object.")

    results = []
    depth_map_cache = {}
    print(f"[DEBUG] Initialized depth_map_cache: {type(depth_map_cache)}")

    ref_analytical_depth = 3.0 # Initialize to a default value (e.g., 3.0m)
    focal_length_px = (img_width / 2.0) / math.tan(ASSUMED_FOV_DEG * math.pi / 360.0)

    # --- NEW BLOCK: MiDaS Normalization ---
    print("\n4.1. Calibrating MiDaS depth map to metric scale...")
    metric_depth_map_key = image_path + '_metric'

    # 1. Get raw MiDaS depth map and reference object's MiDaS median (inverse depth).
    ref_midas_median = estimate_depth_with_midas(
        image_path, ref_segment['bbox_data'], img_width, img_height,
        midas, midas_transform, depth_map_cache
    )

    # Check if the raw depth map was successfully cached
    raw_midas_depth_map = depth_map_cache.get(image_path)

    if raw_midas_depth_map is None:
        print("✗ WARNING: Failed to generate raw MiDaS depth map. Falling back to pinhole model only.")
    else:
        # 2. Calculate analytical depth for the reference object (The metric anchor)
        ref_longest_dim_m = max(ref_size_m)
        ref_bbox_longest_px = max(ref_segment['bbox_data']['width'], ref_segment['bbox_data']['height'])
        
        # Update the reference analytical depth variable
        ref_analytical_depth = (ref_longest_dim_m * focal_length_px) / ref_bbox_longest_px

        print(f"   Reference MiDaS median (inverse depth): {ref_midas_median:.3f}")
        print(f"   Reference analytical depth (meters): {ref_analytical_depth:.3f}m")

        # 3. Normalize the raw depth map to metric scale
        metric_depth_map = normalize_midas_to_metric_depth(
            raw_midas_depth_map, ref_analytical_depth, ref_midas_median
        )

        # 4. Cache the normalized map for feature calculation
        depth_map_cache[metric_depth_map_key] = metric_depth_map
        print(f"✓ Normalized metric depth map cached with key: {metric_depth_map_key}")
    # --- END NEW BLOCK ---
    
    for target_segment in valid_segments:
        target_id = target_segment['id']
        target_label = target_segment['label']
        
        # Use a preliminary depth estimate based on the longest 3D dimension and longest 2D dimension
        target_size_m = estimated_sizes_m[target_id]

        try:
            target_size_m_float = [float(dim) for dim in target_size_m]
        except (TypeError, ValueError) as e:
            print(f"Error parsing dimensions for ID {target_id} ({target_label}): {target_size_m}. Skipping.")
            print(f"Details: {e}")
            continue # Skip this object if dimensions are invalid
            
        # 2. Correctly extract the longest dimension (a single float)
        longest_3d_dim = max(target_size_m_float) 
        
        # Use a preliminary depth estimate based on the longest 3D dimension and longest 2D dimension
        # ... (rest of the calculation) ...
        # longest_2d_dim_px = max(target_segment['bbox_data']['width'], target_segment['bbox_data']['height'])
        
        # # Approximate focal length for analytical depth
        focal_length_px = (img_width / 2.0) / math.tan(ASSUMED_FOV_DEG * math.pi / 360.0)
        # analytical_depth = (longest_3d_dim * focal_length_px) / longest_2d_dim_px if longest_2d_dim_px > 0 else 10.0

        print(f"   Estimating depth for {target_label} using MiDaS...")
        analytical_depth = (max(target_size_m_float) * focal_length_px) / max(target_segment['bbox_data']['width'], target_segment['bbox_data']['height'])

        try:
            # 1. Use cached normalized map if available, otherwise use pinhole model
            metric_depth_map_key = image_path + '_metric'
            
            if metric_depth_map_key in depth_map_cache:
                metric_depth_map = depth_map_cache[metric_depth_map_key]
                
                # ✅ USE SEGMENTATION MASK, NOT BOUNDING BOX
                target_mask = (segmentation_map.cpu().numpy() == target_id)
                target_depth_values = metric_depth_map[target_mask]
                
                if target_depth_values.size > 0:
                    # ✅ OUTLIER FILTERING
                    q1 = np.percentile(target_depth_values, 25)
                    q3 = np.percentile(target_depth_values, 75)
                    iqr = q3 - q1
                    
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    filtered_depth_values = target_depth_values[
                        (target_depth_values >= lower_bound) & 
                        (target_depth_values <= upper_bound)
                    ]
                    
                    if filtered_depth_values.size > max(10, target_depth_values.size * 0.3):
                        outliers_removed = target_depth_values.size - filtered_depth_values.size
                        print(f"  Filtered {outliers_removed} outliers ({outliers_removed/target_depth_values.size*100:.1f}%)")
                        target_depth_values = filtered_depth_values
                    
                    # Update analytical_depth with MiDaS median
                    analytical_depth = np.median(target_depth_values)
                    depth_mean = np.mean(target_depth_values)
                    depth_std = np.std(target_depth_values)
                    
                    print(f"\n[DEBUG] Depth estimation for {target_label}:")
                    print(f"  MiDaS median: {analytical_depth:.3f}m")
                    print(f"  MiDaS mean: {depth_mean:.3f}m")
                    print(f"  MiDaS std: {depth_std:.3f}m")
                    print(f"  Mask pixel count: {target_depth_values.size}")
                    print(f"  Reference analytical depth: {ref_analytical_depth:.3f}m")
                    print(f"  Relative depth ratio: {analytical_depth / ref_analytical_depth:.3f}")
                    
                    depth_diff_pct = abs(depth_mean - analytical_depth) / analytical_depth * 100
                    if depth_diff_pct > 20:
                        print(f"  ⚠️  High variance in MiDaS depth ({depth_diff_pct:.1f}%)")
                        print(f"      Consider using median only for this object")
                    else:
                        print(f"  ✓ Depth estimates consistent ({depth_diff_pct:.1f}% difference)")
                else:
                    print(f"\n[DEBUG] Empty segmentation mask for {target_label}, using pinhole fallback")
                    print(f"  Pinhole depth: {analytical_depth:.2f}m")
            else:
                print(f"\n[DEBUG] No MiDaS map, using pinhole model: {analytical_depth:.2f}m")
                
        except Exception as e:
            print(f"   Error in MiDaS depth estimation: {e}")
            import traceback
            traceback.print_exc()
            print(f"   Fallback pinhole-based depth: {analytical_depth:.2f}m")
            
            # Fallback to pinhole camera model
            target_longest_dim_m = max(target_size_m_float)
            target_bbox_longest_px = max(target_segment['bbox_data']['width'], target_segment['bbox_data']['height'])
            analytical_depth = (target_longest_dim_m * focal_length_px) / target_bbox_longest_px if target_bbox_longest_px > 0 else 3.0
            print(f"   Fallback pinhole-based depth: {analytical_depth:.2f}m")

        
        # Since the model is relative, we'll use a pseudo-relative depth based on the analytical estimate
        # relative_depth = analytical_depth / ref_analytical_depth.
        # However, for simplicity and to prevent division by zero, we'll use the target's analytical
        # depth as the initial 'dist_to_ref' in the feature calculator.
    
        # The key change: The model was trained on TARGET vs REF features.
        # We need to compute features for the target AND reference object and their relationship.
        
        # **This is where the relative prediction is tricky for a single image!**
        # The simplest way is to run a second round of feature creation after getting the first
        # set of predictions, or **SIMPLIFY THE FEATURE VECTOR** for the first pass.
        # Since the training script is complex, we must stick to the structure.
        
        # To simulate a Target-Ref pair, we create a pseudo-row that combines *both* objects' data.
        # This requires manually merging target and reference attributes into one dict.
        target_depth_for_features = analytical_depth
        ref_depth_for_features = ref_analytical_depth

        print(f"  [DEBUG] Depths for features:")
        print(f"    Target depth: {target_depth_for_features:.3f}m")
        print(f"    Reference depth: {ref_depth_for_features:.3f}m")
        print(f"    Relative depth: {target_depth_for_features / ref_depth_for_features:.3f}")

        target_ref_row = {
            # Target bounding box
            'target_bbox_center_x': target_segment['bbox_data']['center_x'] / img_width,
            'target_bbox_center_y': target_segment['bbox_data']['center_y'] / img_height,
            'target_bbox_width': target_segment['bbox_data']['width'] / img_width,
            'target_bbox_height': target_segment['bbox_data']['height'] / img_height,
            
            # Reference bounding box
            'ref_bbox_center_x': ref_segment['bbox_data']['center_x'] / img_width,
            'ref_bbox_center_y': ref_segment['bbox_data']['center_y'] / img_height,
            'ref_bbox_width': ref_segment['bbox_data']['width'] / img_width,
            'ref_bbox_height': ref_segment['bbox_data']['height'] / img_height,
            
            # Depth values - CRITICAL FOR UNPROJECTION
            'target_depth_median': target_depth_for_features,  # ← ADD THIS
            'dist_to_target': target_depth_for_features,       # Target's depth
            'dist_to_ref': ref_depth_for_features,             # Reference's depth
            'relative_depth': target_depth_for_features / ref_depth_for_features,
            
            # Camera features (camera at origin)
            'field_of_view': ASSUMED_FOV_DEG,
            'camera_horizon': 0.0,
            'agent_rot_y': 0.0,
            'agent_pos_x': 0.0,
            'agent_pos_y': 0.0,
            'agent_pos_z': 0.0,
            'agent_rot_x': 0.0,
            'agent_rot_z': 0.0,
            
            # Segmentation data for feature calculation
            'target_segment_id': target_segment['id'],
            'ref_segment_id': ref_segment['id'],
            'segmentation_map': segmentation_map,
            'normalized_metric_depth_map': depth_map_cache.get(metric_depth_map_key)
        }
        
        metric_depth_map_key = image_path + '_metric'
        if depth_map_cache and metric_depth_map_key in depth_map_cache:
            depth_map = depth_map_cache[metric_depth_map_key]
            
            # Target depth region: NO LONGER NEEDED HERE. The mask logic will handle it.
            target_ref_row['normalized_metric_depth_map'] = depth_map # Pass the metric map
        else:
            target_ref_row['normalized_metric_depth_map'] = None

        feature_dict = calculate_comprehensive_geometric_features(
            target_ref_row, img_width, img_height
        )

        print(f"  Feature depth_mean: {feature_dict.get('target_depth_mean', 'MISSING')}")
        print(f"  Feature depth_median: {feature_dict.get('target_depth_median', 'MISSING')}") # Log the median (more stable)
        print(f"  Feature unproj_x: {feature_dict.get('unproj_x', 'MISSING')}")

        # Convert to numpy array in the correct order (matching training feature_names)
        if feature_names:
            feature_vector = np.array([feature_dict.get(name, 0.0) for name in feature_names])
        else:
            feature_vector = np.array(list(feature_dict.values()))

        # Check critical depth features
        if feature_names:
            depth_feature_indices = [i for i, name in enumerate(feature_names) if 'depth' in name.lower()]
            print(f"  Depth feature indices: {depth_feature_indices[:5]}")
            print(f"  Depth feature values: {[feature_vector[i] for i in depth_feature_indices[:5]]}")
            
            unproj_indices = [i for i, name in enumerate(feature_names) if 'unproj' in name.lower()]
            print(f"  Unprojection feature indices: {unproj_indices}")
            print(f"  Unprojection values: {[feature_vector[i] for i in unproj_indices]}")

        print(f"\n[DEBUG] Inference features (raw): {feature_vector[:10]}")
        features_scaled = feature_scaler.transform(feature_vector.reshape(1, -1))
        print(f"[DEBUG] Inference features (scaled): {features_scaled[0, :10]}")

        print(f"\n[DIAGNOSTIC] Feature breakdown for {target_label}:")
        print(f"  Feature names (first 20): {feature_names[:20] if feature_names else 'N/A'}")
        print(f"  Raw features (first 20): {feature_vector[:20]}")
        print(f"  Scaled features (first 20): {features_scaled[0, :20]}")

        # Handle the reference object case: set all predictions to be 0/1/analytical depth
        if target_id == ref_segment['id']:
            # Reference object - assign analytical depth and derived position
            ref_x = (target_segment['bbox_data']['center_x'] - img_width/2) * analytical_depth / focal_length_px
            ref_y = -(target_segment['bbox_data']['center_y'] - img_height/2) * analytical_depth / focal_length_px
            ref_z = analytical_depth
            
            results.append({
                'id': target_id, 'label': target_label, 'is_reference': True,
                'rel_depth_pred': 1.0, 'world_coord_pred': np.array([ref_x, ref_y, ref_z]), 
                'angle_pred_deg': 0.0, 'coord_uncertainty': np.array([0.0, 0.0, 0.0]), 
                'angle_uncertainty': 0.0, 'reference_id': ref_segment['id']
            })
            continue

        # --- 5. MODEL INFERENCE ---
        
        # Scale and convert to tensor
        features_scaled = feature_scaler.transform(feature_vector.reshape(1, -1))
        features_tensor = torch.FloatTensor(features_scaled).to(DEVICE)
        
        with torch.no_grad():
            if use_uncertainty:
                depth_mean, coord_mean, angle_mean, depth_var, coord_var, angle_var = model(features_tensor, return_uncertainty=True)
            else:
                depth_mean, coord_mean, angle_mean = model(features_tensor)
                depth_var = torch.zeros_like(depth_mean)
                coord_var = torch.zeros_like(coord_mean)
                angle_var = torch.zeros_like(angle_mean)
                
            # Convert back from log space and unscale coordinates
            rel_depth_pred = torch.expm1(depth_mean.cpu()).numpy().item()
            world_coords_scaled = coord_mean.cpu().numpy()
            world_coord_pred = coord_scaler.inverse_transform(world_coords_scaled)[0]
            
            # Convert angle from radians to degrees
            angle_pred_rad = angle_mean.cpu().numpy().item()
            angle_pred_deg = np.rad2deg(angle_pred_rad)
            
            # Uncertainty
            coord_uncertainty = coord_var.cpu().numpy()[0]
            angle_uncertainty = angle_var.cpu().numpy().item()

        results.append({
            'id': target_id, 'label': target_label, 'is_reference': False,
            'rel_depth_pred': rel_depth_pred, 'world_coord_pred': world_coord_pred, 
            'angle_pred_deg': angle_pred_deg, 
            'coord_uncertainty': coord_uncertainty, 'angle_uncertainty': angle_uncertainty,
            'reference_id': ref_segment['id']
        })

    # --- 6. DISPLAY FINAL RESULTS ---
    print("\n\n--- 6. FINAL 6D POSE RECONSTRUCTION RESULTS ---")
    
    # Estimate absolute depth and coordinates using the reference object's analytical depth
    ref_result = next(res for res in results if res['is_reference'])
    ref_analytical_depth = ref_result['rel_depth_pred'] # This holds the initial analytical depth for the reference
    
    final_output = []
    
    for res in results:
        # Calculate absolute depth using the reference depth
        estimated_abs_depth = ref_analytical_depth * res['rel_depth_pred']
        
        # The model predicts absolute world coordinates directly, so we use them
        world_x, world_y, world_z = res['world_coord_pred']
        
        # Log the output
        ref_status = " (REFERENCE)" if res['is_reference'] else ""
        print(f"Object: {res['label']} (ID: {res['id']}){ref_status}")
        print(f"  > Position (X,Y,Z): ({world_x:.3f}, {world_y:.3f}, {world_z:.3f}) meters")
        print(f"  > Orientation (Yaw): {res['angle_pred_deg']:.1f}°")
        print(f"  > Rel. Depth (Target/Ref): {res['rel_depth_pred']:.3f}")
        
        if use_uncertainty:
            print(f"  > Position Uncertainty: ({res['coord_uncertainty'][0]:.4f}, {res['coord_uncertainty'][1]:.4f}, {res['coord_uncertainty'][2]:.4f})")
            print(f"  > Angle Uncertainty: {res['angle_uncertainty']:.4f}")
        
        print("-" * 20)
        
        final_output.append({
            'id': res['id'],
            'label': res['label'],
            'position_m': [world_x, world_y, world_z],
            'orientation_yaw_deg': res['angle_pred_deg'],
            'relative_depth': res['rel_depth_pred'],
            'abs_depth_estimate_m': estimated_abs_depth,
            'is_reference': res['is_reference'],
            'dimensions_m': estimated_sizes_m[res['id']]
        })

    # Optional: Save final output to JSON
    output_filename = os.path.splitext(os.path.basename(image_path))[0] + "_3d_pose_results.json"
    with open(output_filename, 'w') as f:
        json.dump(final_output, f, indent=4, cls=NumpyEncoder)
    print(f"\n✓ 3D pose results saved to {output_filename}")

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


if __name__ == "__main__":
    
    # --- INSTRUCTIONS ---
    print("INSTRUCTIONS:")
    print("1. Ensure your trained model 'best_enhanced_ai2thor_model_with_angles.pth' is in this directory.")
    print("2. Set your GEMINI_API_KEY environment variable.")
    print("3. Update 'YOUR_LOCAL_IMAGE_PATH' below to the path of your image.")
    print("4. Install required libraries: pip install torch transformers pillow numpy pandas scikit-learn google-genai")
    print("-" * 50)
    
    # --- USER CONFIGURATION ---
    # !!! CHANGE THIS TO YOUR ACTUAL IMAGE PATH !!!
    YOUR_LOCAL_IMAGE_PATH = r"/Users/adamrolander/WorldBuilder/inference/room_test.png" 
    
    run_enhanced_inference_pipeline(
        image_path=YOUR_LOCAL_IMAGE_PATH,
        model_checkpoint_path='best_enhanced_ai2thor_model_with_angles.pth'
    )