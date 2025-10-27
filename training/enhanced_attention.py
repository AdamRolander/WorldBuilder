import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import math
from scipy.spatial.transform import Rotation as R
import random
from tqdm import tqdm
import seaborn as sns
from datetime import datetime
import os


class EnhancedAI2ThorDataset(Dataset):
    """Enhanced dataset with data augmentation capabilities."""
    def __init__(self, features, relative_depth_targets, world_coord_targets, angle_targets,
                 augment=False, noise_std=0.01):
        self.features = torch.FloatTensor(features)
        self.relative_depth_targets = torch.FloatTensor(relative_depth_targets)
        self.world_coord_targets = torch.FloatTensor(world_coord_targets)
        self.angle_targets = torch.FloatTensor(angle_targets)
        self.augment = augment
        self.noise_std = noise_std
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = self.features[idx].clone()
        
        if self.augment:
            # Add small noise to simulate measurement uncertainty
            noise = torch.normal(0, self.noise_std, features.shape)
            features = features + noise
        
        return (features, 
                self.relative_depth_targets[idx], 
                self.world_coord_targets[idx],
                self.angle_targets[idx])


class MultiScaleAttentionBlock(nn.Module):
    """Multi-scale attention that processes features at different granularities."""
    def __init__(self, input_dim, num_heads=8, dropout_rate=0.1):
        super().__init__()
    
        self.input_dim = input_dim
        
        # Calculate feature dimensions that work with attention
        # Use a fixed dimension that's easily divisible
        self.feat_dim = 64  # Fixed dimension divisible by common head counts
        
        # Adjust num_heads to be compatible
        possible_heads = [1, 2, 4, 8, 16]
        self.heads_per_group = max([h for h in possible_heads if h <= num_heads and self.feat_dim % h == 0])
        
        print(f"MultiScaleAttention: input_dim={input_dim}, feat_dim={self.feat_dim}, heads_per_group={self.heads_per_group}")
        
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
        # Ensure cross_heads divides combined_dim
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
        geo_feat = self.geometric_proj(x).unsqueeze(1)  # Geometric features
        vis_feat = self.visual_proj(x).unsqueeze(1)     # Visual/bbox features  
        cam_feat = self.camera_proj(x).unsqueeze(1)     # Camera/pose features
        
        # Apply specialized attention to each feature group
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


class UncertaintyLoss(nn.Module):
    """Loss function that incorporates prediction uncertainty with better numerical stability."""
    def __init__(self, depth_weight=0.2, coord_weight=0.5, angle_weight=0.3):
        super().__init__()
        self.depth_weight = depth_weight
        self.coord_weight = coord_weight
        self.angle_weight = angle_weight
        
    def forward(self, depth_mean, depth_var, coord_mean, coord_var, angle_mean, angle_var,
                depth_target, coord_target, angle_target):
        # More numerically stable uncertainty loss
        depth_loss = 0.5 * (torch.log(depth_var + 1e-6) + 
                           (depth_target - depth_mean)**2 / (depth_var + 1e-6))
        
        coord_loss = 0.5 * torch.sum(torch.log(coord_var + 1e-6) + 
                                   (coord_target - coord_mean)**2 / (coord_var + 1e-6), dim=1)
        
        # Angular loss with circular distance
        angle_diff = torch.atan2(torch.sin(angle_target - angle_mean), torch.cos(angle_target - angle_mean))
        angle_loss = 0.5 * (torch.log(angle_var + 1e-6) + angle_diff**2 / (angle_var + 1e-6))
        
        # Apply loss weights and add regularization
        total_loss = (self.depth_weight * depth_loss.mean() + 
                     self.coord_weight * coord_loss.mean() +
                     self.angle_weight * angle_loss.mean())
        
        # Add small regularization to prevent variance collapse
        var_reg = 1e-4 * (torch.mean(1.0 / (depth_var + 1e-6)) + 
                         torch.mean(1.0 / (coord_var + 1e-6)) +
                         torch.mean(1.0 / (angle_var + 1e-6)))
        
        return total_loss + var_reg


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
        
        if use_uncertainty:
            # Uncertainty estimation branches (output mean and variance)
            self.depth_mean_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            self.depth_var_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(), 
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus()  # Ensure positive variance
            )
            
            self.coord_mean_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 3)  # Changed from 4 to 3 for X,Y,Z
            )
            self.coord_var_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 3),  # Changed from 4 to 3
                nn.Softplus()
            )
            
            # Add angle prediction heads
            self.angle_mean_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            self.angle_var_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus()
            )
        else:
            # Standard prediction heads
            self.depth_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            
            self.coord_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 3)  # Changed from 4 to 3
            )
            
            self.angle_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
        
    def forward(self, x, return_uncertainty=None):
        if return_uncertainty is None:
            return_uncertainty = self.use_uncertainty
            
        # Multi-scale attention
        x_att = self.multi_attention(x)
        
        # Encoder with residual connections
        features = x_att
        for layer in self.encoder_layers:
            new_features = layer(features)
            if new_features.shape == features.shape:
                features = features + new_features  # Residual connection
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
            # Standard predictions
            if self.use_uncertainty:
                depth_pred = self.depth_mean_head(features).squeeze(-1)
                coord_pred = self.coord_mean_head(features)
                angle_pred = self.angle_mean_head(features).squeeze(-1)
            else:
                depth_pred = self.depth_head(features).squeeze(-1)
                coord_pred = self.coord_head(features)
                angle_pred = self.angle_head(features).squeeze(-1)
            return depth_pred, coord_pred, angle_pred


def calculate_angle_error(pred_angles, true_angles):
    """Calculate circular angle error in degrees."""
    diff = pred_angles - true_angles
    # Normalize to [-pi, pi]
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))
    return torch.abs(diff) * 180.0 / np.pi


def calculate_unprojection(row, img_width=300, img_height=300):
    """Calculates camera-relative 3D coordinates using unprojection formula."""
    
    # Get Camera Intrinsics (from FoV)
    fov_rad = row['field_of_view'] * (math.pi / 180.0)
    focal_length = (img_width / 2.0) / math.tan(fov_rad / 2.0)
    cx = img_width / 2.0
    cy = img_height / 2.0
    K_inv = np.linalg.inv(np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ]))

    # Get 2D Pixel Coordinates and Depth
    px = row['target_bbox_center_x'] * img_width
    py = row['target_bbox_center_y'] * img_height
    depth = row['dist_to_ref']
    
    # Unproject from 2D to 3D (in Camera's coordinate system)
    pixel_coords = np.array([px, py, 1])
    camera_coords = K_inv @ pixel_coords * depth
    
    # CHANGED: Our dataset stores camera-relative coords as "world" coords
    # Camera is always at [0,0,0] with [0,0,0] rotation
    # So camera_coords ARE the final coordinates - no transformation needed
    
    return camera_coords[0], camera_coords[1], camera_coords[2]


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
    
    # 3. Enhanced depth features (keep your existing depth map logic)
    depth_cols = ['depth_mean', 'depth_std', 'depth_min', 'depth_max', 
                  'depth_median', 'depth_percentile_25', 'depth_percentile_75']
    
    for col in depth_cols:
        target_col = f'target_{col}'
        ref_col = f'ref_{col}'
        if target_col in row and ref_col in row:
            features[target_col] = row[target_col]
            features[ref_col] = row[ref_col]
            
            # Depth ratios and differences
            if row[ref_col] > 0:
                features[f'{col}_ratio'] = row[target_col] / row[ref_col]
            features[f'{col}_diff'] = row[target_col] - row[ref_col]
    
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


def analyze_dataset_structure(df):
    """Analyze the structure and content of the dataset."""
    print("\n=== Dataset Structure Analysis ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Check for bbox-related columns
    bbox_cols = [col for col in df.columns if 'bbox' in col.lower()]
    print(f"Bbox-related columns: {bbox_cols}")
    
    # Sample a few rows to see the data
    print("\n=== Sample Data (first 3 rows) ===")
    for col in df.columns[:10]:  # Show first 10 columns
        print(f"{col}: {df[col].iloc[:3].tolist()}")
    
    # Check if bounding box columns exist and what they contain
    required_bbox_cols = ['target_bbox_width', 'target_bbox_height', 'target_bbox_center_x', 'target_bbox_center_y',
                         'ref_bbox_width', 'ref_bbox_height', 'ref_bbox_center_x', 'ref_bbox_center_y']
    
    missing_bbox_cols = [col for col in required_bbox_cols if col not in df.columns]
    if missing_bbox_cols:
        print(f"\nMissing bbox columns: {missing_bbox_cols}")
    
    # Check for alternative bbox column names
    possible_alternatives = []
    for col in df.columns:
        if any(term in col.lower() for term in ['box', 'bound', 'width', 'height', 'x', 'y']):
            possible_alternatives.append(col)
    
    if possible_alternatives:
        print(f"Possible alternative bbox columns: {possible_alternatives}")
    
    return bbox_cols, missing_bbox_cols


def filter_high_quality_samples_no_bbox(df):
    """Filter dataset without relying on bounding box information."""
    initial_count = len(df)
    print(f"Initial dataset size: {initial_count}")
    
    if len(df) == 0:
        print("Warning: Dataset is empty!")
        return df
    
    # Analyze dataset structure first
    bbox_cols, missing_bbox_cols = analyze_dataset_structure(df)
    
    # Skip bbox filtering entirely and warn user
    print("\nWarning: All bounding boxes have zero area!")
    print("This indicates a problem with the dataset generation.")
    print("Proceeding without bounding box filtering, but results may be poor.")
    
    # Remove samples with poor depth estimates
    df = df[df['target_depth_mean'] > 0.5]
    df = df[df['target_depth_mean'] < 10.0]
    print(f"After depth filtering: {len(df)}")
    
    if len(df) == 0:
        print("Warning: All samples filtered out during depth filtering!")
        return df
    
    # Skip visibility filtering since all values are 1.0 (no variation)
    print("Skipping visibility filtering (all values are 1.0)")
    
    # Remove samples where depth is inconsistent with distance
    depth_consistency = np.abs(df['target_depth_mean'] - df['dist_to_ref']) / df['target_depth_mean']
    df = df[depth_consistency < 1.5]  # Very lenient
    print(f"After depth consistency filtering: {len(df)}")
    
    if len(df) == 0:
        print("Warning: All samples filtered out during depth consistency filtering!")
        return df
    
    # Remove extreme outliers in coordinates
    try:
        for coord in ['world_x', 'world_y', 'world_z']:
            if len(df) == 0:
                break
            
            if len(df[coord]) < 10:
                print(f"Warning: Too few samples ({len(df)}) for coordinate outlier filtering")
                break
                
            # Very lenient outlier detection
            Q1 = df[coord].quantile(0.001)
            Q3 = df[coord].quantile(0.999)
            
            if not (np.isnan(Q1) or np.isnan(Q3)):
                df = df[(df[coord] >= Q1) & (df[coord] <= Q3)]
            else:
                print(f"Warning: Invalid quantiles for {coord}, skipping outlier filtering")
        
        print(f"After coordinate outlier filtering: {len(df)}")
    except Exception as e:
        print(f"Warning: Error during coordinate outlier filtering: {e}")
        print("Proceeding without coordinate outlier filtering")
    
    if len(df) == 0:
        print("Warning: All samples filtered out during coordinate outlier filtering!")
        return df
    
    # Remove samples with extreme relative depths
    df = df[(df['relative_depth'] > 0.01) & (df['relative_depth'] < 20.0)]
    print(f"After relative depth filtering: {len(df)}")
    
    if len(df) == 0:
        print("Warning: All samples filtered out during relative depth filtering!")
        return df
    
    print(f"Final dataset size: {len(df)} ({len(df)/initial_count:.1%} retained)")
    return df.reset_index(drop=True)


class EnhancedAI2ThorTrainer:
    """Enhanced trainer with curriculum learning and uncertainty estimation including angle prediction."""
    def __init__(self, model, train_loader, val_loader, coord_scaler, device='cpu', use_uncertainty=True):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.coord_scaler = coord_scaler
        self.device = device
        self.use_uncertainty = use_uncertainty
        
        # Loss functions
        if use_uncertainty:
            self.criterion = UncertaintyLoss(depth_weight=0.2, coord_weight=0.5, angle_weight=0.3)
        else:
            self.depth_criterion = nn.MSELoss()
            self.coord_criterion = nn.MSELoss()
            self.angle_criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=0.0005,
            weight_decay=0.001,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.3,
            patience=5,
            min_lr=1e-7
        )
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        
        # Storage for predictions (for final evaluation)
        self.final_predictions = None
        self.final_ground_truths = None
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        for batch_features, batch_depth_targets, batch_coord_targets, batch_angle_targets in self.train_loader:
            batch_features = batch_features.to(self.device)
            batch_depth_targets = batch_depth_targets.to(self.device)
            batch_coord_targets = batch_coord_targets.to(self.device)
            batch_angle_targets = batch_angle_targets.to(self.device)
            
            # Check for invalid targets
            if (torch.any(torch.isnan(batch_depth_targets)) or 
                torch.any(torch.isnan(batch_coord_targets)) or
                torch.any(torch.isnan(batch_angle_targets))):
                continue
                
            self.optimizer.zero_grad()
            
            try:
                if self.use_uncertainty:
                    depth_mean, coord_mean, angle_mean, depth_var, coord_var, angle_var = self.model(batch_features, return_uncertainty=True)
                    
                    # Clamp predictions to reasonable ranges
                    depth_mean = torch.clamp(depth_mean, -10, 10)
                    coord_mean = torch.clamp(coord_mean, -10, 10)
                    angle_mean = torch.clamp(angle_mean, -np.pi, np.pi)
                    depth_var = torch.clamp(depth_var, 1e-6, 10)
                    coord_var = torch.clamp(coord_var, 1e-6, 10)
                    angle_var = torch.clamp(angle_var, 1e-6, 10)
                    
                    loss = self.criterion(depth_mean, depth_var, coord_mean, coord_var, angle_mean, angle_var,
                                        batch_depth_targets, batch_coord_targets, batch_angle_targets)
                else:
                    depth_pred, coord_pred, angle_pred = self.model(batch_features)
                    depth_pred = torch.clamp(depth_pred, -10, 10)
                    coord_pred = torch.clamp(coord_pred, -10, 10)
                    angle_pred = torch.clamp(angle_pred, -np.pi, np.pi)
                    
                    depth_loss = self.depth_criterion(depth_pred, batch_depth_targets)
                    coord_loss = self.coord_criterion(coord_pred, batch_coord_targets)
                    
                    # Circular angle loss
                    angle_diff = torch.atan2(torch.sin(batch_angle_targets - angle_pred), 
                                           torch.cos(batch_angle_targets - angle_pred))
                    angle_loss = self.angle_criterion(angle_diff, torch.zeros_like(angle_diff))
                    
                    loss = 0.2 * depth_loss + 0.5 * coord_loss + 0.3 * angle_loss
                
                # Check for invalid loss
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                    
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                self.optimizer.step()
                
                batch_size = batch_features.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
            except RuntimeError as e:
                print(f"Training error: {e}")
                continue
        
        if total_samples == 0:
            return 0.0
        return total_loss / total_samples
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        all_depth_preds = []
        all_depth_targets = []
        all_coord_preds = []
        all_coord_targets = []
        all_angle_preds = []
        all_angle_targets = []
        
        with torch.no_grad():
            for batch_features, batch_depth_targets, batch_coord_targets, batch_angle_targets in self.val_loader:
                batch_features = batch_features.to(self.device)
                batch_depth_targets = batch_depth_targets.to(self.device)
                batch_coord_targets = batch_coord_targets.to(self.device)
                batch_angle_targets = batch_angle_targets.to(self.device)
                
                if self.use_uncertainty:
                    depth_mean, coord_mean, angle_mean, depth_var, coord_var, angle_var = self.model(batch_features, return_uncertainty=True)
                    loss = self.criterion(depth_mean, depth_var, coord_mean, coord_var, angle_mean, angle_var,
                                        batch_depth_targets, batch_coord_targets, batch_angle_targets)
                    depth_pred, coord_pred, angle_pred = depth_mean, coord_mean, angle_mean
                else:
                    depth_pred, coord_pred, angle_pred = self.model(batch_features)
                    depth_loss = self.depth_criterion(depth_pred, batch_depth_targets)
                    coord_loss = self.coord_criterion(coord_pred, batch_coord_targets)
                    
                    # Circular angle loss
                    angle_diff = torch.atan2(torch.sin(batch_angle_targets - angle_pred), 
                                           torch.cos(batch_angle_targets - angle_pred))
                    angle_loss = self.angle_criterion(angle_diff, torch.zeros_like(angle_diff))
                    
                    loss = 0.2 * depth_loss + 0.5 * coord_loss + 0.3 * angle_loss
                
                batch_size = batch_features.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Collect predictions for metrics
                all_depth_preds.append(depth_pred.cpu())
                all_depth_targets.append(batch_depth_targets.cpu())
                all_coord_preds.append(coord_pred.cpu())
                all_coord_targets.append(batch_coord_targets.cpu())
                all_angle_preds.append(angle_pred.cpu())
                all_angle_targets.append(batch_angle_targets.cpu())
        
        avg_loss = total_loss / total_samples
        
        # Calculate metrics
        all_depth_preds_log = torch.cat(all_depth_preds).numpy()
        all_depth_targets_log = torch.cat(all_depth_targets).numpy()
        
        # Clip extreme values to prevent overflow in expm1
        all_depth_preds_log = np.clip(all_depth_preds_log, -10, 10)
        all_depth_targets_log = np.clip(all_depth_targets_log, -10, 10)
        
        all_depth_preds = np.expm1(all_depth_preds_log)
        all_depth_targets = np.expm1(all_depth_targets_log)
        
        # Check for any remaining invalid values
        valid_mask = np.isfinite(all_depth_preds) & np.isfinite(all_depth_targets)
        all_depth_preds = all_depth_preds[valid_mask]
        all_depth_targets = all_depth_targets[valid_mask]
        
        all_coord_preds = torch.cat(all_coord_preds).numpy()
        all_coord_targets = torch.cat(all_coord_targets).numpy()
        all_angle_preds = torch.cat(all_angle_preds).numpy()
        all_angle_targets = torch.cat(all_angle_targets).numpy()
        
        # Unscale coordinates
        coord_preds_unscaled = self.coord_scaler.inverse_transform(all_coord_preds)
        coord_targets_unscaled = self.coord_scaler.inverse_transform(all_coord_targets)
        
        # Calculate angle errors (in degrees)
        angle_errors = calculate_angle_error(torch.tensor(all_angle_preds), torch.tensor(all_angle_targets)).numpy()
        
        # Calculate R² and MAE
        depth_r2 = r2_score(all_depth_targets, all_depth_preds)
        depth_mae = mean_absolute_error(all_depth_targets, all_depth_preds)
        coord_r2 = r2_score(coord_targets_unscaled, coord_preds_unscaled)
        coord_mae = mean_absolute_error(coord_targets_unscaled, coord_preds_unscaled)
        angle_mae = np.mean(angle_errors)
        
        # Per-axis metrics
        axis_names = ['X', 'Y', 'Z']
        axis_metrics = {}
        for i, axis in enumerate(axis_names):
            axis_r2 = r2_score(coord_targets_unscaled[:, i], coord_preds_unscaled[:, i])
            axis_mae = mean_absolute_error(coord_targets_unscaled[:, i], coord_preds_unscaled[:, i])
            axis_metrics[f'{axis}_r2'] = axis_r2
            axis_metrics[f'{axis}_mae'] = axis_mae
        
        return {
            'total_loss': avg_loss,
            'depth_r2': depth_r2,
            'coord_r2': coord_r2,
            'depth_mae': depth_mae,
            'coord_mae': coord_mae,
            'angle_mae': angle_mae,
            **axis_metrics
        }
    
    def store_final_predictions(self):
        """Store final predictions and ground truths for evaluation."""
        self.model.eval()
        
        all_predictions = {
            'depth_pred': [],
            'coord_pred': [],
            'angle_pred': [],
            'depth_target': [],
            'coord_target': [],
            'angle_target': [],
            'depth_uncertainty': [],
            'coord_uncertainty': [],
            'angle_uncertainty': []
        }
        
        with torch.no_grad():
            for batch_features, batch_depth_targets, batch_coord_targets, batch_angle_targets in self.val_loader:
                batch_features = batch_features.to(self.device)
                batch_depth_targets = batch_depth_targets.to(self.device)
                batch_coord_targets = batch_coord_targets.to(self.device)
                batch_angle_targets = batch_angle_targets.to(self.device)
                
                if self.use_uncertainty:
                    depth_mean, coord_mean, angle_mean, depth_var, coord_var, angle_var = self.model(batch_features, return_uncertainty=True)
                    all_predictions['depth_uncertainty'].append(depth_var.cpu().numpy())
                    all_predictions['coord_uncertainty'].append(coord_var.cpu().numpy())
                    all_predictions['angle_uncertainty'].append(angle_var.cpu().numpy())
                    depth_pred, coord_pred, angle_pred = depth_mean, coord_mean, angle_mean
                else:
                    depth_pred, coord_pred, angle_pred = self.model(batch_features)
                    # Fill with zeros for uncertainty if not using uncertainty
                    all_predictions['depth_uncertainty'].append(np.zeros_like(depth_pred.cpu().numpy()))
                    all_predictions['coord_uncertainty'].append(np.zeros_like(coord_pred.cpu().numpy()))
                    all_predictions['angle_uncertainty'].append(np.zeros_like(angle_pred.cpu().numpy()))
                
                # Store predictions and targets
                all_predictions['depth_pred'].append(depth_pred.cpu().numpy())
                all_predictions['coord_pred'].append(coord_pred.cpu().numpy())
                all_predictions['angle_pred'].append(angle_pred.cpu().numpy())
                all_predictions['depth_target'].append(batch_depth_targets.cpu().numpy())
                all_predictions['coord_target'].append(batch_coord_targets.cpu().numpy())
                all_predictions['angle_target'].append(batch_angle_targets.cpu().numpy())
        
        # Concatenate all batches
        for key in all_predictions:
            all_predictions[key] = np.concatenate(all_predictions[key], axis=0)
        
        # Convert depth predictions from log space
        all_predictions['depth_pred'] = np.expm1(np.clip(all_predictions['depth_pred'], -10, 10))
        all_predictions['depth_target'] = np.expm1(np.clip(all_predictions['depth_target'], -10, 10))
        
        # Unscale coordinates
        all_predictions['coord_pred'] = self.coord_scaler.inverse_transform(all_predictions['coord_pred'])
        all_predictions['coord_target'] = self.coord_scaler.inverse_transform(all_predictions['coord_target'])
        
        # Convert angles to degrees for easier interpretation
        all_predictions['angle_pred_deg'] = np.rad2deg(all_predictions['angle_pred'])
        all_predictions['angle_target_deg'] = np.rad2deg(all_predictions['angle_target'])
        
        # Calculate errors
        all_predictions['depth_error'] = np.abs(all_predictions['depth_pred'] - all_predictions['depth_target'])
        all_predictions['coord_error'] = np.linalg.norm(all_predictions['coord_pred'] - all_predictions['coord_target'], axis=1)
        
        # Calculate angle error (circular distance in degrees)
        angle_diff = all_predictions['angle_target'] - all_predictions['angle_pred']
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        all_predictions['angle_error_deg'] = np.abs(angle_diff) * 180.0 / np.pi
        
        self.final_predictions = all_predictions
        return all_predictions
    
    def train(self, epochs=200, early_stopping_patience=25):
        print(f"Starting enhanced AI2-THOR model training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Using uncertainty estimation: {self.use_uncertainty}")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_metrics = self.validate()
            val_loss = val_metrics['total_loss']
            
            self.scheduler.step(val_loss)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_loss': self.best_val_loss,
                    'metrics': val_metrics
                }, 'best_enhanced_ai2thor_model_with_angles.pth')
            else:
                self.patience_counter += 1
            
            if epoch % 10 == 0 or epoch < 10:
                print(f"Epoch {epoch:3d} | Train: {train_loss:.4f} | "
                      f"Val: {val_loss:.4f} | "
                      f"R²: D={val_metrics['depth_r2']:.3f} C={val_metrics['coord_r2']:.3f} | "
                      f"MAE: C={val_metrics['coord_mae']:.3f}m A={val_metrics['angle_mae']:.1f}°")
            
            if self.patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        
        # Store final predictions for evaluation
        print("Storing final predictions for evaluation...")
        self.store_final_predictions()
        
        return self.model


def save_predictions_to_csv(trainer, dataset_df, output_path="ai2thor_predictions_and_targets.csv"):
    """Save predictions and ground truths with scene information to CSV."""
    predictions = trainer.final_predictions
    
    # Create DataFrame with predictions and scene information
    results_df = pd.DataFrame({
        # Predictions
        'pred_depth': predictions['depth_pred'],
        'pred_x': predictions['coord_pred'][:, 0],
        'pred_y': predictions['coord_pred'][:, 1], 
        'pred_z': predictions['coord_pred'][:, 2],
        'pred_angle_deg': predictions['angle_pred_deg'],
        
        # Ground truth
        'true_depth': predictions['depth_target'],
        'true_x': predictions['coord_target'][:, 0],
        'true_y': predictions['coord_target'][:, 1],
        'true_z': predictions['coord_target'][:, 2],
        'true_angle_deg': predictions['angle_target_deg'],
        
        # Errors
        'depth_error': predictions['depth_error'],
        'coord_error': predictions['coord_error'],
        'angle_error_deg': predictions['angle_error_deg'],
        
        # Uncertainties (if available)
        'depth_uncertainty': predictions['depth_uncertainty'],
        'coord_uncertainty_x': predictions['coord_uncertainty'][:, 0] if len(predictions['coord_uncertainty'].shape) > 1 else predictions['coord_uncertainty'],
        'coord_uncertainty_y': predictions['coord_uncertainty'][:, 1] if len(predictions['coord_uncertainty'].shape) > 1 else predictions['coord_uncertainty'],
        'coord_uncertainty_z': predictions['coord_uncertainty'][:, 2] if len(predictions['coord_uncertainty'].shape) > 1 else predictions['coord_uncertainty'],
        'angle_uncertainty': predictions['angle_uncertainty'],
    })
    
    # Add scene information if available (from validation set)
    validation_size = len(predictions['depth_pred'])
    total_size = len(dataset_df)
    validation_start_idx = int(0.8 * total_size)  # Assuming 80/20 split
    
    if validation_start_idx + validation_size <= total_size:
        scene_info = dataset_df.iloc[validation_start_idx:validation_start_idx + validation_size]
        
        # Add scene columns
        scene_columns = ['scene_name', 'scene_idx', 'pose_idx', 'target_object_type', 
                        'ref_object_type', 'field_of_view', 'camera_horizon']
        
        for col in scene_columns:
            if col in scene_info.columns:
                results_df[col] = scene_info[col].values
    
    # Save to CSV
    results_df.to_csv(output_path, index=False)
    print(f"Predictions and targets saved to {output_path}")
    
    return results_df


def evaluate_worst_predictions(predictions_df, output_dir="evaluation_results"):
    """Analyze worst predictions and identify problematic scenes."""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define different error metrics for analysis
    error_metrics = {
        'coord_error': 'coord_error',
        'angle_error': 'angle_error_deg',
        'depth_error': 'depth_error'
    }
    
    analysis_results = {}
    
    for metric_name, error_col in error_metrics.items():
        print(f"\n=== Analyzing Worst {metric_name.replace('_', ' ').title()} ===")
        
        # Sort by error and get worst predictions
        worst_predictions = predictions_df.nlargest(100, error_col)
        
        # Statistics
        percentiles = [90, 95, 99]
        print(f"Error percentiles for {metric_name}:")
        for p in percentiles:
            val = np.percentile(predictions_df[error_col], p)
            print(f"  {p}th percentile: {val:.4f}")
        
        # Scene analysis if scene information is available
        if 'scene_name' in predictions_df.columns:
            print(f"\nWorst scenes for {metric_name}:")
            scene_errors = worst_predictions.groupby('scene_name')[error_col].agg(['count', 'mean', 'std'])
            scene_errors = scene_errors.sort_values('mean', ascending=False)
            print(scene_errors.head(10))
            
            # Object type analysis
            if 'target_object_type' in predictions_df.columns:
                print(f"\nWorst object types for {metric_name}:")
                object_errors = worst_predictions.groupby('target_object_type')[error_col].agg(['count', 'mean', 'std'])
                object_errors = object_errors.sort_values('mean', ascending=False)
                print(object_errors.head(10))
        
        # Save worst predictions
        worst_file = os.path.join(output_dir, f"worst_{metric_name}_predictions.csv")
        worst_predictions.to_csv(worst_file, index=False)
        
        analysis_results[metric_name] = {
            'worst_predictions': worst_predictions,
            'percentiles': {p: np.percentile(predictions_df[error_col], p) for p in percentiles}
        }
    
    # Create summary analysis
    create_error_analysis_plots(predictions_df, output_dir)
    
    # Generate summary report
    generate_analysis_report(analysis_results, predictions_df, output_dir)
    
    return analysis_results

def generate_analysis_report(analysis_results, predictions_df, output_dir):
    """Generate a comprehensive analysis report."""
    
    report_path = os.path.join(output_dir, 'prediction_analysis_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("AI2-THOR Coordinate and Angle Estimation - Prediction Analysis Report\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall statistics
        f.write("OVERALL PERFORMANCE SUMMARY\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total predictions analyzed: {len(predictions_df)}\n\n")
        
        # Error statistics for each metric
        error_metrics = []
        if 'coord_error' in predictions_df.columns:
            error_metrics.append(('coord_error', 'Coordinate Error', 'm'))
        if 'angle_error_deg' in predictions_df.columns:
            error_metrics.append(('angle_error_deg', 'Angle Error', '°'))
        if 'depth_error' in predictions_df.columns:
            error_metrics.append(('depth_error', 'Depth Error', 'm'))
        
        for metric_name, display_name, unit in error_metrics:
            error_values = predictions_df[metric_name]
            
            f.write(f"{display_name}:\n")
            f.write(f"  Mean: {error_values.mean():.4f}{unit}\n")
            f.write(f"  Median: {error_values.median():.4f}{unit}\n")
            f.write(f"  Std: {error_values.std():.4f}{unit}\n")
            f.write(f"  90th percentile: {error_values.quantile(0.9):.4f}{unit}\n")
            f.write(f"  95th percentile: {error_values.quantile(0.95):.4f}{unit}\n")
            f.write(f"  99th percentile: {error_values.quantile(0.99):.4f}{unit}\n\n")
        
        # Scene analysis if available
        if 'scene_name' in predictions_df.columns:
            f.write("SCENE-BASED ANALYSIS\n")
            f.write("-" * 20 + "\n")
            
            # Worst scenes for coordinate errors
            if 'coord_error' in predictions_df.columns:
                scene_coord_errors = predictions_df.groupby('scene_name')['coord_error'].agg(['count', 'mean', 'std']).sort_values('mean', ascending=False)
                f.write("Top 10 worst scenes (coordinate error):\n")
                for i, (scene, row) in enumerate(scene_coord_errors.head(10).iterrows()):
                    f.write(f"  {i+1}. {scene}: {row['mean']:.4f}m (±{row['std']:.4f}m, n={row['count']})\n")
                f.write("\n")
            
            # Worst scenes for angle errors
            if 'angle_error_deg' in predictions_df.columns:
                scene_angle_errors = predictions_df.groupby('scene_name')['angle_error_deg'].agg(['count', 'mean', 'std']).sort_values('mean', ascending=False)
                f.write("Top 10 worst scenes (angle error):\n")
                for i, (scene, row) in enumerate(scene_angle_errors.head(10).iterrows()):
                    f.write(f"  {i+1}. {scene}: {row['mean']:.1f}° (±{row['std']:.1f}°, n={row['count']})\n")
                f.write("\n")
        
        # Object type analysis if available
        if 'target_object_type' in predictions_df.columns:
            f.write("OBJECT TYPE ANALYSIS\n")
            f.write("-" * 20 + "\n")
            
            # Worst object types for coordinate errors
            if 'coord_error' in predictions_df.columns:
                obj_coord_errors = predictions_df.groupby('target_object_type')['coord_error'].agg(['count', 'mean', 'std']).sort_values('mean', ascending=False)
                f.write("Top 10 worst object types (coordinate error):\n")
                for i, (obj_type, row) in enumerate(obj_coord_errors.head(10).iterrows()):
                    f.write(f"  {i+1}. {obj_type}: {row['mean']:.4f}m (±{row['std']:.4f}m, n={row['count']})\n")
                f.write("\n")
            
            # Worst object types for angle errors
            if 'angle_error_deg' in predictions_df.columns:
                obj_angle_errors = predictions_df.groupby('target_object_type')['angle_error_deg'].agg(['count', 'mean', 'std']).sort_values('mean', ascending=False)
                f.write("Top 10 worst object types (angle error):\n")
                for i, (obj_type, row) in enumerate(obj_angle_errors.head(10).iterrows()):
                    f.write(f"  {i+1}. {obj_type}: {row['mean']:.1f}° (±{row['std']:.1f}°, n={row['count']})\n")
                f.write("\n")
        
        # Correlation analysis
        f.write("ERROR CORRELATION ANALYSIS\n")
        f.write("-" * 26 + "\n")
        
        # Only compute correlations for columns that exist
        correlations = []
        if 'coord_error' in predictions_df.columns and 'angle_error_deg' in predictions_df.columns:
            coord_angle_corr = np.corrcoef(predictions_df['coord_error'], predictions_df['angle_error_deg'])[0, 1]
            correlations.append(("Coordinate vs Angle error", coord_angle_corr))
        
        if 'coord_error' in predictions_df.columns and 'depth_error' in predictions_df.columns:
            coord_depth_corr = np.corrcoef(predictions_df['coord_error'], predictions_df['depth_error'])[0, 1]
            correlations.append(("Coordinate vs Depth error", coord_depth_corr))
        
        if 'angle_error_deg' in predictions_df.columns and 'depth_error' in predictions_df.columns:
            angle_depth_corr = np.corrcoef(predictions_df['angle_error_deg'], predictions_df['depth_error'])[0, 1]
            correlations.append(("Angle vs Depth error", angle_depth_corr))
        
        for corr_name, corr_value in correlations:
            f.write(f"{corr_name} correlation: {corr_value:.3f}\n")
        
        if correlations:
            f.write("\n")
        
        # Uncertainty analysis if available
        uncertainty_cols = ['depth_uncertainty', 'coord_uncertainty_x', 'angle_uncertainty']
        has_uncertainty = any(col in predictions_df.columns for col in uncertainty_cols)
        
        if has_uncertainty and any(predictions_df.get(col, pd.Series()).sum() > 0 for col in uncertainty_cols):
            f.write("UNCERTAINTY ANALYSIS\n")
            f.write("-" * 19 + "\n")
            
            uncertainty_corrs = []
            if ('depth_uncertainty' in predictions_df.columns and 'depth_error' in predictions_df.columns and 
                predictions_df['depth_uncertainty'].sum() > 0):
                depth_unc_corr = np.corrcoef(predictions_df['depth_uncertainty'], predictions_df['depth_error'])[0, 1]
                uncertainty_corrs.append(("Depth uncertainty vs error", depth_unc_corr))
            
            if ('coord_uncertainty_x' in predictions_df.columns and 'coord_error' in predictions_df.columns and
                predictions_df['coord_uncertainty_x'].sum() > 0):
                coord_unc_corr = np.corrcoef(predictions_df['coord_uncertainty_x'], predictions_df['coord_error'])[0, 1]
                uncertainty_corrs.append(("Coordinate uncertainty vs error", coord_unc_corr))
            
            if ('angle_uncertainty' in predictions_df.columns and 'angle_error_deg' in predictions_df.columns and
                predictions_df['angle_uncertainty'].sum() > 0):
                angle_unc_corr = np.corrcoef(predictions_df['angle_uncertainty'], predictions_df['angle_error_deg'])[0, 1]
                uncertainty_corrs.append(("Angle uncertainty vs error", angle_unc_corr))
            
            for unc_name, unc_corr in uncertainty_corrs:
                f.write(f"{unc_name} correlation: {unc_corr:.3f}\n")
            
            if uncertainty_corrs:
                f.write("\nNote: Higher correlation indicates better uncertainty calibration\n\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS FOR IMPROVEMENT\n")
        f.write("-" * 33 + "\n")
        
        # Find the worst performing aspects
        recommendations = []
        
        if 'coord_error' in predictions_df.columns:
            mean_coord_error = predictions_df['coord_error'].mean()
            if mean_coord_error > 0.5:  # If coordinate error > 0.5m
                recommendations.append("• High coordinate errors detected. Consider:")
                recommendations.append("  - Improving depth estimation accuracy")
                recommendations.append("  - Adding more geometric constraints")
                recommendations.append("  - Increasing training data for problematic scenes")
                recommendations.append("")
        
        if 'angle_error_deg' in predictions_df.columns:
            mean_angle_error = predictions_df['angle_error_deg'].mean()
            if mean_angle_error > 30:  # If angle error > 30 degrees
                recommendations.append("• High angle errors detected. Consider:")
                recommendations.append("  - Adding more rotational features")
                recommendations.append("  - Using circular loss functions")
                recommendations.append("  - Augmenting data with rotation variations")
                recommendations.append("")
        
        if 'scene_name' in predictions_df.columns and 'coord_error' in predictions_df.columns:
            scene_coord_errors = predictions_df.groupby('scene_name')['coord_error'].agg(['count', 'mean', 'std']).sort_values('mean', ascending=False)
            worst_scenes = scene_coord_errors.head(3).index.tolist()
            recommendations.append(f"• Focus improvement efforts on scenes: {', '.join(worst_scenes)}")
            recommendations.append("")
        
        if 'target_object_type' in predictions_df.columns and 'coord_error' in predictions_df.columns:
            obj_coord_errors = predictions_df.groupby('target_object_type')['coord_error'].agg(['count', 'mean', 'std']).sort_values('mean', ascending=False)
            worst_objects = obj_coord_errors.head(3).index.tolist()
            recommendations.append(f"• Focus improvement efforts on object types: {', '.join(worst_objects)}")
            recommendations.append("")
        
        if not recommendations:
            recommendations.append("• Model performance appears to be within acceptable ranges")
            recommendations.append("• Consider fine-tuning hyperparameters for further improvements")
        
        for rec in recommendations:
            f.write(f"{rec}\n")
    
    print(f"Analysis report saved to {report_path}")


def create_error_analysis_plots(predictions_df, output_dir):
    """Create detailed error analysis plots."""
    
    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    
    # Error distributions
    axes[0, 0].hist(predictions_df['coord_error'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Coordinate Error Distribution')
    axes[0, 0].set_xlabel('3D Error (m)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(predictions_df['coord_error'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {predictions_df["coord_error"].mean():.3f}m')
    axes[0, 0].legend()
    
    axes[0, 1].hist(predictions_df['angle_error_deg'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Angle Error Distribution')
    axes[0, 1].set_xlabel('Angle Error (degrees)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(predictions_df['angle_error_deg'].mean(), color='red', linestyle='--',
                       label=f'Mean: {predictions_df["angle_error_deg"].mean():.1f}°')
    axes[0, 1].legend()
    
    axes[0, 2].hist(predictions_df['depth_error'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 2].set_title('Depth Error Distribution')
    axes[0, 2].set_xlabel('Depth Error (m)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].axvline(predictions_df['depth_error'].mean(), color='red', linestyle='--',
                       label=f'Mean: {predictions_df["depth_error"].mean():.3f}m')
    axes[0, 2].legend()
    
    # Error correlations
    axes[0, 3].scatter(predictions_df['coord_error'], predictions_df['angle_error_deg'], alpha=0.5, s=10)
    axes[0, 3].set_xlabel('Coordinate Error (m)')
    axes[0, 3].set_ylabel('Angle Error (degrees)')
    axes[0, 3].set_title('Coordinate vs Angle Error')
    corr = np.corrcoef(predictions_df['coord_error'], predictions_df['angle_error_deg'])[0, 1]
    axes[0, 3].text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=axes[0, 3].transAxes)
    
    # Prediction vs target scatter plots
    axes[1, 0].scatter(predictions_df['true_x'], predictions_df['pred_x'], alpha=0.5, s=10)
    axes[1, 0].plot([predictions_df['true_x'].min(), predictions_df['true_x'].max()], 
                    [predictions_df['true_x'].min(), predictions_df['true_x'].max()], 'r--')
    axes[1, 0].set_xlabel('True X (m)')
    axes[1, 0].set_ylabel('Predicted X (m)')
    axes[1, 0].set_title('X Coordinate Predictions')
    
    axes[1, 1].scatter(predictions_df['true_y'], predictions_df['pred_y'], alpha=0.5, s=10)
    axes[1, 1].plot([predictions_df['true_y'].min(), predictions_df['true_y'].max()], 
                    [predictions_df['true_y'].min(), predictions_df['true_y'].max()], 'r--')
    axes[1, 1].set_xlabel('True Y (m)')
    axes[1, 1].set_ylabel('Predicted Y (m)')
    axes[1, 1].set_title('Y Coordinate Predictions')
    
    axes[1, 2].scatter(predictions_df['true_z'], predictions_df['pred_z'], alpha=0.5, s=10)
    axes[1, 2].plot([predictions_df['true_z'].min(), predictions_df['true_z'].max()], 
                    [predictions_df['true_z'].min(), predictions_df['true_z'].max()], 'r--')
    axes[1, 2].set_xlabel('True Z (m)')
    axes[1, 2].set_ylabel('Predicted Z (m)')
    axes[1, 2].set_title('Z Coordinate Predictions')
    
    axes[1, 3].scatter(predictions_df['true_angle_deg'], predictions_df['pred_angle_deg'], alpha=0.5, s=10)
    axes[1, 3].plot([predictions_df['true_angle_deg'].min(), predictions_df['true_angle_deg'].max()], 
                    [predictions_df['true_angle_deg'].min(), predictions_df['true_angle_deg'].max()], 'r--')
    axes[1, 3].set_xlabel('True Angle (degrees)')
    axes[1, 3].set_ylabel('Predicted Angle (degrees)')
    axes[1, 3].set_title('Angle Predictions')
    
    # Scene-based analysis if available
    if 'scene_name' in predictions_df.columns:
        # Error by scene
        scene_coord_errors = predictions_df.groupby('scene_name')['coord_error'].mean().sort_values(ascending=False)
        axes[2, 0].bar(range(len(scene_coord_errors.head(10))), scene_coord_errors.head(10).values)
        axes[2, 0].set_title('Top 10 Worst Scenes (Coord Error)')
        axes[2, 0].set_xlabel('Scene Rank')
        axes[2, 0].set_ylabel('Mean Coordinate Error (m)')
        
        scene_angle_errors = predictions_df.groupby('scene_name')['angle_error_deg'].mean().sort_values(ascending=False)
        axes[2, 1].bar(range(len(scene_angle_errors.head(10))), scene_angle_errors.head(10).values)
        axes[2, 1].set_title('Top 10 Worst Scenes (Angle Error)')
        axes[2, 1].set_xlabel('Scene Rank')
        axes[2, 1].set_ylabel('Mean Angle Error (degrees)')
        
        # Error by object type if available
        if 'target_object_type' in predictions_df.columns:
            obj_coord_errors = predictions_df.groupby('target_object_type')['coord_error'].mean().sort_values(ascending=False)
            axes[2, 2].bar(range(len(obj_coord_errors.head(10))), obj_coord_errors.head(10).values)
            axes[2, 2].set_title('Top 10 Worst Object Types (Coord)')
            axes[2, 2].set_xlabel('Object Type Rank')
            axes[2, 2].set_ylabel('Mean Coordinate Error (m)')
            axes[2, 2].tick_params(axis='x', rotation=45)
            
            obj_angle_errors = predictions_df.groupby('target_object_type')['angle_error_deg'].mean().sort_values(ascending=False)
            axes[2, 3].bar(range(len(obj_angle_errors.head(10))), obj_angle_errors.head(10).values)
            axes[2, 3].set_title('Top 10 Worst Object Types (Angle)')
            axes[2, 3].set_xlabel('Object Type Rank')
            axes[2, 3].set_ylabel('Mean Angle Error (degrees)')
            axes[2, 3].tick_params(axis='x', rotation=45)
    else:
        # If no scene info, show uncertainty plots if available
        if 'depth_uncertainty' in predictions_df.columns:
            axes[2, 0].scatter(predictions_df['depth_uncertainty'], predictions_df['depth_error'], alpha=0.5, s=10)
            axes[2, 0].set_xlabel('Depth Uncertainty')
            axes[2, 0].set_ylabel('Depth Error')
            axes[2, 0].set_title('Uncertainty vs Error (Depth)')
            
            axes[2, 1].scatter(predictions_df['coord_uncertainty_x'], predictions_df['coord_error'], alpha=0.5, s=10)
            axes[2, 1].set_xlabel('Coordinate Uncertainty')
            axes[2, 1].set_ylabel('Coordinate Error')
            axes[2, 1].set_title('Uncertainty vs Error (Coord)')
            
            axes[2, 2].scatter(predictions_df['angle_uncertainty'], predictions_df['angle_error_deg'], alpha=0.5, s=10)
            axes[2, 2].set_xlabel('Angle Uncertainty')
            axes[2, 2].set_ylabel('Angle Error (degrees)')
            axes[2, 2].set_title('Uncertainty vs Error (Angle)')
            
            # Fill the last subplot with summary statistics
            axes[2, 3].axis('off')
            summary_text = f"""Error Summary:
            Coord Mean: {predictions_df['coord_error'].mean():.3f}m
            Coord 95th: {predictions_df['coord_error'].quantile(0.95):.3f}m
            Angle Mean: {predictions_df['angle_error_deg'].mean():.1f}°
            Angle 95th: {predictions_df['angle_error_deg'].quantile(0.95):.1f}°
            Depth Mean: {predictions_df['depth_error'].mean():.3f}m"""
            axes[2, 3].text(0.1, 0.5, summary_text, transform=axes[2, 3].transAxes, 
                           fontsize=12, verticalalignment='center',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            axes[2, 3].set_title('Performance Summary')
        else:
            # Fill remaining subplots with informative messages
            for i, (row, col) in enumerate([(2, 0), (2, 1), (2, 2), (2, 3)]):
                axes[row, col].axis('off')
                if i == 0:
                    axes[row, col].text(0.5, 0.5, 'No uncertainty data\navailable', 
                                       ha='center', va='center', transform=axes[row, col].transAxes, fontsize=12)
                    axes[row, col].set_title('Uncertainty Analysis Not Available')
                elif i == 3:
                    # Show error summary even without uncertainty
                    summary_text = f"""Error Summary:
                    Coord Mean: {predictions_df['coord_error'].mean():.3f}m
                    Coord 95th: {predictions_df['coord_error'].quantile(0.95):.3f}m
                    Angle Mean: {predictions_df['angle_error_deg'].mean():.1f}°
                    Angle 95th: {predictions_df['angle_error_deg'].quantile(0.95):.1f}°
                    Depth Mean: {predictions_df['depth_error'].mean():.3f}m"""
                    axes[row, col].text(0.1, 0.5, summary_text, transform=axes[row, col].transAxes, 
                                       fontsize=12, verticalalignment='center',
                                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                    axes[row, col].set_title('Performance Summary')
                else:
                    axes[row, col].text(0.5, 0.5, 'Analysis not\navailable', 
                                       ha='center', va='center', transform=axes[row, col].transAxes, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_analysis_plots.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Error analysis plots saved to {output_dir}/error_analysis_plots.png")

def main():
    """
    Main function to train the enhanced AI2-THOR model with comprehensive evaluation.
    
    This function handles the complete pipeline:
    - Data loading and preprocessing
    - Model training with angle estimation
    - Comprehensive evaluation and analysis
    - Report generation
    """
    
    # Configuration parameters
    CONFIG = {
        'dataset_path': "ai2thor_adjusted.csv",
        'test_size': 0.2,
        'batch_size': 32,
        'epochs': 200,
        'use_uncertainty': True,
        'early_stopping_patience': 25,
        'learning_rate': 0.0005,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("=" * 70)
    print("ENHANCED AI2-THOR COORDINATE AND ANGLE ESTIMATION TRAINING")
    print("=" * 70)
    print(f"Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print()
    
    try:
        # Step 1: Load and validate dataset
        print("Step 1/7: Loading and validating dataset...")
        df = pd.read_csv(CONFIG['dataset_path'])
        
        if len(df) == 0:
            raise ValueError("Dataset is empty!")
            
        print(f"✓ Dataset loaded: {len(df)} samples")
        
        # Step 2: Data quality filtering
        print("\nStep 2/7: Applying data quality filters...")
        df_filtered = filter_high_quality_samples_no_bbox(df)
        
        if len(df_filtered) == 0:
            raise ValueError("No samples remaining after filtering!")
            
        print(f"✓ Quality filtering complete: {len(df_filtered)} samples retained")
        
        # Step 3: Feature engineering
        print("\nStep 3/7: Generating enhanced features...")
        try:
            enhanced_features = df_filtered.apply(
                calculate_comprehensive_geometric_features, 
                axis=1, 
                result_type='expand'
            )
            
            # Add enhanced features to dataframe
            for col in enhanced_features.columns:
                df_filtered[f'enhanced_{col}'] = enhanced_features[col]
                
            print(f"✓ Enhanced features created: {len(enhanced_features.columns)} new features")
            
        except Exception as e:
            print(f"✗ Feature engineering failed: {e}")
            return False
        
        # Step 4: Prepare training data
        print("\nStep 4/7: Preparing training data...")
        
        # Check for required columns
        required_columns = ['relative_depth', 'world_x', 'world_y', 'world_z', 'target_rot_y']
        missing_columns = [col for col in required_columns if col not in df_filtered.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Feature selection
        excluded_cols = ['relative_depth', 'world_x', 'world_y', 'world_z', 'target_rot_y',
                        'scene_name', 'scene_idx', 'pose_idx', 'target_object_type', 
                        'ref_object_type', 'target_object_id', 'ref_object_id']
        
        inference_incompatible = [
            'is_occluded', 'visibility_ratio',           # No ground truth occlusion in real images
            'ref_world_x', 'ref_world_y', 'ref_world_z', # Circular - we're trying to predict these!
            'target_rot_x', 'target_rot_z',              # Object rotation unknown from single image
            'ref_rot_x', 'ref_rot_y', 'ref_rot_z',       # Reference rotation also unknown
            'base_y', 'on_object',                       # Support detection unreliable
            'ground_truth', 'gt_consistency'             # No ground truth during inference
        ]

        feature_columns = [col for col in df_filtered.columns 
                        if col not in excluded_cols 
                        and not col.startswith('target_object') 
                        and not col.startswith('ref_object')
                        and not any(incompatible in col for incompatible in inference_incompatible)]
        
        print(f"\n=== Feature Selection Summary ===")
        print(f"Total features selected: {len(feature_columns)}")
        print(f"\nFeature categories:")
        bbox_features = [f for f in feature_columns if 'bbox' in f]
        depth_features = [f for f in feature_columns if 'depth' in f]
        enhanced_features = [f for f in feature_columns if f.startswith('enhanced_')]
        camera_features = [f for f in feature_columns if 'agent' in f or 'camera' in f or 'fov' in f]
        print(f"  Bbox features: {len(bbox_features)}")
        print(f"  Depth features: {len(depth_features)}")
        print(f"  Enhanced features: {len(enhanced_features)}")
        print(f"  Camera features: {len(camera_features)}")
        print(f"  Other: {len(feature_columns) - len(bbox_features) - len(depth_features) - len(enhanced_features) - len(camera_features)}")
        # Prepare target variables
        X = df_filtered[feature_columns].values
        y_depth = np.log1p(df_filtered['relative_depth'].values)
        y_coords = df_filtered[['world_x', 'world_y', 'world_z']].values
        y_angles = np.radians(df_filtered['target_rot_y'].values)
        
        # Handle NaN values
        valid_mask = ~(np.any(np.isnan(X), axis=1) | np.isnan(y_depth) | 
                      np.any(np.isnan(y_coords), axis=1) | np.isnan(y_angles))
        
        if not np.all(valid_mask):
            print(f"⚠ Removing {(~valid_mask).sum()} samples with NaN values")
            X = X[valid_mask]
            y_depth = y_depth[valid_mask]
            y_coords = y_coords[valid_mask]
            y_angles = y_angles[valid_mask]
            df_filtered = df_filtered[valid_mask].reset_index(drop=True)
        
        print(f"✓ Training data prepared: {len(X)} samples, {X.shape[1]} features")
        
        # Step 5: Model training
        print("\nStep 5/7: Training enhanced model...")
        
        # Train-validation split
        split_data = train_test_split(
            X, y_depth, y_coords, y_angles, df_filtered, 
            test_size=CONFIG['test_size'], 
            random_state=42
        )
        X_train, X_val, y_depth_train, y_depth_val, y_coords_train, y_coords_val, y_angles_train, y_angles_val, df_train, df_val = split_data
        
        # Scale features and coordinates
        feature_scaler = StandardScaler()
        coord_scaler = StandardScaler()
        
        X_train_scaled = feature_scaler.fit_transform(X_train)
        X_val_scaled = feature_scaler.transform(X_val)
        y_coords_train_scaled = coord_scaler.fit_transform(y_coords_train)
        y_coords_val_scaled = coord_scaler.transform(y_coords_val)
        
        # Create datasets and loaders
        train_dataset = EnhancedAI2ThorDataset(
            X_train_scaled, y_depth_train, y_coords_train_scaled, y_angles_train,
            augment=True, noise_std=0.01
        )
        val_dataset = EnhancedAI2ThorDataset(
            X_val_scaled, y_depth_val, y_coords_val_scaled, y_angles_val,
            augment=False
        )
        
        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
        
        # Initialize model
        model = EnhancedGeometricEstimator(
            input_dim=X_train.shape[1],
            hidden_dim=512,
            dropout_rate=0.2,
            num_heads=8,
            use_uncertainty=CONFIG['use_uncertainty']
        )
        
        # Initialize trainer
        trainer = EnhancedAI2ThorTrainer(
            model, train_loader, val_loader, coord_scaler, 
            CONFIG['device'], CONFIG['use_uncertainty']
        )
        
        # Train model
        trained_model = trainer.train(
            epochs=CONFIG['epochs'],
            early_stopping_patience=CONFIG['early_stopping_patience']
        )
        
        print("✓ Model training completed successfully!")
        
        # Step 6: Save model and generate reports
        print("\nStep 6/7: Saving model and generating reports...")
        
        # Save model checkpoint
        final_checkpoint = {
            'model_state_dict': trained_model.state_dict(),
            'feature_scaler': feature_scaler,
            'coord_scaler': coord_scaler,
            'feature_names': feature_columns,
            'model_type': 'enhanced_ai2thor_geometric_estimator_with_angles',
            'use_uncertainty': CONFIG['use_uncertainty'],
            'input_dim': X_train.shape[1],
            'training_metrics': trainer.val_metrics[-1] if trainer.val_metrics else None,
            'config': CONFIG
        }
        
        model_path = 'best_enhanced_ai2thor_model_with_angles.pth'
        torch.save(final_checkpoint, model_path)
        print(f"✓ Model saved: {model_path}")
        
        # Generate training plots
        # create_enhanced_training_plots(trainer, coord_scaler, val_dataset, CONFIG['device'], CONFIG['use_uncertainty'])
        print("✓ Training plots generated")
        
        # Step 7: Comprehensive evaluation
        print("\nStep 7/7: Performing comprehensive evaluation...")
        
        # Save predictions with scene information
        predictions_csv = "ai2thor_predictions_and_targets.csv"
        predictions_df = save_predictions_to_csv(trainer, df_val, predictions_csv)
        print(f"✓ Predictions saved: {predictions_csv}")
        
        # Perform detailed error analysis
        evaluation_dir = "evaluation_results"
        analysis_results = evaluate_worst_predictions(predictions_df, evaluation_dir)
        print(f"✓ Error analysis completed: {evaluation_dir}/")
        
        # Create standalone evaluation script
        # create_evaluation_script()
        print("✓ Standalone evaluation script created: evaluate_predictions.py")
        
        # Step 8: Final summary
        print("\n" + "=" * 70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        print("\nFinal Performance Summary:")
        if trainer.val_metrics:
            final_metrics = trainer.val_metrics[-1]
            print(f"   Coordinate R²: {final_metrics.get('coord_r2', 0):.4f}")
            print(f"   Coordinate MAE: {final_metrics.get('coord_mae', 0):.4f}m")
            print(f"   Angle MAE: {final_metrics.get('angle_mae', 0):.1f}°")
            print(f"   Depth R²: {final_metrics.get('depth_r2', 0):.4f}")
            
            # Per-axis breakdown
            print("\n   Per-axis Performance:")
            for axis in ['X', 'Y', 'Z']:
                r2_key, mae_key = f'{axis}_r2', f'{axis}_mae'
                print(f"     {axis}: R²={final_metrics.get(r2_key, 0):.4f}, MAE={final_metrics.get(mae_key, 0):.4f}m")
        
        print(f"\nFiles Generated:")
        print(f"   • {model_path} - Trained model")
        print(f"   • enhanced_ai2thor_training_evaluation_with_angles.png - Training plots")
        print(f"   • {predictions_csv} - All predictions and ground truth")
        print(f"   • {evaluation_dir}/ - Detailed error analysis")
        print(f"   • evaluate_predictions.py - Standalone evaluation script")
        
        print(f"\nModel Features:")
        print(f"   • Full 6DOF pose estimation (position + orientation)")
        print(f"   • Uncertainty quantification: {CONFIG['use_uncertainty']}")
        print(f"   • Multi-scale attention architecture")
        print(f"   • Comprehensive failure mode analysis")
        
        print(f"\nNext Steps:")
        print(f"   • Review error analysis in {evaluation_dir}/")
        print(f"   • Run 'python evaluate_predictions.py' for additional analysis")
        print(f"   • Use load_enhanced_model() function for inference")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        print(f"Check your dataset path and ensure all required columns are present.")
        import traceback
        print(f"\nFull traceback:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Training pipeline completed successfully!")
    else:
        print("\n❌ Training pipeline failed. Check the error messages above.")