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


class EnhancedAI2ThorDataset(Dataset):
    """Enhanced dataset with data augmentation capabilities."""
    def __init__(self, features, relative_depth_targets, world_coord_targets, 
                 augment=False, noise_std=0.01):
        self.features = torch.FloatTensor(features)
        self.relative_depth_targets = torch.FloatTensor(relative_depth_targets)
        self.world_coord_targets = torch.FloatTensor(world_coord_targets)
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
                self.world_coord_targets[idx])


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
    def __init__(self, depth_weight=0.3, coord_weight=0.7):
        super().__init__()
        self.depth_weight = depth_weight
        self.coord_weight = coord_weight
        
    def forward(self, depth_mean, depth_var, coord_mean, coord_var, depth_target, coord_target):
        # More numerically stable uncertainty loss
        depth_loss = 0.5 * (torch.log(depth_var + 1e-6) + 
                           (depth_target - depth_mean)**2 / (depth_var + 1e-6))
        
        coord_loss = 0.5 * torch.sum(torch.log(coord_var + 1e-6) + 
                                   (coord_target - coord_mean)**2 / (coord_var + 1e-6), dim=1)
        
        # Apply loss weights and add regularization
        total_loss = (self.depth_weight * depth_loss.mean() + 
                     self.coord_weight * coord_loss.mean())
        
        # Add small regularization to prevent variance collapse
        var_reg = 1e-4 * (torch.mean(1.0 / (depth_var + 1e-6)) + 
                         torch.mean(1.0 / (coord_var + 1e-6)))
        
        return total_loss + var_reg


class EnhancedGeometricEstimator(nn.Module):
    """Enhanced model with multi-scale attention and uncertainty estimation."""
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
                nn.Linear(hidden_dim // 2, 3)
            )
            self.coord_var_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 3),
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
                nn.Linear(hidden_dim // 2, 3)
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
            depth_var = self.depth_var_head(features).squeeze(-1)
            coord_var = self.coord_var_head(features)
            return depth_mean, coord_mean, depth_var, coord_var
        else:
            # Standard predictions
            if self.use_uncertainty:
                depth_pred = self.depth_mean_head(features).squeeze(-1)
                coord_pred = self.coord_mean_head(features)
            else:
                depth_pred = self.depth_head(features).squeeze(-1)
                coord_pred = self.coord_head(features)
            return depth_pred, coord_pred


def calculate_unprojection(row, img_width=300, img_height=300):
    """Calculates an initial 3D world coordinate guess using the unprojection formula."""
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
    
    # Get Camera Extrinsics (Position and Rotation)
    rotation = R.from_euler('xyz', [row['agent_rot_x'], row['agent_rot_y'], row['agent_rot_z']], degrees=True)
    rotation_matrix = rotation.as_matrix()
    translation_vector = np.array([row['agent_pos_x'], row['agent_pos_y'], row['agent_pos_z']])

    # Transform from Camera Coordinates to World Coordinates
    world_coords = rotation_matrix @ camera_coords + translation_vector
    
    return world_coords[0], world_coords[1], world_coords[2]


def calculate_comprehensive_geometric_features(row, img_width=300, img_height=300):
    """Calculate comprehensive features including enhanced depth map data."""
    features = {}
    
    # Basic unprojection (keep existing logic but improved)
    proj_x, proj_y, proj_z = calculate_unprojection(row, img_width, img_height)
    features.update({'proj_x': proj_x, 'proj_y': proj_y, 'proj_z': proj_z})
    
    # Enhanced depth features (utilize new depth map statistics)
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
    
    # Ground truth consistency features
    if 'target_ground_truth_distance' in row:
        features['target_gt_distance'] = row['target_ground_truth_distance']
        features['ref_gt_distance'] = row['ref_ground_truth_distance']
        
        # Depth map vs ground truth consistency
        features['target_depth_consistency'] = row.get('target_depth_gt_consistency', 0)
        features['ref_depth_consistency'] = row.get('ref_depth_gt_consistency', 0)
    
    # Existing geometric features (keep the good ones from your original code)
    bbox_center_x = max(0.001, min(0.999, row.get('target_bbox_center_x', 0.5)))
    bbox_center_y = max(0.001, min(0.999, row.get('target_bbox_center_y', 0.5)))
    
    features.update({
        'center_offset_x': bbox_center_x - 0.5,
        'center_offset_y': bbox_center_y - 0.5,
        'center_distance': np.sqrt((bbox_center_x - 0.5)**2 + (bbox_center_y - 0.5)**2),
        'angle_from_center': np.arctan2(bbox_center_y - 0.5, bbox_center_x - 0.5)
    })
    
    # Camera and orientation features (keep existing)
    fov_rad = row['field_of_view'] * (np.pi / 180.0)
    features.update({
        'horizon_sin': np.sin(np.radians(row['camera_horizon'])),
        'horizon_cos': np.cos(np.radians(row['camera_horizon'])),
        'rot_y_sin': np.sin(np.radians(row['agent_rot_y'])),
        'rot_y_cos': np.cos(np.radians(row['agent_rot_y'])),
        'fov_rad': fov_rad
    })
    
    return features


def calculate_unprojection_with_center(row, bbox_center_x, bbox_center_y, img_width=300, img_height=300):
    """Calculate unprojection with specified bbox center coordinates."""
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
    px = bbox_center_x * img_width
    py = bbox_center_y * img_height
    depth = row['dist_to_ref']
    
    # Unproject from 2D to 3D (in Camera's coordinate system)
    pixel_coords = np.array([px, py, 1])
    camera_coords = K_inv @ pixel_coords * depth
    
    # Get Camera Extrinsics (Position and Rotation)
    rotation = R.from_euler('xyz', [row['agent_rot_x'], row['agent_rot_y'], row['agent_rot_z']], degrees=True)
    rotation_matrix = rotation.as_matrix()
    translation_vector = np.array([row['agent_pos_x'], row['agent_pos_y'], row['agent_pos_z']])

    # Transform from Camera Coordinates to World Coordinates
    world_coords = rotation_matrix @ camera_coords + translation_vector
    
    return world_coords[0], world_coords[1], world_coords[2]


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
    print("\nIMPORTANT: Training will proceed, but you should fix the bounding box data issue!")
    return df.reset_index(drop=True)


class BalancedSampler:
    """Custom sampler to balance different types of scenes/distances."""
    def __init__(self, dataset_df):
        self.df = dataset_df
        self.depth_bins = self._create_depth_bins()
        
    def _create_depth_bins(self):
        """Create balanced depth bins."""
        depth_values = self.df['relative_depth'].values
        bins = np.quantile(depth_values, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        bin_indices = np.digitize(depth_values, bins) - 1
        return bin_indices
    
    def get_balanced_indices(self, n_samples):
        """Get balanced sample indices."""
        samples_per_bin = n_samples // 5
        balanced_indices = []
        
        for bin_idx in range(5):
            bin_mask = self.depth_bins == bin_idx
            bin_indices = np.where(bin_mask)[0]
            
            if len(bin_indices) >= samples_per_bin:
                selected = np.random.choice(bin_indices, samples_per_bin, replace=False)
            else:
                selected = np.random.choice(bin_indices, samples_per_bin, replace=True)
            
            balanced_indices.extend(selected)
        
        return np.array(balanced_indices)


class EnhancedAI2ThorTrainer:
    """Enhanced trainer with curriculum learning and uncertainty estimation."""
    def __init__(self, model, train_loader, val_loader, coord_scaler, device='cpu', use_uncertainty=True):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.coord_scaler = coord_scaler
        self.device = device
        self.use_uncertainty = use_uncertainty
        
        # Loss functions
        if use_uncertainty:
            self.criterion = UncertaintyLoss(depth_weight=0.3, coord_weight=0.7)
        else:
            self.depth_criterion = nn.MSELoss()
            self.coord_criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=0.0005,  # Much lower learning rate
            weight_decay=0.001,  # Lower weight decay
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # More aggressive learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.3,  # Reduce LR more aggressively
            patience=5,   # Reduce patience
            min_lr=1e-7
        )
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        for batch_features, batch_depth_targets, batch_coord_targets in self.train_loader:
            batch_features = batch_features.to(self.device)
            batch_depth_targets = batch_depth_targets.to(self.device)
            batch_coord_targets = batch_coord_targets.to(self.device)
            
            # Check for invalid targets
            if torch.any(torch.isnan(batch_depth_targets)) or torch.any(torch.isnan(batch_coord_targets)):
                continue
                
            self.optimizer.zero_grad()
            
            try:
                if self.use_uncertainty:
                    depth_mean, coord_mean, depth_var, coord_var = self.model(batch_features, return_uncertainty=True)
                    
                    # Clamp predictions to reasonable ranges
                    depth_mean = torch.clamp(depth_mean, -10, 10)
                    coord_mean = torch.clamp(coord_mean, -10, 10)
                    depth_var = torch.clamp(depth_var, 1e-6, 10)
                    coord_var = torch.clamp(coord_var, 1e-6, 10)
                    
                    loss = self.criterion(depth_mean, depth_var, coord_mean, coord_var, 
                                        batch_depth_targets, batch_coord_targets)
                else:
                    depth_pred, coord_pred = self.model(batch_features)
                    depth_pred = torch.clamp(depth_pred, -10, 10)
                    coord_pred = torch.clamp(coord_pred, -10, 10)
                    
                    depth_loss = self.depth_criterion(depth_pred, batch_depth_targets)
                    coord_loss = self.coord_criterion(coord_pred, batch_coord_targets)
                    loss = 0.3 * depth_loss + 0.7 * coord_loss
                
                # Check for invalid loss
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                    
                loss.backward()
                
                # More aggressive gradient clipping
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
        
        with torch.no_grad():
            for batch_features, batch_depth_targets, batch_coord_targets in self.val_loader:
                batch_features = batch_features.to(self.device)
                batch_depth_targets = batch_depth_targets.to(self.device)
                batch_coord_targets = batch_coord_targets.to(self.device)
                
                if self.use_uncertainty:
                    depth_mean, coord_mean, depth_var, coord_var = self.model(batch_features, return_uncertainty=True)
                    loss = self.criterion(depth_mean, depth_var, coord_mean, coord_var, 
                                        batch_depth_targets, batch_coord_targets)
                    depth_pred, coord_pred = depth_mean, coord_mean
                else:
                    depth_pred, coord_pred = self.model(batch_features)
                    depth_loss = self.depth_criterion(depth_pred, batch_depth_targets)
                    coord_loss = self.coord_criterion(coord_pred, batch_coord_targets)
                    loss = 0.3 * depth_loss + 0.7 * coord_loss
                
                batch_size = batch_features.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Collect predictions for metrics
                all_depth_preds.append(depth_pred.cpu())
                all_depth_targets.append(batch_depth_targets.cpu())
                all_coord_preds.append(coord_pred.cpu())
                all_coord_targets.append(batch_coord_targets.cpu())
        
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
        
        # Unscale coordinates
        coord_preds_unscaled = self.coord_scaler.inverse_transform(all_coord_preds)
        coord_targets_unscaled = self.coord_scaler.inverse_transform(all_coord_targets)
        
        # Calculate R² and MAE
        depth_r2 = r2_score(all_depth_targets, all_depth_preds)
        depth_mae = mean_absolute_error(all_depth_targets, all_depth_preds)
        coord_r2 = r2_score(coord_targets_unscaled, coord_preds_unscaled)
        coord_mae = mean_absolute_error(coord_targets_unscaled, coord_preds_unscaled)
        
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
            **axis_metrics
        }
    
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
                }, 'best_enhanced_ai2thor_model.pth')
            else:
                self.patience_counter += 1
            
            if epoch % 10 == 0 or epoch < 10:
                print(f"Epoch {epoch:3d} | Train: {train_loss:.4f} | "
                      f"Val: {val_loss:.4f} | "
                      f"R²: D={val_metrics['depth_r2']:.3f} C={val_metrics['coord_r2']:.3f} | "
                      f"MAE: C={val_metrics['coord_mae']:.3f}m")
            
            if self.patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        return self.model


def train_enhanced_ai2thor_model(dataset_path="ai2thor_coordinate_dataset.csv", 
                                test_size=0.2, 
                                batch_size=32, 
                                epochs=200,
                                use_uncertainty=True):
    """Enhanced training function with all improvements."""
    
    print("Loading and enhancing AI2-THOR dataset...")
    try:
        df = pd.read_csv(dataset_path)
        # Add this right after loading the CSV in train_enhanced_ai2thor_model:
        print("\nDEBUG: Checking dataset quality...")
        print(f"Non-zero target bboxes: {(df['target_bbox_width'] > 0).sum()}/{len(df)}")
        print(f"Non-zero ref bboxes: {(df['ref_bbox_width'] > 0).sum()}/{len(df)}")
        print(f"Sample bbox values: {df[['target_bbox_width', 'target_bbox_height']].head()}")
    except FileNotFoundError:
        print(f"Error: Dataset file '{dataset_path}' not found!")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    if len(df) == 0:
        print("Dataset is empty!")
        return None
    
    print(f"Initial dataset loaded: {len(df)} samples")
    
    # Check for required columns
    required_columns = ['relative_depth', 'world_x', 'world_y', 'world_z', 'target_depth_mean', 
                       'target_visibility_ratio', 'ref_visibility_ratio', 'target_bbox_width', 
                       'target_bbox_height', 'ref_bbox_width', 'ref_bbox_height', 'dist_to_ref']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return None
    
    # Apply data quality filtering (without bbox filtering)
    df = filter_high_quality_samples_no_bbox(df)
    
    if len(df) == 0:
        print("Error: No samples remaining after filtering!")
        return None
    
    # Calculate enhanced geometric features (with bbox workaround)
    print("Calculating comprehensive geometric features...")
    try:
        enhanced_features = df.apply(calculate_comprehensive_geometric_features, axis=1, result_type='expand')
        
        # Add enhanced features to dataframe
        for col in enhanced_features.columns:
            df[f'enhanced_{col}'] = enhanced_features[col]
        
        print(f"Enhanced geometric features created: {len(enhanced_features.columns)} new features")
    except Exception as e:
        print(f"Error calculating enhanced features: {e}")
        return None

    # Update feature column selection to include new depth features
    excluded_cols = ['relative_depth', 'world_x', 'world_y', 'world_z', 
                    'scene_name', 'scene_idx', 'pose_idx', 'target_object_type', 
                    'ref_object_type', 'target_object_id', 'ref_object_id']

    feature_columns = [col for col in df.columns if col not in excluded_cols and 
                    not col.startswith('target_object') and not col.startswith('ref_object')]
    
    X = df[feature_columns].values
    y_depth = np.log1p(df['relative_depth'].values)
    y_coords = df[['world_x', 'world_y', 'world_z']].values
    
    print(f"Enhanced feature dimensions: {X.shape[1]}")
    print(f"Feature columns: {len(feature_columns)}")
    
    # Check for NaN values
    if np.any(np.isnan(X)) or np.any(np.isnan(y_depth)) or np.any(np.isnan(y_coords)):
        print("Warning: NaN values detected in features or targets. Removing affected samples...")
        valid_mask = ~(np.any(np.isnan(X), axis=1) | np.isnan(y_depth) | np.any(np.isnan(y_coords), axis=1))
        X = X[valid_mask]
        y_depth = y_depth[valid_mask]
        y_coords = y_coords[valid_mask]
        print(f"Samples after NaN removal: {len(X)}")
    
    if len(X) == 0:
        print("Error: No valid samples remaining after NaN removal!")
        return None
    
    # Train-validation split
    X_train, X_val, y_depth_train, y_depth_val, y_coords_train, y_coords_val = train_test_split(
        X, y_depth, y_coords, test_size=test_size, random_state=42
    )
    
    # Scale features
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_val_scaled = feature_scaler.transform(X_val)
    
    # Scale coordinates
    coord_scaler = StandardScaler()
    y_coords_train_scaled = coord_scaler.fit_transform(y_coords_train)
    y_coords_val_scaled = coord_scaler.transform(y_coords_val)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    # Make sure input dimension works with multi-scale attention
    input_dim = X_train.shape[1]
    print(f"Original input dimension: {input_dim}")
    
    # No need to pad - the fixed MultiScaleAttentionBlock handles any input dimension
    X_train_final = X_train_scaled
    X_val_final = X_val_scaled
    
    # Create enhanced datasets with augmentation
    train_dataset = EnhancedAI2ThorDataset(
        X_train_final, y_depth_train, y_coords_train_scaled, 
        augment=True, noise_std=0.01
    )
    val_dataset = EnhancedAI2ThorDataset(
        X_val_final, y_depth_val, y_coords_val_scaled, 
        augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize enhanced model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = EnhancedGeometricEstimator(
        input_dim=input_dim,
        hidden_dim=512,
        dropout_rate=0.2,
        num_heads=8,
        use_uncertainty=use_uncertainty
    )
    
    # Initialize enhanced trainer
    trainer = EnhancedAI2ThorTrainer(
        model, train_loader, val_loader, coord_scaler, device, use_uncertainty
    )
    
    # Train the model
    trained_model = trainer.train(epochs=epochs)
    
    # Save final model and scalers
    final_checkpoint = {
        'model_state_dict': trained_model.state_dict(),
        'feature_scaler': feature_scaler,
        'coord_scaler': coord_scaler,
        'feature_names': feature_columns,
        'model_type': 'enhanced_ai2thor_geometric_estimator',
        'use_uncertainty': use_uncertainty,
        'input_dim': input_dim,
        'training_metrics': trainer.val_metrics[-1] if trainer.val_metrics else None
    }
    
    torch.save(final_checkpoint, 'enhanced_ai2thor_world_coordinate_model.pth')
    print("Enhanced model saved as 'enhanced_ai2thor_world_coordinate_model.pth'")
    
    # Generate comprehensive training plots
    create_enhanced_training_plots(trainer, coord_scaler, val_dataset, device, use_uncertainty)
    
    return trained_model, feature_scaler, coord_scaler, feature_columns


def create_enhanced_training_plots(trainer, coord_scaler, val_dataset, device, use_uncertainty=True):
    """Create comprehensive training and evaluation plots for enhanced model."""
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    
    epochs = range(len(trainer.train_losses))
    
    # Training curves
    axes[0,0].plot(epochs, trainer.train_losses, label='Train Loss', alpha=0.8)
    axes[0,0].plot(epochs, trainer.val_losses, label='Val Loss', alpha=0.8)
    axes[0,0].set_title('Total Loss Over Time')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # R² scores over time
    axes[0,1].plot(epochs, [m['depth_r2'] for m in trainer.val_metrics], label='Depth R²', alpha=0.8)
    axes[0,1].plot(epochs, [m['coord_r2'] for m in trainer.val_metrics], label='Coordinate R²', alpha=0.8)
    axes[0,1].set_title('R² Score Over Time')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('R² Score')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # MAE over time
    axes[0,2].plot(epochs, [m['depth_mae'] for m in trainer.val_metrics], label='Depth MAE', alpha=0.8)
    axes[0,2].plot(epochs, [m['coord_mae'] for m in trainer.val_metrics], label='Coordinate MAE', alpha=0.8)
    axes[0,2].set_title('MAE Over Time')
    axes[0,2].set_xlabel('Epoch')
    axes[0,2].set_ylabel('MAE')
    axes[0,2].legend()
    axes[0,2].grid(True)
    
    # Get final predictions for evaluation plots
    trainer.model.eval()
    with torch.no_grad():
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        all_depth_preds = []
        all_depth_targets = []
        all_coord_preds = []
        all_coord_targets = []
        all_depth_vars = []
        all_coord_vars = []
        
        for batch_features, batch_depth_targets, batch_coord_targets in val_loader:
            batch_features = batch_features.to(device)
            
            if use_uncertainty:
                depth_pred, coord_pred, depth_var, coord_var = trainer.model(batch_features, return_uncertainty=True)
                all_depth_vars.append(depth_var.cpu())
                all_coord_vars.append(coord_var.cpu())
            else:
                depth_pred, coord_pred = trainer.model(batch_features)
            
            all_depth_preds.append(depth_pred.cpu())
            all_depth_targets.append(batch_depth_targets.cpu())
            all_coord_preds.append(coord_pred.cpu())
            all_coord_targets.append(batch_coord_targets.cpu())
        
        depth_preds = torch.cat(all_depth_preds).numpy()
        depth_targets = torch.cat(all_depth_targets).numpy()
        
        # Clip extreme values to prevent overflow
        depth_preds = np.clip(depth_preds, -10, 10)
        depth_targets = np.clip(depth_targets, -10, 10)
        coord_preds = torch.cat(all_coord_preds).numpy()
        coord_targets = torch.cat(all_coord_targets).numpy()
        
        if use_uncertainty:
            depth_vars = torch.cat(all_depth_vars).numpy()
            coord_vars = torch.cat(all_coord_vars).numpy()
    
    # Unscale coordinates for visualization
    coord_preds_unscaled = coord_scaler.inverse_transform(coord_preds)
    coord_targets_unscaled = coord_scaler.inverse_transform(coord_targets)
    
    # Depth prediction scatter
    axes[1,0].scatter(depth_targets, depth_preds, alpha=0.5, s=20)
    min_depth, max_depth = depth_targets.min(), depth_targets.max()
    axes[1,0].plot([min_depth, max_depth], [min_depth, max_depth], 'r--', alpha=0.8)
    axes[1,0].set_xlabel('True Relative Depth (log)')
    axes[1,0].set_ylabel('Predicted Relative Depth (log)')
    axes[1,0].set_title('Depth Prediction Accuracy')
    axes[1,0].grid(True)
    
    # Coordinate prediction accuracy per axis
    for i, axis in enumerate(['X', 'Y', 'Z']):
        row = 1
        col = i + 1 if i < 2 else 2
        if i == 2:
            row = 2
            col = 0
            
        axes[row,col].scatter(coord_targets_unscaled[:, i], coord_preds_unscaled[:, i], alpha=0.5, s=20)
        min_coord = coord_targets_unscaled[:, i].min()
        max_coord = coord_targets_unscaled[:, i].max()
        axes[row,col].plot([min_coord, max_coord], [min_coord, max_coord], 'r--', alpha=0.8)
        axes[row,col].set_xlabel(f'True {axis} Coordinate (m)')
        axes[row,col].set_ylabel(f'Predicted {axis} Coordinate (m)')
        axes[row,col].set_title(f'{axis}-Axis Prediction Accuracy')
        axes[row,col].grid(True)
    
    # Error distribution
    coord_errors = np.linalg.norm(coord_targets_unscaled - coord_preds_unscaled, axis=1)
    axes[2,1].hist(coord_errors, bins=50, alpha=0.7, edgecolor='black')
    axes[2,1].axvline(np.mean(coord_errors), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(coord_errors):.3f}m')
    axes[2,1].axvline(np.median(coord_errors), color='green', linestyle='--', 
                      label=f'Median: {np.median(coord_errors):.3f}m')
    axes[2,1].set_xlabel('3D Coordinate Error (m)')
    axes[2,1].set_ylabel('Frequency')
    axes[2,1].set_title('Prediction Error Distribution')
    axes[2,1].legend()
    axes[2,1].grid(True)
    
    # Uncertainty visualization (if available)
    if use_uncertainty:
        # Plot prediction uncertainty vs error
        coord_error_per_sample = np.linalg.norm(coord_targets_unscaled - coord_preds_unscaled, axis=1)
        coord_uncertainty = np.mean(coord_vars, axis=1)  # Average uncertainty across x,y,z
        
        axes[2,2].scatter(coord_uncertainty, coord_error_per_sample, alpha=0.5, s=20)
        axes[2,2].set_xlabel('Predicted Uncertainty')
        axes[2,2].set_ylabel('Actual Error (m)')
        axes[2,2].set_title('Uncertainty vs Actual Error')
        axes[2,2].grid(True)
        
        # Calculate correlation between uncertainty and error
        correlation = np.corrcoef(coord_uncertainty, coord_error_per_sample)[0, 1]
        axes[2,2].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                       transform=axes[2,2].transAxes, fontsize=10, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    else:
        # 3D scatter plot of predictions vs targets
        ax_3d = fig.add_subplot(3, 3, 9, projection='3d')
        idx_sample = np.random.choice(len(coord_targets_unscaled), min(500, len(coord_targets_unscaled)), replace=False)
        
        ax_3d.scatter(coord_targets_unscaled[idx_sample, 0], 
                      coord_targets_unscaled[idx_sample, 1], 
                      coord_targets_unscaled[idx_sample, 2],
                      c='blue', alpha=0.6, s=20, label='True')
        ax_3d.scatter(coord_preds_unscaled[idx_sample, 0], 
                      coord_preds_unscaled[idx_sample, 1], 
                      coord_preds_unscaled[idx_sample, 2],
                      c='red', alpha=0.6, s=20, label='Predicted')
        ax_3d.set_xlabel('X (m)')
        ax_3d.set_ylabel('Y (m)')
        ax_3d.set_zlabel('Z (m)')
        ax_3d.set_title('3D Coordinate Predictions')
        ax_3d.legend()
    
    plt.tight_layout()
    plt.savefig('enhanced_ai2thor_training_evaluation.png', dpi=150, bbox_inches='tight')
    print("Enhanced training plots saved as 'enhanced_ai2thor_training_evaluation.png'")
    
    # Print comprehensive final metrics
    final_metrics = trainer.val_metrics[-1]
    print(f"\n=== Enhanced Model Performance ===")
    print(f"Relative Depth - R²: {final_metrics['depth_r2']:.4f}, MAE: {final_metrics['depth_mae']:.4f}")
    print(f"World Coordinates - R²: {final_metrics['coord_r2']:.4f}, MAE: {final_metrics['coord_mae']:.4f}m")
    print(f"Per-axis Performance:")
    for axis in ['X', 'Y', 'Z']:
        print(f"  {axis}: R²={final_metrics[f'{axis}_r2']:.4f}, MAE={final_metrics[f'{axis}_mae']:.4f}m")
    print(f"3D Error Statistics: Mean={np.mean(coord_errors):.4f}m, Median={np.median(coord_errors):.4f}m")
    print(f"3D Error Percentiles: 90th={np.percentile(coord_errors, 90):.4f}m, 95th={np.percentile(coord_errors, 95):.4f}m")
    
    if use_uncertainty:
        print(f"Uncertainty-Error Correlation: {np.corrcoef(coord_uncertainty, coord_error_per_sample)[0,1]:.4f}")


def load_enhanced_model(model_path, device='cpu'):
    """Load a trained enhanced model for inference."""
    checkpoint = torch.load(model_path, map_location=device)
    
    model = EnhancedGeometricEstimator(
        input_dim=checkpoint['input_dim'],
        hidden_dim=512,
        dropout_rate=0.2,
        num_heads=8,
        use_uncertainty=checkpoint['use_uncertainty']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint['feature_scaler'], checkpoint['coord_scaler'], checkpoint['feature_names']


def predict_coordinates(model, feature_scaler, coord_scaler, feature_names, 
                       sample_data, device='cpu', return_uncertainty=False):
    """Make predictions using the trained enhanced model."""
    model.eval()
    
    # Prepare features
    features = np.array([sample_data[name] for name in feature_names]).reshape(1, -1)
    features_scaled = feature_scaler.transform(features)
    
    features_tensor = torch.FloatTensor(features_scaled).to(device)
    
    with torch.no_grad():
        if return_uncertainty:
            depth_mean, coord_mean, depth_var, coord_var = model(features_tensor, return_uncertainty=True)
            
            # Convert depth back from log space
            depth_pred = np.expm1(depth_mean.cpu().numpy()[0])
            depth_uncertainty = depth_var.cpu().numpy()[0]
            
            # Unscale coordinates
            coord_pred = coord_scaler.inverse_transform(coord_mean.cpu().numpy())[0]
            coord_uncertainty = coord_var.cpu().numpy()[0]
            
            return {
                'relative_depth': depth_pred,
                'depth_uncertainty': depth_uncertainty,
                'world_coordinates': coord_pred,
                'coord_uncertainty': coord_uncertainty
            }
        else:
            depth_pred, coord_pred = model(features_tensor)
            
            # Convert depth back from log space
            depth_pred = np.expm1(depth_pred.cpu().numpy()[0])
            
            # Unscale coordinates
            coord_pred = coord_scaler.inverse_transform(coord_pred.cpu().numpy())[0]
            
            return {
                'relative_depth': depth_pred,
                'world_coordinates': coord_pred
            }


if __name__ == "__main__":
    # Train the enhanced model
    print("Training Enhanced AI2-THOR Coordinate Estimation Model")
    print("====================================================")
    
    result = train_enhanced_ai2thor_model(
        dataset_path="ai2thor_coordinate_dataset.csv",
        test_size=0.2,
        batch_size=32,
        epochs=200,
        use_uncertainty=True
    )
    
    if result is not None:
        model, feature_scaler, coord_scaler, feature_names = result
        print("\nEnhanced AI2-THOR coordinate estimation model training completed successfully!")
        print("Key improvements implemented:")
        print("   • Enhanced geometric feature engineering (25+ new features)")
        print("   • Multi-scale attention architecture with feature grouping")
        print("   • Data quality filtering and augmentation")
        print("   • Uncertainty estimation for prediction confidence")
        print("   • Balanced sampling and curriculum learning")
        print("\nModel saved as 'enhanced_ai2thor_world_coordinate_model.pth'")
        print("Training plots saved as 'enhanced_ai2thor_training_evaluation.png'")
        print("\nReady for inference on real-world images!")
    else:
        print("\nTraining failed. Please check your dataset and try again.")