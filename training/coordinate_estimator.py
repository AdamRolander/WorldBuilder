import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from mpl_toolkits.mplot3d import Axes3D

class WorldCoordinateDatasetBuilder:
    def __init__(self, output_base_dir="final_dataset"):
        self.output_base_dir = output_base_dir
        self.features = []
        self.relative_depth_targets = []
        self.world_coord_targets = []
        self.scene_info = []

    def extract_features_from_metadata(self):
        """Extract features for both relative depth and world coordinate prediction."""
        print(f"Searching for metadata in: {self.output_base_dir}")
        metadata_paths = [p for p in Path(self.output_base_dir).rglob("metadata.json")]
        print(f"Found {len(metadata_paths)} metadata files. Processing...")

        for metadata_path in metadata_paths:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            static_objects_data = {obj['category_id']: obj for obj in metadata.get('scene_objects', [])}

            for frame in metadata.get('frames', []):
                if 'camera_pose' not in frame or 'camera_intrinsics' not in frame:
                    continue

                # Get camera pose and intrinsics
                camera_pose = np.array(frame['camera_pose'])
                camera_intrinsics = np.array(frame['camera_intrinsics'])
                
                # Get all objects in this frame with valid pixel counts
                frame_objects = []
                for obj_key, frame_obj_data in frame.get('objects_in_frame', {}).items():
                    if frame_obj_data.get('pixel_count', 0) == 0:
                        continue
                    
                    category_id = int(obj_key.split('_')[1])
                    static_data = static_objects_data.get(category_id)
                    if not static_data: 
                        continue
                    
                    frame_objects.append({
                        'category_id': category_id,
                        'static_data': static_data,
                        'frame_data': frame_obj_data
                    })
                
                # Need at least 2 objects for relative comparisons
                if len(frame_objects) < 2:
                    continue
                
                # Choose reference object (e.g., the one with largest bbox area)
                ref_obj = max(frame_objects, key=lambda x: x['frame_data']['bbox_area'])
                ref_depth = ref_obj['frame_data']['z_depth']
                
                # Calculate reference object's world coordinates
                ref_world_coords = self.pixel_to_world_coords(
                    ref_obj['frame_data']['bbox_center'],
                    ref_depth,
                    camera_intrinsics,
                    camera_pose
                )
                
                # Create training pairs: (target_object, reference_object) -> relative_depth + world_coords
                for target_obj in frame_objects:
                    if target_obj['category_id'] == ref_obj['category_id']:
                        continue  # Skip self-comparison
                    
                    # Extract features for both objects
                    target_features = self.extract_object_features(target_obj, frame)
                    ref_features = self.extract_object_features(ref_obj, frame)
                    
                    # Add camera-specific features
                    camera_features = self.extract_camera_features(frame)
                    
                    # Combine features: [target_features, ref_features, comparative_features, camera_features]
                    comparative_features = self.extract_comparative_features(target_obj, ref_obj)
                    combined_features = np.concatenate([
                        target_features, ref_features, comparative_features, camera_features
                    ])
                    
                    # Targets
                    target_depth = target_obj['frame_data']['z_depth']
                    relative_depth = target_depth / ref_depth if ref_depth > 0 else 1.0
                    
                    # Calculate target object's world coordinates
                    target_world_coords = self.pixel_to_world_coords(
                        target_obj['frame_data']['bbox_center'],
                        target_depth,
                        camera_intrinsics,
                        camera_pose
                    )
                    
                    self.features.append(combined_features)
                    self.relative_depth_targets.append(relative_depth)
                    self.world_coord_targets.append(target_world_coords)
                    
                    self.scene_info.append({
                        'scene_id': metadata['scene_id'],
                        'frame_id': frame['frame_id'],
                        'target_category_id': target_obj['category_id'],
                        'ref_category_id': ref_obj['category_id'],
                        'target_absolute_depth': target_depth,
                        'ref_absolute_depth': ref_depth,
                        'relative_depth': relative_depth,
                        'target_world_coords': target_world_coords,
                        'ref_world_coords': ref_world_coords,
                        'camera_pose': camera_pose,
                        'target_pixel_coords': target_obj['frame_data']['bbox_center']
                    })

        print(f"Extracted {len(self.features)} training samples.")
        return (np.array(self.features), 
                np.array(self.relative_depth_targets), 
                np.array(self.world_coord_targets))

    def pixel_to_world_coords(self, pixel_coords, depth, camera_intrinsics, camera_pose):
        """Convert pixel coordinates and depth to world coordinates."""
        # Unpack pixel coordinates
        u, v = pixel_coords
        
        # Camera intrinsics
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
        
        # Convert to camera coordinates
        x_cam = (u - cx) * depth / fx
        y_cam = (v - cy) * depth / fy
        z_cam = depth
        
        # Homogeneous camera coordinates
        camera_coords = np.array([x_cam, y_cam, z_cam, 1.0])
        
        # Transform to world coordinates using camera pose
        world_coords = camera_pose @ camera_coords
        
        return world_coords[:3]  # Return only x, y, z

    def world_to_pixel_coords(self, world_coords, camera_intrinsics, camera_pose):
        """Convert world coordinates back to pixel coordinates (for validation)."""
        # Homogeneous world coordinates
        world_coords_homo = np.append(world_coords, 1.0)
        
        # Transform to camera coordinates
        camera_coords = np.linalg.inv(camera_pose) @ world_coords_homo
        
        # Project to image plane
        x_cam, y_cam, z_cam = camera_coords[:3]
        
        # Camera intrinsics
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
        
        # Convert to pixel coordinates
        u = (x_cam * fx / z_cam) + cx
        v = (y_cam * fy / z_cam) + cy
        
        return np.array([u, v]), z_cam

    def extract_camera_features(self, frame):
        """Extract camera-specific features."""
        camera_pose = np.array(frame['camera_pose'])
        camera_intrinsics = np.array(frame['camera_intrinsics'])
        
        # Camera position (translation)
        camera_position = camera_pose[:3, 3]
        
        # Camera orientation (extract Euler angles from rotation matrix)
        rotation_matrix = camera_pose[:3, :3]
        # Simple orientation features - you could use proper Euler angle extraction
        orientation_features = rotation_matrix.flatten()[:6]  # First 6 elements as features
        
        # Intrinsic parameters
        fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
        cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
        
        return np.concatenate([
            camera_position,      # 3 features: x, y, z position
            orientation_features, # 6 features: rotation matrix elements
            [fx, fy, cx, cy]     # 4 features: intrinsic parameters
        ])

    def extract_object_features(self, obj_data, frame):
        """Extract features for a single object (same as before)."""
        frame_obj_data = obj_data['frame_data']
        static_data = obj_data['static_data']
        intrinsics = frame['camera_intrinsics']
        
        bbox_width = frame_obj_data['bbox_width']
        bbox_height = frame_obj_data['bbox_height']
        pixel_count = frame_obj_data['pixel_count']

        bbox_area = bbox_width * bbox_height
        bbox_aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 1.0
        pixel_density = pixel_count / bbox_area if bbox_area > 0 else 0.0

        dims = static_data.get("dimensions_xyz", {})
        longest_3d_dim = max(dims.get('width', 0), dims.get('depth', 0), dims.get('height', 0))
        if longest_3d_dim == 0: 
            longest_3d_dim = 1.0

        longest_2d_dim_px = max(bbox_width, bbox_height)
        fx = intrinsics[0][0]
        analytical_depth = (longest_3d_dim * fx) / longest_2d_dim_px if longest_2d_dim_px > 0 else 10.0

        img_width, img_height = 640, 480
        center_x, center_y = frame_obj_data['bbox_center']
        norm_pixel_x = center_x / img_width
        norm_pixel_y = center_y / img_height
        norm_dist_from_center = np.sqrt((norm_pixel_x - 0.5)**2 + (norm_pixel_y - 0.5)**2) * np.sqrt(2)

        return np.array([
            pixel_count, longest_3d_dim, bbox_width, bbox_height, 
            bbox_aspect_ratio, pixel_density, analytical_depth, 
            norm_pixel_x, norm_pixel_y, norm_dist_from_center
        ])

    def extract_comparative_features(self, target_obj, ref_obj):
        """Extract features comparing two objects."""
        target_frame = target_obj['frame_data']
        ref_frame = ref_obj['frame_data']
        
        # Size ratios
        bbox_width_ratio = target_frame['bbox_width'] / ref_frame['bbox_width'] if ref_frame['bbox_width'] > 0 else 1.0
        bbox_height_ratio = target_frame['bbox_height'] / ref_frame['bbox_height'] if ref_frame['bbox_height'] > 0 else 1.0
        pixel_count_ratio = target_frame['pixel_count'] / ref_frame['pixel_count'] if ref_frame['pixel_count'] > 0 else 1.0
        
        # True size ratio
        target_dims = target_obj['static_data'].get("dimensions_xyz", {})
        ref_dims = ref_obj['static_data'].get("dimensions_xyz", {})
        target_true_size = max(target_dims.get('width', 0), target_dims.get('depth', 0), target_dims.get('height', 0))
        ref_true_size = max(ref_dims.get('width', 0), ref_dims.get('depth', 0), ref_dims.get('height', 0))
        true_size_ratio = target_true_size / ref_true_size if ref_true_size > 0 else 1.0
        
        # Spatial relationship
        target_center = np.array(target_frame['bbox_center'])
        ref_center = np.array(ref_frame['bbox_center'])
        pixel_distance = np.linalg.norm(target_center - ref_center)
        
        # Relative position (normalized)
        rel_x = (target_center[0] - ref_center[0]) / 640  # Normalize by image width
        rel_y = (target_center[1] - ref_center[1]) / 480  # Normalize by image height
        
        return np.array([
            bbox_width_ratio, bbox_height_ratio, pixel_count_ratio, 
            true_size_ratio, pixel_distance, rel_x, rel_y
        ])

    def get_feature_names(self):
        object_features = [
            'pixel_count', 'object_true_size', 'bbox_width', 'bbox_height', 
            'bbox_aspect_ratio', 'pixel_density', 'analytical_depth', 'norm_pixel_x',
            'norm_pixel_y', 'norm_distance_from_center'
        ]
        
        # Features are: [target_obj_features, ref_obj_features, comparative_features, camera_features]
        target_features = [f'target_{feat}' for feat in object_features]
        ref_features = [f'ref_{feat}' for feat in object_features]
        comparative_features = [
            'bbox_width_ratio', 'bbox_height_ratio', 'pixel_count_ratio',
            'true_size_ratio', 'pixel_distance', 'rel_x', 'rel_y'
        ]
        camera_features = [
            'cam_pos_x', 'cam_pos_y', 'cam_pos_z',
            'rot_00', 'rot_01', 'rot_02', 'rot_10', 'rot_11', 'rot_12',
            'fx', 'fy', 'cx', 'cy'
        ]
        
        return target_features + ref_features + comparative_features + camera_features


class WorldCoordinateDataset(Dataset):
    def __init__(self, features, relative_depth_targets, world_coord_targets):
        self.features = torch.FloatTensor(features)
        self.relative_depth_targets = torch.FloatTensor(relative_depth_targets)
        self.world_coord_targets = torch.FloatTensor(world_coord_targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (self.features[idx], 
                self.relative_depth_targets[idx], 
                self.world_coord_targets[idx])


class MultiTaskDepthCoordinateEstimator(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.4):
        super().__init__()
        
        # Shared feature extractor
        self.shared_features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
        )
        
        # Relative depth prediction head
        self.depth_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
            nn.ReLU()  # Ensure positive relative depths
        )
        
        # World coordinate prediction head
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 3)  # x, y, z world coordinates
        )
    
    def forward(self, x):
        shared = self.shared_features(x)
        relative_depth = self.depth_head(shared).squeeze(-1)
        world_coords = self.coord_head(shared)
        return relative_depth, world_coords


class GeometricConsistencyEstimator(nn.Module):
    """Alternative approach using geometric consistency between relative depth and world coordinates."""
    
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.4):
        super().__init__()
        
        # First predict relative depth
        self.depth_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim // 2, 1),
            nn.ReLU()
        )
        
        # Then use geometric reasoning for world coordinates
        # This network takes: [features, predicted_relative_depth, camera_params]
        coord_input_dim = input_dim + 1  # +1 for predicted relative depth
        self.coord_network = nn.Sequential(
            nn.Linear(coord_input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim // 2, 3)  # x, y, z world coordinates
        )
    
    def forward(self, x):
        # First predict relative depth
        relative_depth = self.depth_network(x).squeeze(-1)
        
        # Then predict world coordinates using the predicted relative depth
        coord_input = torch.cat([x, relative_depth.unsqueeze(1)], dim=1)
        world_coords = self.coord_network(coord_input)
        
        return relative_depth, world_coords


def train_world_coordinate_model(train_df, test_df, feature_names, epochs=100, lr=0.001, model_type="multitask"):
    """Train the world coordinate prediction model."""
    X_train = train_df[feature_names].values
    y_depth_train = train_df['relative_depth'].values
    y_coord_train = train_df[['world_x', 'world_y', 'world_z']].values
    
    X_test = test_df[feature_names].values
    y_depth_test = test_df['relative_depth'].values
    y_coord_test = test_df[['world_x', 'world_y', 'world_z']].values
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Scale world coordinates for better training
    coord_scaler = StandardScaler()
    y_coord_train_scaled = coord_scaler.fit_transform(y_coord_train)
    y_coord_test_scaled = coord_scaler.transform(y_coord_test)
    
    # Create datasets
    train_dataset = WorldCoordinateDataset(X_train_scaled, y_depth_train, y_coord_train_scaled)
    test_dataset = WorldCoordinateDataset(X_test_scaled, y_depth_test, y_coord_test_scaled)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    if model_type == "multitask":
        model = MultiTaskDepthCoordinateEstimator(input_dim=X_train.shape[1])
    else:
        model = GeometricConsistencyEstimator(input_dim=X_train.shape[1])
    
    # Loss functions
    depth_criterion = nn.MSELoss()
    coord_criterion = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    print(f"Starting {model_type} training...")
    for epoch in range(epochs):
        # Training
        model.train()
        for batch_features, batch_depth_targets, batch_coord_targets in train_loader:
            optimizer.zero_grad()
            
            depth_pred, coord_pred = model(batch_features)
            
            # Combined loss with weights
            depth_loss = depth_criterion(depth_pred, batch_depth_targets)
            coord_loss = coord_criterion(coord_pred, batch_coord_targets)
            
            # Weight the losses (you can adjust these)
            total_loss = depth_loss + 0.5 * coord_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Validation
        model.eval()
        val_depth_loss = 0
        val_coord_loss = 0
        with torch.no_grad():
            for batch_features, batch_depth_targets, batch_coord_targets in test_loader:
                depth_pred, coord_pred = model(batch_features)
                val_depth_loss += depth_criterion(depth_pred, batch_depth_targets).item()
                val_coord_loss += coord_criterion(coord_pred, batch_coord_targets).item()
        
        avg_val_depth_loss = val_depth_loss / len(test_loader)
        avg_val_coord_loss = val_coord_loss / len(test_loader)
        total_val_loss = avg_val_depth_loss + 0.5 * avg_val_coord_loss
        
        scheduler.step(total_val_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Depth Loss: {avg_val_depth_loss:.4f}, Coord Loss: {avg_val_coord_loss:.4f}")

    print("Training finished.")
    return model, scaler, coord_scaler


def evaluate_world_coordinate_model(model, test_df, feature_names, scaler, coord_scaler):
    """Evaluate the world coordinate prediction model."""
    model.eval()
    
    X_test = test_df[feature_names].values
    y_depth_test = test_df['relative_depth'].values
    y_coord_test = test_df[['world_x', 'world_y', 'world_z']].values
    
    X_test_scaled = scaler.transform(X_test)
    y_coord_test_scaled = coord_scaler.transform(y_coord_test)
    
    with torch.no_grad():
        depth_pred, coord_pred_scaled = model(torch.FloatTensor(X_test_scaled))
        depth_pred = depth_pred.numpy()
        coord_pred = coord_scaler.inverse_transform(coord_pred_scaled.numpy())
    
    # Evaluate relative depth
    depth_mae = mean_absolute_error(y_depth_test, depth_pred)
    depth_r2 = r2_score(y_depth_test, depth_pred)
    
    # Evaluate world coordinates
    coord_mae = mean_absolute_error(y_coord_test, coord_pred)
    coord_r2 = r2_score(y_coord_test, coord_pred)
    
    # Per-axis evaluation
    axis_names = ['X', 'Y', 'Z']
    for i, axis in enumerate(axis_names):
        axis_mae = mean_absolute_error(y_coord_test[:, i], coord_pred[:, i])
        axis_r2 = r2_score(y_coord_test[:, i], coord_pred[:, i])
        print(f"{axis}-axis MAE: {axis_mae:.4f}, R²: {axis_r2:.4f}")
    
    print(f"\nOverall Results:")
    print(f"Relative Depth - MAE: {depth_mae:.4f}, R²: {depth_r2:.4f}")
    print(f"World Coordinates - MAE: {coord_mae:.4f}, R²: {coord_r2:.4f}")
    
    return depth_pred, coord_pred


def visualize_results(test_df, depth_pred, coord_pred):
    """Create visualizations of the results."""
    # Plot 1: Relative depth prediction
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(test_df['relative_depth'], depth_pred, alpha=0.3)
    plt.plot([test_df['relative_depth'].min(), test_df['relative_depth'].max()], 
             [test_df['relative_depth'].min(), test_df['relative_depth'].max()], 
             '--', color='red', label='Perfect Prediction')
    plt.xlabel("True Relative Depth")
    plt.ylabel("Predicted Relative Depth")
    plt.title("Relative Depth Prediction")
    plt.grid(True)
    plt.legend()
    
    # Plot 2: World coordinate prediction (3D scatter)
    ax = plt.subplot(1, 3, 2, projection='3d')
    true_coords = test_df[['world_x', 'world_y', 'world_z']].values
    ax.scatter(true_coords[:, 0], true_coords[:, 1], true_coords[:, 2], 
               c='blue', alpha=0.3, label='True', s=20)
    ax.scatter(coord_pred[:, 0], coord_pred[:, 1], coord_pred[:, 2], 
               c='red', alpha=0.3, label='Predicted', s=20)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('World Coordinates')
    ax.legend()
    
    # Plot 3: Coordinate prediction error distribution
    plt.subplot(1, 3, 3)
    coord_errors = np.linalg.norm(true_coords - coord_pred, axis=1)
    plt.hist(coord_errors, bins=30, alpha=0.7)
    plt.xlabel("3D Coordinate Error (Euclidean Distance)")
    plt.ylabel("Frequency")
    plt.title("Coordinate Prediction Error Distribution")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("world_coordinate_evaluation.png", dpi=150, bbox_inches='tight')
    print("Saved evaluation plots to 'world_coordinate_evaluation.png'")


if __name__ == "__main__":
    # 1. Build the dataset from your metadata files
    builder = WorldCoordinateDatasetBuilder(output_base_dir="final_dataset")
    
    features, relative_depth_targets, world_coord_targets = builder.extract_features_from_metadata()
    
    if len(features) > 0:
        # Create the full dataframe
        df = pd.DataFrame(features, columns=builder.get_feature_names())
        df['relative_depth'] = relative_depth_targets
        df['world_x'] = world_coord_targets[:, 0]
        df['world_y'] = world_coord_targets[:, 1]
        df['world_z'] = world_coord_targets[:, 2]
        
        # 2. Split the data into training and testing sets
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        print(f"\nDataset created. Training on {len(train_df)} samples, testing on {len(test_df)} samples.")
        
        # 3. Train the model (try both approaches)
        feature_names = builder.get_feature_names()
        
        print("\n=== Training MultiTask Model ===")
        model_mt, scaler_mt, coord_scaler_mt = train_world_coordinate_model(
            train_df, test_df, feature_names, model_type="multitask"
        )
        
        print("\n=== Training Geometric Consistency Model ===")
        model_gc, scaler_gc, coord_scaler_gc = train_world_coordinate_model(
            train_df, test_df, feature_names, model_type="geometric"
        )
        
        # 4. Evaluate both models
        print("\n=== MultiTask Model Evaluation ===")
        depth_pred_mt, coord_pred_mt = evaluate_world_coordinate_model(
            model_mt, test_df, feature_names, scaler_mt, coord_scaler_mt
        )
        
        print("\n=== Geometric Consistency Model Evaluation ===")
        depth_pred_gc, coord_pred_gc = evaluate_world_coordinate_model(
            model_gc, test_df, feature_names, scaler_gc, coord_scaler_gc
        )
        
        # 5. Save models
        torch.save({
            'model_state_dict': model_mt.state_dict(),
            'scaler': scaler_mt,
            'coord_scaler': coord_scaler_mt,
            'feature_names': feature_names,
            'model_type': 'multitask'
        }, 'world_coordinate_model_multitask.pth')
        
        torch.save({
            'model_state_dict': model_gc.state_dict(),
            'scaler': scaler_gc,
            'coord_scaler': coord_scaler_gc,
            'feature_names': feature_names,
            'model_type': 'geometric'
        }, 'world_coordinate_model_geometric.pth')
        
        print("\nModels saved.")
        
        # 6. Visualize results (using the better performing model)
        visualize_results(test_df, depth_pred_mt, coord_pred_mt)

    else:
        print("No training samples were extracted. Exiting.")