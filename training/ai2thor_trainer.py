import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

class AI2ThorCoordinateDataset(Dataset):
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


class AI2ThorGeometricEstimator(nn.Module):
    """Enhanced geometric consistency model optimized for AI2-THOR data."""
    
    def __init__(self, input_dim, hidden_dim=256, dropout_rate=0.2):
        super().__init__()
        
        # Input feature attention mechanism
        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),  # More stable than sigmoid for gradients
            nn.Dropout(dropout_rate * 0.5)
        )
        
        # Shared feature processing
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate)
        )
        
        # Depth estimation branch
        self.depth_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus()  # Ensures positive depth ratios
        )
        
        # World coordinate estimation branch
        # Uses both shared features and predicted depth
        coord_input_dim = hidden_dim + 1  # +1 for depth
        self.coord_branch = nn.Sequential(
            nn.Linear(coord_input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim // 4, 3)  # x, y, z coordinates
        )
    
    def forward(self, x):
        # Apply attention weighting to input features
        attention_weights = self.feature_attention(x)
        x_weighted = x * attention_weights
        
        # Extract shared features
        shared_features = self.shared_encoder(x_weighted)
        
        # Predict relative depth
        relative_depth = self.depth_branch(shared_features).squeeze(-1)
        
        # Predict world coordinates using shared features + depth
        coord_input = torch.cat([shared_features, relative_depth.unsqueeze(1)], dim=1)
        world_coords = self.coord_branch(coord_input)
        
        return relative_depth, world_coords


class AI2ThorTrainer:
    def __init__(self, model, train_loader, val_loader, coord_scaler, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.coord_scaler = coord_scaler
        self.device = device
        
        # Loss functions
        self.depth_criterion = nn.MSELoss()
        self.coord_criterion = nn.MSELoss()
        
        # Optimizer - using AdamW for better generalization
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=0.002, 
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=8, 
            min_lr=1e-6
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
        total_depth_loss = 0
        total_coord_loss = 0
        total_loss = 0
        total_samples = 0
        
        for batch_idx, (batch_features, batch_depth_targets, batch_coord_targets) in enumerate(self.train_loader):
            batch_features = batch_features.to(self.device)
            batch_depth_targets = batch_depth_targets.to(self.device)
            batch_coord_targets = batch_coord_targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            depth_pred, coord_pred = self.model(batch_features)
            
            # Calculate losses
            depth_loss = self.depth_criterion(depth_pred, batch_depth_targets)
            coord_loss = self.coord_criterion(coord_pred, batch_coord_targets)
            
            # Weighted combination - emphasize coordinate accuracy
            total_batch_loss = 0.3 * depth_loss + 0.7 * coord_loss
            
            # Backward pass
            total_batch_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Track losses
            batch_size = batch_features.size(0)
            total_depth_loss += depth_loss.item() * batch_size
            total_coord_loss += coord_loss.item() * batch_size
            total_loss += total_batch_loss.item() * batch_size
            total_samples += batch_size
        
        return (total_depth_loss / total_samples, 
                total_coord_loss / total_samples,
                total_loss / total_samples)
    
    def validate(self):
        self.model.eval()
        total_depth_loss = 0
        total_coord_loss = 0
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
                
                depth_pred, coord_pred = self.model(batch_features)
                
                depth_loss = self.depth_criterion(depth_pred, batch_depth_targets)
                coord_loss = self.coord_criterion(coord_pred, batch_coord_targets)
                
                batch_size = batch_features.size(0)
                total_depth_loss += depth_loss.item() * batch_size
                total_coord_loss += coord_loss.item() * batch_size
                total_samples += batch_size
                
                # Collect predictions for metrics
                all_depth_preds.append(depth_pred.cpu())
                all_depth_targets.append(batch_depth_targets.cpu())
                all_coord_preds.append(coord_pred.cpu())
                all_coord_targets.append(batch_coord_targets.cpu())
        
        avg_depth_loss = total_depth_loss / total_samples
        avg_coord_loss = total_coord_loss / total_samples
        avg_total_loss = 0.3 * avg_depth_loss + 0.7 * avg_coord_loss
        
        # Calculate R² and MAE metrics
        all_depth_preds_log = torch.cat(all_depth_preds).numpy()
        all_depth_targets_log = torch.cat(all_depth_targets).numpy()

        # --- Convert predictions and targets back to original scale for metrics ---
        all_depth_preds = np.expm1(all_depth_preds_log)
        all_depth_targets = np.expm1(all_depth_targets_log)

        all_coord_preds = torch.cat(all_coord_preds).numpy()
        all_coord_targets = torch.cat(all_coord_targets).numpy()
        
        # Unscale coordinates for meaningful metrics
        coord_preds_unscaled = self.coord_scaler.inverse_transform(all_coord_preds)
        coord_targets_unscaled = self.coord_scaler.inverse_transform(all_coord_targets)
        
        # Depth metrics
        depth_r2 = r2_score(all_depth_targets, all_depth_preds)
        depth_mae = mean_absolute_error(all_depth_targets, all_depth_preds)
        
        # Coordinate metrics
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
            'total_loss': avg_total_loss,
            'depth_loss': avg_depth_loss,
            'coord_loss': avg_coord_loss,
            'depth_r2': depth_r2,
            'coord_r2': coord_r2,
            'depth_mae': depth_mae,
            'coord_mae': coord_mae,
            **axis_metrics
        }
    
    def train(self, epochs=150, early_stopping_patience=20):
        print(f"Starting AI2-THOR model training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(epochs):
            # Training phase
            train_depth_loss, train_coord_loss, train_total_loss = self.train_epoch()
            
            # Validation phase
            val_metrics = self.validate()
            val_total_loss = val_metrics['total_loss']
            
            # Update learning rate
            self.scheduler.step(val_total_loss)
            
            # Store metrics
            self.train_losses.append((train_depth_loss, train_coord_loss, train_total_loss))
            self.val_losses.append(val_total_loss)
            self.val_metrics.append(val_metrics)
            
            # Early stopping check
            if val_total_loss < self.best_val_loss:
                self.best_val_loss = val_total_loss
                self.patience_counter = 0
                
                # Save best model
                self.save_checkpoint('best_ai2thor_model.pth', epoch, val_metrics)
            else:
                self.patience_counter += 1
            
            # Progress reporting
            if epoch % 10 == 0 or epoch < 10:
                print(f"Epoch {epoch:3d} | "
                      f"Train: D={train_depth_loss:.4f} C={train_coord_loss:.4f} | "
                      f"Val: D={val_metrics['depth_loss']:.4f} C={val_metrics['coord_loss']:.4f} | "
                      f"R²: D={val_metrics['depth_r2']:.3f} C={val_metrics['coord_r2']:.3f} | "
                      f"MAE: C={val_metrics['coord_mae']:.3f}m")
            
            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
        
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        return self.model
    
    def save_checkpoint(self, filepath, epoch, metrics):
        """Save model checkpoint with metadata."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': metrics
        }, filepath)

def calculate_unprojection(row, img_width=300, img_height=300):
    """
    Calculates an initial 3D world coordinate guess using the unprojection formula.
    """
    # --- 1. Get Camera Intrinsics (from FoV) ---
    fov_rad = row['field_of_view'] * (math.pi / 180.0)
    focal_length = (img_width / 2.0) / math.tan(fov_rad / 2.0)
    cx = img_width / 2.0
    cy = img_height / 2.0
    K_inv = np.linalg.inv(np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ]))

    # --- 2. Get 2D Pixel Coordinates and Depth ---
    # Un-normalize the bounding box center to get pixel coordinates
    px = row['target_bbox_center_x'] * img_width
    py = row['target_bbox_center_y'] * img_height
    depth = row['dist_to_ref'] # Use the mean depth from the depth map
    
    # --- 3. Unproject from 2D to 3D (in Camera's coordinate system) ---
    pixel_coords = np.array([px, py, 1])
    camera_coords = K_inv @ pixel_coords * depth
    
    # --- 4. Get Camera Extrinsics (Position and Rotation) ---
    # Create a 3D rotation matrix from the agent's Euler angles
    # AI2-THOR uses (pitch, yaw, roll) which corresponds to (X, Y, Z) rotation order
    rotation = R.from_euler('xyz', [row['agent_rot_x'], row['agent_rot_y'], row['agent_rot_z']], degrees=True)
    rotation_matrix = rotation.as_matrix()
    
    # Agent's position is the translation vector
    translation_vector = np.array([row['agent_pos_x'], row['agent_pos_y'], row['agent_pos_z']])

    # --- 5. Transform from Camera Coordinates to World Coordinates ---
    # P_world = R * P_camera + T
    world_coords = rotation_matrix @ camera_coords + translation_vector
    
    return world_coords[0], world_coords[1], world_coords[2]

def train_ai2thor_model(dataset_path="ai2thor_coordinate_dataset.csv", 
                       test_size=0.2, 
                       batch_size=32, 
                       epochs=150):
    """Main training function for AI2-THOR coordinate estimation model."""
    
    print("Loading AI2-THOR dataset...")
    df = pd.read_csv(dataset_path)
    
    if len(df) == 0:
        print("Dataset is empty!")
        return None
    
    print(f"Dataset loaded: {len(df)} samples")
    
    # --- NEW: APPLY THE UNPROJECTION FORMULA TO CREATE NEW FEATURES ---
    print("Calculating initial guess coordinates via unprojection...")
    proj_coords = df.apply(calculate_unprojection, axis=1, result_type='expand')
    df[['proj_x', 'proj_y', 'proj_z']] = proj_coords
    print("Unprojection features created.")
    
    # Prepare features and targets
    # The feature_columns logic will automatically pick up proj_x, proj_y, proj_z
    feature_columns = [col for col in df.columns if col not in 
                      ['relative_depth', 'world_x', 'world_y', 'world_z'] + 
                      ['scene_name', 'scene_idx', 'pose_idx', 'target_object_type', 
                       'ref_object_type', 'target_object_id', 'ref_object_id']]
    
    X = df[feature_columns].values
    y_depth = np.log1p(df['relative_depth'].values) # Keep the log transform for depth
    y_coords = df[['world_x', 'world_y', 'world_z']].values
    
    print(f"Feature dimensions (including unprojection guess): {X.shape[1]}")
    
    # Train-validation split
    X_train, X_val, y_depth_train, y_depth_val, y_coords_train, y_coords_val = train_test_split(
        X, y_depth, y_coords, test_size=test_size, random_state=42, stratify=None
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
    
    # Create datasets and loaders
    train_dataset = AI2ThorCoordinateDataset(X_train_scaled, y_depth_train, y_coords_train_scaled)
    val_dataset = AI2ThorCoordinateDataset(X_val_scaled, y_depth_val, y_coords_val_scaled)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = AI2ThorGeometricEstimator(
        input_dim=X_train.shape[1],
        hidden_dim=512, # Keep the larger model
        dropout_rate=0.2
    )
    
    # Initialize trainer
    trainer = AI2ThorTrainer(model, train_loader, val_loader, coord_scaler, device)
    
    # Train the model
    trained_model = trainer.train(epochs=epochs)
    
    # Save final model and scalers
    final_checkpoint = {
        'model_state_dict': trained_model.state_dict(),
        'feature_scaler': feature_scaler,
        'coord_scaler': coord_scaler,
        'feature_names': feature_columns,
        'model_type': 'ai2thor_geometric_estimator',
        'training_metrics': trainer.val_metrics[-1] if trainer.val_metrics else None
    }
    
    torch.save(final_checkpoint, 'ai2thor_world_coordinate_model.pth')
    print("Final model saved as 'ai2thor_world_coordinate_model.pth'")
    
    # Generate training plots
    create_training_plots(trainer, coord_scaler, val_dataset, device)
    
    return trained_model, feature_scaler, coord_scaler, feature_columns


def create_training_plots(trainer, coord_scaler, val_dataset, device):
    """Create comprehensive training and evaluation plots."""
    
    # Training curves
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    epochs = range(len(trainer.train_losses))
    
    # Loss curves
    train_depth = [loss[0] for loss in trainer.train_losses]
    train_coord = [loss[1] for loss in trainer.train_losses]
    val_total = trainer.val_losses
    
    axes[0,0].plot(epochs, train_depth, label='Train Depth Loss', alpha=0.8)
    axes[0,0].plot(epochs, [m['depth_loss'] for m in trainer.val_metrics], label='Val Depth Loss', alpha=0.8)
    axes[0,0].set_title('Depth Loss Over Time')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('MSE Loss')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    axes[0,1].plot(epochs, train_coord, label='Train Coord Loss', alpha=0.8)
    axes[0,1].plot(epochs, [m['coord_loss'] for m in trainer.val_metrics], label='Val Coord Loss', alpha=0.8)
    axes[0,1].set_title('Coordinate Loss Over Time')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('MSE Loss')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # R² scores
    axes[0,2].plot(epochs, [m['depth_r2'] for m in trainer.val_metrics], label='Depth R²', alpha=0.8)
    axes[0,2].plot(epochs, [m['coord_r2'] for m in trainer.val_metrics], label='Coordinate R²', alpha=0.8)
    axes[0,2].set_title('R² Score Over Time')
    axes[0,2].set_xlabel('Epoch')
    axes[0,2].set_ylabel('R² Score')
    axes[0,2].legend()
    axes[0,2].grid(True)
    
    # Get final predictions for scatter plots
    trainer.model.eval()
    with torch.no_grad():
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        all_depth_preds = []
        all_depth_targets = []
        all_coord_preds = []
        all_coord_targets = []
        
        for batch_features, batch_depth_targets, batch_coord_targets in val_loader:
            batch_features = batch_features.to(device)
            depth_pred, coord_pred = trainer.model(batch_features)
            
            all_depth_preds.append(depth_pred.cpu())
            all_depth_targets.append(batch_depth_targets.cpu())
            all_coord_preds.append(coord_pred.cpu())
            all_coord_targets.append(batch_coord_targets.cpu())
        
        depth_preds = torch.cat(all_depth_preds).numpy()
        depth_targets = torch.cat(all_depth_targets).numpy()
        coord_preds = torch.cat(all_coord_preds).numpy()
        coord_targets = torch.cat(all_coord_targets).numpy()
    
    # Unscale coordinates
    coord_preds_unscaled = coord_scaler.inverse_transform(coord_preds)
    coord_targets_unscaled = coord_scaler.inverse_transform(coord_targets)
    
    # Depth prediction scatter
    axes[1,0].scatter(depth_targets, depth_preds, alpha=0.5, s=20)
    min_depth, max_depth = depth_targets.min(), depth_targets.max()
    axes[1,0].plot([min_depth, max_depth], [min_depth, max_depth], 'r--', alpha=0.8)
    axes[1,0].set_xlabel('True Relative Depth')
    axes[1,0].set_ylabel('Predicted Relative Depth')
    axes[1,0].set_title('Depth Prediction Accuracy')
    axes[1,0].grid(True)
    
    # 3D coordinate prediction
    ax_3d = fig.add_subplot(2, 3, 5, projection='3d')
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
    
    # Error distribution
    coord_errors = np.linalg.norm(coord_targets_unscaled - coord_preds_unscaled, axis=1)
    axes[1,2].hist(coord_errors, bins=50, alpha=0.7, edgecolor='black')
    axes[1,2].axvline(np.mean(coord_errors), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(coord_errors):.3f}m')
    axes[1,2].axvline(np.median(coord_errors), color='green', linestyle='--', 
                      label=f'Median: {np.median(coord_errors):.3f}m')
    axes[1,2].set_xlabel('3D Coordinate Error (m)')
    axes[1,2].set_ylabel('Frequency')
    axes[1,2].set_title('Prediction Error Distribution')
    axes[1,2].legend()
    axes[1,2].grid(True)
    
    plt.tight_layout()
    plt.savefig('ai2thor_training_evaluation.png', dpi=150, bbox_inches='tight')
    print("Training plots saved as 'ai2thor_training_evaluation.png'")
    
    # Print final metrics
    final_metrics = trainer.val_metrics[-1]
    print(f"\n=== Final Model Performance ===")
    print(f"Relative Depth - R²: {final_metrics['depth_r2']:.4f}, MAE: {final_metrics['depth_mae']:.4f}")
    print(f"World Coordinates - R²: {final_metrics['coord_r2']:.4f}, MAE: {final_metrics['coord_mae']:.4f}m")
    print(f"Per-axis MAE: X={final_metrics['X_mae']:.4f}m, Y={final_metrics['Y_mae']:.4f}m, Z={final_metrics['Z_mae']:.4f}m")
    print(f"Mean 3D error: {np.mean(coord_errors):.4f}m, Median: {np.median(coord_errors):.4f}m")


if __name__ == "__main__":
    # Train the model
    model, feature_scaler, coord_scaler, feature_names = train_ai2thor_model(
        dataset_path="ai2thor_coordinate_dataset.csv",
        test_size=0.2,
        batch_size=32,
        epochs=150
    )
    
    if model is not None:
        print("\nAI2-THOR coordinate estimation model training completed successfully!")
        print("Model saved and ready for inference on real-world images.")