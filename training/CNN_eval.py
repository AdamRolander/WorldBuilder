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
import math
from scipy.spatial.transform import Rotation as R
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms
import os
from tqdm import tqdm

# --- 1. Redefine the Core Classes (Must match the training script exactly) ---

class AI2ThorHybridDataset(Dataset):
    def __init__(self, df, geometric_features_scaled, image_size=224):
        self.df = df.reset_index(drop=True)
        self.geometric_features = torch.FloatTensor(geometric_features_scaled)
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        target_image_path = row['target_image_path']
        try:
            target_image = Image.open(target_image_path).convert("RGB")
            target_tensor = self.transform(target_image)
        except FileNotFoundError:
            target_tensor = torch.zeros((3, 224, 224))
        
        geometrics = self.geometric_features[idx]
        depth_target = torch.FloatTensor([np.log1p(row['relative_depth'])])
        coord_target = torch.FloatTensor([row['world_x'], row['world_y'], row['world_z']])
        return target_tensor, geometrics, depth_target.squeeze(), coord_target

class AI2ThorHybridEstimator(nn.Module):
    def __init__(self, geometric_dim, visual_embed_dim=256, hidden_dim=512, dropout_rate=0.4):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        for param in mobilenet.parameters():
            param.requires_grad = False
        num_mobilenet_features = mobilenet.classifier[1].in_features
        mobilenet.classifier = nn.Sequential(nn.Linear(num_mobilenet_features, visual_embed_dim), nn.ReLU(), nn.Dropout(dropout_rate))
        self.visual_branch = mobilenet
        combined_dim = visual_embed_dim + geometric_dim
        self.shared_encoder = nn.Sequential(nn.Linear(combined_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim), nn.Dropout(dropout_rate), nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.BatchNorm1d(hidden_dim // 2), nn.Dropout(dropout_rate))
        self.depth_branch = nn.Linear(hidden_dim // 2, 1)
        self.coord_branch = nn.Linear(hidden_dim // 2, 3)

    def forward(self, image_input, geometric_input):
        visual_features = self.visual_branch(image_input)
        combined_features = torch.cat([visual_features, geometric_input], dim=1)
        shared_output = self.shared_encoder(combined_features)
        relative_depth = self.depth_branch(shared_output).squeeze(-1)
        world_coords = self.coord_branch(shared_output)
        return relative_depth, world_coords

def calculate_unprojection(row, img_width=300, img_height=300):
    # This helper function is needed to recreate the features
    depth = row['dist_to_ref']
    if depth <= 0: return 0, 0, 0
    fov_rad = row['field_of_view'] * (math.pi / 180.0)
    focal_length = (img_width / 2.0) / math.tan(fov_rad / 2.0)
    K_inv = np.linalg.inv(np.array([
        [focal_length, 0, img_width / 2.0],
        [0, focal_length, img_height / 2.0],
        [0, 0, 1]
    ]))
    px = row['target_bbox_center_x'] * img_width
    py = row['target_bbox_center_y'] * img_height
    camera_coords = K_inv @ np.array([px, py, 1]) * depth
    rotation = R.from_euler('xyz', [row['agent_rot_x'], row['agent_rot_y'], row['agent_rot_z']], degrees=True)
    translation_vector = np.array([row['agent_pos_x'], row['agent_pos_y'], row['agent_pos_z']])
    world_coords = rotation.as_matrix() @ camera_coords + translation_vector
    return world_coords[0], world_coords[1], world_coords[2]


# --- 2. The Main Evaluation Function ---
def evaluate_model(model_path, dataset_path, batch_size=32):
    print("--- Starting Model Evaluation ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and preprocess data just like in training
    df = pd.read_csv(dataset_path)
    print("Calculating unprojection features...")
    proj_coords = df.apply(calculate_unprojection, axis=1, result_type='expand')
    df[['proj_x', 'proj_y', 'proj_z']] = proj_coords
    
    geo_feature_columns = [col for col in df.columns if col not in 
                           ['target_image_path', 'ref_image_path', 'target_object_type', 
                            'ref_object_type', 'relative_depth', 'world_x', 'world_y', 'world_z']]
    
    # Recreate the train/validation split to get the correct validation set
    # Using the same random_state is key to getting the same split as training
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Recreate scalers by fitting them on the training data part
    print("Recreating data scalers...")
    geo_scaler = StandardScaler().fit(train_df[geo_feature_columns])
    coord_scaler = StandardScaler().fit(train_df[['world_x', 'world_y', 'world_z']])
    
    # Transform the validation data
    X_val_geo_scaled = geo_scaler.transform(val_df[geo_feature_columns])
    
    # Create dataset and loader for the validation set
    val_dataset = AI2ThorHybridDataset(val_df, X_val_geo_scaled)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model and load state
    print(f"Loading model from {model_path}...")
    model = AI2ThorHybridEstimator(geometric_dim=len(geo_feature_columns))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Run inference
    print("Running inference on validation set...")
    all_depth_preds, all_depth_targets = [], []
    all_coord_preds, all_coord_targets = [], []
    with torch.no_grad():
        for image_batch, geo_batch, depth_targets, coord_targets in tqdm(val_loader, desc="Evaluating"):
            image_batch = image_batch.to(device)
            geo_batch = geo_batch.to(device)
            
            depth_pred, coord_pred_scaled = model(image_batch, geo_batch)
            
            # Un-log depth predictions
            all_depth_preds.append(np.expm1(depth_pred.cpu().numpy()))
            all_depth_targets.append(np.expm1(depth_targets.cpu().numpy()))
            
            # Un-scale coordinate predictions
            all_coord_preds.append(coord_scaler.inverse_transform(coord_pred_scaled.cpu().numpy()))
            all_coord_targets.append(coord_targets.numpy())

    # Concatenate results from all batches
    depth_preds = np.concatenate(all_depth_preds)
    depth_targets = np.concatenate(all_depth_targets)
    coord_preds = np.concatenate(all_coord_preds)
    coord_targets = np.concatenate(all_coord_targets)
    
    print("Evaluation complete.")
    return depth_preds, depth_targets, coord_preds, coord_targets

# --- 3. Plotting and Metrics Function ---
def create_evaluation_plots(depth_preds, depth_targets, coord_preds, coord_targets):
    print("\n--- Generating Plots and Final Metrics ---")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Depth prediction scatter
    axes[0].scatter(depth_targets, depth_preds, alpha=0.5, s=20)
    min_val, max_val = min(depth_targets.min(), depth_preds.min()), max(depth_targets.max(), depth_preds.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    axes[0].set_xlabel('True Relative Depth')
    axes[0].set_ylabel('Predicted Relative Depth')
    axes[0].set_title('Depth Prediction Accuracy')
    axes[0].grid(True)
    
    # 3D coordinate prediction
    ax_3d = fig.add_subplot(1, 3, 2, projection='3d')
    sample_size = min(500, len(coord_targets))
    idx_sample = np.random.choice(len(coord_targets), sample_size, replace=False)
    ax_3d.scatter(coord_targets[idx_sample, 0], coord_targets[idx_sample, 1], coord_targets[idx_sample, 2], c='blue', alpha=0.6, s=20, label='True')
    ax_3d.scatter(coord_preds[idx_sample, 0], coord_preds[idx_sample, 1], coord_preds[idx_sample, 2], c='red', alpha=0.6, s=20, label='Predicted')
    ax_3d.set_xlabel('X (m)'); ax_3d.set_ylabel('Y (m)'); ax_3d.set_zlabel('Z (m)')
    ax_3d.set_title('3D Coordinate Predictions'); ax_3d.legend()
    
    # Error distribution
    coord_errors = np.linalg.norm(coord_targets - coord_preds, axis=1)
    axes[2].hist(coord_errors, bins=50, alpha=0.7, edgecolor='black')
    mean_error = np.mean(coord_errors)
    median_error = np.median(coord_errors)
    axes[2].axvline(mean_error, color='red', linestyle='--', label=f'Mean: {mean_error:.4f}m')
    axes[2].axvline(median_error, color='green', linestyle='--', label=f'Median: {median_error:.4f}m')
    axes[2].set_xlabel('3D Coordinate Error (m)'); axes[2].set_ylabel('Frequency')
    axes[2].set_title('Prediction Error Distribution'); axes[2].legend(); axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('final_evaluation_plots.png', dpi=150)
    print("Evaluation plots saved as 'final_evaluation_plots.png'")
    
    # Print final metrics
    depth_r2 = r2_score(depth_targets, depth_preds)
    depth_mae = mean_absolute_error(depth_targets, depth_preds)
    coord_r2 = r2_score(coord_targets, coord_preds)
    coord_mae = mean_absolute_error(coord_targets, coord_preds)
    
    print(f"\n=== Final Model Performance ===")
    print(f"Relative Depth - R²: {depth_r2:.4f}, MAE: {depth_mae:.4f}")
    print(f"World Coordinates - R²: {coord_r2:.4f}, MAE: {coord_mae:.4f}m")
    print(f"Mean 3D error: {mean_error:.4f}m, Median: {median_error:.4f}m")

# --- 4. Script Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    # Make sure these paths are correct
    MODEL_PATH = "best_hybrid_model.pth"
    DATASET_PATH = "data/master.csv"
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATASET_PATH):
        print("Error: Model file or dataset CSV not found. Please check paths.")
    else:
        # Run the full evaluation pipeline
        d_preds, d_targets, c_preds, c_targets = evaluate_model(MODEL_PATH, DATASET_PATH)
        create_evaluation_plots(d_preds, d_targets, c_preds, c_targets)