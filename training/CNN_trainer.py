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

# --- Helper function for unprojection feature calculation ---
def calculate_unprojection(row, img_width=300, img_height=300):
    """Calculates an initial 3D world coordinate guess using the unprojection formula."""
    # Use dist_to_ref as a robust proxy for absolute depth in meters
    depth = row['dist_to_ref']
    if depth <= 0: return 0, 0, 0 # Safety check

    fov_rad = row['field_of_view'] * (math.pi / 180.0)
    focal_length = (img_width / 2.0) / math.tan(fov_rad / 2.0)
    cx = img_width / 2.0
    cy = img_height / 2.0
    K_inv = np.linalg.inv(np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]]))

    px = row['target_bbox_center_x'] * img_width
    py = row['target_bbox_center_y'] * img_height
    
    pixel_coords = np.array([px, py, 1])
    camera_coords = K_inv @ pixel_coords * depth
    
    rotation = R.from_euler('xyz', [row['agent_rot_x'], row['agent_rot_y'], row['agent_rot_z']], degrees=True)
    rotation_matrix = rotation.as_matrix()
    translation_vector = np.array([row['agent_pos_x'], row['agent_pos_y'], row['agent_pos_z']])
    
    world_coords = rotation_matrix @ camera_coords + translation_vector
    return world_coords[0], world_coords[1], world_coords[2]


# --- 1. The Dataset for loading images and geometric data ---
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
            print(f"Warning: Image not found at {target_image_path}. Using a placeholder.")
            target_tensor = torch.zeros((3, 224, 224)) # Placeholder
        
        geometrics = self.geometric_features[idx]
        
        # Apply log transform to depth target
        depth_target = torch.FloatTensor([np.log1p(row['relative_depth'])])
        coord_target = torch.FloatTensor([row['world_x'], row['world_y'], row['world_z']])
        
        return target_tensor, geometrics, depth_target.squeeze(), coord_target

# --- 2. The Hybrid CNN + MLP Model ---
class AI2ThorHybridEstimator(nn.Module):
    def __init__(self, geometric_dim, visual_embed_dim=256, hidden_dim=512, dropout_rate=0.4):
        super().__init__()
        
        # Visual Branch (CNN)
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        for param in mobilenet.parameters():
            param.requires_grad = False
        
        num_mobilenet_features = mobilenet.classifier[1].in_features
        mobilenet.classifier = nn.Sequential(
            nn.Linear(num_mobilenet_features, visual_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.visual_branch = mobilenet
        
        # Combined MLP Branch
        combined_dim = visual_embed_dim + geometric_dim
        self.shared_encoder = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout_rate)
        )
        
        self.depth_branch = nn.Linear(hidden_dim // 2, 1)
        self.coord_branch = nn.Linear(hidden_dim // 2, 3)

    def forward(self, image_input, geometric_input):
        visual_features = self.visual_branch(image_input)
        combined_features = torch.cat([visual_features, geometric_input], dim=1)
        shared_output = self.shared_encoder(combined_features)
        relative_depth = self.depth_branch(shared_output).squeeze(-1)
        world_coords = self.coord_branch(shared_output)
        return relative_depth, world_coords

# --- 3. The Trainer Class (updated for hybrid inputs) ---
class AI2ThorTrainer:
    def __init__(self, model, train_loader, val_loader, coord_scaler, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.coord_scaler = coord_scaler
        self.device = device
        self.depth_criterion = nn.MSELoss()
        self.coord_criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5)
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for image_batch, geo_batch, depth_targets, coord_targets in self.train_loader:
            image_batch = image_batch.to(self.device)
            geo_batch = geo_batch.to(self.device)
            depth_targets = depth_targets.to(self.device)
            coord_targets = coord_targets.to(self.device)
            
            self.optimizer.zero_grad()
            depth_pred, coord_pred = self.model(image_batch, geo_batch)
            
            depth_loss = self.depth_criterion(depth_pred, depth_targets)
            coord_loss = self.coord_criterion(coord_pred, coord_targets)
            
            batch_loss = 0.3 * depth_loss + 0.7 * coord_loss
            batch_loss.backward()
            self.optimizer.step()
            total_loss += batch_loss.item()
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_val_loss = 0
        # (For brevity, the detailed validation logic for calculating R2/MAE is omitted,
        # but it would loop through val_loader similar to train_epoch and compute metrics)
        with torch.no_grad():
            for image_batch, geo_batch, depth_targets, coord_targets in self.val_loader:
                image_batch = image_batch.to(self.device)
                geo_batch = geo_batch.to(self.device)
                depth_targets = depth_targets.to(self.device)
                coord_targets = coord_targets.to(self.device)
                depth_pred, coord_pred = self.model(image_batch, geo_batch)
                depth_loss = self.depth_criterion(depth_pred, depth_targets)
                coord_loss = self.coord_criterion(coord_pred, coord_targets)
                batch_loss = 0.3 * depth_loss + 0.7 * coord_loss
                total_val_loss += batch_loss.item()
        return total_val_loss / len(self.val_loader)
        
    def train(self, epochs=100, early_stopping_patience=15):
        print("Starting hybrid CNN model training...")
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), 'best_hybrid_model.pth')
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        print("Training complete!")
        self.model.load_state_dict(torch.load('best_hybrid_model.pth'))
        return self.model


# --- 4. Main Training Orchestrator ---
def main(dataset_path="data/master.csv"):
    df = pd.read_csv(dataset_path)
    
    # Feature Engineering: Add unprojection guess
    print("Calculating unprojection features...")
    proj_coords = df.apply(calculate_unprojection, axis=1, result_type='expand')
    df[['proj_x', 'proj_y', 'proj_z']] = proj_coords

    # Define geometric feature columns (all columns except targets and identifiers)
    geo_feature_columns = [col for col in df.columns if col not in 
                           ['target_image_path', 'ref_image_path', 'target_object_type', 
                            'ref_object_type', 'relative_depth', 'world_x', 'world_y', 'world_z']]
    
    # Split data first to prevent data leakage
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Scale geometric features (fit on train, transform both)
    geo_scaler = StandardScaler()
    X_train_geo_scaled = geo_scaler.fit_transform(train_df[geo_feature_columns])
    X_val_geo_scaled = geo_scaler.transform(val_df[geo_feature_columns])

    # Scale coordinate targets
    coord_scaler = StandardScaler()
    y_train_coords_scaled = coord_scaler.fit_transform(train_df[['world_x', 'world_y', 'world_z']])
    y_val_coords_scaled = coord_scaler.transform(val_df[['world_x', 'world_y', 'world_z']])
    
    # Create Datasets
    train_dataset = AI2ThorHybridDataset(train_df, X_train_geo_scaled)
    val_dataset = AI2ThorHybridDataset(val_df, X_val_geo_scaled)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Initialize and train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AI2ThorHybridEstimator(geometric_dim=len(geo_feature_columns))
    trainer = AI2ThorTrainer(model, train_loader, val_loader, coord_scaler, device)
    
    # Note: The detailed plotting and final metric evaluation from your
    # previous script would be added here to complete the pipeline.
    # For brevity, this example focuses on the core training loop.
    trained_model = trainer.train()

if __name__ == "__main__":
    main()