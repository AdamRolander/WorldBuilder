"""
Depth-aware PCA rotation estimation for 3D objects.
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import List, Dict


def estimate_pca_rotations(
    segments: List[Dict],
    depth_map: np.ndarray,
    img_width: int,
    img_height: int,
    fov_deg: float = 60.0,
    downsample: bool = True
) -> List[Dict]:
    """
    Estimate horizontal orientations using depth-aware PCA.
    
    This method:
    1. Unprojects 2D mask points to 3D using depth map
    2. Runs PCA on horizontal (X-Z) plane only
    3. Returns PC1 as primary orientation vector
    
    Args:
        segments: List of segment dictionaries with masks
        depth_map: Normalized depth map from MiDaS
        img_width: Image width in pixels
        img_height: Image height in pixels
        fov_deg: Camera field of view in degrees
        downsample: Whether to downsample points for efficiency
        
    Returns:
        Updated segments with rotation_y_deg and rotation_confidence added
    """
    print("\n" + "="*80)
    print("PCA-BASED ROTATION ESTIMATION (Depth-Aware)")
    print("="*80 + "\n")
    
    for seg in segments:
        # Get 2D pixel coordinates where mask is True
        coords_2d = np.argwhere(seg['mask'])  # Shape: (N, 2) -> (row, col)
        
        if len(coords_2d) < 10:
            seg['rotation_y_deg'] = 0.0
            seg['rotation_confidence'] = 0.0
            seg['pca_vector'] = np.array([0.0, 0.0, 1.0])
            print(f"{seg['label']} (ID: {seg['id']}): Too few points, skipping")
            continue
        
        # Downsample points to reduce noise
        if downsample and len(coords_2d) > 500:
            step = max(1, len(coords_2d) // 500)
            coords_2d = coords_2d[::step]
        
        # Get depth values at these pixel locations
        depth_values = depth_map[coords_2d[:, 0], coords_2d[:, 1]]
        
        # Unproject to 3D world coordinates
        points_3d = unproject_to_3d(coords_2d, depth_values, img_width, img_height, fov_deg)
        
        # Extract horizontal (X, Z) coordinates only - ignore vertical component
        points_horizontal = points_3d[:, [0, 2]]  # Only X and Z columns
        
        # Run PCA on purely horizontal data
        pca_2d = PCA(n_components=2)
        pca_2d.fit(points_horizontal)
        
        # Get principal component in horizontal plane
        pc1_horizontal = pca_2d.components_[0]  # 2D vector: [X, Z]
        confidence = pca_2d.explained_variance_ratio_[0]
        
        # Convert to 3D format: [X, 0, Z] with Y=0 (horizontal plane)
        pca_vector = np.array([pc1_horizontal[0], 0.0, pc1_horizontal[1]])
        
        # Calculate rotation angle around Y axis
        # Angle from forward axis (-Z) in XZ plane
        # atan2(X, -Z) gives angle where 0° = facing camera, 90° = facing right
        rotation_rad = np.arctan2(pca_vector[0], -pca_vector[2])
        rotation_deg = np.degrees(rotation_rad)
        
        # Normalize to [0, 360)
        if rotation_deg < 0:
            rotation_deg += 360
        
        # Add to segment
        seg['rotation_y_deg'] = float(rotation_deg)
        seg['rotation_confidence'] = float(confidence)
        seg['pca_vector'] = pca_vector  # Store for debugging
        
        print(f"{seg['label']} (ID: {seg['id']})")
        print(f"  PCA vector: [{pca_vector[0]:6.3f}, {pca_vector[1]:6.3f}, {pca_vector[2]:6.3f}]")
        print(f"  Rotation: {rotation_deg:.1f}° (confidence: {confidence:.3f})")
    
    print(f"\n{'='*80}\n")
    
    return segments


def unproject_to_3d(
    pixel_coords: np.ndarray,
    depth_values: np.ndarray,
    img_width: int,
    img_height: int,
    fov_deg: float
) -> np.ndarray:
    """
    Convert 2D pixel coordinates + depth to 3D world coordinates.
    
    Args:
        pixel_coords: Nx2 array of (row, col) pixel coordinates
        depth_values: N array of normalized depth values [0, 1]
        img_width: Image width in pixels
        img_height: Image height in pixels
        fov_deg: Camera field of view in degrees
    
    Returns:
        Nx3 array of (X, Y, Z) world coordinates in meters
    """
    # Camera intrinsics (simple pinhole model)
    focal_length = img_width / (2 * np.tan(np.radians(fov_deg) / 2))
    cx = img_width / 2
    cy = img_height / 2
    
    # Extract pixel coordinates
    rows = pixel_coords[:, 0]  # y in image
    cols = pixel_coords[:, 1]  # x in image
    
    # Convert normalized depth [0,1] to approximate meters
    # Assuming depth=0 means close (0.5m), depth=1 means far (10m)
    Z = 0.5 + depth_values * 9.5
    
    # Unproject to 3D
    X = (cols - cx) * Z / focal_length
    Y = (rows - cy) * Z / focal_length
    
    # Stack into Nx3 array
    points_3d = np.column_stack([X, Y, Z])
    
    return points_3d