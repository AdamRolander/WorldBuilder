"""
Geometric calculations for 3D unprojection.
"""

import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from typing import Dict, Tuple

## Pinhole camera model unprojection function ##
def unproject_to_3d(
    pixel_x: float,
    pixel_y: float,
    depth: float,
    img_width: int,
    img_height: int,
    fov_deg: float,
    camera_position: np.ndarray = None,
    camera_rotation_deg: np.ndarray = None
) -> np.ndarray:
    """
    Unproject 2D pixel coordinates to 3D camera coordinates.
    Uses simple pinhole camera model.
    
    Args:
        pixel_x: Pixel x coordinate
        pixel_y: Pixel y coordinate  
        depth: Depth in meters
        img_width: Image width in pixels
        img_height: Image height in pixels
        fov_deg: Field of view in degrees
        camera_position: Unused (kept for compatibility)
        camera_rotation_deg: Unused (kept for compatibility)
        
    Returns:
        3D camera coordinates as [x, y, z] array
    """
    # Calculate focal length
    fov_rad = fov_deg * (math.pi / 180.0)
    fx = (img_width / 2.0) / math.tan(fov_rad / 2.0)
    fy = fx  # Assume square pixels
    
    # Principal point (image center)
    cx = img_width / 2.0
    cy = img_height / 2.0
    
    # Unproject to 3D camera coordinates
    x_cam = (pixel_x - cx) * depth / fx
    y_cam = -(pixel_y - cy) * depth / fy  # Y inverted (pixel to camera space)
    z_cam = -depth                         # Camera looks down -Z axis
    
    return np.array([x_cam, y_cam, z_cam])

#### Rotated and translated unprojection function ####
# def unproject_to_3d(
#     pixel_x: float,
#     pixel_y: float,
#     depth: float,
#     img_width: int,
#     img_height: int,
#     fov_deg: float,
#     camera_position: np.ndarray = None,
#     camera_rotation_deg: np.ndarray = None
# ) -> np.ndarray:
#     """
#     Unproject 2D pixel coordinates to 3D world coordinates.
    
#     Args:
#         pixel_x: Pixel x coordinate
#         pixel_y: Pixel y coordinate  
#         depth: Depth in meters
#         img_width: Image width in pixels
#         img_height: Image height in pixels
#         fov_deg: Field of view in degrees
#         camera_position: Camera position [x, y, z] (default: origin)
#         camera_rotation_deg: Camera rotation [rx, ry, rz] in degrees (default: no rotation)
        
#     Returns:
#         3D world coordinates as [x, y, z] array
#     """
#     if camera_position is None:
#         camera_position = np.array([0.0, 0.0, 0.0])
#     if camera_rotation_deg is None:
#         camera_rotation_deg = np.array([0.0, 0.0, 0.0])
    
#     # Calculate camera intrinsics
#     fov_rad = fov_deg * (math.pi / 180.0)
#     focal_length = (img_width / 2.0) / math.tan(fov_rad / 2.0)
#     cx = img_width / 2.0
#     cy = img_height / 2.0
    
#     # Intrinsic matrix inverse
#     K_inv = np.linalg.inv(np.array([
#         [focal_length, 0, cx],
#         [0, focal_length, cy],
#         [0, 0, 1]
#     ]))
    
#     # Unproject to camera coordinates
#     pixel_coords = np.array([pixel_x, pixel_y, 1])
#     camera_coords = K_inv @ pixel_coords * depth
    
#     # Transform to world coordinates
#     rotation = R.from_euler('xyz', camera_rotation_deg, degrees=True)
#     rotation_matrix = rotation.as_matrix()
#     world_coords = rotation_matrix @ camera_coords + camera_position
    
#     return world_coords


def calculate_analytical_depth(
    object_size_meters: float,
    object_size_pixels: float,
    img_width: int,
    fov_deg: float
) -> float:
    """
    Calculate depth using pinhole camera model.
    
    Args:
        object_size_meters: Real-world size of object (meters)
        object_size_pixels: Object size in image (pixels)
        img_width: Image width (pixels)
        fov_deg: Field of view (degrees)
        
    Returns:
        Estimated depth in meters
    """
    fov_rad = fov_deg * (math.pi / 180.0)
    focal_length = (img_width / 2.0) / math.tan(fov_rad / 2.0)
    
    if object_size_pixels <= 0:
        return 3.0  # Default fallback
    
    depth = (object_size_meters * focal_length) / object_size_pixels
    return depth