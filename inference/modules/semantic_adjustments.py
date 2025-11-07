"""
Semantic adjustments for 3D pose estimation.
"""

import numpy as np
from typing import Dict, List, Optional


# Objects that typically sit on the floor
FLOOR_OBJECTS = {
    'couch', 'sofa', 'chair', 'table', 'cabinet', 'potted plant', 
    'plant', 'lamp', 'light', 'refrigerator', 'bed', 'desk', 'bookshelf'
}
#tv

# Objects that typically hang or are elevated
ELEVATED_OBJECTS = {
    'ceiling', 'light-ceiling', 'chandelier', 'ceiling fan'
}


def should_use_bottom_center(label: str) -> bool:
    """
    Determine if we should unproject from bottom center vs geometric center.
    
    Args:
        label: Object label
        
    Returns:
        True if should use bottom center (for floor objects)
    """
    label_lower = label.lower()
    return any(floor_obj in label_lower for floor_obj in FLOOR_OBJECTS)

def should_use_elevated_depth_adjustment(label: str) -> bool:
    """
    Determine if object is elevated and should inherit depth from objects below.
    """
    label_lower = label.lower()
    elevated_keywords = {'light', 'chandelier', 'ceiling', 'lamp-ceiling'}
    return any(keyword in label_lower for keyword in elevated_keywords)

def estimate_ground_plane_y(segments_with_poses: List[Dict], 
                           floor_object_labels: set = FLOOR_OBJECTS) -> float:
    """
    Estimate ground plane Y coordinate from floor objects.
    
    Args:
        segments_with_poses: List of segments with 3D positions
        floor_object_labels: Set of labels for floor objects
        
    Returns:
        Estimated ground plane Y coordinate
    """
    floor_y_values = []
    
    for seg in segments_with_poses:
        label_lower = seg['label'].lower()
        if any(floor_obj in label_lower for floor_obj in floor_object_labels):
            # For floor objects, estimate base Y using dimensions
            pos_y = seg['position_m'][1]
            height = seg['dimensions_m'][2]
            base_y = pos_y - (height / 2)  # Subtract half height to get base
            floor_y_values.append(base_y)
    
    if floor_y_values:
        # Use median to be robust to outliers
        return np.median(floor_y_values)
    
    return 0.0  # Default to origin if no floor objects found


def adjust_position_to_ground_plane(position: np.ndarray, 
                                   dimensions: List[float],
                                   ground_y: float,
                                   label: str) -> np.ndarray:
    """
    Adjust object position so its base sits on the ground plane.
    
    Args:
        position: Original 3D position [x, y, z]
        dimensions: Object dimensions [length, width, height]
        ground_y: Ground plane Y coordinate
        label: Object label
        
    Returns:
        Adjusted position with base on ground plane
    """
    if not should_use_bottom_center(label):
        return position
    
    height = dimensions[2]
    adjusted_position = position.copy()
    
    # Set Y so that bottom of object is at ground_y
    adjusted_position[1] = ground_y + (height / 2)
    
    return adjusted_position


def get_unproject_point(bbox: Dict, use_bottom: bool) -> tuple:
    """
    Get the pixel coordinates to use for unprojection.
    
    Args:
        bbox: Bounding box dictionary with center and dimensions
        use_bottom: Whether to use bottom center vs geometric center
        
    Returns:
        (pixel_x, pixel_y) tuple
    """
    if use_bottom:
        # Use bottom-center of bounding box
        pixel_x = bbox['center_x']
        pixel_y = bbox['y_min'] + bbox['height']  # Bottom edge
        return pixel_x, pixel_y
    else:
        # Use geometric center
        return bbox['center_x'], bbox['center_y']
    
def extract_depth_from_bottom_region(
    depth_map: np.ndarray,
    mask: np.ndarray,
    bbox: Dict,
    bottom_fraction: float = 0.3
) -> Optional[float]:
    """
    Extract depth from bottom portion of mask.
    
    Args:
        depth_map: Metric depth map
        mask: Boolean mask (numpy array)
        bbox: Bounding box dictionary
        bottom_fraction: Fraction of height to use from bottom
        
    Returns:
        Median depth from bottom region or None
    """
    # Define bottom region  
    y_min = bbox['y_min']
    height = bbox['height']
    bottom_y_threshold = int(y_min + height * (1 - bottom_fraction))
    
    # Create boolean mask for bottom region
    mask_bottom = mask.copy()
    mask_bottom[:bottom_y_threshold, :] = False
    
    depth_values = depth_map[mask_bottom]
    
    if depth_values.size == 0:
        return None
    
    return np.median(depth_values)