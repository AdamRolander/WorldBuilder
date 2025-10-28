"""
Depth estimation using MiDaS.
"""

import torch
import cv2
import numpy as np
from typing import Dict, Optional


def load_midas_model(device: str = "cuda"):
    """
    Load MiDaS depth estimation model.
    
    Args:
        device: Device to load model on
        
    Returns:
        Tuple of (model, transform_function)
    """
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.to(device)
    midas.eval()
    
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform
    
    return midas, transform


def estimate_depth_with_midas(
    image_path: str,
    midas_model,
    midas_transform,
    depth_map_cache: Optional[Dict] = None
) -> np.ndarray:
    """
    Estimate depth map for an entire image using MiDaS.
    
    Args:
        image_path: Path to input image
        midas_model: Loaded MiDaS model
        midas_transform: MiDaS transform function
        depth_map_cache: Optional cache dictionary
        
    Returns:
        Depth map as numpy array (inverse depth/disparity)
    """
    # Check cache
    if depth_map_cache is not None and image_path in depth_map_cache:
        return depth_map_cache[image_path]
    
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Transform and prepare input
    input_batch = midas_transform(img)
    if input_batch.dim() == 3:
        input_batch = input_batch.unsqueeze(0)
    
    device = next(midas_model.parameters()).device
    input_batch = input_batch.to(device)
    
    # Predict depth
    with torch.no_grad():
        prediction = midas_model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth_map = prediction.cpu().numpy()
    
    # Cache result
    if depth_map_cache is not None:
        depth_map_cache[image_path] = depth_map
    
    return depth_map


def normalize_depth_to_metric(
    midas_depth_map: np.ndarray,
    reference_depth_meters: float,
    reference_midas_value: float
) -> np.ndarray:
    """
    Convert MiDaS inverse depth (disparity) to metric depth.
    
    MiDaS outputs inverse depth where higher values = closer objects.
    We calibrate using a reference object with known metric depth.
    
    Args:
        midas_depth_map: Raw MiDaS output (inverse depth)
        reference_depth_meters: Known metric depth of reference object
        reference_midas_value: MiDaS inverse depth value for reference
        
    Returns:
        Metric depth map in meters
    """
    safe_midas_map = np.clip(midas_depth_map, 0.1, None)
    k = reference_depth_meters * reference_midas_value
    metric_depth_map = k / safe_midas_map
    
    return metric_depth_map


def extract_depth_from_mask(
    depth_map: np.ndarray,
    mask: torch.Tensor,
    filter_outliers: bool = True
) -> Dict[str, float]:
    """
    Extract depth statistics from a masked region.
    
    Args:
        depth_map: Metric depth map
        mask: Boolean mask tensor
        filter_outliers: Whether to filter outliers using IQR
        
    Returns:
        Dictionary with depth statistics (median, mean, std, etc.)
    """
    mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else mask
    depth_values = depth_map[mask_np]
    
    if depth_values.size == 0:
        return None
    
    # Filter outliers using IQR method
    if filter_outliers and depth_values.size >= 10:
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
        if filtered.size > max(10, depth_values.size * 0.3):
            depth_values = filtered
    
    median = np.median(depth_values)
    mean = np.mean(depth_values)
    std = np.std(depth_values)

    # Sanity check: if median is unreasonable, use mean of all non-outlier values
    # if median > 50 or median < 0.1:
    #     # Clip to reasonable range (0.5m to 20m for indoor scenes)
    #     clipped_values = np.clip(depth_values, 0.5, 20.0)
    #     median = np.median(clipped_values)
    #     print(f"  WARNING: Depth outlier detected, clipping to range [0.5, 20.0]m")

    return {
        'mean': mean,
        'median': median,
        'std': std,
        'min': np.min(depth_values),
        'max': np.max(depth_values),
        'p25': np.percentile(depth_values, 25),
        'p75': np.percentile(depth_values, 75)
    }