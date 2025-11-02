"""
Multi-reference depth calibration for improved accuracy.
"""

import numpy as np
from typing import List, Dict, Tuple
from geometry import calculate_analytical_depth
from depth_estimation import extract_depth_from_mask


def calculate_calibration_constants(
    segments: List[Dict],
    estimated_sizes: Dict,
    raw_depth_map: np.ndarray,
    img_width: int,
    fov_deg: float,
    min_confidence: float = 0.6
) -> List[Dict]:
    """
    Calculate calibration constants from multiple reference objects.
    
    Args:
        segments: List of valid segments
        estimated_sizes: Size estimation results
        raw_depth_map: Raw MiDaS depth map (inverse depth)
        img_width: Image width
        fov_deg: Field of view
        min_confidence: Minimum size confidence to use as reference
        
    Returns:
        List of calibration dictionaries with k values and metadata
    """
    calibration_points = []
    
    for segment in segments:
        seg_id = segment['id']
        if seg_id not in estimated_sizes:
            continue
            
        label = segment['label']
        bbox = segment['bbox']
        size_data = estimated_sizes[seg_id]
        
        # Skip low-confidence size estimates
        if size_data.get('confidence', 0) < min_confidence:
            continue
        
        dims = size_data['dimensions_meters']
        
        # Calculate analytical depth using both height and width
        # This gives us two independent estimates
        analytical_depth_height = calculate_analytical_depth(
            dims[2],  # height
            bbox['height'],
            img_width,
            fov_deg
        )
        
        analytical_depth_width = calculate_analytical_depth(
            max(dims[0], dims[1]),  # length or width (whichever is larger)
            max(bbox['width'], bbox['height']),  # corresponding pixel dimension
            img_width,
            fov_deg
        )
        
        # Average the two estimates
        analytical_depth = (analytical_depth_height + analytical_depth_width) / 2.0
        
        # Get MiDaS inverse depth value for this object
        depth_stats = extract_depth_from_mask(raw_depth_map, segment['mask'])
        midas_value = depth_stats['median']
        
        # Calculate k = depth * midas_value (the calibration constant)
        k = analytical_depth * midas_value
        
        calibration_points.append({
            'id': seg_id,
            'label': label,
            'k': k,
            'analytical_depth': analytical_depth,
            'midas_value': midas_value,
            'confidence': size_data.get('confidence', 0.5),
            'mask_area': segment['mask_area']
        })
        
        print(f"  {label} (ID {seg_id}): k={k:.2f}, depth={analytical_depth:.2f}m, confidence={size_data.get('confidence', 0.5):.2f}")
    
    return calibration_points


def get_robust_calibration_constant(
    calibration_points: List[Dict],
    method: str = "weighted_median"
) -> Tuple[float, Dict]:
    """
    Get robust calibration constant from multiple measurements.
    
    Args:
        calibration_points: List of calibration dictionaries
        method: 'weighted_median', 'median', 'mean', or 'confidence_weighted'
        
    Returns:
        Tuple of (k_value, metadata_dict)
    """
    if not calibration_points:
        print("WARNING: No calibration points available")
        return None, {}
    
    k_values = np.array([p['k'] for p in calibration_points])
    confidences = np.array([p['confidence'] for p in calibration_points])
    areas = np.array([p['mask_area'] for p in calibration_points])
    
    if method == "median":
        k_final = np.median(k_values)
    elif method == "mean":
        k_final = np.mean(k_values)
    elif method == "weighted_median":
        # Weight by mask area (larger objects are more reliable)
        weights = areas / areas.sum()
        sorted_idx = np.argsort(k_values)
        sorted_weights = weights[sorted_idx]
        cumsum = np.cumsum(sorted_weights)
        median_idx = np.argmax(cumsum >= 0.5)
        k_final = k_values[sorted_idx[median_idx]]
    elif method == "confidence_weighted":
        # Weight by confidence scores
        weights = confidences / confidences.sum()
        k_final = np.sum(k_values * weights)
    else:
        k_final = np.median(k_values)
    
    # Calculate statistics
    k_std = np.std(k_values)
    k_min = np.min(k_values)
    k_max = np.max(k_values)
    
    metadata = {
        'k_mean': float(np.mean(k_values)),
        'k_median': float(np.median(k_values)),
        'k_std': float(k_std),
        'k_min': float(k_min),
        'k_max': float(k_max),
        'num_references': len(calibration_points),
        'method': method
    }
    
    print(f"\nCalibration constant k = {k_final:.2f} (method: {method})")
    print(f"  Mean: {metadata['k_mean']:.2f}, Std: {k_std:.2f}")
    print(f"  Range: [{k_min:.2f}, {k_max:.2f}]")
    print(f"  Using {len(calibration_points)} reference objects")
    
    return k_final, metadata


def normalize_depth_with_constant(
    raw_depth_map: np.ndarray,
    k: float
) -> np.ndarray:
    """
    Convert MiDaS inverse depth to metric depth using calibration constant.
    
    Args:
        raw_depth_map: Raw MiDaS output (inverse depth)
        k: Calibration constant
        
    Returns:
        Metric depth map in meters
    """
    safe_depth_map = np.clip(raw_depth_map, 0.1, None)
    metric_depth_map = k / safe_depth_map
    return metric_depth_map