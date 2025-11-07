"""
Spatial pattern recognition and adjustment.
"""

import numpy as np
from typing import List, Dict
from sklearn.cluster import DBSCAN


def detect_object_groups(segments: List[Dict], results: List[Dict]) -> List[List[int]]:
    """
    Detect groups of similar objects that might form patterns.
    
    Returns:
        List of groups, where each group is a list of object IDs
    """
    # Group by label
    label_groups = {}
    for result in results:
        label = result['label'].lower()
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(result['id'])
    
    # Return groups of 2+ similar objects
    groups = []
    for label, ids in label_groups.items():
        if len(ids) >= 2:
            groups.append(ids)
    
    return groups


def detect_arrangement_type(positions: np.ndarray) -> Dict:
    """
    Detect if objects are in a linear or circular arrangement.
    
    Args:
        positions: Nx3 array of [x, y, z] positions
        
    Returns:
        Dict with arrangement info
    """
    if len(positions) < 3:
        return {'type': 'none'}
    
    # Calculate center
    center = np.mean(positions, axis=0)
    
    # Check for circular arrangement (equal distances from center)
    distances_from_center = np.sqrt((positions[:, 0] - center[0])**2 + 
                                   (positions[:, 2] - center[2])**2)
    mean_dist = np.mean(distances_from_center)
    std_dist = np.std(distances_from_center)
    circular_score = std_dist / (mean_dist + 1e-6)
    
    # Check for linear arrangement (fit a line through XZ positions)
    xz_positions = positions[:, [0, 2]]  # Just X and Z
    
    # Fit line using PCA
    centered = xz_positions - np.mean(xz_positions, axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    # Principal component ratio (how "line-like" are the points?)
    ev_ratio = eigenvalues[0] / (eigenvalues[1] + 1e-6)
    
    # Calculate distances from fitted line
    line_direction = eigenvectors[:, 0]
    projections = centered @ line_direction
    reconstructed = np.outer(projections, line_direction)
    distances_from_line = np.linalg.norm(centered - reconstructed, axis=1)
    mean_line_dist = np.mean(distances_from_line)
    
    # Determine arrangement type
    is_linear = ev_ratio > 8.0 and mean_line_dist < 0.3  # Tight line
    is_circular = circular_score < 0.3  # Equal distances from center
    
    if is_linear:
        # Calculate line parameters
        line_center = np.mean(xz_positions, axis=0)
        line_angle = np.degrees(np.arctan2(line_direction[1], line_direction[0]))
        
        return {
            'type': 'linear',
            'center_xz': line_center,
            'direction': line_direction,
            'angle': line_angle,
            'ev_ratio': ev_ratio,
            'mean_dist_from_line': mean_line_dist
        }
    elif is_circular:
        angles = np.arctan2(positions[:, 2] - center[2], 
                           positions[:, 0] - center[0])
        return {
            'type': 'circular',
            'center': center,
            'radius': mean_dist,
            'angles': np.degrees(angles),
            'circular_score': circular_score
        }
    else:
        return {'type': 'scattered'}


def adjust_circular_pattern_rotations(
    results: List[Dict],
    group_ids: List[int],
    pattern_info: Dict
) -> List[Dict]:
    """
    Adjust rotations of objects in a circular pattern to face the center.
    
    Args:
        results: List of result dictionaries
        group_ids: IDs of objects in the circular pattern
        pattern_info: Output from 
        
    Returns:
        Updated results with adjusted rotations
    """
    if not pattern_info['is_circular']:
        return results
    
    center = pattern_info['center']
    
    for result in results:
        if result['id'] in group_ids:
            pos = np.array(result['position_m'])
            
            # Calculate angle from object to center
            dx = center[0] - pos[0]
            dz = center[2] - pos[2]
            angle_to_center = np.degrees(np.arctan2(dz, dx))
            
            # Object should face the center
            # In your coordinate system: 0° = -Z, 90° = -X
            # Angle to center in world → rotation_y
            rotation_y = (angle_to_center + 90) % 360
            
            print(f"  Pattern: {result['label']} (ID {result['id']}) adjusted to face center: {rotation_y:.1f}°")
            result['rotation_y_deg'] = rotation_y
    
    return results


def apply_pattern_recognition(segments: List[Dict], results: List[Dict]) -> List[Dict]:
    """
    Main function: detect patterns and adjust object poses accordingly.
    """
    print("\n=== Pattern Recognition ===")
    
    # Detect groups of similar objects
    groups = detect_object_groups(segments, results)
    print(f"Detected {len(groups)} object groups")
    
    for group_ids in groups:
        if len(group_ids) < 3:
            continue
            
        # Get positions for this group
        positions = []
        labels = []
        for result in results:
            if result['id'] in group_ids:
                positions.append(result['position_m'])
                labels.append(result['label'])
        
        positions = np.array(positions)
        label = labels[0] if labels else 'unknown'
        
        # Detect arrangement type
        arrangement = detect_arrangement_type(positions)
        
        if arrangement['type'] == 'linear':
            print(f"  Linear pattern detected: {len(group_ids)}x {label}")
            print(f"    Center: ({arrangement['center_xz'][0]:.2f}, {arrangement['center_xz'][1]:.2f})")
            print(f"    Line angle: {arrangement['angle']:.1f}°")
            print(f"    Linearity score: {arrangement['ev_ratio']:.2f}")
            
            # For linear arrangements (like chairs along island edge):
            # All should face perpendicular to the line (toward the table)
            perpendicular_angle = (arrangement['angle'] + 90) % 360
            
            for result in results:
                if result['id'] in group_ids:
                    print(f"    {result['label']} (ID {result['id']}) aligned perpendicular to line: {perpendicular_angle:.1f}°")
                    result['rotation_y_deg'] = perpendicular_angle
            
        elif arrangement['type'] == 'circular':
            print(f"  Circular pattern detected: {len(group_ids)}x {label}")
            print(f"    Center: ({arrangement['center'][0]:.2f}, {arrangement['center'][2]:.2f})")
            print(f"    Radius: {arrangement['radius']:.2f}m")
            
            # Adjust rotations to face center (original logic)
            results = adjust_circular_pattern_rotations(results, group_ids, arrangement)
        
        else:
            print(f"  Scattered arrangement: {len(group_ids)}x {label} (no adjustment)")
    
    return results