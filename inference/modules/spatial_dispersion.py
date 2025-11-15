"""
Spatial dispersion to reduce object clustering.
"""

import numpy as np
from typing import List, Dict


def calculate_scene_extent(results: List[Dict]) -> Dict:
    """Calculate spatial extent of the scene."""
    positions = np.array([r['position_m'] for r in results])
    
    x_range = np.max(positions[:, 0]) - np.min(positions[:, 0])
    z_range = np.max(positions[:, 2]) - np.min(positions[:, 2])
    
    return {
        'x_range': x_range,
        'z_range': z_range,
        'horizontal_extent': max(x_range, z_range)
    }


def apply_dispersion(results: List[Dict], dispersion_strength: float = 0.5) -> List[Dict]:
    """
    Apply repulsion forces to spread out clustered objects.
    
    Args:
        results: List of object results with positions and dimensions
        dispersion_strength: Multiplier on minimum separation (0.5 = half object size buffer)
    
    Returns:
        Updated results with dispersed positions
    """
    print("\n" + "="*80)
    print("SPATIAL DISPERSION")
    print("="*80 + "\n")
    
    positions = np.array([r['position_m'] for r in results])
    adjusted_positions = positions.copy()
    
    moved_count = 0
    total_movement = 0
    
    for i, result_i in enumerate(results):
        repulsion = np.zeros(3)
        dims_i = result_i['dimensions_m']
        size_i = max(dims_i[0], dims_i[1])  # Horizontal footprint (length or width)
        
        for j, result_j in enumerate(results):
            if i == j:
                continue
            
            dims_j = result_j['dimensions_m']
            size_j = max(dims_j[0], dims_j[1])
            
            # Minimum separation = sum of radii + buffer
            min_separation = (size_i + size_j) / 2 * (1 + dispersion_strength)
            
            # Vector from j to i (horizontal plane only)
            diff = positions[i] - positions[j]
            diff[1] = 0  # Ignore vertical
            distance = np.linalg.norm(diff)
            
            # If too close, apply repulsion
            if distance < min_separation and distance > 0.01:
                # Repulsion strength increases as objects get closer
                overlap = min_separation - distance
                strength = overlap / min_separation
                direction = diff / distance
                repulsion += direction * overlap * strength
        
        # Apply repulsion (only X-Z plane)
        repulsion_magnitude = np.linalg.norm(repulsion[:2])
        if repulsion_magnitude > 0.01:
            adjusted_positions[i, 0] += repulsion[0]
            adjusted_positions[i, 2] += repulsion[2]
            moved_count += 1
            total_movement += repulsion_magnitude
            
            print(f"  {result_i['label']} (ID {result_i['id']}): moved {repulsion_magnitude:.3f}m")
    
    # Update results
    for i, result in enumerate(results):
        result['position_m'] = adjusted_positions[i]
    
    print(f"\n{'='*80}")
    print(f"Dispersed {moved_count}/{len(results)} objects")
    print(f"Average movement: {total_movement/max(moved_count, 1):.3f}m")
    print(f"{'='*80}\n")
    
    return results