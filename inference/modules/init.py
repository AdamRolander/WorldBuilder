"""
Modular inference components for 3D pose estimation.
"""

from .segmentation import segment_image
from .depth_estimation import estimate_depth_with_midas, normalize_depth_to_metric
from .size_estimation import estimate_object_sizes
from .geometry import unproject_to_3d, calculate_bounding_box_from_mask

__all__ = [
    'segment_image',
    'estimate_depth_with_midas',
    'normalize_depth_to_metric',
    'estimate_object_sizes',
    'unproject_to_3d',
    'calculate_bounding_box_from_mask'
]