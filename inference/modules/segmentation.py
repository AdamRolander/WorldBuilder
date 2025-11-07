"""
Panoptic segmentation using Mask2Former.
"""

import torch
from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor
from PIL import Image
from typing import List, Dict, Any


def calculate_bounding_box_from_mask(mask_tensor: torch.Tensor, img_width: int, img_height: int) -> Dict[str, float]:
    """
    Calculate normalized bounding box from a boolean mask tensor.
    
    Args:
        mask_tensor: Boolean mask tensor
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        Dictionary with normalized bbox coordinates and dimensions
    """
    if not mask_tensor.any():
        return None
        
    coords = torch.nonzero(mask_tensor)
    y_min, x_min = coords.min(dim=0).values
    y_max, x_max = coords.max(dim=0).values
    
    width = (x_max - x_min + 1).item()
    height = (y_max - y_min + 1).item()
    
    return {
        'x_min': x_min.item(),
        'y_min': y_min.item(),
        'width': width,
        'height': height,
        'center_x': x_min.item() + width / 2,
        'center_y': y_min.item() + height / 2,
        'center_x_norm': (x_min.item() + width / 2) / img_width,
        'center_y_norm': (y_min.item() + height / 2) / img_height,
        'width_norm': width / img_width,
        'height_norm': height / img_height
    }


def segment_image(image_path: str, device: str = "cuda") -> Dict[str, Any]:
    """
    Perform panoptic segmentation on an image.
    
    Args:
        image_path: Path to input image
        device: Device to run model on ('cuda' or 'cpu')
        
    Returns:
        Dictionary containing:
            - segmentation_map: Tensor with segment IDs
            - segments: List of segment dictionaries with id, label, bbox, mask_area
            - img_width: Image width
            - img_height: Image height
    """
    # Structural elements to exclude // go back to include bottle and curb crazy depth scores
    EXCLUDED_LABELS = {
        'wall', 'floor', 'ceiling', 'window', 'door', 'sky', 'ground',
        'wall-other-merged', 'building', 'bottle'
    }
    
    # Load image
    img = Image.open(image_path).convert("RGB")
    img_width, img_height = img.size
    
    # Load segmentation model
    model_name = "facebook/mask2former-swin-large-coco-panoptic"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name).to(device)
    
    # Run segmentation
    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt").to(device)
        outputs = model(**inputs)
        panoptic_result = processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[img.size[::-1]]
        )[0]
    
    segmentation_map = panoptic_result["segmentation"]
    segments_info = panoptic_result["segments_info"]
    
    # Process segments
    segments = []
    for segment in segments_info:
        segment_id = segment['id']
        label_id = segment['label_id']
        segment_label = model.config.id2label[label_id].lower()
        
        # Skip structural elements
        if any(excluded in segment_label for excluded in EXCLUDED_LABELS):
            continue
            
        mask_tensor = (segmentation_map == segment_id).cpu()
        bbox_data = calculate_bounding_box_from_mask(mask_tensor, img_width, img_height)
        mask_area = mask_tensor.sum().item()
        
        if bbox_data is not None and mask_area > 0:
            segments.append({
                'id': segment_id,
                'label': model.config.id2label[label_id],
                'bbox': bbox_data,
                'mask_area': mask_area,
                'mask': mask_tensor
            })
    
    return {
        'segmentation_map': segmentation_map,
        'segments': segments,
        'img_width': img_width,
        'img_height': img_height
    }