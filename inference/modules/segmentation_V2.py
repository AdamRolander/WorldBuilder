"""
Enhanced segmentation with instance/panoptic fusion and mask splitting.
"""

import torch
import numpy as np
from PIL import Image
from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor
from scipy import ndimage
from typing import List, Dict, Any


def create_bbox_data(x_min, y_min, width, height, img_width, img_height):
    """Create bbox dictionary with both pixel and normalized coordinates."""
    return {
        'x_min': int(x_min),
        'y_min': int(y_min),
        'width': int(width),
        'height': int(height),
        'center_x': float(x_min + width / 2),
        'center_y': float(y_min + height / 2),
        'center_x_norm': float(x_min + width / 2) / img_width,
        'center_y_norm': float(y_min + height / 2) / img_height,
        'width_norm': float(width) / img_width,
        'height_norm': float(height) / img_height,
    }


def segment_image_hybrid(image_path: str, device: str = "cuda") -> tuple:
    """
    Hybrid segmentation combining instance and panoptic models.
    
    Args:
        image_path: Path to input image
        device: Device to run models on
        
    Returns:
        Tuple of (segments, img_width, img_height, img_np)
    """
    EXCLUDED_LABELS = {
        'wall', 'floor', 'ceiling', 'window', 'door', 'sky', 'ground',
        'wall-other-merged', 'building', 'bottle'
    }
    
    img = Image.open(image_path).convert("RGB")
    img_width, img_height = img.size
    img_np = np.array(img)
    
    segments = []
    next_id = 0
    
    # === PART 1: Instance Segmentation ===
    print("Running instance segmentation...")
    model_name = "facebook/mask2former-swin-large-coco-instance"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name).to(device)
    
    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt").to(device)
        outputs = model(**inputs)
        instance_result = processor.post_process_instance_segmentation(
            outputs, 
            target_sizes=[img.size[::-1]],
            threshold=0.5
        )[0]
    
    segmentation_map_instance = instance_result["segmentation"]
    segments_info_instance = instance_result["segments_info"]
    
    for segment in segments_info_instance:
        segment_id = segment['id']
        label_id = segment['label_id']
        score = segment.get('score', 1.0)
        
        segment_label = model.config.id2label[label_id].lower()
        
        if any(excluded in segment_label for excluded in EXCLUDED_LABELS):
            continue
        
        mask_tensor = (segmentation_map_instance == segment_id)
        mask_np = mask_tensor.cpu().numpy()
        
        coords = np.argwhere(mask_np)
        if len(coords) == 0:
            continue
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        
        bbox_data = create_bbox_data(x_min, y_min, width, height, img_width, img_height)
        
        segments.append({
            'id': next_id,
            'label': model.config.id2label[label_id],
            'bbox': bbox_data,
            'mask_area': int(mask_np.sum()),
            'mask': mask_np,
            'confidence': float(score),
            'source': 'instance'
        })
        next_id += 1
    
    print(f"  Found {len(segments)} instances")
    
    # === PART 2: Panoptic Segmentation (for structural elements) ===
    print("Running panoptic segmentation...")
    model_name = "facebook/mask2former-swin-large-coco-panoptic"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name).to(device)
    
    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt").to(device)
        outputs = model(**inputs)
        panoptic_result = processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[img.size[::-1]]
        )[0]
    
    segmentation_map_panoptic = panoptic_result["segmentation"]
    segments_info_panoptic = panoptic_result["segments_info"]
    
    # Only keep "stuff" classes from panoptic
    STUFF_CLASSES = {'cabinet', 'light', 'shelf', 'counter', 'countertop'}
    
    stuff_count = 0
    for segment in segments_info_panoptic:
        segment_id = segment['id']
        label_id = segment['label_id']
        
        segment_label = model.config.id2label[label_id].lower()
        
        if not any(stuff in segment_label for stuff in STUFF_CLASSES):
            continue
        
        if any(excluded in segment_label for excluded in EXCLUDED_LABELS):
            continue
        
        mask_tensor = (segmentation_map_panoptic == segment_id)
        mask_np = mask_tensor.cpu().numpy()
        
        coords = np.argwhere(mask_np)
        if len(coords) == 0:
            continue
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(0)
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        
        bbox_data = create_bbox_data(x_min, y_min, width, height, img_width, img_height)
        
        segments.append({
            'id': next_id,
            'label': model.config.id2label[label_id],
            'bbox': bbox_data,
            'mask_area': int(mask_np.sum()),
            'mask': mask_np,
            'confidence': 0.9,
            'source': 'panoptic'
        })
        next_id += 1
        stuff_count += 1
    
    print(f"  Found {stuff_count} structural elements")
    print(f"Total: {len(segments)} segments")
    
    return segments, img_width, img_height, img_np


def split_merged_masks(segments: List[Dict], img_width: int, img_height: int, min_area_pixels: int = 500) -> List[Dict]:
    """
    Split merged panoptic masks using connected components analysis.
    
    Args:
        segments: List of segment dictionaries
        img_width: Image width
        img_height: Image height
        min_area_pixels: Minimum area for a component to be kept
        
    Returns:
        Updated segments list with split instances
    """
    new_segments = []
    next_id = max(seg['id'] for seg in segments) + 1
    
    for seg in segments:
        if seg['source'] != 'panoptic':
            new_segments.append(seg)
            continue
        
        SPLITTABLE_LABELS = {'light', 'cabinet', 'shelf', 'counter'}
        
        if not any(label in seg['label'].lower() for label in SPLITTABLE_LABELS):
            new_segments.append(seg)
            continue
        
        labeled_mask, num_components = ndimage.label(seg['mask'])
        
        if num_components <= 1:
            new_segments.append(seg)
            continue
        
        print(f"Splitting {seg['label']} (ID: {seg['id']}) into {num_components} components")
        
        for component_id in range(1, num_components + 1):
            component_mask = (labeled_mask == component_id)
            component_area = component_mask.sum()
            
            if component_area < min_area_pixels:
                continue
            
            coords = np.argwhere(component_mask)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            width = x_max - x_min + 1
            height = y_max - y_min + 1
            
            bbox_data = create_bbox_data(x_min, y_min, width, height, img_width, img_height)
            
            new_segments.append({
                'id': next_id,
                'label': seg['label'],
                'bbox': bbox_data,
                'mask_area': int(component_area),
                'mask': component_mask,
                'confidence': seg['confidence'],
                'source': 'panoptic_split'
            })
            next_id += 1
    
    return new_segments


def segment_and_split(image_path: str, device: str = "cuda") -> Dict[str, Any]:
    """
    Main entry point: Segmentation with mask splitting.
    
    Args:
        image_path: Path to input image
        device: Device to run models on
        
    Returns:
        Dictionary containing:
            - segments: List of segment dictionaries with masks and bboxes
            - img_width: Image width
            - img_height: Image height
            - img_np: Image as numpy array
    """
    print(f"\n{'='*80}")
    print("HYBRID SEGMENTATION PIPELINE")
    print(f"{'='*80}\n")
    
    # Step 1: Hybrid segmentation
    segments, img_width, img_height, img_np = segment_image_hybrid(image_path, device)
    
    # Step 2: Split merged masks
    print("\nSplitting merged masks...")
    segments = split_merged_masks(segments, img_width, img_height, min_area_pixels=500)
    
    print(f"\n{'='*80}")
    print(f"COMPLETE: {len(segments)} objects detected")
    print(f"{'='*80}\n")
    
    return {
        'segments': segments,
        'img_width': img_width,
        'img_height': img_height,
        'img_np': img_np
    }