"""
Enhanced segmentation with instance/panoptic fusion, mask splitting, and rotation estimation.
"""

import torch
import cv2
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


def aggressive_spatial_split(segments: List[Dict], img_width: int, img_height: int) -> List[Dict]:
    """
    Apply spatial heuristics to split complex masks (L-shapes, horizontal arrangements).
    
    Args:
        segments: List of segment dictionaries
        img_width: Image width
        img_height: Image height
        
    Returns:
        Updated segments list with additional splits
    """
    
    def detect_l_shape(mask, bbox):
        """Detect if a mask is L-shaped by checking density ratios."""
        height = bbox['height']
        width = bbox['width']
        y_min = bbox['y_min']
        x_min = bbox['x_min']
        
        left_half = mask[y_min:y_min + height, x_min:x_min + width//2]
        right_half = mask[y_min:y_min + height, x_min + width//2:x_min + width]
        
        left_density = left_half.sum() / (left_half.size + 1e-6)
        right_density = right_half.sum() / (right_half.size + 1e-6)
        
        density_ratio = max(left_density, right_density) / (min(left_density, right_density) + 0.1)
        
        return density_ratio > 2.0, density_ratio
    
    def split_l_shaped_cabinet(mask, bbox):
        """Split L-shaped cabinet by finding optimal horizontal cut."""
        best_split_y = None
        best_score = 0
        
        for split_y in range(bbox['y_min'] + bbox['height']//4, 
                            bbox['y_min'] + 3*bbox['height']//4, 
                            10):
            top_mask = mask.copy()
            bot_mask = mask.copy()
            
            top_mask[split_y:, :] = False
            bot_mask[:split_y, :] = False
            
            top_area = top_mask.sum()
            bot_area = bot_mask.sum()
            
            if top_area > 5000 and bot_area > 5000:
                score = min(top_area, bot_area)
                if score > best_score:
                    best_score = score
                    best_split_y = split_y
        
        if best_split_y is not None:
            top_mask = mask.copy()
            bot_mask = mask.copy()
            top_mask[best_split_y:, :] = False
            bot_mask[:best_split_y, :] = False
            return [top_mask, bot_mask]
        
        return [mask]
    
    def split_horizontally_arranged_objects(mask, bbox, num_expected=3):
        """Split horizontally arranged objects (like pendant lights)."""
        x_min = bbox['x_min']
        width = bbox['width']
        
        strip_width = width // num_expected
        submasks = []
        
        for i in range(num_expected):
            x_start = x_min + i * strip_width
            x_end = x_min + (i + 1) * strip_width if i < num_expected - 1 else x_min + width
            
            submask = np.zeros_like(mask)
            submask[:, x_start:x_end] = mask[:, x_start:x_end]
            
            if submask.sum() > 500:
                submasks.append(submask)
        
        return submasks
    
    new_segments = []
    next_id = max(seg['id'] for seg in segments) + 1
    
    print("Applying spatial splitting heuristics...")
    
    for seg in segments:
        label_lower = seg['label'].lower()
        
        # Strategy 1: Split L-shaped cabinets
        if 'cabinet' in label_lower and seg['source'] in ['panoptic', 'panoptic_split']:
            is_l_shape, ratio = detect_l_shape(seg['mask'], seg['bbox'])
            
            if is_l_shape:
                print(f"  Detected L-shaped {seg['label']} (ID: {seg['id']})")
                submasks = split_l_shaped_cabinet(seg['mask'], seg['bbox'])
                
                if len(submasks) > 1:
                    for submask in submasks:
                        coords = np.argwhere(submask)
                        if len(coords) == 0:
                            continue
                        
                        y_min, x_min = coords.min(axis=0)
                        y_max, x_max = coords.max(axis=0)
                        width = x_max - x_min + 1
                        height = y_max - y_min + 1
                        
                        bbox_data = create_bbox_data(x_min, y_min, width, height, img_width, img_height)
                        
                        new_segments.append({
                            'id': next_id,
                            'label': seg['label'],
                            'bbox': bbox_data,
                            'mask_area': int(submask.sum()),
                            'mask': submask,
                            'confidence': seg['confidence'],
                            'source': 'spatial_split'
                        })
                        next_id += 1
                    continue
        
        # Strategy 2: Split horizontal light arrangements
        if 'light' in label_lower and seg['bbox']['width'] > seg['bbox']['height'] * 1.5:
            print(f"  Detected horizontal light arrangement {seg['label']} (ID: {seg['id']})")
            submasks = split_horizontally_arranged_objects(seg['mask'], seg['bbox'], num_expected=3)
            
            if len(submasks) > 1:
                for submask in submasks:
                    coords = np.argwhere(submask)
                    if len(coords) == 0:
                        continue
                    
                    y_min, x_min = coords.min(axis=0)
                    y_max, x_max = coords.max(axis=0)
                    width = x_max - x_min + 1
                    height = y_max - y_min + 1
                    
                    bbox_data = create_bbox_data(x_min, y_min, width, height, img_width, img_height)
                    
                    new_segments.append({
                        'id': next_id,
                        'label': seg['label'],
                        'bbox': bbox_data,
                        'mask_area': int(submask.sum()),
                        'mask': submask,
                        'confidence': seg['confidence'],
                        'source': 'spatial_split'
                    })
                    next_id += 1
                continue
        
        new_segments.append(seg)
    
    return new_segments


def resolve_bbox_overlaps(segments: List[Dict], img_width: int, img_height: int) -> List[Dict]:
    """
    Resolve bounding box overlaps when masks don't actually overlap.
    
    Args:
        segments: List of segment dictionaries
        img_width: Image width
        img_height: Image height
        
    Returns:
        Updated segments with adjusted bounding boxes
    """
    
    def bbox_overlap_iou(bbox1, bbox2):
        """Calculate IoU between two bounding boxes."""
        x1_min, y1_min = bbox1['x_min'], bbox1['y_min']
        x1_max = x1_min + bbox1['width']
        y1_max = y1_min + bbox1['height']
        
        x2_min, y2_min = bbox2['x_min'], bbox2['y_min']
        x2_max = x2_min + bbox2['width']
        y2_max = y2_min + bbox2['height']
        
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        
        overlap_area = x_overlap * y_overlap
        bbox1_area = bbox1['width'] * bbox1['height']
        bbox2_area = bbox2['width'] * bbox2['height']
        union_area = bbox1_area + bbox2_area - overlap_area
        
        return overlap_area / union_area if union_area > 0 else 0
    
    def masks_overlap(mask1, mask2):
        """Check if two masks have overlapping pixels."""
        return np.logical_and(mask1, mask2).sum() > 0
    
    def compute_tight_bbox(mask, exclude_mask=None):
        """Compute tighter bounding box by excluding another mask's region."""
        if exclude_mask is not None:
            adjusted_mask = np.logical_and(mask, ~exclude_mask)
        else:
            adjusted_mask = mask
        
        coords = np.argwhere(adjusted_mask)
        
        if len(coords) == 0:
            return None
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        
        return create_bbox_data(x_min, y_min, width, height, img_width, img_height)
    
    structural_elements = [s for s in segments if s['source'] in ['panoptic', 'panoptic_split', 'spatial_split']]
    resolved_segments = segments.copy()
    
    print("Resolving bounding box overlaps...")
    
    for i, seg1 in enumerate(structural_elements):
        for j, seg2 in enumerate(structural_elements):
            if i >= j:
                continue
            
            iou = bbox_overlap_iou(seg1['bbox'], seg2['bbox'])
            
            if iou > 0.1:
                if not masks_overlap(seg1['mask'], seg2['mask']):
                    area1 = seg1['bbox']['width'] * seg1['bbox']['height']
                    area2 = seg2['bbox']['width'] * seg2['bbox']['height']
                    
                    larger_seg = seg1 if area1 > area2 else seg2
                    smaller_seg = seg2 if area1 > area2 else seg1
                    
                    new_bbox = compute_tight_bbox(larger_seg['mask'], smaller_seg['mask'])
                    
                    if new_bbox is not None:
                        for idx, seg in enumerate(resolved_segments):
                            if seg['id'] == larger_seg['id']:
                                resolved_segments[idx]['bbox'] = new_bbox
                                resolved_segments[idx]['bbox_adjusted'] = True
                                break
    
    return resolved_segments


def estimate_rotation_mar(segments: List[Dict]) -> List[Dict]:
    """
    Estimate rotation for all segments using Minimum Area Rectangle method.
    
    Args:
        segments: List of segment dictionaries
        
    Returns:
        Segments with rotation_deg and rotation_confidence added
    """
    print("Computing rotations...")
    
    for seg in segments:
        mask = seg['mask']
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            seg['rotation_deg'] = 0.0
            seg['rotation_confidence'] = 0.0
            continue
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        if len(largest_contour) < 5:
            seg['rotation_deg'] = 0.0
            seg['rotation_confidence'] = 0.0
            continue
        
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]
        width, height = rect[1]
        
        # Adjust angle based on aspect ratio
        if width < height:
            angle = angle + 90
        
        angle = angle % 360
        
        # Confidence based on elongation
        aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
        confidence = min(aspect_ratio / 3.0, 1.0)
        
        seg['rotation_deg'] = float(angle)
        seg['rotation_confidence'] = float(confidence)
    
    return segments


def segment_and_estimate_rotations(image_path: str, device: str = "cuda") -> Dict[str, Any]:
    """
    Main entry point: Complete segmentation pipeline with rotation estimation.
    
    Args:
        image_path: Path to input image
        device: Device to run models on
        
    Returns:
        Dictionary containing:
            - segments: List of segment dictionaries with masks, bboxes, and rotations
            - img_width: Image width
            - img_height: Image height
    """
    print(f"\n{'='*80}")
    print("ENHANCED SEGMENTATION PIPELINE")
    print(f"{'='*80}\n")
    
    # Step 1: Hybrid segmentation
    segments, img_width, img_height, img_np = segment_image_hybrid(image_path, device)
    
    # Step 2: Split merged masks
    print("\nSplitting merged masks...")
    segments = split_merged_masks(segments, img_width, img_height, min_area_pixels=500)
    
    # Step 3: Aggressive spatial splitting
    print("\nApplying spatial splitting...")
    segments = aggressive_spatial_split(segments, img_width, img_height)
    
    # Step 4: Resolve overlaps
    print("\nResolving overlaps...")
    segments = resolve_bbox_overlaps(segments, img_width, img_height)
    
    # Step 5: Estimate rotations
    print("\nEstimating rotations...")
    segments = estimate_rotation_mar(segments)
    
    print(f"\n{'='*80}")
    print(f"COMPLETE: {len(segments)} objects detected")
    print(f"{'='*80}\n")
    
    return {
        'segments': segments,
        'img_width': img_width,
        'img_height': img_height
    }