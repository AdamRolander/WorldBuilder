import sys
from pathlib import Path
import warnings

# Add sam3_repo to Python path
project_root = Path(__file__).parent.parent
sam3_path = project_root / "sam3_repo"
if str(sam3_path) not in sys.path:
    sys.path.insert(0, str(sam3_path))

import torch
import numpy as np
from PIL import Image
from typing import List, Dict

# Import SAM 3 components
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class SAM3Segmenter:
    """Text-prompted segmentation with SAM 3."""

    # Singleton model handle — see SAM3DReconstructor for the rationale.
    _shared_model = None
    _shared_processor = None
    _shared_key = None  # (checkpoint_path, device) — reload if either changes.

    def __init__(self, checkpoint_path: str = None, device: str = None):
        """Initialize (or reuse) SAM 3 model."""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.device = device

        if checkpoint_path is None:
            checkpoint_path = str(Path(__file__).parent.parent / "checkpoints/sam3.pt")

        key = (checkpoint_path, device)
        if SAM3Segmenter._shared_model is not None and SAM3Segmenter._shared_key == key:
            self.model = SAM3Segmenter._shared_model
            self.processor = SAM3Segmenter._shared_processor
            print(f"\n✓ SAM 3 (reusing cached model on {device})")
        else:
            print(f"\nLoading SAM 3...")
            print(f"  Checkpoint: {checkpoint_path}")
            self.model = build_sam3_image_model(
                checkpoint_path=checkpoint_path,
                device=device,
                eval_mode=True,
                load_from_HF=False,
                enable_segmentation=True,
                compile=False
            )
            self.processor = Sam3Processor(self.model)
            SAM3Segmenter._shared_model = self.model
            SAM3Segmenter._shared_processor = self.processor
            SAM3Segmenter._shared_key = key
            print("✓ SAM 3 loaded successfully")

    def _to_device(self, device: str):
        """Park / wake SAM 3 model to manage shared GPU memory across the
        three-model pipeline. The processor wraps the model, so moving
        the model is enough — processor state is metadata."""
        import gc, torch
        self.model.to(device)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def segment_objects(
        self,
        image_path: str,
        object_descriptions: List[Dict[str, str]],
        output_dir: str = "outputs/masks",
        score_threshold: float = 0.35,
        min_mask_pixels: int = 100,
    ) -> List[Dict]:
        """Segment all instances of each described object type.

        SAM 3 returns multiple masks per text prompt (one per detected instance).
        We keep every instance whose score exceeds `score_threshold`, producing
        one result dict per instance (with a unique sequential id).
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\nLoading image: {image_path}")
        image = Image.open(image_path).convert('RGB')
        inference_state = self.processor.set_image(image)

        results = []
        next_id = 1

        print(f"\nSegmenting {len(object_descriptions)} object types...")
        print("=" * 60)

        for i, obj in enumerate(object_descriptions, 1):
            label = obj['label']
            text_prompt = obj.get('description', label)
            expected = obj.get('expected_instances', 'multiple')
            print(f"\n[{i}/{len(object_descriptions)}] {label} (expected: {expected})")

            try:
                output = self.processor.set_text_prompt(
                    state=inference_state, prompt=text_prompt
                )
                masks = output["masks"]
                scores = output["scores"]

                if (len(masks) == 0 or float(scores.max()) < score_threshold) \
                        and label != text_prompt:
                    print(f"  ↻ retrying with label only: '{label}'")
                    output = self.processor.set_text_prompt(
                        state=inference_state, prompt=label)
                    masks = output["masks"]
                    scores = output["scores"]
                    if isinstance(masks, torch.Tensor): masks = masks.cpu().numpy()
                    if isinstance(scores, torch.Tensor): scores = scores.cpu().numpy()
                    if len(masks) == 0:
                        print("  ⚠️  Still no masks")
                        continue

                if isinstance(masks, torch.Tensor):
                    masks = masks.cpu().numpy()
                if isinstance(scores, torch.Tensor):
                    scores = scores.cpu().numpy()

                valid = np.where(scores > score_threshold)[0]
                if len(valid) == 0:
                    valid = np.array([int(np.argmax(scores))])
                    print(f"  ⚠️  No masks above {score_threshold}; using best "
                          f"(score={scores[valid[0]]:.2f})")
                else:
                    valid = valid[np.argsort(-scores[valid])]

                if expected == 'single':
                    valid = valid[:1]
                    print(f"  ✓ expected=single → keeping top mask only")
                else:
                    # NMS: drop instances whose mask overlaps (IoU > 0.5) a
                    # higher-confidence mask of the same label.
                    kept_idxs = []
                    kept_masks = []
                    for ci in valid:
                        cm = masks[ci]
                        if cm.ndim == 3: cm = cm[0]
                        cm_bin = cm > 0.5
                        overlap = False
                        for km in kept_masks:
                            inter = np.logical_and(cm_bin, km).sum()
                            if inter == 0: continue
                            iou = inter / np.logical_or(cm_bin, km).sum()
                            if iou > 0.5:
                                overlap = True; break
                        if not overlap:
                            kept_idxs.append(ci)
                            kept_masks.append(cm_bin)
                    dropped = len(valid) - len(kept_idxs)
                    valid = np.array(kept_idxs)
                    if dropped:
                        print(f"  ✓ {len(valid)} instance(s) after NMS (dropped {dropped} duplicate)")
                    else:
                        print(f"  ✓ {len(valid)} instance(s)")

                kept = 0
                for inst_idx in valid:
                    mask = masks[inst_idx]
                    score = float(scores[inst_idx])
                    if mask.ndim == 3:
                        mask = mask[0]

                    mask_binary = (mask > 0.5).astype(np.uint8)
                    if mask_binary.sum() < min_mask_pixels:
                        continue

                    ys, xs = np.where(mask_binary > 0)
                    bbox = [int(xs.min()), int(ys.min()),
                            int(xs.max()), int(ys.max())]

                    safe_label = label.replace(' ', '_').replace('/', '_')
                    if len(valid) > 1:
                        fname = f"{next_id:03d}_{safe_label}_i{kept}.png"
                    else:
                        fname = f"{next_id:03d}_{safe_label}.png"
                    mask_path = output_path / fname
                    Image.fromarray(mask_binary * 255).save(mask_path)

                    results.append({
                        'id': next_id,
                        'label': label,
                        'description': text_prompt,
                        'instance_idx': kept,
                        'mask_path': str(mask_path),
                        'mask_array': mask_binary,
                        'bbox': bbox,
                        'confidence': score,
                    })
                    print(f"    [{next_id:3d}] instance {kept}: "
                          f"conf={score:.2f} bbox={bbox}")
                    next_id += 1
                    kept += 1

            except Exception as e:
                print(f"  ❌ Error: {str(e)[:200]}")
                continue

        # Cross-label NMS: a single physical object can be picked up by
        # multiple Gemini labels ("desk" + "wooden table" + "school desk"
        # all firing on the same desk). Within-label NMS above can't see
        # those. Sort by confidence, drop any later instance that overlaps
        # a kept one.
        print("\nCross-label NMS pass...")
        results.sort(key=lambda r: -r['confidence'])
        deduped = []
        for r in results:
            rm = r['mask_array'].astype(bool)
            is_dup = False
            for k in deduped:
                km = k['mask_array'].astype(bool)
                inter = np.logical_and(rm, km).sum()
                if inter == 0:
                    continue
                union = np.logical_or(rm, km).sum()
                if union > 0 and inter / union > 0.5:
                    is_dup = True
                    print(f"  ✗ dropping '{r['label']}' (conf={r['confidence']:.2f}) "
                          f"— duplicate of '{k['label']}' (conf={k['confidence']:.2f})")
                    break
            if not is_dup:
                deduped.append(r)
        dropped = len(results) - len(deduped)
        # Renumber ids contiguously for downstream stages
        for i, r in enumerate(deduped, 1):
            r['id'] = i
        results = deduped

        print("\n" + "=" * 60)
        if dropped:
            print(f"✓ Removed {dropped} cross-label duplicate(s)")
        print(f"✓ Segmented {len(results)} total instances from "
              f"{len(object_descriptions)} object types")
        return results
