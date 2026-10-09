"""Stage 2: text-prompted instance segmentation with SAM 3.

Given the cleaned stage-1 object list, produce one binary mask per physical
object instance. This is the hand-off that decides what SAM 3D gets to
reconstruct, so most of the "the local VLM misses objects" complaints land
here rather than in the VLM itself. Design notes (see docs/AUDIT.md §2 for
the measurements behind them):

* ``Sam3Processor`` pre-filters detections at its own ``confidence_threshold``
  (default 0.5) *before* we ever see them. The old code then applied a 0.35
  threshold to what was left, which did nothing. We now push our threshold
  into the processor so the 0.3–0.5 band is actually available, and rely on
  NMS to remove the duplicates a lower threshold produces.
* SAM 3 is trained on short noun phrases. The old code prompted with the
  VLM's free-text *description* first ("Green metal storage cabinet with
  multiple drawers and wheels") and only fell back to the label if nothing
  came back. We now query the label and a shortened description and take
  the union (``prompt_strategy="union"``), which costs one extra forward
  pass per object type on an already-encoded image.
* ``expected_instances == "single"`` used to hard-truncate to the top mask.
  The local VLM marks nearly everything "single", so a room with three
  chairs got one. It is now a soft prior: extra instances survive if their
  score is within ``single_keep_ratio`` of the best one.
* If the VLM supplied a ``bbox_2d`` and both text prompts fail, the box is
  used as a geometric prompt (``Sam3Processor.add_geometric_prompt``). This
  is the fallback that lets a grounding-capable local VLM recover objects
  SAM 3's text encoder doesn't know.
* Masks with several disconnected fragments are trimmed to their dominant
  component when the fragments are small noise, and masks covering most of
  the frame (a "floor" that slipped through) are rejected.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

# Vendored SAM 3 clone lives at <repo>/sam3_repo (README §1).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SAM3_PATH = _PROJECT_ROOT / "sam3_repo"
if str(_SAM3_PATH) not in sys.path:
    sys.path.insert(0, str(_SAM3_PATH))

import torch  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")


# --------------------------------------------------------------------------
# Pure-numpy helpers (unit-tested without a GPU)
# --------------------------------------------------------------------------

def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def nms_masks(entries: List[Dict], iou_threshold: float = 0.5,
              key: str = "mask_array") -> List[Dict]:
    """Greedy NMS over binary masks. Entries need ``confidence`` and ``key``.
    Returns the survivors sorted by descending confidence; each survivor
    gains ``suppressed`` (labels of what it absorbed) for diagnostics."""
    ordered = sorted(entries, key=lambda e: -float(e.get("confidence", 0.0)))
    kept: List[Dict] = []
    for e in ordered:
        m = e[key].astype(bool)
        dup_of = None
        for k in kept:
            if mask_iou(m, k[key].astype(bool)) > iou_threshold:
                dup_of = k
                break
        if dup_of is None:
            e.setdefault("suppressed", [])
            kept.append(e)
        else:
            dup_of.setdefault("suppressed", []).append(e.get("label", "?"))
    return kept


def clean_mask(mask: np.ndarray, dominant_fraction: float = 0.85) -> np.ndarray:
    """Drop small disconnected fragments when one component clearly dominates.
    Keeps genuinely multi-part masks (e.g. a chair seen through a table)."""
    try:
        from scipy import ndimage
    except ImportError:  # pragma: no cover
        return mask
    mask = mask.astype(bool)
    labels, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
    biggest = int(np.argmax(sizes)) + 1
    if sizes[biggest - 1] / sizes.sum() >= dominant_fraction:
        return labels == biggest
    return mask


def apply_single_prior(scores: np.ndarray, keep_ratio: float = 0.85) -> np.ndarray:
    """Indices to keep for an ``expected_instances == 'single'`` label.
    Always keeps the best; keeps others only if nearly as confident."""
    if len(scores) == 0:
        return np.array([], dtype=int)
    order = np.argsort(-scores)
    top = scores[order[0]]
    return order[scores[order] >= keep_ratio * top]


def short_phrase(description: str, label: str, max_words: int = 6) -> Optional[str]:
    """A second, more specific prompt derived from the VLM description.
    Returns None when it would add nothing over the label."""
    d = (description or "").strip().rstrip(".").lower()
    if not d or d == label.lower():
        return None
    words = d.split()
    if len(words) > max_words:
        # Keep the head noun phrase: last max_words tokens usually end in the noun.
        d = " ".join(words[:max_words])
    return d if d != label.lower() else None


def bbox_to_cxcywh_norm(bbox_xyxy: Sequence[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    return [((x1 + x2) / 2) / width, ((y1 + y2) / 2) / height,
            (x2 - x1) / width, (y2 - y1) / height]


# --------------------------------------------------------------------------
# SAM 3 wrapper
# --------------------------------------------------------------------------

class SAM3Segmenter:
    """Text-prompted segmentation with SAM 3 (process-wide singleton model)."""

    _shared_model = None
    _shared_processor = None
    _shared_key = None  # (checkpoint_path, device)

    def __init__(self, checkpoint_path: Optional[str] = None,
                 device: Optional[str] = None):
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if checkpoint_path is None:
            checkpoint_path = str(_PROJECT_ROOT / "checkpoints" / "sam3.pt")

        key = (checkpoint_path, device)
        if SAM3Segmenter._shared_model is not None and SAM3Segmenter._shared_key == key:
            self.model = SAM3Segmenter._shared_model
            self.processor = SAM3Segmenter._shared_processor
            print(f"\n✓ SAM 3 (reusing cached model on {device})")
            return

        print(f"\nLoading SAM 3...\n  Checkpoint: {checkpoint_path}")
        self.model = build_sam3_image_model(
            checkpoint_path=checkpoint_path, device=device, eval_mode=True,
            load_from_HF=False, enable_segmentation=True, compile=False,
        )
        self.processor = Sam3Processor(self.model, device=device)
        SAM3Segmenter._shared_model = self.model
        SAM3Segmenter._shared_processor = self.processor
        SAM3Segmenter._shared_key = key
        print("✓ SAM 3 loaded successfully")

    # -- VRAM management ----------------------------------------------------
    def _to_device(self, device: str):
        """Park/wake the model so SAM 3 and SAM 3D never share the GPU."""
        import gc
        self.model.to(device)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # -- low-level queries ---------------------------------------------------
    @staticmethod
    def _unpack(state: Dict):
        masks = state["masks"]
        scores = state["scores"]
        boxes = state.get("boxes")
        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.detach().cpu().numpy()
        masks = np.asarray(masks)
        if masks.ndim == 4:      # (N, 1, H, W)
            masks = masks[:, 0]
        return masks.astype(bool), np.asarray(scores, dtype=float), boxes

    def query_text(self, state: Dict, prompt: str):
        with torch.inference_mode():
            out = self.processor.set_text_prompt(state=state, prompt=prompt)
        return self._unpack(out)

    def query_box(self, state: Dict, bbox_xyxy: Sequence[float], width: int, height: int):
        """Geometric (box) prompt with no text. Resets prompts afterwards so
        the box does not leak into the next label's text query."""
        self.processor.reset_all_prompts(state)
        try:
            with torch.inference_mode():
                out = self.processor.add_geometric_prompt(
                    box=bbox_to_cxcywh_norm(bbox_xyxy, width, height),
                    label=True, state=state)
            return self._unpack(out)
        finally:
            self.processor.reset_all_prompts(state)

    # -- main entry point ----------------------------------------------------
    def segment_objects(
        self,
        image_path: str,
        object_descriptions: List[Dict],
        output_dir: str = "outputs/masks",
        score_threshold: float = 0.3,
        min_mask_pixels: int = 100,
        max_mask_fraction: float = 0.6,
        prompt_strategy: str = "union",     # "union" | "label" | "description"
        single_keep_ratio: float = 0.85,    # 1.0 reproduces the old hard top-1
        max_instances_per_label: int = 12,
        nms_iou: float = 0.5,
        use_box_fallback: bool = True,
    ) -> List[Dict]:
        """Segment every instance of every described object type.

        Returns one dict per instance with ``id, label, description,
        instance_idx, mask_path, mask_array, bbox, confidence, prompt_used,
        source``. Ids are contiguous after cross-label NMS.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\nLoading image: {image_path}")
        image = Image.open(image_path).convert("RGB")
        W, H = image.size
        max_pixels = max_mask_fraction * W * H

        # Our threshold, not the processor's default 0.5 (see module docstring).
        self.processor.confidence_threshold = float(score_threshold)
        with torch.inference_mode():
            state = self.processor.set_image(image)

        candidates: List[Dict] = []
        print(f"\nSegmenting {len(object_descriptions)} object types "
              f"(strategy={prompt_strategy}, thr={score_threshold})")
        print("=" * 60)

        for i, obj in enumerate(object_descriptions, 1):
            label = str(obj["label"]).strip()
            description = str(obj.get("description", "") or "")
            expected = obj.get("expected_instances", "multiple")
            bbox_hint = obj.get("bbox_2d")
            print(f"\n[{i}/{len(object_descriptions)}] {label} (expected: {expected})")

            prompts: List[str] = []
            if prompt_strategy in ("union", "label"):
                prompts.append(label)
            if prompt_strategy in ("union", "description"):
                alt = short_phrase(description, label)
                if alt:
                    prompts.append(alt)
            if not prompts:
                prompts = [label]

            per_label: List[Dict] = []
            try:
                for p in prompts:
                    masks, scores, boxes = self.query_text(state, p)
                    for m, s in zip(masks, scores, strict=False):
                        per_label.append({"label": label, "description": description,
                                          "mask_array": m, "confidence": float(s),
                                          "prompt_used": p, "source": "text"})
                    print(f"  '{p}': {len(scores)} candidate(s)"
                          + (f", best={scores.max():.2f}" if len(scores) else ""))

                if not per_label and use_box_fallback and bbox_hint:
                    masks, scores, boxes = self.query_box(state, bbox_hint, W, H)
                    if len(scores):
                        j = int(np.argmax(scores))
                        per_label.append({"label": label, "description": description,
                                          "mask_array": masks[j], "confidence": float(scores[j]),
                                          "prompt_used": f"box{[round(v) for v in bbox_hint]}",
                                          "source": "box"})
                        print(f"  ↳ box fallback recovered 1 mask (score={scores[j]:.2f})")
                    else:
                        print("  ↳ box fallback found nothing")
            except Exception as e:  # keep going; one bad label must not kill the scene
                print(f"  ❌ Error: {str(e)[:200]}")
                continue

            if not per_label:
                print("  ⚠️  no masks")
                continue

            # Size sanity, fragment clean-up.
            filtered = []
            for c in per_label:
                m = clean_mask(c["mask_array"])
                area = int(m.sum())
                if area < min_mask_pixels:
                    continue
                if area > max_pixels:
                    print(f"  ✗ rejecting a mask covering {100 * area / (W * H):.0f}% of the frame")
                    continue
                c["mask_array"] = m
                filtered.append(c)
            if not filtered:
                print("  ⚠️  all masks rejected by size filters")
                continue

            # Within-label NMS merges the two prompts' views of the same object.
            filtered = nms_masks(filtered, iou_threshold=nms_iou)

            if expected == "single":
                keep = apply_single_prior(np.array([c["confidence"] for c in filtered]),
                                          keep_ratio=single_keep_ratio)
                filtered = [filtered[k] for k in keep]
                print(f"  ✓ single prior → {len(filtered)} instance(s)")
            else:
                filtered = filtered[:max_instances_per_label]
                print(f"  ✓ {len(filtered)} instance(s) after NMS")
            candidates.extend(filtered)

        # Cross-label NMS: "desk" + "wooden table" + "workbench" on one object.
        print("\nCross-label NMS pass...")
        before = len(candidates)
        results = nms_masks(candidates, iou_threshold=nms_iou)
        for r in results:
            for s in r.get("suppressed", []):
                if s != r["label"]:
                    print(f"  ✗ '{s}' merged into '{r['label']}' (conf={r['confidence']:.2f})")

        # Stable ordering (by label then confidence) keeps ids meaningful.
        results.sort(key=lambda r: (r["label"], -r["confidence"]))
        per_label_counter: Dict[str, int] = {}
        for new_id, r in enumerate(results, 1):
            k = per_label_counter.get(r["label"], 0)
            per_label_counter[r["label"]] = k + 1
            m = r["mask_array"].astype(np.uint8)
            ys, xs = np.where(m)
            r["id"] = new_id
            r["instance_idx"] = k
            r["bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            r["area_px"] = int(m.sum())
            safe = r["label"].replace(" ", "_").replace("/", "_")
            fname = f"{new_id:03d}_{safe}_i{k}.png" if k or per_label_counter else f"{new_id:03d}_{safe}.png"
            mask_path = output_path / fname
            Image.fromarray(m * 255).save(mask_path)
            r["mask_path"] = str(mask_path)
            r["mask_array"] = m
            r.pop("suppressed", None)

        print("\n" + "=" * 60)
        if before - len(results):
            print(f"✓ Removed {before - len(results)} cross-label duplicate(s)")
        print(f"✓ Segmented {len(results)} total instances from "
              f"{len(object_descriptions)} object types")
        return results
