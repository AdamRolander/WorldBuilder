import numpy as np

from src.segmentation import (
    apply_single_prior,
    bbox_to_cxcywh_norm,
    clean_mask,
    mask_iou,
    nms_masks,
    short_phrase,
)


def _blob(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_mask_iou():
    a = _blob(10, 10, 0, 5, 0, 10)
    b = _blob(10, 10, 0, 10, 0, 10)
    assert np.isclose(mask_iou(a, b), 0.5)
    assert mask_iou(a, ~b) == 0.0


def test_nms_keeps_highest_and_records_suppressed():
    a = {"label": "desk", "confidence": 0.9, "mask_array": _blob(20, 20, 0, 10, 0, 20)}
    b = {"label": "table", "confidence": 0.7, "mask_array": _blob(20, 20, 0, 11, 0, 20)}
    c = {"label": "chair", "confidence": 0.8, "mask_array": _blob(20, 20, 15, 20, 0, 5)}
    kept = nms_masks([b, c, a], iou_threshold=0.5)
    assert [k["label"] for k in kept] == ["desk", "chair"]
    assert kept[0]["suppressed"] == ["table"]


def test_clean_mask_drops_noise_but_keeps_real_multipart():
    m = _blob(50, 50, 0, 40, 0, 40)
    m[48, 48] = True                       # one speck
    cleaned = clean_mask(m)
    assert cleaned.sum() == 40 * 40
    two = _blob(50, 50, 0, 20, 0, 50) | _blob(50, 50, 30, 50, 0, 50)  # 50/50 split
    assert clean_mask(two).sum() == two.sum()


def test_single_prior_is_soft():
    scores = np.array([0.9, 0.88, 0.4])
    assert list(apply_single_prior(scores, 0.85)) == [0, 1]
    assert list(apply_single_prior(scores, 1.0)) == [0]
    assert list(apply_single_prior(np.array([]))) == []


def test_short_phrase():
    assert short_phrase("", "sofa") is None
    assert short_phrase("Sofa", "sofa") is None
    assert short_phrase("green metal storage cabinet with multiple drawers and wheels", "cabinet") \
        == "green metal storage cabinet with multiple"
    assert short_phrase("red mug", "mug") == "red mug"


def test_bbox_norm():
    assert bbox_to_cxcywh_norm([0, 0, 100, 50], 200, 100) == [0.25, 0.25, 0.5, 0.5]
