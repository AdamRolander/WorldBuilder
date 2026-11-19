"""Which placed objects are failed reconstructions to drop."""
from src.placement import failed_reconstructions


def test_mesh_that_misses_its_mask_is_dropped():
    out = failed_reconstructions([{"id": 1, "fit_iou": 0.02}, {"id": 2, "fit_iou": 0.6}])
    assert list(out) == [1]


def test_box_stand_in_is_kept_while_something_rests_on_it():
    recs = [{"id": 1, "fit_iou": 0.02, "proxy": "box"}, {"id": 2, "fit_iou": 0.7, "supported_by": 1}]
    assert failed_reconstructions(recs) == {}


def test_box_stand_in_that_carries_nothing_is_dropped():
    recs = [{"id": 1, "fit_iou": 0.02, "proxy": "box"}, {"id": 2, "fit_iou": 0.7, "supported_by": None}]
    assert list(failed_reconstructions(recs)) == [1]


def test_shared_instance_is_not_judged_here():
    assert failed_reconstructions([{"id": 1, "fit_iou": 0.02, "instance_of": 5}]) == {}
