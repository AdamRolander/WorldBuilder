"""Gravity refinement, robust floor height and the stage-3 cache."""
import json

import numpy as np

from src import room_layout as rl
from src import stage3_cache
from tests.test_room_layout import FLOOR_Y, raycast_room, rot_x, rot_y


def test_gravity_refinement_recovers_up_from_a_tilted_start():
    R = rot_x(-14.0) @ rot_y(25.0)
    P_cam, colors, depth, P_true, hit = raycast_room(R)
    valid = np.isfinite(P_cam).all(-1)
    N = rl.compute_normals(P_cam, valid)
    true_up = R @ np.array([0, 1.0, 0])
    start = rot_x(6.0) @ true_up                        # start 6° off
    up, support = rl.refine_gravity(N, valid, start)
    assert support > 0.8
    assert np.degrees(np.arccos(min(1.0, float(up @ true_up)))) < 0.5


def test_floor_mask_spanning_two_levels_does_not_tilt_or_raise_the_floor():
    """The bathroom case: the floor mask includes a raised tray on one side.
    A plane through both levels is tilted; the layout must not be."""
    R = rot_x(-20.0)
    P_cam, colors, depth, P_true, hit = raycast_room(R)
    valid = np.isfinite(P_cam).all(-1)
    floor = hit == "floor"
    cols = np.arange(valid.shape[1])[None, :] > 0.6 * valid.shape[1]
    tray = floor & cols
    P = P_cam.copy()
    P[tray] += 0.12 * (R @ np.array([0, 1.0, 0]))        # raised 12 cm
    plane = rl.estimate_floor(P, valid, floor, rng=np.random.default_rng(0))
    layout = rl.estimate_layout(P, valid, structural_masks={"floor": floor}, rng=np.random.default_rng(0))
    up_true = R @ np.array([0, 1.0, 0])
    np.testing.assert_allclose(layout.R_total @ up_true, [0, 1, 0], atol=1.5e-2)
    assert abs(layout.floor_y - FLOOR_Y) < 0.03          # the lower level, not the average
    assert any("gravity refined" in n for n in layout.notes)
    assert plane is not None


def test_stage3_cache_roundtrip(tmp_path):
    models = tmp_path / "scene" / "3d_models"
    rng = np.random.default_rng(0)
    P = rng.normal(size=(300, 500, 3)).astype(np.float32)
    valid = rng.random((300, 500)) > 0.1
    Kn = np.array([[1.1, 0, 0.5], [0, 0.9, 0.5], [0, 0, 1.0]])
    stage3_cache.save_pointmap(models, P, valid, Kn, (3000, 5000))
    pm = stage3_cache.load_pointmap(models)
    assert pm["points"].shape == (300, 500, 3) and pm["image_hw"] == (3000, 5000)
    np.testing.assert_allclose(pm["points"][valid], P[valid], atol=1e-6)
    assert not pm["points"][~valid].any()
    # larger than MAX_SIDE: subsampled by index, never interpolated
    big = np.arange(2048 * 1536 * 3, dtype=np.float32).reshape(1536, 2048, 3)
    stage3_cache.save_pointmap(models, big, np.ones((1536, 2048), bool), Kn, (1536, 2048))
    pm = stage3_cache.load_pointmap(models)
    assert max(pm["valid"].shape) == stage3_cache.MAX_SIDE
    assert np.isin(pm["points"][..., 0], big[..., 0]).all()
    # raw.json: paths follow the scene directory when it is moved
    (tmp_path / "scene" / "masks").mkdir()
    (tmp_path / "scene" / "masks" / "001_chair_i0.png").write_bytes(b"")
    stage3_cache.save_raw(models, [{"id": 1, "label": "chair", "model_path": "/old/place/stage3/001_chair.ply",
                                    "mask_path": "/old/place/masks/001_chair_i0.png"}], [], "photo.jpg")
    raw = stage3_cache.load_raw(models)
    assert raw["objects"][0]["model_path"] == str(models / "stage3" / "001_chair.ply")
    assert raw["objects"][0]["mask_path"] == str(tmp_path / "scene" / "masks" / "001_chair_i0.png")
    assert json.loads((models / "stage3" / "raw.json").read_text())["source_image"] == "photo.jpg"
