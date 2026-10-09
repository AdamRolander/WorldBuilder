"""Synthetic-scene tests for src/placement.py.

A floor, a back wall, a table and a lamp standing on it are ray-cast into
a point map with per-pixel object ids. Objects are then deliberately
mis-posed (wrong depth, floating, tilted) and the placement stage has to
bring them back using only the point map and the masks.
"""
import numpy as np
import pytest

from src import placement as pl
from src import room_layout as rl
from src.geometry import matrix_to_quat, quat_to_matrix

K = np.array([[1.0, 0, 0.5], [0, 1.3, 0.5], [0, 0, 1.0]])
W, H = 240, 180
FLOOR_Y, WALL_Z, CEIL_Y = -1.2, 4.0, 1.4
BOXES = {                      # id -> (min, max) in the aligned frame; camera at the origin
    1: (np.array([-0.6, FLOOR_Y, 2.0]), np.array([0.6, -0.5, 3.0])),      # table
    2: (np.array([-0.12, -0.5, 2.3]), np.array([0.12, -0.15, 2.54])),     # lamp on the table
    3: (np.array([0.9, 0.2, 3.9]), np.array([1.4, 0.7, 4.0])),            # picture on the wall
}
LABELS = {1: "table", 2: "lamp", 3: "picture"}


def _rays():
    u = (np.arange(W) + 0.5) / W
    v = (np.arange(H) + 0.5) / H
    gu, gv = np.meshgrid(u, v)
    d_cv = np.stack([(gu - K[0, 2]) / K[0, 0], (gv - K[1, 2]) / K[1, 1], np.ones_like(gu)], -1)
    return d_cv @ rl.OPENCV_TO_WORLD.T          # camera-frame world == aligned frame here


def raycast_scene():
    d = _rays()
    t_best = np.full((H, W), np.inf)
    owner = np.zeros((H, W), np.int32)
    with np.errstate(divide="ignore", invalid="ignore"):
        for t in (FLOOR_Y / d[..., 1], WALL_Z / d[..., 2]):
            ok = (t > 0) & (t < t_best)
            t_best[ok] = t[ok]
        for rid, (mn, mx) in BOXES.items():
            t0 = (mn - 0) / d
            t1 = (mx - 0) / d
            near = np.nanmax(np.minimum(t0, t1), axis=-1)
            far = np.nanmin(np.maximum(t0, t1), axis=-1)
            hit = (near <= far) & (near > 0) & (near < t_best)
            t_best[hit] = near[hit]
            owner[hit] = rid
    P = d * t_best[..., None]
    return P, np.isfinite(P).all(-1), owner


def box_cloud(mn, mx, n=28):
    """Dense surface samples of a box (stands in for a SAM 3D mesh)."""
    a = np.linspace(0, 1, n)
    g1, g2 = np.meshgrid(a, a)
    g1, g2 = g1.ravel(), g2.ravel()
    faces = []
    for axis in range(3):
        for val in (0.0, 1.0):
            p = np.zeros((len(g1), 3))
            p[:, axis] = val
            p[:, (axis + 1) % 3] = g1
            p[:, (axis + 2) % 3] = g2
            faces.append(p)
    return mn + np.vstack(faces) * (mx - mn)


def to_model(points_world):
    """Inverse of geometry.world_vertices for an identity pose."""
    p = np.asarray(points_world)
    return np.stack([p[:, 0], p[:, 2], -p[:, 1]], 1)


@pytest.fixture()
def scene():
    P, valid, owner = raycast_scene()
    masks = {rid: owner == rid for rid in BOXES}
    assert all(m.sum() > 60 for m in masks.values())
    ev = pl.build_evidence(P, valid, K, np.eye(3), floor_y=FLOOR_Y, room_height=CEIL_Y - FLOOR_Y, masks=masks)
    results, verts = [], {}
    for rid, (mn, mx) in BOXES.items():
        c = 0.5 * (mn + mx)
        verts[rid] = to_model(box_cloud(mn, mx) - c)          # model centred on its own origin
        results.append({"id": rid, "label": LABELS[rid], "confidence": 0.9, "translation": c.tolist(),
                        "rotation_quaternion": [1.0, 0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]})
    return ev, results, verts


def _bounds(r, verts):
    o = pl.PlacedObject(r, verts[r["id"]])
    return o.lo, o.hi


def test_correct_poses_are_left_alone(scene):
    ev, results, verts = scene
    diag = pl.place_objects(results, verts, ev, verbose=False)
    for r in results:
        lo, hi = _bounds(r, verts)
        np.testing.assert_allclose(lo, BOXES[r["id"]][0], atol=0.03)
        np.testing.assert_allclose(hi, BOXES[r["id"]][1], atol=0.03)
        assert r["fit_iou"] > 0.8
    by = {r["id"]: r for r in results}
    assert by[1]["support"] == pl.FLOOR
    assert by[2]["support"] == "object" and by[2]["supported_by"] == 1
    assert by[3]["support"] is None            # wall-mounted: nothing under it carries it
    assert not diag["removed"]


def test_wrong_depth_is_refit_along_the_rays(scene):
    ev, results, verts = scene
    lamp = next(r for r in results if r["id"] == 2)
    k = 1.3                                   # 30 % too far and too big: same image
    lamp["translation"] = (np.array(lamp["translation"]) * k).tolist()
    lamp["scale"] = [k, k, k]
    obj = pl.PlacedObject(lamp, verts[2])
    applied = pl.refit_depth(ev, obj)
    assert applied == pytest.approx(1 / k, rel=0.03)
    np.testing.assert_allclose(obj.lo, BOXES[2][0], atol=0.04)


def test_floating_object_lands_on_its_supporter(scene):
    ev, results, verts = scene
    lamp = next(r for r in results if r["id"] == 2)
    lamp["translation"][1] += 0.08
    pl.place_objects(results, verts, ev, verbose=False)
    lo, _ = _bounds(lamp, verts)
    assert lo[1] == pytest.approx(-0.5, abs=0.02)
    assert lamp["supported_by"] == 1


def test_small_tilt_is_removed_and_large_tilt_kept(scene):
    ev, results, verts = scene
    table = next(r for r in results if r["id"] == 1)
    a = np.radians(8)
    tilt = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    table["rotation_quaternion"] = matrix_to_quat(quat_to_matrix(table["rotation_quaternion"]) @ tilt.T).tolist()
    obj = pl.PlacedObject(table, verts[1])
    assert pl.correct_upright(ev, obj)
    R = quat_to_matrix(table["rotation_quaternion"])
    assert np.max(np.abs(R[:, 1])) == pytest.approx(1.0, abs=1e-6)
    # 40°: beyond what pose noise explains
    b = np.radians(40)
    big = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    table["rotation_quaternion"] = matrix_to_quat(big.T).tolist()
    assert not pl.correct_upright(ev, pl.PlacedObject(table, verts[1]))


def test_upright_rotation_is_axis_agnostic():
    # an object whose model X axis is "up", leaning 10°
    a = np.radians(10)
    lean = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    R_obj = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.0]]) @ lean.T
    R_new, tilt = pl.upright_rotation(R_obj)
    assert tilt == pytest.approx(10, abs=1e-6)
    np.testing.assert_allclose(np.abs(R_new[0]), [0, 1, 0], atol=1e-9)
    np.testing.assert_allclose(R_new @ R_new.T, np.eye(3), atol=1e-9)


def test_same_volume_same_label_is_a_duplicate(scene):
    ev, results, verts = scene
    twin = dict(results[0], id=9)
    twin["translation"] = (np.array(twin["translation"]) + [0.05, 0, 0.05]).tolist()
    verts[9] = verts[1]
    ev.masks[9] = ev.masks[1]
    results.append(twin)
    diag = pl.place_objects(results, verts, ev, verbose=False)
    assert len(diag["duplicates"]) == 1
    assert sorted(r["id"] for r in results) == [1, 2, 3] or sorted(r["id"] for r in results) == [2, 3, 9]


def test_mesh_that_misses_its_mask_is_removed(scene):
    ev, results, verts = scene
    pic = next(r for r in results if r["id"] == 3)
    pic["translation"] = [-1.5, -0.9, 2.0]                 # nowhere near its mask
    diag = pl.place_objects(results, verts, ev, verbose=False)
    assert 3 in diag["removed"] and all(r["id"] != 3 for r in results)


def test_contained_parts_and_groups():
    def m(r0, r1, c0, c1):
        a = np.zeros((40, 40), bool)
        a[r0:r1, c0:c1] = True
        return a

    desk, top = m(5, 30, 5, 30), m(5, 12, 6, 29)
    labels = {1: "desk", 2: "desk", 3: "lamp"}
    assert pl.contained_parts({1: desk, 2: top, 3: m(6, 10, 8, 12)}, labels) == {2: 1}     # lamp: other label
    # two desks plus one mask spanning both: the spanning mask goes
    left, right, both = m(5, 30, 2, 19), m(5, 30, 21, 38), m(5, 30, 2, 38)
    out = pl.contained_parts({1: both, 2: left, 3: right}, {1: "desk", 2: "desk", 3: "desk"})
    assert list(out) == [1]


def test_structural_duplicates_spare_the_rug():
    floor = np.zeros((40, 40), bool); floor[20:, :] = True
    carpet = np.zeros((40, 40), bool); carpet[21:, 1:39] = True
    rug = np.zeros((40, 40), bool); rug[28:34, 10:25] = True
    out = pl.structural_duplicates({1: carpet, 2: rug}, {"floor": floor})
    assert out == {1: "floor"}


# ---------------------------------------------------------------------------
# Second pass: settling, walls, visibility, silhouette search
# ---------------------------------------------------------------------------

ROOM = {"min": [-2.0, FLOOR_Y, -0.1], "max": [2.0, CEIL_Y, WALL_Z],
        "sources": {"x_min": "extent", "x_max": "extent", "z_min": "extent", "z_max": "wall(9999)"},
        "ceiling_y": CEIL_Y}


def _objs(results, verts):
    return {r["id"]: pl.PlacedObject(r, verts[r["id"]]) for r in results}


def test_unsupported_object_settles_onto_what_is_under_it(scene):
    ev, results, verts = scene
    o = _objs(results, verts)
    o[1].r["support"] = pl.FLOOR
    o[2].translate([0, 0.06, 0])                       # lamp floats; no contact was found for it
    o[3].r["support"] = None
    settled = pl.settle_unsupported(ev, list(o.values()), ROOM)
    assert 2 in settled and o[2].r["supported_by"] == 1
    assert o[2].lo[1] == pytest.approx(-0.5, abs=0.02)
    # the picture hangs 1.4 above the floor and is 0.5 tall: far beyond a settling gap
    assert 3 not in settled and o[3].lo[1] == pytest.approx(0.2, abs=1e-6)


def test_stretch_down_keeps_the_top(scene):
    ev, results, verts = scene
    lamp = _objs(results, verts)[2]
    top, bottom = float(lamp.hi[1]), float(lamp.lo[1])
    assert lamp.stretch_down(1.5)
    assert lamp.hi[1] == pytest.approx(top, abs=1e-6)
    assert lamp.lo[1] == pytest.approx(top - 1.5 * (top - bottom), abs=1e-6)


def test_object_through_an_observed_wall_comes_back_inside(scene):
    ev, results, verts = scene
    o = _objs(results, verts)
    o[3].slide_along_rays(1.12)                        # too deep: pokes through the back wall
    assert o[3].hi[2] > WALL_Z + 0.2
    assert pl.keep_inside_walls(ev, list(o.values()), ROOM) == 1
    assert o[3].hi[2] <= WALL_Z + 1e-3
    np.testing.assert_allclose(o[3].lo, BOXES[3][0], atol=0.06)   # and it is where the photo has it


def test_visible_object_is_moved_in_front_of_what_hides_it(scene):
    ev, results, verts = scene
    o = _objs(results, verts)
    o[2].r["support"] = "object"
    o[1].translate([0, 0.3, 0])                        # the table's mesh now stands in front of the lamp
    z_before = float(o[2].lo[2])
    moved = pl.enforce_visibility(ev, list(o.values()))
    assert 2 in moved and o[2].lo[2] < z_before
    # an object pinned to the floor by an observed contact is not slid
    o2 = _objs(*scene[1:])
    o2[2].r["support"] = pl.FLOOR
    o2[1].translate([0, 0.3, 0])
    assert 2 not in pl.enforce_visibility(ev, list(o2.values()))


def test_silhouette_search_recovers_a_shifted_object_and_respects_occlusion(scene):
    from src import pose_fit
    ev, results, verts = scene
    o = _objs(results, verts)
    good = pose_fit.fit_score(ev, o[2].fast_verts, ev.masks[2])
    o[2].translate([0.1, 0.05, 0])
    bad = pose_fit.fit_score(ev, o[2].fast_verts, ev.masks[2])
    assert good > 0.8 > bad
    assert pose_fit.refine_pose(ev, o[2]) is not None
    np.testing.assert_allclose(o[2].lo[:2], BOXES[2][0][:2], atol=0.03)
    # pushed far away and scaled up, the silhouette is unchanged but the depth is wrong
    o[3].slide_along_rays(0.5)
    assert pose_fit.fit_score(ev, o[3].fast_verts, ev.masks[3]) < 0.2
    # the table is partly hidden by the lamp standing on it: that must not count against it
    assert pose_fit.fit_score(ev, o[1].fast_verts, ev.masks[1]) > 0.6
    assert pose_fit.hidden_fraction(ev, o[1].fast_verts, ev.masks[1]) > 0.0


def test_placement_with_room_and_image_runs_all_steps(scene):
    ev, results, verts = scene
    img = np.full(ev.hw + (3,), 128, np.uint8)
    diag = pl.place_objects(results, verts, ev, verbose=False, room=ROOM, image_small=img)
    for key in ("pose_refined", "settled", "instances", "moved_in_front"):
        assert key in diag
    assert len(results) == 3 and not diag["removed"]
