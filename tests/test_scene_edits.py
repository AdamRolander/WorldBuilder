"""Manual scene edits (src/scene_edits.py): the layer the viewer saves and
the GLB exports apply."""
import json

import numpy as np
import pytest
import trimesh

from src import scene_edits as se


def _edit(**kw):
    e = {"translate": [0, 0, 0], "rotate_deg": [0, 0, 0], "scale": [1, 1, 1], "pivot": [0, 0, 0], "hidden": False}
    e.update(kw)
    return e


def test_apply_rotates_and_scales_about_the_pivot_then_translates():
    v = np.array([[2.0, 1.0, 1.0]])
    out = se.apply(v, _edit(pivot=[1, 1, 1], rotate_deg=[0, 90, 0], scale=[2, 2, 2], translate=[0, 0.5, 0]))
    # local (1,0,0) -> scaled (2,0,0) -> yaw +90° about Y -> (0,0,-2)
    np.testing.assert_allclose(out, [[1.0, 1.5, -1.0]], atol=1e-9)


def test_euler_order_matches_three_js_xyz():
    R = se.euler_xyz([30, 40, 50])
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    # three.js Euler(x, y, z, 'XYZ'): element m[0][2] = sin(y), m[1][2] = -sin(x)cos(y)
    assert np.isclose(R[0, 2], np.sin(np.radians(40)))
    assert np.isclose(R[1, 2], -np.sin(np.radians(30)) * np.cos(np.radians(40)))


def test_sanitize_drops_junk_and_clamps():
    dirty = {"objects": {"7": {"translate": [1, "x", float("nan")], "scale": [0, 5, 1e9], "hidden": "yes",
                               "__proto__": {"a": 1}},
                         "../etc": {"translate": [1, 1, 1]},
                         "9": {"translate": [0, 0, 0]}},                 # identity: not stored
             "hidden_nodes": ["room_relief_02", "../../x", 5, "a b"], "extra": "ignored"}
    clean = se.sanitize(dirty)
    assert set(clean["objects"]) == {"7"}
    e = clean["objects"]["7"]
    assert e["translate"] == [1.0, 0.0, 0.0] and e["scale"] == [1e-3, 5.0, 1e3] and e["hidden"] is False
    assert clean["hidden_nodes"] == ["room_relief_02"]
    assert se.sanitize("not a dict")["objects"] == {}


def test_edited_glb_moves_hides_and_refreshes(tmp_path):
    sc = trimesh.Scene()
    for name, x in (("001_chair", 0.0), ("002_lamp", 2.0), ("room_relief_00", 4.0), ("room_floor", 6.0)):
        sc.add_geometry(trimesh.creation.box(extents=(1, 1, 1), transform=trimesh.transformations.translation_matrix([x, 0, 0])),
                        node_name=name, geom_name=name)
    sc.export(str(tmp_path / "scene_lite.glb"))
    assert se.edited_glb(tmp_path) is None                               # no edits yet
    se.save(tmp_path, {"objects": {"1": _edit(translate=[0, 3, 0]), "2": _edit(hidden=True)},
                       "hidden_nodes": ["room_relief_00"]})
    out = se.edited_glb(tmp_path)
    got = trimesh.load(str(out), process=False)
    assert set(got.geometry) == {"001_chair", "room_floor"}
    np.testing.assert_allclose(got.geometry["001_chair"].bounds.mean(0), [0, 3, 0], atol=1e-6)
    np.testing.assert_allclose(got.geometry["room_floor"].bounds.mean(0), [6, 0, 0], atol=1e-6)
    assert json.loads((tmp_path / se.FILENAME).read_text())["version"] == 1
    assert se.edited_glb(tmp_path) == out                                # cached while nothing changed
    se.save(tmp_path, {})                                                # clearing edits removes the file
    assert not (tmp_path / se.FILENAME).exists() and se.edited_glb(tmp_path) is None


def _quad(lo, hi, axis):
    """A textured square in the plane ``axis = lo[axis]`` spanning lo..hi, uv 0..1."""
    import trimesh.visual
    a, b = [k for k in range(3) if k != axis]
    V = np.zeros((4, 3))
    V[:, axis] = lo[axis]
    for i, (u, v) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
        V[i, a] = lo[a] + u * (hi[a] - lo[a])
        V[i, b] = lo[b] + v * (hi[b] - lo[b])
    m = trimesh.Trimesh(V, [[0, 1, 2], [0, 2, 3]], process=False)
    m.visual = trimesh.visual.TextureVisuals(uv=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float))
    return m


def test_sanitize_keeps_copies_crops_nodes_and_room():
    clean = se.sanitize({
        "objects": {"40": {"source": 3}, "41": {"source": 40}, "42": {"source": True},
                    "5": {"crop": [[0, 0, 0], [1, 1, 1]]}, "6": {"crop": [[1, 0, 0], [0, 1, 1]]},
                    "1000": {"translate": [1, 0, 0]}},
        "nodes": {"room_builtin_03": {"translate": [0, 1, 0]}, "room_wall_x_min": {"translate": [1, 0, 0]},
                  "room_relief_01": {"translate": [0, 0, 0]}},
        "room": {"min": [0, 0, 0], "max": [-0.5, 0, 0]}})
    assert clean["objects"]["40"]["source"] == 3
    assert "41" not in clean["objects"]                     # a copy of a copy
    assert "42" not in clean["objects"] and "6" not in clean["objects"] and "1000" not in clean["objects"]
    assert clean["objects"]["5"]["crop"] == [[0, 0, 0], [1, 1, 1]]
    assert set(clean["nodes"]) == {"room_builtin_03"}       # walls are moved with "room"; identity dropped
    assert clean["room"] == {"min": [0, 0, 0], "max": [-0.5, 0, 0]}


def test_crop_cuts_faces_and_keeps_texture_coordinates():
    m = _quad([0, 0, 0], [0, 2, 2], axis=0)
    out = se.crop_mesh(m, [[-1, -1, -1], [1, 1, 3]])        # keep y <= 1: the lower half
    np.testing.assert_allclose(out.bounds, [[0, 0, 0], [0, 1, 2]], atol=1e-9)
    assert out.visual.uv[:, 0].max() == pytest.approx(0.5)   # u runs along y: cut, not squeezed
    np.testing.assert_allclose(out.area, 2.0)


def test_room_shrinks_by_cutting_the_texture_and_grows_by_stretching_it():
    lo, hi = np.array([0.0, 0, 0]), np.array([4.0, 2, 4])
    floor = _quad(lo, np.array([4.0, 0, 4]), axis=1)
    small = se.resize_shell(floor, lo, hi, lo, hi + [-2, 0, 0])
    assert small.bounds[1, 0] == pytest.approx(2.0)
    ucol = 0 if np.ptp(floor.visual.uv[:, 0][[0, 1]]) else 1   # which uv axis follows x
    assert small.visual.uv[:, ucol].max() == pytest.approx(0.5)
    big = se.resize_shell(floor, lo, hi, lo, hi + [2, 0, 0])
    assert big.bounds[1, 0] == pytest.approx(6.0)
    assert big.visual.uv[:, ucol].max() == pytest.approx(1.0)


def test_copies_nodes_and_room_in_the_export(tmp_path):
    sc = trimesh.Scene()
    sc.add_geometry(trimesh.creation.box(extents=(1, 1, 1)), node_name="003_chair", geom_name="003_chair")
    sc.add_geometry(trimesh.creation.box(extents=(1, 1, 1), transform=trimesh.transformations.translation_matrix([3, 0, 0])),
                    node_name="room_builtin_00_body", geom_name="room_builtin_00_body")
    sc.add_geometry(_quad(np.array([-5.0, -0.5, -5]), np.array([5.0, -0.5, 5]), axis=1), node_name="room_floor", geom_name="room_floor")
    sc.export(str(tmp_path / "scene_lite.glb"))
    (tmp_path / "3d_models").mkdir()
    layout = {"bounds_min": [-5, -0.5, -5], "bounds_max": [5, 2, 5], "floor_y": -0.5,
              "builtins": [{"min": [2.5, -0.5, -0.5], "max": [3.5, 0.5, 0.5]}]}
    (tmp_path / "3d_models" / "layout.json").write_text(json.dumps(layout))
    se.save(tmp_path, {"objects": {"40": _edit(source=3, translate=[0, 0, 2])},
                       "nodes": {"room_builtin_00": _edit(pivot=[3, 0, 0], scale=[2, 1, 1])},
                       "room": {"min": [1, 0, 0], "max": [0, 0, 0]}})
    got = trimesh.load(str(se.edited_glb(tmp_path)), process=False)
    np.testing.assert_allclose(got.geometry["040_chair"].bounds.mean(0), [0, 0, 2], atol=1e-6)
    np.testing.assert_allclose(got.geometry["003_chair"].bounds.mean(0), [0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(got.geometry["room_builtin_00_body"].extents, [2, 1, 1], atol=1e-6)
    assert got.geometry["room_floor"].bounds[0, 0] == pytest.approx(-4.0)
    lay = se.edited_layout(layout, se.load(tmp_path))
    assert lay["bounds_min"][0] == pytest.approx(-4.0)
    np.testing.assert_allclose([lay["builtins"][0]["min"][0], lay["builtins"][0]["max"][0]], [2.0, 4.0])
    assert se.copy_sources(se.load(tmp_path)) == {40: 3}


def test_extend_stretches_one_end_and_keeps_the_other():
    # a "table": a thin top at y in [0.9, 1] and a leg from 0.2 to 0.9 (it should reach 0)
    top = trimesh.creation.box(extents=(1, 0.1, 1), transform=trimesh.transformations.translation_matrix([0, 0.95, 0]))
    leg = trimesh.creation.box(extents=(0.1, 0.7, 0.1), transform=trimesh.transformations.translation_matrix([0, 0.55, 0]))
    m = trimesh.util.concatenate([top, leg])
    lo, hi = m.bounds
    e = se.sanitize({"objects": {"1": _edit(pivot=list((lo + hi) / 2),
                                            extend=[lo.tolist(), hi.tolist(), [lo[0], 0.0, lo[2]], hi.tolist()])}})["objects"]["1"]
    v = se.stretch(m.vertices, e)
    assert v[:, 1].min() == pytest.approx(0.0) and v[:, 1].max() == pytest.approx(1.0)
    top_v = v[np.asarray(m.vertices)[:, 1] >= 0.9 - 1e-9]
    np.testing.assert_allclose(top_v[:, 1].min(), 0.9)               # the top keeps its thickness
    assert se.sanitize({"objects": {"1": {"extend": [[0, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 1]]}}})["objects"] == {}
