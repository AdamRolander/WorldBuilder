"""Manual scene edits (src/scene_edits.py): the layer the viewer saves and
the GLB exports apply."""
import json

import numpy as np
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
