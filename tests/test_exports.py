"""Lite-mesh baking and the MuJoCo export, on synthetic scenes (CPU)."""
import json

import numpy as np
import pytest
import trimesh
from PIL import Image

from src import mesh_bake


def _two_tone_sphere(subdivisions=5):
    m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=0.5)
    col = np.where(m.vertices[:, [1]] > 0, [[220, 30, 30, 255]], [[30, 30, 220, 255]]).astype(np.uint8)
    return trimesh.Trimesh(vertices=m.vertices, faces=m.faces, vertex_colors=col, process=False)


def test_lite_mesh_keeps_shape_and_colour():
    pytest.importorskip("fast_simplification")
    pytest.importorskip("xatlas")
    src = _two_tone_sphere()
    lite = mesh_bake.lite_mesh(src, target_faces=600, tex_size=256)
    assert len(lite.faces) <= 700 < len(src.faces)
    np.testing.assert_allclose(lite.extents, src.extents, atol=0.03)
    tex = np.asarray(lite.visual.material.baseColorTexture.convert("RGB")).astype(int)
    uv = lite.visual.uv
    px = np.clip((uv[:, 0] * tex.shape[1]).astype(int), 0, tex.shape[1] - 1)
    py = np.clip(((1 - uv[:, 1]) * tex.shape[0]).astype(int), 0, tex.shape[0] - 1)
    c = tex[py, px]
    top, bottom = lite.vertices[:, 1] > 0.15, lite.vertices[:, 1] < -0.15
    assert np.mean(c[top, 0] > c[top, 2]) > 0.95          # red hemisphere stays red
    assert np.mean(c[bottom, 2] > c[bottom, 0]) > 0.95    # blue stays blue


def test_texture_budget_scales_with_size():
    assert mesh_bake.texture_budget(0.05, 5.0) == 256
    assert mesh_bake.texture_budget(2.5, 5.0) == 1024
    assert mesh_bake.texture_budget(1.0, 5.0) in (256, 512)


def _write_scene(tmp_path):
    """A 4×3×5 room with a table on the floor, a box on the table, a box on an
    unmodelled surface and a picture on the wall, in the aligned frame
    (X left, Y up, Z forward; floor at y = -1)."""
    scene = tmp_path / "scene_q"
    (scene / "3d_models").mkdir(parents=True)
    tex = Image.fromarray(np.full((8, 8, 3), 128, np.uint8))

    def box(lo, hi):
        m = trimesh.creation.box(extents=np.subtract(hi, lo))
        m.apply_translation(np.add(lo, hi) / 2)
        uv = np.random.default_rng(0).random((len(m.vertices), 2))
        m.visual = trimesh.visual.TextureVisuals(uv=uv, material=trimesh.visual.material.PBRMaterial(baseColorTexture=tex))
        return m

    sc = trimesh.Scene()
    objs = [
        (1, "table", [-0.5, -1.0, 2.0], [0.5, -0.3, 3.0], "floor", None),
        (2, "box", [-0.1, -0.3, 2.4], [0.1, -0.1, 2.6], "object", 1),
        (3, "mug", [1.2, -0.2, 2.0], [1.3, -0.1, 2.1], "surface", None),
        (4, "picture", [-0.3, 0.5, 3.95], [0.3, 0.9, 4.0], None, None),
    ]
    results = []
    for rid, label, lo, hi, support, sup_by in objs:
        name = f"{rid:03d}_{label}"
        sc.add_geometry(box(lo, hi), node_name=name, geom_name=name)
        results.append({"id": rid, "label": label, "support": support, "supported_by": sup_by, "status": "ok"})
    quad = trimesh.Trimesh(vertices=[[-2, -1, 0], [2, -1, 0], [2, -1, 4], [-2, -1, 4]], faces=[[0, 2, 1], [0, 3, 2]],
                           process=False)
    quad.visual = trimesh.visual.TextureVisuals(uv=[[0, 0], [1, 0], [1, 1], [0, 1]],
                                                material=trimesh.visual.material.PBRMaterial(baseColorTexture=tex))
    sc.add_geometry(quad, node_name="room_floor", geom_name="room_floor")
    sc.export(str(scene / "scene_lite.glb"), file_type="glb")
    (scene / "reconstruction_results.json").write_text(json.dumps({"objects": results, "failed": [], "metadata": {}}))
    (scene / "3d_models" / "layout.json").write_text(json.dumps(
        {"bounds_min": [-2, -1, 0], "bounds_max": [2, 2, 4], "floor_y": -1.0, "estimated_metric_scale": 1.0}))
    np.savez(scene / "3d_models" / "scene_pointmap.npz", points=np.zeros((1, 3)), colors=np.zeros((1, 3)),
             intrinsics=np.array([[1.0, 0, 0.5], [0, 1.3, 0.5], [0, 0, 1]]), R_total=np.eye(3), image_hw=[480, 640])
    return scene


def test_mujoco_export_loads_and_rests(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    from src.mujoco_export import check_model, export_mujoco, export_stable
    scene = _write_scene(tmp_path)
    xml = export_mujoco(scene, verbose=False)
    model = mujoco.MjModel.from_xml_path(str(xml))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(1, model.nbody)]
    assert names == ["obj_001_table", "obj_002_box", "obj_003_mug", "obj_004_picture"]
    free = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)}
    assert free == {"obj_001_table_free", "obj_002_box_free", "obj_003_mug_free"}      # the picture is welded
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pad_obj_003_mug") >= 0  # nothing modelled under the mug
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "photo") >= 0
    # frame: aligned (x left, y up, z forward) -> (x forward, y left, z up), floor at z = 0
    table = model.body("obj_001_table").pos
    np.testing.assert_allclose(table, [2.5, 0.0, 0.35], atol=1e-6)
    res = check_model(xml, seconds=1.0, verbose=False)
    assert res["finite"] and res["max_displacement_m"] < 0.03, res
    # metric scale is applied to everything
    xml2 = export_mujoco(scene, tmp_path / "scaled", scale=2.0, verbose=False)
    m2 = mujoco.MjModel.from_xml_path(str(xml2))
    np.testing.assert_allclose(m2.body("obj_001_table").pos, 2 * table, atol=1e-6)
    # the MJX variant uses primitives only for collision
    xml3, stab = export_stable(scene, tmp_path / "mjx", mjx=True, verbose=False)
    m3 = mujoco.MjModel.from_xml_path(str(xml3))
    coll = [g for g in range(m3.ngeom) if m3.geom_contype[g]]
    allowed = {int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_PLANE)}
    assert coll and all(int(m3.geom_type[g]) in allowed for g in coll)
    assert stab["welded_for_stability"] == []
