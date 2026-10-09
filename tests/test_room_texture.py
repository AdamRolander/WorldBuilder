"""Tests for the textured shell and relief (src/room_texture.py)."""
import numpy as np

from src import room_layout as rl
from src import room_texture as rt
from tests.test_room_layout import K, plane_color, raycast_room, rot_x


def test_largest_rectangle():
    m = np.zeros((10, 12), bool)
    m[2:7, 3:11] = True
    m[4, 6] = False
    r0, c0, r1, c1 = rt.largest_rectangle(m)
    assert m[r0:r1, c0:c1].all()
    assert (r1 - r0) * (c1 - c0) == 20          # 5 rows × 4 cols beside the hole beats 2 × 8
    assert rt.largest_rectangle(np.zeros((4, 4), bool)) == (0, 0, 0, 0)


def test_fill_keeps_what_was_seen_and_continues_the_material():
    rng = np.random.default_rng(0)
    th, tw = 120, 200
    yy, xx = np.mgrid[:th, :tw]
    tex = np.stack([120 + 40 * ((xx // 10) % 2), 100 + 0 * xx, 80 + 20 * ((yy // 10) % 2)], -1).astype(np.uint8)
    tex = np.clip(tex + rng.integers(-3, 4, tex.shape), 0, 255).astype(np.uint8)
    vis = np.zeros((th, tw), bool)
    vis[20:90, 30:120] = True
    out, how = rt.fill_hidden(np.where(vis[..., None], tex, 0).astype(np.uint8), vis)
    assert how == "tiled"
    # well inside the seen area nothing changes
    np.testing.assert_array_equal(out[40:70, 50:100], tex[40:70, 50:100])
    # the unseen area carries the same two-tone material, not black and not a flat colour
    far = out[:, 150:].reshape(-1, 3).astype(float)
    assert far[:, 0].min() > 100 and far[:, 0].std() > 10
    # a patch with large-scale structure must not be stamped around
    half = tex.copy(); half[:, :75] = 250
    out2, how2 = rt.fill_hidden(half, vis)
    assert how2 == "median"
    # nothing seen at all
    out3, how3 = rt.fill_hidden(tex, np.zeros_like(vis), fallback_rgb=(1, 2, 3))
    assert how3 == "fallback" and tuple(out3[0, 0]) == (1, 2, 3)


def _layout():
    R = rot_x(-15.0)
    P_cam, colors, depth, P_true, hit = raycast_room(R)
    valid = np.isfinite(P_cam).all(-1)
    layout = rl.estimate_layout(P_cam, valid, rng=np.random.default_rng(0))
    return R, P_cam, valid, colors, depth, hit, layout


def test_plane_textures_show_the_true_surface():
    R, P_cam, valid, colors, depth, hit, layout = _layout()
    planes = rt.room_planes(layout)
    for name, true_name in (("floor", "floor"), ("wall_z_max", "z_max")):
        pl = planes[name]
        tex, vis = rt.project_plane_texture(pl, layout.R_total, K, colors, depth, long_side=160)
        assert vis.mean() > 0.3
        th, tw = vis.shape
        rows, cols = np.nonzero(vis)
        sel = np.random.default_rng(1).choice(len(rows), 400, replace=False)
        bad = 0
        for r, c in zip(rows[sel], cols[sel], strict=True):
            s = (c + 0.5) / tw * pl["su"]
            t = (1 - (r + 0.5) / th) * pl["sv"]
            p_true = ((pl["o"] + s * pl["au"] + t * pl["av"]) @ layout.R_total) @ R
            if np.abs(tex[r, c].astype(int) - np.array(plane_color(true_name, p_true))).max() > 45:
                bad += 1
        assert bad / 400 < 0.1, f"{name}: {bad}/400 texels wrong"


def test_shell_meshes_have_uvs_and_face_inward():
    R, P_cam, valid, colors, depth, hit, layout = _layout()
    meshes, stats = rt.build_shell(layout, K, colors, depth, long_side=128)
    assert set(meshes) == {"room_floor", "room_ceiling", "room_wall_x_min", "room_wall_x_max",
                           "room_wall_z_max", "room_wall_z_min"}
    centre = 0.5 * (np.asarray(layout.bounds_min) + np.asarray(layout.bounds_max))
    for name, m in meshes.items():
        assert m.visual.uv.shape == (4, 2)
        to_centre = centre - m.triangles_center[0]
        assert float(m.face_normals[0] @ to_centre) > 0, name
    assert stats["wall_z_min"]["fill"] in ("fallback", "median")      # behind the camera
    ply = rt.shell_vertex_color_mesh(meshes, cells=16)
    assert len(ply.vertices) > 100 and ply.visual.kind == "vertex"


def test_relief_meshes_a_counter_but_not_the_shell_or_objects():
    R, P_cam, valid, colors, depth, hit, layout = _layout()
    P_a = P_cam @ layout.R_total.T
    # Raise a "counter": a block of pixels on the floor lifted by 0.5 along +Y.
    Hh, Ww = valid.shape
    counter = np.zeros_like(valid)
    counter[int(0.72 * Hh):int(0.9 * Hh), int(0.3 * Ww):int(0.7 * Ww)] = True
    counter &= hit == "floor"
    P_a = P_a.copy()
    P_a[counter, 1] += 0.5
    owner = np.zeros(valid.shape, np.int32)
    obj = np.zeros_like(valid); obj[int(0.78 * Hh):int(0.82 * Hh), int(0.45 * Ww):int(0.5 * Ww)] = True
    owner[obj] = 7                                # a small object on the counter
    mesh, stats = rt.build_relief(P_a, valid, owner, layout, colors)
    assert mesh is not None and stats["faces"] > 50
    V = np.asarray(mesh.vertices)
    assert np.all(V[:, 1] > layout.floor_y + 0.3)                  # only the counter, not the floor
    assert stats["filled_hole_px"] > 0                              # the object's hole was closed
    cen = mesh.triangles_center
    assert np.mean(np.sum(mesh.face_normals * -cen, axis=1) > 0) > 0.95   # faces look at the camera
    assert mesh.visual.uv.min() >= 0 and mesh.visual.uv.max() <= 1
