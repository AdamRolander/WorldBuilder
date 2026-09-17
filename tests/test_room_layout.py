"""Synthetic-room tests for src/room_layout.py.

We ray-cast a box room of known dimensions through MoGe-style normalised
intrinsics from a camera with a known pitch/yaw, colour every plane with a
position-dependent pattern, and check that the layout stage recovers the
gravity rotation, floor height, ceiling height, wall positions and — most
importantly — paints each plane vertex with the colour of the true surface
at that location. That last check pins the projection math and all frame
conventions end to end, which eyeballing textures cannot.
"""
import math

import numpy as np
import pytest

from src import room_layout as rl
from src.geometry import world_vertices

# Ground-truth room in the upright ("aligned") frame. Camera at origin.
FLOOR_Y, CEIL_Y = -1.2, 1.3
X_MIN, X_MAX = -2.0, 2.5
Z_MIN, Z_MAX = -0.8, 5.0
K = np.array([[1.2, 0, 0.5], [0, 1.6, 0.5], [0, 0, 1.0]])   # normalised, fx/fy differ
W, H = 240, 180


def plane_color(name, p):
    """Deterministic colour pattern per plane, as uint8."""
    x, y, z = p
    if name == "floor":
        return (40 + 40 * (math.floor(x * 2) % 3), 90, 60 + 30 * (math.floor(z * 2) % 4))
    if name == "ceiling":
        return (230, 230, 230)
    if name == "x_min":
        return (200, 60 + 30 * (math.floor(z) % 3), 60)
    if name == "x_max":
        return (60, 200, 60 + 30 * (math.floor(y * 2) % 3))
    if name == "z_max":
        return (60 + 30 * (math.floor(x) % 4), 60, 200)
    return (120, 120, 120)  # z_min (behind camera)


def rot_x(deg):
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def rot_y(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def raycast_room(R_cam_from_true):
    """Return (points_world_camframe HxWx3, colors HxWx3, depth_opencv HxW).

    ``R_cam_from_true`` rotates true-frame points into the camera-frame
    world (the frame SAM 3D poses live in)."""
    us = (np.arange(W) + 0.5) / W
    vs = (np.arange(H) + 0.5) / H
    gu, gv = np.meshgrid(us, vs)
    # OpenCV ray directions, then to SAM 3D world (x,y negated).
    d_cv = np.stack([(gu - K[0, 2]) / K[0, 0], (gv - K[1, 2]) / K[1, 1], np.ones_like(gu)], -1)
    d_w = d_cv @ rl.OPENCV_TO_WORLD.T
    # Rays into the true frame: p_true = R^T p_cam.
    d_true = d_w @ R_cam_from_true          # (R^T applied to row vectors)
    planes = {
        "floor": (np.array([0, 1, 0.]), FLOOR_Y), "ceiling": (np.array([0, 1, 0.]), CEIL_Y),
        "x_min": (np.array([1, 0, 0.]), X_MIN), "x_max": (np.array([1, 0, 0.]), X_MAX),
        "z_min": (np.array([0, 0, 1.]), Z_MIN), "z_max": (np.array([0, 0, 1.]), Z_MAX),
    }
    best_t = np.full((H, W), np.inf)
    hit_name = np.full((H, W), "", dtype=object)
    for name, (n, c) in planes.items():
        denom = d_true @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = c / denom
        ok = (t > 1e-6) & (t < best_t)
        best_t[ok] = t[ok]
        hit_name[ok] = name
    P_true = d_true * best_t[..., None]
    colors = np.zeros((H, W, 3), np.uint8)
    for i in range(H):
        for j in range(W):
            colors[i, j] = plane_color(hit_name[i, j], P_true[i, j])
    P_cam = P_true @ R_cam_from_true.T
    depth = (P_cam @ rl.OPENCV_TO_WORLD.T)[..., 2]
    return P_cam, colors, depth, P_true, hit_name


@pytest.mark.parametrize("pitch_deg,yaw_deg", [(0.0, 0.0), (-18.0, 0.0), (-10.0, 20.0)])
def test_layout_recovers_room(pitch_deg, yaw_deg):
    R = rot_x(pitch_deg) @ rot_y(yaw_deg)            # true -> camera-frame world
    P_cam, colors, depth, P_true, hit = raycast_room(R)
    valid = np.isfinite(P_cam).all(-1)
    layout = rl.estimate_layout(P_cam, valid, rng=np.random.default_rng(0))

    # Gravity + yaw: R_total should undo R up to a multiple of 90° about Y.
    Rt = layout.R_total
    up_cam = R @ np.array([0, 1, 0.])
    np.testing.assert_allclose(Rt @ up_cam, [0, 1, 0], atol=2e-2)
    assert abs(layout.floor_y - FLOOR_Y) < 0.03
    assert layout.floor.source in ("geometric", "geometric-loose")
    if (hit == "ceiling").sum() > 500:                 # ceiling in view -> measured
        assert layout.ceiling_source == "geometric"
        assert abs(layout.ceiling_y - CEIL_Y) < 0.05
    else:                                              # out of view -> declared prior
        assert layout.ceiling_source == "prior"

    # Walls: only sides the camera could see are detectable; those must match
    # the true wall to within 5 cm. Express the true room in the recovered
    # frame (which may differ by a multiple of 90° about Y) first.
    M = Rt @ R                                    # true -> aligned
    corners = np.array([[x, y, z] for x in (X_MIN, X_MAX) for y in (FLOOR_Y, CEIL_Y) for z in (Z_MIN, Z_MAX)]) @ M.T
    t_min, t_max = corners.min(0), corners.max(0)
    detected = 0
    for side, src in layout.wall_sources.items():
        if not src.startswith("wall"):
            continue
        axis = 0 if side.startswith("x") else 2
        got = layout.bounds_min[axis] if side.endswith("min") else layout.bounds_max[axis]
        want = t_min[axis] if side.endswith("min") else t_max[axis]
        assert abs(got - want) < 0.05, f"{side}: {got} vs {want}"
        detected += 1
    assert detected >= 1                          # the far wall is always in view
    assert layout.wall_sources["z_max"].startswith("wall") or layout.wall_sources["x_min"].startswith("wall") \
        or layout.wall_sources["x_max"].startswith("wall") or layout.wall_sources["z_min"].startswith("wall")
    # The room must contain every observed point.
    P_a = P_cam[valid] @ Rt.T
    assert np.all(P_a.min(0) >= np.array(layout.bounds_min) - 0.1)
    assert np.all(P_a.max(0) <= np.array(layout.bounds_max) + 0.1)


def test_textures_sample_the_true_surface():
    pitch, yaw = -15.0, 0.0
    R = rot_x(pitch) @ rot_y(yaw)
    P_cam, colors, depth, P_true, hit = raycast_room(R)
    valid = np.isfinite(P_cam).all(-1)
    layout = rl.estimate_layout(P_cam, valid, rng=np.random.default_rng(0))
    mesh, stats = rl.build_textured_room(layout, K, colors, depth, cells=48)
    assert stats["z_max"]["visible_fraction"] > 0.5
    assert stats["floor"]["visible_fraction"] > 0.3
    assert stats["z_min"]["visible_fraction"] == 0.0      # behind the camera

    # Every textured vertex that projects into the image must carry the colour
    # of the true surface at its own position (within nearest-pixel error).
    V = np.asarray(mesh.vertices)
    C = np.asarray(mesh.visual.vertex_colors)[:, :3]
    Rt = layout.R_total
    V_true = (V @ Rt) @ R          # aligned -> camera-frame -> true (row-vector inverses)
    u, v, z = rl.project_world_to_pixels(V @ Rt, K, W, H)
    inside = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    checked = 0
    mismatches = 0
    for i in np.flatnonzero(inside):
        # which plane is this vertex on?
        p = V_true[i]
        cands = {"floor": abs(p[1] - FLOOR_Y), "ceiling": abs(p[1] - CEIL_Y),
                 "x_min": abs(p[0] - X_MIN), "x_max": abs(p[0] - X_MAX), "z_max": abs(p[2] - Z_MAX)}
        name = min(cands, key=cands.get)
        if cands[name] > 0.08:
            continue
        # only count vertices the raycast actually saw on that plane
        ui, vi = int(round(u[i])), int(round(v[i]))
        if not (0 <= ui < W and 0 <= vi < H) or hit[vi, ui] != name:
            continue
        expected = np.array(plane_color(name, p))
        checked += 1
        if np.abs(C[i].astype(int) - expected).max() > 45:   # pattern edges tolerate 1-cell error
            mismatches += 1
    assert checked > 500
    assert mismatches / checked < 0.08, f"{mismatches}/{checked} vertices got the wrong colour"


def test_rotate_pose_is_consistent_with_world_vertices():
    rng = np.random.default_rng(5)
    q = rng.normal(size=4); q /= np.linalg.norm(q)
    pose = {"translation": rng.normal(size=3).tolist(), "rotation_quaternion": q.tolist(),
            "scale": rng.uniform(0.5, 2, 3).tolist(), "label": "x"}
    Rw = rot_x(-17) @ rot_y(31)
    v = rng.normal(size=(30, 3))
    before = world_vertices(v, pose) @ Rw.T
    after = world_vertices(v, rl.rotate_pose(pose, Rw))
    np.testing.assert_allclose(after, before, atol=1e-9)
    assert rl.rotate_pose(pose, Rw)["label"] == "x"


def test_gravity_rotation_and_yaw_helpers():
    n = np.array([0.1, 0.95, -0.2]); n /= np.linalg.norm(n)
    G = rl.gravity_rotation(n)
    np.testing.assert_allclose(G @ n, [0, 1, 0], atol=1e-12)
    # normals of two walls at 30° and 120° -> yaw 30° (mod 90)
    normals = np.array([[math.cos(math.radians(30)), 0, math.sin(math.radians(30))]] * 60 +
                       [[math.cos(math.radians(120)), 0, math.sin(math.radians(120))]] * 60)
    yaw, support = rl.dominant_yaw_deg(normals)
    assert support > 0.99
    assert (yaw - 30) % 90 == pytest.approx(0, abs=1e-6) or (yaw - 30) % 90 == pytest.approx(90, abs=1e-6)


def test_upward_photo_uses_ceiling_for_gravity():
    """Camera pitched up 12°: no floor pixel is in view. Gravity must still
    come out right (from the ceiling) and the failure must be reported."""
    R = rot_x(12.0) @ rot_y(20.0)
    P_cam, colors, depth, P_true, hit = raycast_room(R)
    assert not (hit == "floor").any()
    valid = np.isfinite(P_cam).all(-1)
    layout = rl.estimate_layout(P_cam, valid, rng=np.random.default_rng(0))
    assert layout.floor.source == "fallback-ceiling"
    np.testing.assert_allclose(layout.R_total @ (R @ np.array([0, 1, 0.])), [0, 1, 0], atol=2e-2)
    assert any("no floor visible" in n for n in layout.notes)


def test_intrinsics_recovered_from_pointmap():
    R = rot_x(-12.0) @ rot_y(5.0)
    P_cam, colors, depth, P_true, hit = raycast_room(R)
    valid = np.isfinite(P_cam).all(-1)
    K_est = rl.intrinsics_from_pointmap(P_cam, valid)
    np.testing.assert_allclose(K_est, K, atol=1e-6)
