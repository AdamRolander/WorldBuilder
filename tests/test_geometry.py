import numpy as np
import pytest

from src.geometry import (
    glb_to_gaussian_frame,
    matrix_to_quat,
    quat_multiply,
    quat_to_matrix,
    rotation_between,
    world_vertices,
)

pytorch3d = pytest.importorskip("pytorch3d")


def _rand_quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def test_quat_to_matrix_matches_pytorch3d():
    import torch
    from pytorch3d.transforms import quaternion_to_matrix
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = _rand_quat(rng)
        ref = quaternion_to_matrix(torch.tensor(q, dtype=torch.float64)).numpy()
        np.testing.assert_allclose(quat_to_matrix(q), ref, atol=1e-12)


def test_world_vertices_matches_upstream_compose_transform():
    """The exact composition SAM 3D / the old CUDA code used."""
    import torch
    from pytorch3d.transforms import Transform3d, quaternion_to_matrix
    rng = np.random.default_rng(1)
    for _ in range(10):
        q = _rand_quat(rng)
        s = rng.uniform(0.2, 3.0, size=3)
        t = rng.normal(size=3) * 2
        v = rng.normal(size=(50, 3))
        vg = glb_to_gaussian_frame(v)
        T = (Transform3d(dtype=torch.float64)
             .scale(torch.tensor(s)[None])
             .rotate(quaternion_to_matrix(torch.tensor(q)[None]))
             .translate(torch.tensor(t)[None]))
        ref = T.transform_points(torch.tensor(vg)[None])[0].numpy()
        pose = {"translation": t.tolist(), "rotation_quaternion": q.tolist(), "scale": s.tolist()}
        np.testing.assert_allclose(world_vertices(v, pose), ref, atol=1e-9)


def test_matrix_quat_roundtrip():
    rng = np.random.default_rng(2)
    for _ in range(50):
        q = _rand_quat(rng)
        if q[0] < 0:
            q = -q
        np.testing.assert_allclose(matrix_to_quat(quat_to_matrix(q)), q, atol=1e-9)


def test_quat_multiply_matches_matrix_product():
    rng = np.random.default_rng(3)
    a, b = _rand_quat(rng), _rand_quat(rng)
    np.testing.assert_allclose(quat_to_matrix(quat_multiply(a, b)),
                               quat_to_matrix(a) @ quat_to_matrix(b), atol=1e-12)


@pytest.mark.parametrize("a,b", [
    ([0, 0, 1], [0, 1, 0]),
    ([0, -1, 0], [0, 1, 0]),      # antiparallel
    ([0.3, 0.9, 0.1], [0, 1, 0]),
    ([1, 0, 0], [1, 0, 0]),       # identity
])
def test_rotation_between(a, b):
    R = rotation_between(a, b)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)
    an = np.asarray(a, float) / np.linalg.norm(a)
    bn = np.asarray(b, float) / np.linalg.norm(b)
    np.testing.assert_allclose(R @ an, bn, atol=1e-12)
