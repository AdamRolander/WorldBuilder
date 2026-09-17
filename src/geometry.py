"""Pure-numpy pose math shared by the scene-assembly stages.

SAM 3D Objects returns each object as (translation, rotation quaternion,
scale) in the camera frame, and its GLB mesh in a canonical frame whose Y
and Z axes are swapped relative to the Gaussian frame the pose refers to.
The composition used upstream (``compose_transform`` from sam3d_objects,
built on ``pytorch3d.transforms.Transform3d``) is, in row-vector form::

    world = (v * scale) @ R + t

with ``R = quaternion_to_matrix(q)`` for a (w, x, y, z) quaternion. The
previous implementation re-derived this on the GPU in three separate places
(``_bake_world_space_plys``, ``_save_combined_scene`` and
``room_generator._transform_verts``); this module is the single CPU-only
source of truth, and ``tests/test_geometry.py`` checks it against pytorch3d
to float precision.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

__all__ = [
    "quat_to_matrix",
    "matrix_to_quat",
    "glb_to_gaussian_frame",
    "world_vertices",
    "rotation_between",
    "quat_multiply",
]


def quat_to_matrix(q: Sequence[float]) -> np.ndarray:
    """(w, x, y, z) unit quaternion -> 3x3 rotation matrix (column-vector
    convention, identical to ``pytorch3d.transforms.quaternion_to_matrix``)."""
    w, x, y, z = (float(c) for c in q)
    n = w * w + x * x + y * y + z * z
    if n == 0.0:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
        [s * (x * y + w * z), 1 - s * (x * x + z * z), s * (y * z - w * x)],
        [s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x * x + y * y)],
    ])


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> (w, x, y, z) unit quaternion (Shepperd's method)."""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def quat_multiply(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    """Hamilton product a*b for (w, x, y, z) quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def glb_to_gaussian_frame(vertices: np.ndarray) -> np.ndarray:
    """Undo the Y/Z swap between SAM 3D's exported GLB and its pose frame:
    new_y = -old_z, new_z = old_y."""
    v = np.asarray(vertices, dtype=np.float64).copy()
    old_y = v[:, 1].copy()
    v[:, 1] = -v[:, 2]
    v[:, 2] = old_y
    return v


def world_vertices(vertices: np.ndarray, pose: Dict,
                   already_in_gaussian_frame: bool = False) -> np.ndarray:
    """Model-space GLB vertices -> world (camera) space for a pipeline result.

    ``pose`` is any dict with ``translation`` (3,), ``rotation_quaternion``
    (w, x, y, z) and ``scale`` (3,) — i.e. an entry of
    ``reconstruction_results.json``.
    """
    v = np.asarray(vertices, dtype=np.float64)
    if not already_in_gaussian_frame:
        v = glb_to_gaussian_frame(v)
    scale = np.asarray(pose["scale"], dtype=np.float64).reshape(3)
    R = quat_to_matrix(pose["rotation_quaternion"])
    t = np.asarray(pose["translation"], dtype=np.float64).reshape(3)
    return (v * scale) @ R + t


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation matrix (column-vector convention) taking unit vector
    ``a`` onto unit vector ``b``. Handles the antiparallel case."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    c = float(np.dot(a, b))
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        # 180° about any axis perpendicular to a
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + 2 * K @ K
    v = np.cross(a, b)
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)
