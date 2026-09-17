"""Procedural room geometry and ground snapping.

Derives an axis-aligned box room from the bounds of reconstructed objects,
and optionally snaps near-floor objects to a shared floor plane.
"""
import numpy as np
import trimesh
from pathlib import Path
from typing import List, Dict, Optional
import torch
from pytorch3d.transforms import quaternion_to_matrix
from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform

# Substring whitelist of categories that always sit on the floor in indoor
# scenes. Used to override the snap threshold for objects whose SAM 3D
# depth came back wildly off. Conservative on purpose — wall/ceiling-mounted
# things (TV, mirror, painting, light fixture, blackboard) stay out.
FLOOR_SUPPORTED_KEYWORDS = (
    'chair', 'desk', 'table', 'sofa', 'couch', 'bed', 'stool', 'bench',
    'cabinet', 'dresser', 'nightstand', 'bookshelf', 'bookcase',
    'trash', 'bin', 'wastebasket', 'floor lamp',
    'planter', 'potted plant', 'rug', 'carpet',
    'piano', 'refrigerator', 'fridge', 'oven', 'stove',
    'dishwasher', 'washing machine', 'dryer', 'ottoman',
)

# VERTICAL_STRUCTURAL_KEYWORDS = (
#     'pillar', 'column', 'support column', 'support post', 'post',
#     'i-beam', 'i beam', 'steel beam', 'structural beam', 'beam',
#     'truss', 'girder', 'stanchion', 'rafter'
# )

# def _is_vertical_structural(label: str) -> bool:
#     s = label.lower()
#     return any(k in s for k in VERTICAL_STRUCTURAL_KEYWORDS)

def _is_floor_supported(label: str) -> bool:
    s = label.lower()
    return any(k in s for k in FLOOR_SUPPORTED_KEYWORDS)

def _transform_verts(result: Dict, device: str = 'cuda') -> np.ndarray:
    """Apply GLB->Gaussian swap + pose transform -> world-space vertices."""
    mesh = trimesh.load(result['ply_path'])
    v = mesh.vertices.copy()
    old_y, old_z = v[:, 1].copy(), v[:, 2].copy()
    v[:, 1] = -old_z
    v[:, 2] = old_y

    vt = torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
    quat = torch.tensor([result['rotation_quaternion']], dtype=torch.float32, device=device)
    trans = torch.tensor([result['translation']], dtype=torch.float32, device=device)
    scale = torch.tensor([result['scale']], dtype=torch.float32, device=device)
    T = compose_transform(scale=scale,
                          rotation=quaternion_to_matrix(quat),
                          translation=trans)
    return T.transform_points(vt)[0].cpu().numpy()


def compute_per_object_aabb(results: List[Dict]) -> Dict[int, Dict]:
    """World-space AABB per object id."""
    aabbs = {}
    for r in results:
        if not r.get('ply_path') or not Path(r['ply_path']).exists():
            continue
        v = _transform_verts(r)
        aabbs[r['id']] = {'min': v.min(axis=0).tolist(),
                          'max': v.max(axis=0).tolist()}
    return aabbs


def compute_scene_bounds(aabbs: Dict[int, Dict],
                         robust_percentile: float = 1.0) -> Optional[Dict]:
    """Overall scene AABB. `robust_percentile` trims horizontal outliers
    (rogue objects floating off in X/Z), but the vertical axis always uses
    raw min/max so tall fixtures like blackboards, doors, or whiteboards
    stay inside the ceiling. Set robust_percentile=0 to use raw on all axes.
    """
    if not aabbs:
        return None
    mins = np.array([b['min'] for b in aabbs.values()])
    maxs = np.array([b['max'] for b in aabbs.values()])
    if robust_percentile > 0 and len(aabbs) >= 4:
        lo = np.percentile(mins, robust_percentile, axis=0)
        hi = np.percentile(maxs, 100 - robust_percentile, axis=0)
        # Vertical: tall objects are features, not outliers.
        lo[1] = float(mins[:, 1].min())
        hi[1] = float(maxs[:, 1].max())
    else:
        lo, hi = mins.min(axis=0), maxs.max(axis=0)
    return {'min': lo.tolist(), 'max': hi.tolist(),
            'size': (hi - lo).tolist()}

def upright_correct(results: List[Dict], only_snapped: bool = True,
                    upright_threshold_deg: float = 45.0) -> int:
    """Snap near-upright objects to a clean upright pose, preserving yaw.

    SAM 3D's canonical object frame (after the GLB->Gaussian Y/Z swap done
    in reconstruction_3d.py) has +Z as the natural "up" axis — a perfectly
    upright chair has rotation R_yaw(α) @ R_x(-90°), which sends local +Z
    to world +Y. The previous yaw-only stripping discarded the R_x(-90°)
    factor, leaving canonical +Y (back of the chair) pointing skyward.

    For each targeted object, measure where local +Z lands in world. If its
    Y component is within `upright_threshold_deg` of straight up, extract
    yaw α and rebuild the rotation as R_yaw(α) @ R_x(-90°). Objects more
    tilted than that are left alone — better to keep something angled than
    forcibly flip a pose we don't understand.

    Modifies result['rotation_quaternion'] in place. Returns count of
    corrected objects.
    """
    cos_thresh = float(np.cos(np.radians(upright_threshold_deg)))
    cp_h = float(np.sqrt(2) / 2)          # cos(-45°) — half of R_x(-90°)
    sp_h = float(-np.sqrt(2) / 2)         # sin(-45°)
    corrected = 0

    for r in results:
        if only_snapped and not r.get('snapped', False):
            continue

        # PyTorch3D quaternion convention: (w, x, y, z)
        w, x, y, z = r['rotation_quaternion']

        # World Y component of local +Z under current rotation: R[1,2]
        up_y = 2.0 * (y * z - w * x)
        if up_y < cos_thresh:
            # Too tilted — don't risk a wrong correction
            r['snapped'] = False
            continue

        # Yaw α from R = R_yaw(α) @ R_x(-90°):
        #   R[0,0] = cos α  = 1 - 2(y² + z²)
        #   R[0,1] = -sin α = 2(xy - wz)   ⇒  sin α = 2(wz - xy)
        cos_a = 1.0 - 2.0 * (y * y + z * z)
        sin_a = 2.0 * (w * z - x * y)
        yaw = float(np.arctan2(sin_a, cos_a))

        # Rebuild as q_yaw(α) * q_x(-90°)  (Hamilton product, expanded)
        cy_h = float(np.cos(yaw / 2.0))
        sy_h = float(np.sin(yaw / 2.0))
        r['rotation_quaternion'] = [
            cy_h * cp_h,        # w
            cy_h * sp_h,        # x
            sy_h * cp_h,        # y
            -sy_h * sp_h,       # z
        ]
        r['upright_corrected'] = True
        corrected += 1
    return corrected

def snap_ground_objects(results: List[Dict],
                        aabbs: Dict[int, Dict],
                        bounds: Dict,
                        threshold_frac: float = 0.15) -> int:
    """Snap objects near the floor to be exactly on it.

    An object is snapped if either:
      (a) its current bottom is within `threshold_frac * scene_height` of
          the scene floor, OR
      (b) its label matches a known floor-supported category — these always
          snap regardless of how far off SAM 3D's depth estimate was.
    Modifies result['translation'] in place. Returns count snapped.
    """
    floor_y = bounds['min'][1]
    scene_h = bounds['max'][1] - bounds['min'][1]
    tol = threshold_frac * scene_h
    snapped = 0
    for r in results:
        ab = aabbs.get(r['id'])
        if ab is None:
            r['snapped'] = False
            continue
        dy = ab['min'][1] - floor_y
        force = _is_floor_supported(r['label'])
        if abs(dy) <= tol or force:
            r['translation'] = [r['translation'][0],
                                r['translation'][1] - dy,
                                r['translation'][2]]
            r['snapped'] = True
            r['snap_forced'] = bool(force and abs(dy) > tol)
            snapped += 1
        else:
            r['snapped'] = False
    return snapped


def make_box_room(bounds: Dict,
                  padding_xz: float = 0.15,
                  padding_y: float = 0.10,
                  wall_thickness: float = 0.02,
                  floor_color=(180, 170, 150, 255),
                  wall_color=(235, 230, 220, 255),
                  ceiling_color=(245, 245, 245, 255),
                  include_ceiling: bool = True,
                  include_front_wall: bool = True) -> trimesh.Trimesh:
    """Axis-aligned floor, 4 walls, and ceiling as a single colored trimesh.

    Uses manual concatenation to guarantee vertex colors survive the
    PLY export step (trimesh.util.concatenate is inconsistent about this).
    """
    mn = np.array(bounds['min']); mx = np.array(bounds['max'])
    min_x, max_x = mn[0] - padding_xz, mx[0] + padding_xz
    min_z, max_z = mn[2] - padding_xz, mx[2] + padding_xz
    floor_y   = mn[1]
    ceiling_y = mx[1] + padding_y

    sx = max_x - min_x
    sz = max_z - min_z
    h  = ceiling_y - floor_y
    cx = (min_x + max_x) / 2
    cz = (min_z + max_z) / 2
    cy = (floor_y + ceiling_y) / 2

    def plane(extents, center, color):
        b = trimesh.creation.box(extents=extents)
        b.apply_translation(center)
        n = len(b.vertices)
        rgba = np.array(color, dtype=np.uint8)
        if rgba.size == 3:
            rgba = np.concatenate([rgba, [255]])
        b.vertex_colors = np.tile(rgba, (n, 1))  # stored on the mesh, not visual
        return b, rgba

    parts = [
        plane([sx, wall_thickness, sz],
              [cx, floor_y - wall_thickness / 2, cz], floor_color),
        plane([wall_thickness, h, sz],
              [min_x - wall_thickness / 2, cy, cz], wall_color),
        plane([wall_thickness, h, sz],
              [max_x + wall_thickness / 2, cy, cz], wall_color),
        plane([sx, h, wall_thickness],
              [cx, cy, min_z - wall_thickness / 2], wall_color),
    ]
    if include_front_wall:
        parts.append(plane([sx, h, wall_thickness],
                           [cx, cy, max_z + wall_thickness / 2], wall_color))
    if include_ceiling:
        parts.append(plane([sx, wall_thickness, sz],
                           [cx, ceiling_y + wall_thickness / 2, cz],
                           ceiling_color))

    all_v, all_f, all_c = [], [], []
    offset = 0
    for mesh, color in parts:
        v = np.asarray(mesh.vertices)
        f = np.asarray(mesh.faces) + offset
        c = np.tile(color, (len(v), 1)).astype(np.uint8)
        all_v.append(v); all_f.append(f); all_c.append(c)
        offset += len(v)

    return trimesh.Trimesh(
        vertices=np.vstack(all_v),
        faces=np.vstack(all_f),
        vertex_colors=np.vstack(all_c),
        process=False,
    )

def resolve_xz_collisions(results: List[Dict],
                          aabbs: Dict[int, Dict],
                          iterations: int = 8,
                          padding: float = 0.02) -> int:
    """Iteratively separate overlapping object AABBs in the XZ plane.

    Y is left alone (floor-snap handles vertical placement). For each pair
    overlapping in BOTH X and Z, push them apart along the axis of smaller
    overlap. Total displacement = overlap + padding, split by confidence so
    the lower-confidence object moves more. Mutates translations and AABBs
    in place. Returns total push events.
    """
    if not results or not aabbs:
        return 0

    by_id = {r['id']: r for r in results}
    moved_total = 0

    for _ in range(iterations):
        any_move = False
        ids = list(aabbs.keys())
        for i, id_a in enumerate(ids):
            if id_a not in by_id:
                continue
            a = aabbs[id_a]
            for id_b in ids[i + 1:]:
                if id_b not in by_id:
                    continue
                b = aabbs[id_b]
                ox = min(a['max'][0], b['max'][0]) - max(a['min'][0], b['min'][0])
                oz = min(a['max'][2], b['max'][2]) - max(a['min'][2], b['min'][2])
                if ox <= 0 or oz <= 0:
                    continue

                ra, rb = by_id[id_a], by_id[id_b]
                ca = float(ra.get('confidence', 0.5)) or 0.5
                cb = float(rb.get('confidence', 0.5)) or 0.5
                total = ca + cb
                # lower confidence -> larger push share
                push_a_frac = cb / total
                push_b_frac = ca / total

                if ox < oz:
                    push = ox + padding
                    cx_a = 0.5 * (a['min'][0] + a['max'][0])
                    cx_b = 0.5 * (b['min'][0] + b['max'][0])
                    sign_a = -1.0 if cx_a < cx_b else 1.0
                    da, db = sign_a * push * push_a_frac, -sign_a * push * push_b_frac
                    ra['translation'][0] += da
                    rb['translation'][0] += db
                    a['min'][0] += da; a['max'][0] += da
                    b['min'][0] += db; b['max'][0] += db
                else:
                    push = oz + padding
                    cz_a = 0.5 * (a['min'][2] + a['max'][2])
                    cz_b = 0.5 * (b['min'][2] + b['max'][2])
                    sign_a = -1.0 if cz_a < cz_b else 1.0
                    da, db = sign_a * push * push_a_frac, -sign_a * push * push_b_frac
                    ra['translation'][2] += da
                    rb['translation'][2] += db
                    a['min'][2] += da; a['max'][2] += da
                    b['min'][2] += db; b['max'][2] += db

                any_move = True
                moved_total += 1
        if not any_move:
            break
    return moved_total

def stretch_vertical_structural(results: List[Dict],
                                aabbs: Dict[int, Dict],
                                bounds: Dict,
                                min_height_frac: float = 0.6) -> int:
    """Vertically scale floor-to-ceiling structural elements to fill the
    scene height. SAM 3D returns these tiny because tall thin objects have
    weak depth signal in a single image.

    Only stretches objects that are currently shorter than `min_height_frac`
    of scene height — preserves any that came out reasonably-sized. Modifies
    result['scale'] in place. Returns count stretched.

    Note: this scales BEFORE the floor snap pass that follows. We bump scale,
    then let snap_ground_objects re-place the bottom on the floor.
    """
    scene_h = bounds['max'][1] - bounds['min'][1]
    if scene_h <= 0:
        return 0
    target_h = scene_h * 0.95  # leave headroom under ceiling
    stretched = 0

    for r in results:
        if not _is_vertical_structural(r['label']):
            continue
        ab = aabbs.get(r['id'])
        if ab is None:
            continue
        cur_h = ab['max'][1] - ab['min'][1]
        if cur_h <= 0 or cur_h >= scene_h * min_height_frac:
            continue  # already tall enough, leave it
        factor = target_h / cur_h
        # Scale is in local frame, applied before rotation. After upright
        # correction, local +Z is what rotates to world +Y, so to stretch
        # vertically in world space we scale local Z (index 2), not Y.
        r['scale'] = [r['scale'][0],
                      r['scale'][1],
                      r['scale'][2] * factor]
        r['vertical_stretched'] = True
        stretched += 1
    return stretched