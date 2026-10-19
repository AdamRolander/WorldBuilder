"""Manual corrections to a finished scene, kept as a layer on top of it.

The viewer lets a user nudge, turn, resize or remove objects and hide
pieces of the room. Those choices are stored in ``<scene>/edits.json`` and
applied wherever the scene leaves the pipeline: the viewer applies them on
load, and the GLB exports (which Blender, Unreal and the MuJoCo export
read) apply them when written. The pipeline's own output is never modified,
so deleting ``edits.json`` restores the scene, and replaying stage 4 keeps
the edits.

Format (scene units, degrees, three.js ``XYZ`` Euler order)::

    {"version": 1,
     "objects": {"12": {"translate": [dx, dy, dz], "rotate_deg": [rx, ry, rz],
                        "scale": [sx, sy, sz], "pivot": [cx, cy, cz], "hidden": false}},
     "hidden_nodes": ["room_relief_03"]}

An object's vertices become ``pivot + translate + R · (scale ⊙ (v − pivot))``.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

FILENAME = "edits.json"
MAX_OBJECTS = 2000
_NODE_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


def _vec3(x, default, lo, hi) -> List[float]:
    if not isinstance(x, (list, tuple)) or len(x) != 3:
        return list(default)
    out = []
    for v, d in zip(x, default, strict=True):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            out.append(float(d))
        else:
            out.append(float(min(hi, max(lo, v))))
    return out


def sanitize(data) -> Dict:
    """Keep only well-formed, bounded values. The file can arrive over HTTP,
    so nothing in it is trusted: unknown keys are dropped, numbers clamped."""
    out = {"version": 1, "objects": {}, "hidden_nodes": []}
    if not isinstance(data, dict):
        return out
    objs = data.get("objects")
    if isinstance(objs, dict):
        for k, e in list(objs.items())[:MAX_OBJECTS]:
            if not (isinstance(k, (str, int)) and str(k).isdigit() and isinstance(e, dict)):
                continue
            clean = {"translate": _vec3(e.get("translate"), (0, 0, 0), -1e3, 1e3),
                     "rotate_deg": _vec3(e.get("rotate_deg"), (0, 0, 0), -3600, 3600),
                     "scale": _vec3(e.get("scale"), (1, 1, 1), 1e-3, 1e3),
                     "pivot": _vec3(e.get("pivot"), (0, 0, 0), -1e4, 1e4),
                     "hidden": e.get("hidden") is True}
            if is_identity(clean):
                continue
            out["objects"][str(int(k))] = clean
    nodes = data.get("hidden_nodes")
    if isinstance(nodes, list):
        out["hidden_nodes"] = sorted({n for n in nodes[:MAX_OBJECTS] if isinstance(n, str) and _NODE_RE.match(n)})
    return out


def is_identity(e: Dict) -> bool:
    return (not e.get("hidden") and np.allclose(e["translate"], 0) and np.allclose(e["rotate_deg"], 0)
            and np.allclose(e["scale"], 1))


def load(scene_dir: Path) -> Optional[Dict]:
    p = Path(scene_dir) / FILENAME
    if not p.exists():
        return None
    try:
        data = sanitize(json.loads(p.read_text()))
    except (OSError, ValueError):
        return None
    return data if (data["objects"] or data["hidden_nodes"]) else None


def save(scene_dir: Path, data) -> Dict:
    """Write the sanitised edits (or remove the file when nothing is left)."""
    clean = sanitize(data)
    p = Path(scene_dir) / FILENAME
    if clean["objects"] or clean["hidden_nodes"]:
        p.write_text(json.dumps(clean, indent=1))
    elif p.exists():
        p.unlink()
    return clean


def euler_xyz(deg) -> np.ndarray:
    """Rotation matrix for three.js' default Euler order: R = Rx · Ry · Rz."""
    x, y, z = np.radians(np.asarray(deg, float))
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def apply(vertices: np.ndarray, edit: Dict) -> np.ndarray:
    c = np.asarray(edit["pivot"], float)
    local = (np.asarray(vertices, float) - c) * np.asarray(edit["scale"], float)
    return local @ euler_xyz(edit["rotate_deg"]).T + c + np.asarray(edit["translate"], float)


def for_object(edits: Optional[Dict], obj_id) -> Optional[Dict]:
    if not edits or obj_id is None:
        return None
    return edits["objects"].get(str(int(obj_id)))


_ID_RE = re.compile(r"^(\d{3})_")


def edited_glb(scene_dir: Path, lite: bool = True) -> Optional[Path]:
    """Path of the scene GLB with the edits applied, built or refreshed on
    demand. ``None`` when there are no edits or no source GLB (callers then
    use the pipeline's own file)."""
    scene_dir = Path(scene_dir)
    src = scene_dir / ("scene_lite.glb" if lite else "scene.glb")
    edits = load(scene_dir)
    if edits is None or not src.exists():
        return None
    out = src.with_name(src.stem + ".edited.glb")
    newest = max(src.stat().st_mtime, (scene_dir / FILENAME).stat().st_mtime)
    if out.exists() and out.stat().st_mtime >= newest:
        return out
    import trimesh
    sc = trimesh.load(str(src), process=False)
    new = trimesh.Scene()
    new.metadata.update(sc.metadata)
    hidden = set(edits["hidden_nodes"])
    for name, geom in sc.geometry.items():
        if name in hidden:
            continue
        m = _ID_RE.match(name)
        e = for_object(edits, int(m.group(1))) if (m and not name.startswith("room")) else None
        if e is not None:
            if e["hidden"]:
                continue
            nodes = sc.graph.geometry_nodes.get(name, [])
            T = sc.graph.get(nodes[0])[0] if nodes else np.eye(4)
            geom = geom.copy()
            geom.apply_transform(T)
            geom.vertices = apply(geom.vertices, e)
            new.add_geometry(geom, node_name=name, geom_name=name)
            continue
        nodes = sc.graph.geometry_nodes.get(name, [])
        T = sc.graph.get(nodes[0])[0] if nodes else np.eye(4)
        new.add_geometry(geom, node_name=name, geom_name=name, transform=T)
    new.export(str(out), file_type="glb")
    return out


def main(argv=None) -> int:
    """``python -m src.scene_edits <scene>``: write the edited GLBs."""
    import argparse
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("scene", type=Path)
    a = ap.parse_args(argv)
    if load(a.scene) is None:
        print(f"{a.scene}: no edits.json (or it is empty)")
        return 1
    for lite in (True, False):
        p = edited_glb(a.scene, lite=lite)
        if p:
            print(f"✓ {p} ({p.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
