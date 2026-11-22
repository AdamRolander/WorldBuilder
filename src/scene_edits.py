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
                        "scale": [sx, sy, sz], "pivot": [cx, cy, cz], "hidden": false,
                        "crop": [[x0, y0, z0], [x1, y1, z1]]},
                 "41": {"source": 12, ...}},
     "nodes": {"room_builtin_03": {"translate": ..., "rotate_deg": ..., "scale": ..., "pivot": ...}},
     "room": {"min": [dx, dy, dz], "max": [dx, dy, dz]},
     "hidden_nodes": ["room_relief_03"]}

An object's vertices become ``pivot + translate + R · (scale ⊙ (v − pivot))``.
``crop`` (optional) is a box in the object's original coordinates; the mesh is
cut to it before it is moved. An entry with ``source`` is a copy: a new object
made from object ``source``'s original mesh, exported as ``<id>_<label>``.
``nodes`` moves, turns and resizes built-ins and relief pieces the same way
(the key is the node name without its ``_body`` / ``_face`` suffix). ``room``
moves the sides of the room box (floor, walls, ceiling) by the given offsets;
the walls keep the photo's texture where they shrink and stretch it where they
grow.
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
MAX_ID = 999                       # object ids are three digits in node names
_NODE_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
_EDITABLE_NODE_RE = re.compile(r"^room_(builtin|relief)_\d{1,3}$")
SHELL = ("room_floor", "room_ceiling", "room_wall")


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


def _transform(e) -> Dict:
    return {"translate": _vec3(e.get("translate"), (0, 0, 0), -1e3, 1e3),
            "rotate_deg": _vec3(e.get("rotate_deg"), (0, 0, 0), -3600, 3600),
            "scale": _vec3(e.get("scale"), (1, 1, 1), 1e-3, 1e3),
            "pivot": _vec3(e.get("pivot"), (0, 0, 0), -1e4, 1e4)}


def _crop(c) -> Optional[List[List[float]]]:
    if not (isinstance(c, (list, tuple)) and len(c) == 2):
        return None
    lo = _vec3(c[0], (-1e4,) * 3, -1e4, 1e4)
    hi = _vec3(c[1], (1e4,) * 3, -1e4, 1e4)
    if any(a >= b for a, b in zip(lo, hi, strict=True)):
        return None
    return [lo, hi]


def sanitize(data) -> Dict:
    """Keep only well-formed, bounded values. The file can arrive over HTTP,
    so nothing in it is trusted: unknown keys are dropped, numbers clamped."""
    out = {"version": 1, "objects": {}, "nodes": {}, "room": None, "hidden_nodes": []}
    if not isinstance(data, dict):
        return out
    objs = data.get("objects")
    if isinstance(objs, dict):
        for k, e in list(objs.items())[:MAX_OBJECTS]:
            if not (isinstance(k, (str, int)) and str(k).isdigit() and isinstance(e, dict)):
                continue
            if not 0 <= int(k) <= MAX_ID:
                continue
            clean = dict(_transform(e), hidden=e.get("hidden") is True)
            crop = _crop(e.get("crop"))
            if crop:
                clean["crop"] = crop
            src = e.get("source")
            if isinstance(src, int) and not isinstance(src, bool) and 0 <= src <= MAX_ID and src != int(k):
                clean["source"] = src
            if is_identity(clean):
                continue
            out["objects"][str(int(k))] = clean
    # a copy of a copy is a copy of the original: sources must be originals
    for k, e in list(out["objects"].items()):
        if "source" in e and "source" in out["objects"].get(str(e["source"]), {}):
            del out["objects"][k]
    nodes = data.get("nodes")
    if isinstance(nodes, dict):
        for k, e in list(nodes.items())[:MAX_OBJECTS]:
            if isinstance(k, str) and _EDITABLE_NODE_RE.match(k) and isinstance(e, dict):
                clean = dict(_transform(e), hidden=False)
                if not is_identity(clean):
                    del clean["hidden"]
                    out["nodes"][k] = clean
    room = data.get("room")
    if isinstance(room, dict):
        r = {"min": _vec3(room.get("min"), (0, 0, 0), -1e3, 1e3), "max": _vec3(room.get("max"), (0, 0, 0), -1e3, 1e3)}
        if not (np.allclose(r["min"], 0) and np.allclose(r["max"], 0)):
            out["room"] = r
    hidden = data.get("hidden_nodes")
    if isinstance(hidden, list):
        out["hidden_nodes"] = sorted({n for n in hidden[:MAX_OBJECTS] if isinstance(n, str) and _NODE_RE.match(n)})
    return out


def is_identity(e: Dict) -> bool:
    return (not e.get("hidden") and not e.get("crop") and "source" not in e
            and np.allclose(e["translate"], 0) and np.allclose(e["rotate_deg"], 0)
            and np.allclose(e["scale"], 1))


def _empty(d: Dict) -> bool:
    return not (d["objects"] or d["nodes"] or d["room"] or d["hidden_nodes"])


def load(scene_dir: Path) -> Optional[Dict]:
    p = Path(scene_dir) / FILENAME
    if not p.exists():
        return None
    try:
        data = sanitize(json.loads(p.read_text()))
    except (OSError, ValueError):
        return None
    return None if _empty(data) else data


def save(scene_dir: Path, data) -> Dict:
    """Write the sanitised edits (or remove the file when nothing is left)."""
    clean = sanitize(data)
    p = Path(scene_dir) / FILENAME
    if not _empty(clean):
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


def copy_sources(edits: Optional[Dict]) -> Dict[int, int]:
    """{copy id: id of the object it was copied from}."""
    if not edits:
        return {}
    return {int(k): e["source"] for k, e in edits["objects"].items() if "source" in e}


_ID_RE = re.compile(r"^(\d{3})_")


def _with_vertices(mesh, vertices):
    m = mesh.copy()
    m.vertices = vertices
    return m


def crop_mesh(mesh, box):
    """The part of ``mesh`` inside the axis-aligned ``box`` ([lo, hi]). Faces
    crossing a side are cut, with their texture coordinates interpolated;
    meshes coloured per vertex lose the faces whose centre is outside instead."""
    import trimesh
    lo, hi = np.asarray(box[0], float), np.asarray(box[1], float)
    uv = getattr(mesh.visual, "uv", None) if getattr(mesh.visual, "kind", None) == "texture" else None
    if getattr(mesh.visual, "kind", None) == "vertex":
        c = mesh.triangles_center
        keep = np.all((c >= lo) & (c <= hi), axis=1)
        m = mesh.copy()
        m.update_faces(keep)
        m.remove_unreferenced_vertices()
        return m
    V, Fc = np.asarray(mesh.vertices, float), np.asarray(mesh.faces)
    for k in range(3):
        for origin, sign in ((lo, 1.0), (hi, -1.0)):
            if len(Fc) == 0:
                break
            if sign > 0 and V[:, k].min() >= origin[k] or sign < 0 and V[:, k].max() <= origin[k]:
                continue
            n = np.zeros(3)
            n[k] = sign
            out = trimesh.intersections.slice_faces_plane(V, Fc, n, origin, uv=uv)
            V, Fc, uv = out[0], out[1], (out[2] if uv is not None else None)
    m = trimesh.Trimesh(V, Fc, process=False)
    if uv is not None:
        m.visual = trimesh.visual.TextureVisuals(uv=uv, material=mesh.visual.material)
    return m


def _remap(v, lo0, hi0, lo1, hi1):
    span = np.where(np.abs(hi0 - lo0) > 1e-9, hi0 - lo0, 1.0)
    return lo1 + (v - lo0) * (hi1 - lo1) / span


def resize_shell(mesh, lo0, hi0, lo1, hi1):
    """A floor, wall or ceiling of the box ``[lo0, hi0]`` moved with the box's sides to
    ``[lo1, hi1]``. Where a side moves in, the texture is cut off with it (the photo
    is not squeezed); where it moves out, the texture is stretched to fill."""
    lo0, hi0, lo1, hi1 = (np.asarray(x, float) for x in (lo0, hi0, lo1, hi1))
    V = np.asarray(mesh.vertices, float)
    new = _remap(V, lo0, hi0, lo1, hi1)
    m = _with_vertices(mesh, new)
    uv = getattr(mesh.visual, "uv", None) if getattr(mesh.visual, "kind", None) == "texture" else None
    if uv is not None and len(V) >= 3:
        X = np.c_[V, np.ones(len(V))]
        A, *_ = np.linalg.lstsq(X, uv, rcond=None)
        if np.abs(X @ A - uv).max() < 1e-4:          # a flat, linearly mapped piece
            # where the texture comes from: the new position, but not beyond the old box
            q = np.clip(new, np.minimum(lo0, hi0), np.maximum(lo0, hi0))
            m.visual.uv = np.c_[q, np.ones(len(q))] @ A
    return m


def room_box(edits: Optional[Dict], lo, hi):
    """The room box after the edits: (lo, hi)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    r = (edits or {}).get("room")
    if not r:
        return lo, hi
    return lo + np.asarray(r["min"], float), hi + np.asarray(r["max"], float)


def _node_key(name: str) -> Optional[str]:
    m = re.match(r"^(room_(?:builtin|relief)_\d{1,3})(?:_(?:body|face))?$", name)
    return m.group(1) if m else None


def apply_to_meshes(named, edits: Optional[Dict], room_bounds=None):
    """Apply ``edits`` to ``[(node name, mesh in scene coordinates)]`` and return the
    new list: hidden pieces left out, objects cropped and moved, copies added,
    built-ins moved and the room's sides moved (``room_bounds``: the layout's
    ``(bounds_min, bounds_max)``, needed for that last one)."""
    if not edits:
        return list(named)
    hidden = set(edits["hidden_nodes"])
    out, originals = [], {}
    for name, m in named:
        mid = _ID_RE.match(name)
        if mid and not name.startswith("room"):
            originals.setdefault(int(mid.group(1)), []).append((name, m))
    for name, m in named:
        if name in hidden:
            continue
        mid = _ID_RE.match(name)
        if mid and not name.startswith("room"):
            e = for_object(edits, int(mid.group(1)))
            if e is not None:
                if e["hidden"]:
                    continue
                if e.get("crop"):
                    m = crop_mesh(m, e["crop"])
                m = _with_vertices(m, apply(m.vertices, e))
            out.append((name, m))
            continue
        key = _node_key(name)
        if key and key in edits["nodes"]:
            m = _with_vertices(m, apply(m.vertices, edits["nodes"][key]))
        elif name.startswith(SHELL) and edits.get("room") and room_bounds is not None:
            lo0, hi0 = (np.asarray(x, float) for x in room_bounds)
            lo1, hi1 = room_box(edits, lo0, hi0)
            m = resize_shell(m, lo0, hi0, lo1, hi1)
        out.append((name, m))
    for cid, e in sorted(copy_sources(edits).items()):
        ce = edits["objects"][str(cid)]
        if ce["hidden"] or cid in originals:          # the id is taken by a real object
            continue
        for name, m in originals.get(e, []):
            if ce.get("crop"):
                m = crop_mesh(m, ce["crop"])
            out.append((f"{cid:03d}{name[3:]}", _with_vertices(m, apply(m.vertices, ce))))
    return out


def edited_layout(layout: Dict, edits: Optional[Dict]) -> Dict:
    """``layout.json`` as the edits leave it: the room box (and floor height) after
    its sides moved, and the solid built-in boxes moved, resized or left out."""
    if not edits:
        return layout
    out = dict(layout)
    if edits.get("room") and "bounds_min" in layout:
        lo, hi = room_box(edits, layout["bounds_min"], layout["bounds_max"])
        out["bounds_min"], out["bounds_max"] = lo.tolist(), hi.tolist()
        if "floor_y" in layout:
            out["floor_y"] = float(layout["floor_y"]) + edits["room"]["min"][1]
    boxes = []
    hidden = set(edits["hidden_nodes"])
    for i, b in enumerate(layout.get("builtins") or []):
        key = f"room_builtin_{i:02d}"
        if f"{key}_body" in hidden or key in hidden:
            continue
        e = edits["nodes"].get(key)
        if e is not None:
            lo, hi = np.asarray(b["min"], float), np.asarray(b["max"], float)
            corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            c = apply(corners, e)
            b = dict(b, min=c.min(0).tolist(), max=c.max(0).tolist())
        boxes.append(b)
    if "builtins" in layout:
        out["builtins"] = boxes
    return out


def _layout_bounds(scene_dir: Path):
    p = Path(scene_dir) / "3d_models" / "layout.json"
    try:
        lay = json.loads(p.read_text())
        return lay["bounds_min"], lay["bounds_max"]
    except (OSError, ValueError, KeyError):
        return None


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
    named = []
    for name, geom in sc.geometry.items():
        nodes = sc.graph.geometry_nodes.get(name, [])
        T = sc.graph.get(nodes[0])[0] if nodes else np.eye(4)
        g = geom.copy()
        if not np.allclose(T, np.eye(4)):
            g.apply_transform(T)
        named.append((name, g))
    new = trimesh.Scene()
    new.metadata.update(sc.metadata)
    for name, g in apply_to_meshes(named, edits, _layout_bounds(scene_dir)):
        new.add_geometry(g, node_name=name, geom_name=name)
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
