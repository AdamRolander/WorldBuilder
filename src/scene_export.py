"""Export a finished scene as one glTF binary (GLB) for Blender / Unreal / web.

PLY-per-object is fine for the built-in viewer but awkward for DCC tools:
no hierarchy, no names, and vertex colours are not always imported. A GLB
carries every object as a named node (``<id>_<label>``) with vertex colours
as ``COLOR_0``, plus the room as its own node, in a right-handed Y-up frame
that Blender and three.js consume directly (Unreal's importer converts to
Z-up on the way in).

Reads the world-space PLYs the pipeline bakes, so it can also be run after
the fact: ``python -m src.scene_export outputs/<scene>``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import trimesh


def _load_world_mesh(ply_path: Path, name: str) -> Optional[trimesh.Trimesh]:
    if not ply_path.exists():
        return None
    m = trimesh.load(str(ply_path), process=False)
    if not isinstance(m, trimesh.Trimesh) or len(m.faces) == 0:
        return None
    colors = None
    if m.visual.kind == "vertex":
        colors = np.asarray(m.visual.vertex_colors)
    out = trimesh.Trimesh(vertices=np.asarray(m.vertices, dtype=np.float32), faces=m.faces,
                          vertex_colors=colors, process=False)
    out.metadata["name"] = name
    return out


def export_scene_glb(output_dir: Path, results: List[Dict], room_file: str = "3d_models/room.ply",
                     filename: str = "scene.glb", include_room: bool = True) -> Optional[Path]:
    """Write ``<output_dir>/<filename>``. Returns the path or None if empty."""
    output_dir = Path(output_dir)
    scene = trimesh.Scene()
    n = 0
    for r in results:
        p = r.get("ply_path")
        if not p:
            continue
        p = Path(p)
        if not p.is_absolute():
            p = (Path.cwd() / p)
        name = f"{int(r.get('id', n + 1)):03d}_{str(r.get('label', 'object')).replace(' ', '_')}"
        mesh = _load_world_mesh(p, name)
        if mesh is None:
            continue
        mesh.metadata.update({"label": r.get("label"), "confidence": r.get("confidence"),
                              "supported_by": r.get("supported_by")})
        scene.add_geometry(mesh, node_name=name, geom_name=name)
        n += 1
    if include_room:
        room = _load_world_mesh(output_dir / room_file, "room")
        if room is not None:
            scene.add_geometry(room, node_name="room", geom_name="room")
    if n == 0 and len(scene.geometry) == 0:
        return None
    meta = {}
    layout_path = output_dir / "3d_models" / "layout.json"
    if layout_path.exists():
        try:
            lj = json.loads(layout_path.read_text())
            meta["worldbuilder_layout"] = {k: lj.get(k) for k in
                                           ("floor_y", "ceiling_y", "estimated_metric_scale", "yaw_deg")}
        except ValueError:
            pass
    scene.metadata.update({"generator": "WorldBuilder", "units": "scene (scale-invariant; see layout)", **meta})
    out = output_dir / filename
    scene.export(str(out), file_type="glb")
    return out


if __name__ == "__main__":
    for d in sys.argv[1:]:
        d = Path(d)
        data = json.loads((d / "reconstruction_results.json").read_text())
        p = export_scene_glb(d, data["objects"])
        print(f"✓ {p}" if p else f"nothing to export in {d}")
