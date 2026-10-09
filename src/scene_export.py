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
                     filename: str = "scene.glb", include_room: bool = True,
                     lite_faces: Optional[int] = None) -> Optional[Path]:
    """Write ``<output_dir>/<filename>``. Returns the path or None if empty.

    With ``lite_faces`` every object is decimated to that triangle budget
    and its vertex colours are baked into a texture (``src/mesh_bake.py``):
    the file shrinks by one to two orders of magnitude and is what game
    engines, simulators and phones should load.
    """
    output_dir = Path(output_dir)
    scene = trimesh.Scene()
    n = 0
    scene_extent = None
    if lite_faces:
        from src.mesh_bake import lite_mesh, texture_budget
        lp = output_dir / "3d_models" / "layout.json"
        try:
            lj = json.loads(lp.read_text())
            scene_extent = float(np.max(np.asarray(lj["bounds_max"]) - np.asarray(lj["bounds_min"])))
        except (OSError, ValueError, KeyError):
            scene_extent = None
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
        if lite_faces:
            ext = float(np.max(mesh.extents))
            lite = lite_mesh(mesh, target_faces=lite_faces, name=name,
                             tex_size=texture_budget(ext, scene_extent or 4 * ext))
            lite.metadata.update(mesh.metadata)
            mesh = lite
        mesh.metadata.update({"label": r.get("label"), "confidence": r.get("confidence"),
                              "supported_by": r.get("supported_by"), "support": r.get("support")})
        scene.add_geometry(mesh, node_name=name, geom_name=name)
        n += 1
    if include_room:
        room_glb = (output_dir / room_file).with_suffix(".glb")
        if room_glb.exists():
            # textured shell + relief, one named node per surface
            rs = trimesh.load(str(room_glb), process=False)
            for gname, geom in rs.geometry.items():
                scene.add_geometry(geom, node_name=gname, geom_name=gname)
        else:
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


LITE_FACES = 8000


def export_scene_lite(output_dir: Path, results: List[Dict], faces: int = LITE_FACES) -> Optional[Path]:
    """``scene_lite.glb``: same scene, decimated objects with baked textures."""
    return export_scene_glb(output_dir, results, filename="scene_lite.glb", lite_faces=faces)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for d in args:
        d = Path(d)
        data = json.loads((d / "reconstruction_results.json").read_text())
        p = export_scene_glb(d, data["objects"])
        print(f"✓ {p}" if p else f"nothing to export in {d}")
        if "--lite" in sys.argv:
            p = export_scene_lite(d, data["objects"])
            print(f"✓ {p} ({p.stat().st_size / 1e6:.1f} MB)" if p else "no lite export")
