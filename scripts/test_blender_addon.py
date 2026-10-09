"""End-to-end test of the Blender add-on against a running WorldBuilder server.

    blender --background --python scripts/test_blender_addon.py -- \
        --server http://127.0.0.1:5174 --photo test_images/bedroom.jpeg --render /tmp/wb_blender.png

    # import a scene the server already has instead of uploading:
    blender --background --python scripts/test_blender_addon.py -- --scene k1_q --render /tmp/k1.png

Registers the add-on from ``integrations/blender_worldbuilder``, drives the
same functions the UI operators call (upload → poll → download → import),
checks what landed in the .blend, and renders one frame so a human can
look at it. Exits non-zero on any failure, so it can run in CI on a
machine that has Blender and a server.
"""
import argparse
import importlib.util
import sys
import tempfile
import time
from pathlib import Path

import bpy

ROOT = Path(__file__).resolve().parent.parent


def load_addon():
    path = ROOT / "integrations" / "blender_worldbuilder" / "__init__.py"
    spec = importlib.util.spec_from_file_location("worldbuilder", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["worldbuilder"] = mod
    spec.loader.exec_module(mod)
    mod.register()
    return mod


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:5174")
    ap.add_argument("--photo")
    ap.add_argument("--scene", help="existing scene id (skips the upload)")
    ap.add_argument("--detector", default="qwen")
    ap.add_argument("--full", action="store_true", help="download the full GLB instead of the lite one")
    ap.add_argument("--render", type=Path)
    ap.add_argument("--save-blend", type=Path)
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args(argv)

    wb = load_addon()
    assert hasattr(bpy.ops.worldbuilder, "generate") and hasattr(bpy.ops.worldbuilder, "import_existing")
    print(f"[test] add-on registered: {wb.bl_info['name']} {wb.bl_info['version']} on Blender {bpy.app.version_string}")
    health = wb._get_json(f"{a.server}/api/health", timeout=5)
    assert health.get("ok"), health

    t0 = time.time()
    scene_id = a.scene
    if not scene_id:
        assert a.photo, "--photo or --scene is required"
        job = wb.upload_photo(a.server, a.photo, a.detector)
        scene_id = job["scene_id"]
        print(f"[test] uploaded {a.photo} → job {job['job_id']} scene {scene_id}")
        last = None
        while True:
            st = wb._get_json(f"{a.server}/api/jobs/{job['job_id']}")
            if st["stage"] != last:
                print(f"[test]   stage {st['stage']} ({st['status']}) at {time.time() - t0:.0f}s")
                last = st["stage"]
            if st["status"] == "complete":
                break
            if st["status"] == "failed":
                raise SystemExit(f"[test] pipeline failed: {st.get('error')}")
            if time.time() - t0 > a.timeout:
                raise SystemExit("[test] timed out")
            time.sleep(3)
    t_pipeline = time.time() - t0

    # start from an empty file so counts are exact
    bpy.ops.wm.read_factory_settings(use_empty=True)
    wb.register() if not hasattr(bpy.types, "WB_PT_panel") else None
    t1 = time.time()
    glb = wb.download_glb(a.server, scene_id, tempfile.gettempdir(), lite=not a.full)
    scale = wb.metric_scale(a.server, scene_id)
    objs = wb.import_glb(bpy.context, glb, scene_id, import_room=True, scale=scale)
    t_import = time.time() - t1
    manifest = wb._get_json(f"{a.server}/api/scenes/{scene_id}")
    meshes = [o for o in objs if o.type == "MESH"]
    room = [o for o in meshes if o.name.lower().startswith("room")]
    things = [o for o in meshes if not o.name.lower().startswith("room")]
    tris = sum(len(o.data.polygons) for o in meshes)
    textured = sum(1 for o in meshes if any(n.type == "TEX_IMAGE" for m in o.data.materials if m and m.use_nodes
                                            for n in m.node_tree.nodes))
    print(f"[test] GLB {Path(glb).stat().st_size / 1e6:.1f} MB, import {t_import:.1f}s, scale ×{scale:.3f}")
    print(f"[test] {len(things)} objects (server reports {len(manifest['objects'])}), {len(room)} room parts, "
          f"{tris:,} polygons, {textured}/{len(meshes)} meshes with image textures")
    assert len(things) == len(manifest["objects"]), "object count differs from the server manifest"
    assert len(room) >= 5, "room surfaces missing"
    assert f"WorldBuilder {scene_id}" in bpy.data.collections

    if a.render:
        # Camera at the photo's viewpoint: the scene root's origin, looking
        # along +Y in Blender (glTF -Z forward… the importer maps scene +Z to -Y).
        import mathutils
        layout = manifest.get("layout") or {}
        scn = bpy.context.scene
        cam_data = bpy.data.cameras.new("photo_cam")
        cam = bpy.data.objects.new("photo_cam", cam_data)
        scn.collection.objects.link(cam)
        cam.location = (0.0, 0.0, 0.0)
        # scene frame: X left, Y up, Z forward → Blender: X left… glTF import: (x, y, z) → (x, -z, y)
        fwd = mathutils.Vector((0.0, -1.0, 0.0))
        cam.rotation_euler = fwd.to_track_quat("-Z", "Y").to_euler()
        cam_data.lens = 18
        cam_data.clip_start = 0.01
        scn.camera = cam
        scn.render.resolution_x, scn.render.resolution_y = 960, 720
        engines = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
        scn.render.engine = "BLENDER_WORKBENCH"
        scn.display.shading.light = "FLAT"
        scn.display.shading.color_type = "TEXTURE"
        scn.render.filepath = str(a.render)
        bpy.ops.render.render(write_still=True)
        print(f"[test] rendered {a.render} with {scn.render.engine} (available: {engines}); layout notes: "
              f"{(layout.get('notes') or [''])[0]}")
    if a.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(a.save_blend))
    print(f"[test] PASS — pipeline {t_pipeline:.0f}s, import {t_import:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"[test] FAIL: {e}")
        sys.exit(1)
