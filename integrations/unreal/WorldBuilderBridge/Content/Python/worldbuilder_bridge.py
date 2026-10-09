"""WorldBuilder Bridge — Unreal Editor Python side.

Run from the Output Log (Python) or bind to an Editor Utility Widget button:

    import worldbuilder_bridge as wb
    wb.generate_and_import(r"C:/photos/dorm.jpg")                  # blocking
    wb.generate_and_import(r"/home/me/dorm.jpg", detector="qwen")   # fully local server

What it does:
  1. POST the photo to a running WorldBuilder webapp (``/api/upload``).
  2. Poll ``/api/jobs/<id>`` until complete (prints stage names).
  3. Download ``/api/scenes/<id>/glb`` to the project's Saved/WorldBuilder dir.
  4. Import it with Interchange (glTF), then spawn it in the current level.

The GLB is Y-up, metres-ish (scale-invariant; see layout.json). Interchange
converts to Unreal's Z-up, cm frame; ``UNIT_SCALE`` lets you apply the
estimated metric scale from the scene manifest so a 2.6 m ceiling is 260 cm.

Only stdlib + ``unreal``. A C++ module (``Source/WorldBuilderBridge``) is
scaffolded so the plugin packages as a *code plugin* for Fab; it can stay
empty apart from the module boilerplate until an editor UI is wanted.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
import uuid

try:
    import unreal  # type: ignore
except ImportError:  # allows importing this file outside the editor for linting/tests
    unreal = None

SERVER = os.environ.get("WORLDBUILDER_SERVER", "http://127.0.0.1:5174")
UNIT_SCALE_FROM_LAYOUT = True   # multiply by layout.estimated_metric_scale (camera-height prior)

STAGES = {1: "detecting objects", 2: "segmenting", 3: "reconstructing 3D", 4: "laying out room", 5: "finishing"}


def _log(msg: str):
    (unreal.log if unreal else print)(f"[WorldBuilder] {msg}")


def _get_json(url: str, timeout: int = 15):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _multipart(fields: dict, files: dict):
    boundary = uuid.uuid4().hex
    body = bytearray()
    for k, v in fields.items():
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    for k, (filename, data, ctype) in files.items():
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{filename}\"\r\n"
                 f"Content-Type: {ctype}\r\n\r\n").encode() + data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def upload(photo_path: str, detector: str | None = None, server: str = SERVER) -> dict:
    with open(photo_path, "rb") as f:
        data = f.read()
    ext = os.path.splitext(photo_path)[1].lower().lstrip(".") or "jpg"
    body, ctype = _multipart({"detector": detector} if detector else {},
                             {"image": (os.path.basename(photo_path), data, f"image/{ext}")})
    req = urllib.request.Request(f"{server}/api/upload", data=body, method="POST",
                                 headers={"Content-Type": ctype})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def wait_for_job(job_id: str, server: str = SERVER, poll_s: float = 2.0) -> dict:
    last = None
    while True:
        st = _get_json(f"{server}/api/jobs/{job_id}")
        stage = STAGES.get(st.get("stage", 0), "queued")
        if stage != last:
            _log(stage + "…")
            last = stage
        if st["status"] == "complete":
            return st
        if st["status"] == "failed":
            raise RuntimeError(st.get("error") or "pipeline failed")
        time.sleep(poll_s)


def download_scene(scene_id: str, server: str = SERVER) -> tuple[str, dict]:
    if unreal:
        out_dir = os.path.join(unreal.Paths.project_saved_dir(), "WorldBuilder")
    else:
        out_dir = os.path.join(os.getcwd(), "WorldBuilder")
    os.makedirs(out_dir, exist_ok=True)
    glb = os.path.join(out_dir, f"{scene_id}.glb")
    urllib.request.urlretrieve(f"{server}/api/scenes/{urllib.parse.quote(scene_id)}/glb", glb)
    manifest = _get_json(f"{server}/api/scenes/{urllib.parse.quote(scene_id)}")
    return glb, manifest


def import_glb(glb_path: str, scene_id: str, metric_scale: float = 1.0):
    """Interchange import into /Game/WorldBuilder/<scene_id>, then place in the level."""
    if not unreal:
        raise RuntimeError("import_glb must run inside the Unreal Editor")
    dest = f"/Game/WorldBuilder/{scene_id}"
    source = unreal.InterchangeManager.create_source_data(glb_path)
    params = unreal.ImportAssetParameters()
    params.is_automated = True
    mgr = unreal.InterchangeManager.get_interchange_manager_scripted()
    mgr.import_asset(dest, source, params)
    # Interchange runs async; give the asset registry a moment on first import.
    unreal.AssetRegistryHelpers.get_asset_registry().scan_paths_synchronous([dest], True)
    assets = unreal.EditorAssetLibrary.list_assets(dest, recursive=True, include_folder=False)
    meshes = [a for a in assets if unreal.EditorAssetLibrary.find_asset_data(a).asset_class_path.asset_name == "StaticMesh"]
    _log(f"imported {len(meshes)} static meshes under {dest}")
    sub = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    root = sub.spawn_actor_from_class(unreal.Actor, unreal.Vector(0, 0, 0))
    root.set_actor_label(f"WorldBuilder_{scene_id}")
    for path in meshes:
        mesh = unreal.EditorAssetLibrary.load_asset(path)
        actor = sub.spawn_actor_from_object(mesh, unreal.Vector(0, 0, 0))
        actor.set_actor_label(mesh.get_name())
        actor.set_actor_scale3d(unreal.Vector(metric_scale, metric_scale, metric_scale))
        actor.attach_to_actor(root, "", unreal.AttachmentRule.KEEP_WORLD,
                              unreal.AttachmentRule.KEEP_WORLD, unreal.AttachmentRule.KEEP_WORLD, False)
    return root


def generate_and_import(photo_path: str, detector: str | None = None, server: str = SERVER):
    """One call: photo in, actors in the level. Blocks the editor while the
    pipeline runs (3–10 min on a 5090); fine for a first integration, an
    async Editor Utility Widget is the follow-up (ROADMAP DCC-4)."""
    _log(f"uploading {photo_path} to {server}")
    job = upload(photo_path, detector, server)
    wait_for_job(job["job_id"], server)
    glb, manifest = download_scene(job["scene_id"], server)
    scale = 1.0
    layout = (manifest or {}).get("layout") or {}
    if UNIT_SCALE_FROM_LAYOUT and layout.get("estimated_metric_scale"):
        scale = float(layout["estimated_metric_scale"])
        _log(f"applying estimated metric scale ×{scale:.2f} (camera-height prior; adjust if needed)")
    return import_glb(glb, job["scene_id"], scale)
