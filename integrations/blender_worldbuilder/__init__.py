"""WorldBuilder for Blender — photo → 3D scene, via a local WorldBuilder server.

Design: the add-on is a thin HTTP client. All the heavy models (SAM 3, SAM 3D
Objects, MoGe, the VLM) run in the WorldBuilder webapp process on a GPU box
(possibly this machine). That keeps the add-on tiny, GPL-compatible, and
free of Meta's SAM licence terms, which the Extensions platform would not
accept in a bundled form.

Flow:  pick a photo → POST /api/upload → poll /api/jobs/<id> → GET
/api/scenes/<id>/glb → bpy.ops.import_scene.gltf → objects land in a
collection named after the scene with the room as a separate object.

Works on Blender 4.2+ (extension) and as a legacy add-on on 3.6+ via
``bl_info``. Only stdlib + bpy; no wheels.
"""
import json
import os
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid

import bpy
from bpy.props import BoolProperty, StringProperty
from bpy.types import AddonPreferences, Operator, Panel

bl_info = {
    "name": "WorldBuilder",
    "author": "Adam Rolander",
    "version": (0, 1, 0),
    "blender": (3, 6, 0),
    "location": "3D Viewport > Sidebar > WorldBuilder",
    "description": "Turn a photo of a room into a 3D scene via a local WorldBuilder server",
    "category": "Import-Export",
}

_PKG = __package__ or "worldbuilder"


# --------------------------------------------------------------------------
# HTTP helpers (stdlib only; blocking calls run in a worker thread)
# --------------------------------------------------------------------------

def _multipart(fields, files):
    boundary = uuid.uuid4().hex
    body = bytearray()
    for k, v in fields.items():
        body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    for k, (filename, data, ctype) in files.items():
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"; "
                 f"filename=\"{filename}\"\r\nContent-Type: {ctype}\r\n\r\n").encode()
        body += data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def _get_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def upload_photo(server, path, detector):
    with open(path, "rb") as f:
        data = f.read()
    ext = os.path.splitext(path)[1].lower().lstrip(".") or "jpg"
    body, ctype = _multipart({"detector": detector} if detector else {},
                             {"image": (os.path.basename(path), data, f"image/{ext}")})
    req = urllib.request.Request(f"{server}/api/upload", data=body, method="POST",
                                 headers={"Content-Type": ctype})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def download_glb(server, scene_id, dest_dir):
    dest = os.path.join(dest_dir, f"{scene_id}.glb")
    urllib.request.urlretrieve(f"{server}/api/scenes/{urllib.parse.quote(scene_id)}/glb", dest)
    return dest


# --------------------------------------------------------------------------
# Preferences + scene properties
# --------------------------------------------------------------------------

class WB_Preferences(AddonPreferences):
    bl_idname = _PKG
    server_url: StringProperty(name="Server URL", default="http://127.0.0.1:5174",
                               description="Where `python -m webapp.server` is running")
    detector: StringProperty(name="Detector", default="",
                             description="Leave empty for the server default; 'qwen' for fully local, 'gemini' for cloud")

    def draw(self, context):
        col = self.layout.column()
        col.prop(self, "server_url")
        col.prop(self, "detector")
        col.label(text="Run the WorldBuilder webapp on a GPU machine; this add-on only talks HTTP to it.")


def _prefs(context):
    return context.preferences.addons[_PKG].preferences


# --------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------

_STATE = {"status": "", "job": None, "error": None, "glb": None, "scene_id": None}


class WB_OT_generate(Operator):
    """Upload a photo to the WorldBuilder server and import the resulting scene"""
    bl_idname = "worldbuilder.generate"
    bl_label = "Generate scene from photo"
    bl_options = {"REGISTER"}

    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(default="*.jpg;*.jpeg;*.png;*.webp", options={"HIDDEN"})
    import_room: BoolProperty(name="Import room (floor/walls/ceiling)", default=True)

    _timer = None
    _thread = None

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        prefs = _prefs(context)
        server = prefs.server_url.rstrip("/")
        path = bpy.path.abspath(self.filepath)
        if not os.path.exists(path):
            self.report({"ERROR"}, f"File not found: {path}")
            return {"CANCELLED"}
        _STATE.update(status="uploading…", job=None, error=None, glb=None, scene_id=None)

        def work():
            try:
                job = upload_photo(server, path, prefs.detector.strip() or None)
                _STATE["job"] = job
                _STATE["scene_id"] = job["scene_id"]
                stages = {1: "detecting objects", 2: "segmenting", 3: "reconstructing 3D",
                          4: "laying out room", 5: "finishing"}
                import time
                while True:
                    st = _get_json(f"{server}/api/jobs/{job['job_id']}")
                    _STATE["status"] = f"{stages.get(st.get('stage', 0), 'queued')}…"
                    if st["status"] == "complete":
                        break
                    if st["status"] == "failed":
                        raise RuntimeError(st.get("error") or "pipeline failed")
                    time.sleep(2)
                _STATE["status"] = "downloading GLB…"
                _STATE["glb"] = download_glb(server, job["scene_id"], tempfile.gettempdir())
                _STATE["status"] = "importing"
            except urllib.error.URLError as e:
                _STATE["error"] = f"Cannot reach WorldBuilder server at {server}: {e.reason}"
            except Exception as e:  # noqa: BLE001
                _STATE["error"] = str(e)

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()
        self._timer = context.window_manager.event_timer_add(0.5, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
        if _STATE["error"]:
            self._finish(context)
            self.report({"ERROR"}, _STATE["error"])
            return {"CANCELLED"}
        if _STATE["glb"]:
            glb = _STATE["glb"]
            self._finish(context)
            self._import(context, glb, _STATE["scene_id"])
            return {"FINISHED"}
        return {"RUNNING_MODAL"}

    def _finish(self, context):
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

    def _import(self, context, glb, scene_id):
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=glb)
        new = [o for o in bpy.data.objects if o not in before]
        coll = bpy.data.collections.new(f"WorldBuilder {scene_id}")
        context.scene.collection.children.link(coll)
        for o in new:
            for c in list(o.users_collection):
                c.objects.unlink(o)
            coll.objects.link(o)
            if o.name.lower().startswith("room") and not self.import_room:
                bpy.data.objects.remove(o)
        _STATE["status"] = f"imported {len(new)} objects"
        self.report({"INFO"}, f"WorldBuilder: imported {len(new)} objects into '{coll.name}'")


class WB_OT_check_server(Operator):
    """Ping the WorldBuilder server"""
    bl_idname = "worldbuilder.check_server"
    bl_label = "Check server"

    def execute(self, context):
        server = _prefs(context).server_url.rstrip("/")
        try:
            h = _get_json(f"{server}/api/health", timeout=5)
            v = _get_json(f"{server}/api/vlm-status", timeout=5)
            local = f"local VLM: {v.get('model')}" if v.get("ok") else "local VLM offline (Gemini only)"
            self.report({"INFO"}, f"Server OK ({h.get('time')}); {local}")
        except Exception as e:  # noqa: BLE001
            self.report({"ERROR"}, f"Server unreachable: {e}")
        return {"FINISHED"}


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

class WB_PT_panel(Panel):
    bl_label = "WorldBuilder"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "WorldBuilder"

    def draw(self, context):
        col = self.layout.column(align=True)
        col.operator(WB_OT_generate.bl_idname, icon="IMAGE_DATA")
        col.operator(WB_OT_check_server.bl_idname, icon="URL")
        if _STATE["status"]:
            col.label(text=_STATE["status"])
        col.separator()
        col.label(text=f"Server: {_prefs(context).server_url}")


classes = (WB_Preferences, WB_OT_generate, WB_OT_check_server, WB_PT_panel)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
