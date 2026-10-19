"""WorldBuilder for Blender — photo → 3D scene, via a local WorldBuilder server.

Design: the add-on is a thin HTTP client. All the heavy models (SAM 3, SAM 3D
Objects, MoGe, the VLM) run in the WorldBuilder webapp process on a GPU box
(possibly this machine). That keeps the add-on tiny, GPL-compatible, and
free of Meta's SAM licence terms, which the Extensions platform would not
accept in a bundled form.

Flow:  pick a photo → POST /api/upload → poll /api/jobs/<id> → GET
/api/scenes/<id>/glb?lite=1 → bpy.ops.import_scene.gltf → objects land in a
collection named after the scene under one empty (scaled to metres), with
each room surface and the photo relief as separate objects. "Import
existing scene" skips the upload and pulls a scene the server already has.

Works on Blender 4.2+ (extension) and as a legacy add-on on 3.6+ via
``bl_info``. Only stdlib + bpy; no wheels.
"""
import json
import os
import re
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid

import bpy
from bpy.props import BoolProperty, EnumProperty, PointerProperty, StringProperty
from bpy.types import AddonPreferences, Operator, Panel, PropertyGroup

bl_info = {
    "name": "WorldBuilder",
    "author": "Adam Rolander",
    "version": (0, 4, 0),
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


def download_glb(server, scene_id, dest_dir, lite=True):
    """``lite``: decimated meshes with baked textures (tens of MB). The full
    GLB is SAM 3D's raw output with vertex colours and can be hundreds of MB."""
    dest = os.path.join(dest_dir, f"{scene_id}{'_lite' if lite else ''}.glb")
    url = f"{server}/api/scenes/{urllib.parse.quote(scene_id)}/glb" + ("?lite=1" if lite else "")
    urllib.request.urlretrieve(url, dest)
    return dest


def metric_scale(server, scene_id):
    """Scene units -> metres, from the server's layout estimate (1.0 if unknown)."""
    try:
        m = _get_json(f"{server}/api/scenes/{urllib.parse.quote(scene_id)}")
        return float((m.get("layout") or {}).get("estimated_metric_scale") or 1.0)
    except Exception:  # noqa: BLE001
        return 1.0


def import_glb(context, glb, scene_id, import_room=True, scale=1.0):
    """Import a WorldBuilder GLB into its own collection under one empty.

    Kept free of operator state so it can be driven from a script or a test
    (``blender --background --python ...``). Returns the imported objects.
    """
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=glb)
    new = [o for o in bpy.data.objects if o not in before]
    coll = bpy.data.collections.new(f"WorldBuilder {scene_id}")
    context.scene.collection.children.link(coll)
    root = bpy.data.objects.new(f"WorldBuilder {scene_id}", None)
    coll.objects.link(root)
    root.scale = (scale, scale, scale)
    kept = []
    for o in new:
        if o.name.lower().startswith("room") and not import_room:
            bpy.data.objects.remove(o)
            continue
        for c in list(o.users_collection):
            c.objects.unlink(o)
        coll.objects.link(o)
        if o.parent is None:
            o.parent = root
        kept.append(o)
    return kept


# --------------------------------------------------------------------------
# Preferences + scene properties
# --------------------------------------------------------------------------

PHOTO_TYPES = "JPG, PNG, WebP, BMP or TIFF"
DOCS_URL = "https://github.com/AdamRolander/WorldBuilder/blob/main/docs/plugins/BLENDER.md"


class WB_Preferences(AddonPreferences):
    bl_idname = _PKG
    server_url: StringProperty(name="Server URL", default="http://127.0.0.1:5174",
                               description="Where `python -m webapp.server` is running (port 5174 by default)")
    detector: StringProperty(name="Detector", default="",
                             description="A detector name, not a URL. Empty: the server's default; "
                                         "'qwen': fully local (needs vlm_server/server.py running); "
                                         "'gemini': cloud (needs GEMINI_API_KEY in the server's .env)")

    def draw(self, context):
        col = self.layout.column()
        col.prop(self, "server_url")
        col.prop(self, "detector")
        col.label(text="Detector: leave empty, or type qwen (local) or gemini (cloud).")
        col.label(text="Start the WorldBuilder server on a GPU machine; this add-on only talks HTTP to it.")


def _prefs(context):
    return context.preferences.addons[_PKG].preferences


def _server(context):
    return _prefs(context).server_url.rstrip("/")


class WB_Settings(PropertyGroup):
    review_first: BoolProperty(
        name="Review in Browser First", default=True,
        description="When the scene is ready, open it in the web viewer to fix it, then import it "
                    "with Import Into Blender. Off: import straight away")
    import_room: BoolProperty(name="Room", default=True,
                              description="Import the floor, walls, ceiling and built-in furniture")
    full_detail: BoolProperty(
        name="Full Detail", default=False,
        description="SAM 3D's original dense meshes with vertex colours (often 10-20x larger "
                    "and slower in the viewport). Off: lighter meshes with baked textures")
    use_metric_scale: BoolProperty(name="Scale to Metres", default=True,
                                   description="Apply the server's estimate of the scene's scale in metres")


# --------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------

# Shared between the worker thread and the UI. "progress" mirrors the server's
# job progress; "review" holds a scene waiting to be imported after browser review.
_STATE = {"status": "", "job": None, "error": None, "glb": None, "scene_id": None,
          "progress": None, "review": None, "busy": False, "last_import": ""}


def _fmt_minutes(seconds):
    if seconds is None:
        return ""
    if seconds < 60:
        return "under a minute"
    return f"about {round(seconds / 60)} min"


def _fmt_clock(seconds):
    seconds = int(seconds or 0)
    return f"{seconds // 60}:{seconds % 60:02d}"


def _viewer_url(server, scene_id):
    return f"{server}/outputs/{urllib.parse.quote(scene_id)}/viewer.html"


def _import_scene(context, server, scene_id, settings):
    """Download and import one scene with the panel's settings; returns a short summary."""
    lite = not settings.full_detail
    glb = download_glb(server, scene_id, tempfile.gettempdir(), lite=lite)
    scale = metric_scale(server, scene_id) if settings.use_metric_scale else 1.0
    new = import_glb(context, glb, scene_id, settings.import_room, scale)
    n = sum(1 for o in new if o.type == "MESH" and not o.name.lower().startswith("room"))
    mb = os.path.getsize(glb) / 1e6
    _STATE["scene_id"] = scene_id
    _STATE["last_import"] = f"{scene_id}: {n} objects, {mb:.0f} MB ({'full' if not lite else 'lite'})"
    return n, mb


class WB_OT_generate(Operator):
    """Upload a photo to the WorldBuilder server and build a 3D scene from it. """ \
        """Accepts JPG, PNG, WebP, BMP or TIFF. Takes a few minutes"""
    bl_idname = "worldbuilder.generate"
    bl_label = "Generate Scene From Photo"
    bl_options = {"REGISTER"}

    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(default="*.jpg;*.jpeg;*.png;*.webp;*.bmp;*.tif;*.tiff", options={"HIDDEN"})

    _timer = None
    _thread = None

    @classmethod
    def poll(cls, context):
        return not _STATE["busy"]

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        prefs = _prefs(context)
        server = _server(context)
        path = bpy.path.abspath(self.filepath)
        if not os.path.exists(path):
            self.report({"ERROR"}, f"File not found: {path}")
            return {"CANCELLED"}
        _STATE.update(status="Uploading photo…", job=None, error=None, glb=None, scene_id=None,
                      progress=None, review=None, busy=True)

        def work():
            import time
            try:
                job = upload_photo(server, path, prefs.detector.strip() or None)
                _STATE["job"] = job
                _STATE["scene_id"] = job["scene_id"]
                while True:
                    st = _get_json(f"{server}/api/jobs/{job['job_id']}")
                    if st["status"] == "complete":
                        break
                    if st["status"] == "failed":
                        raise RuntimeError(st.get("error") or "pipeline failed")
                    if st["status"] == "queued":
                        _STATE["status"] = "Waiting for the server (another scene is being built)…"
                    else:
                        _STATE["progress"] = st.get("progress")
                        _STATE["status"] = ""
                    time.sleep(2)
                _STATE["progress"] = None
                _STATE["status"] = "Scene ready"
                _STATE["glb"] = "ready"
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
            _STATE.update(busy=False, progress=None, status="")
            self.report({"ERROR"}, _STATE["error"])
            return {"CANCELLED"}
        if _STATE["glb"]:
            self._finish(context)
            _STATE["busy"] = False
            sid, server, settings = _STATE["scene_id"], _server(context), context.scene.worldbuilder
            if settings.review_first:
                _STATE["review"] = sid
                _STATE["status"] = ""
                bpy.ops.wm.url_open(url=_viewer_url(server, sid))
                self.report({"INFO"}, f"WorldBuilder: '{sid}' is open in your browser; "
                                      "press Import Into Blender when you're done")
            else:
                try:
                    n, mb = _import_scene(context, server, sid, settings)
                except Exception as e:  # noqa: BLE001
                    self.report({"ERROR"}, f"Import failed: {e}")
                    return {"CANCELLED"}
                _STATE["status"] = ""
                self.report({"INFO"}, f"WorldBuilder: imported {n} objects into 'WorldBuilder {sid}'")
            return {"FINISHED"}
        return {"RUNNING_MODAL"}

    def _finish(self, context):
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None


class WB_OT_import_reviewed(Operator):
    """Import the scene you reviewed in the browser, with the edits you saved there"""
    bl_idname = "worldbuilder.import_reviewed"
    bl_label = "Import Into Blender"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        sid = _STATE.get("review")
        if not sid:
            return {"CANCELLED"}
        try:
            n, mb = _import_scene(context, _server(context), sid, context.scene.worldbuilder)
        except Exception as e:  # noqa: BLE001
            self.report({"ERROR"}, f"Import failed: {e}")
            return {"CANCELLED"}
        _STATE["review"] = None
        self.report({"INFO"}, f"WorldBuilder: imported {n} objects into 'WorldBuilder {sid}'")
        return {"FINISHED"}


class WB_OT_dismiss_review(Operator):
    """Forget the scene waiting for review (it stays on the server; import it later with Import Existing Scene)"""
    bl_idname = "worldbuilder.dismiss_review"
    bl_label = "Not Now"

    def execute(self, context):
        _STATE["review"] = None
        return {"FINISHED"}


# Blender keeps only references to enum item strings, so the list must outlive the callback.
_SCENE_ITEMS = []


def _scene_items(self, context):
    return _SCENE_ITEMS or [("", "No scenes found on the server", "")]


def _refresh_scene_items(server):
    import datetime
    scenes = _get_json(f"{server}/api/scenes").get("scenes", [])
    _SCENE_ITEMS.clear()
    for sc in scenes:
        when = datetime.datetime.fromtimestamp(sc.get("modified", 0)).strftime("%b %d %H:%M")
        n = sc.get("objects")
        sizes = [f"{sc[k]:.0f} MB {name}" for k, name in (("lite_mb", "lite"), ("full_mb", "full"))
                 if sc.get(k) is not None]
        label = f"{sc['scene_id']}   ({n} objects, {when})" if n is not None else f"{sc['scene_id']}   ({when})"
        _SCENE_ITEMS.append((sc["scene_id"], label, ", ".join(sizes) or sc["scene_id"]))
    return len(scenes)


class WB_OT_import_existing(Operator):
    """Import a scene the server has already built (no new reconstruction). """ \
        """Scenes are the folders in the server's outputs/ directory, newest first"""
    bl_idname = "worldbuilder.import_existing"
    bl_label = "Import Existing Scene"
    bl_options = {"REGISTER", "UNDO"}

    scene: EnumProperty(name="Scene", items=_scene_items,
                        description="Folders in the server's outputs/ directory, newest first")

    def invoke(self, context, event):
        try:
            if not _refresh_scene_items(_server(context)):
                self.report({"ERROR"}, "The server has no finished scenes yet")
                return {"CANCELLED"}
        except Exception as e:  # noqa: BLE001
            self.report({"ERROR"}, f"Server unreachable: {e}")
            return {"CANCELLED"}
        return context.window_manager.invoke_props_dialog(self, width=420)

    def draw(self, context):
        col = self.layout.column()
        col.prop(self, "scene", text="")
        settings = context.scene.worldbuilder
        row = col.row()
        row.prop(settings, "import_room")
        row.prop(settings, "full_detail")
        row.prop(settings, "use_metric_scale")

    def execute(self, context):
        sid = self.scene
        if not sid:
            return {"CANCELLED"}
        try:
            n, mb = _import_scene(context, _server(context), sid, context.scene.worldbuilder)
        except Exception as e:  # noqa: BLE001
            self.report({"ERROR"}, f"Import failed: {e}")
            return {"CANCELLED"}
        self.report({"INFO"}, f"WorldBuilder: imported {n} objects from '{sid}' ({mb:.0f} MB)")
        return {"FINISHED"}


def _scene_of_selection(context):
    """The scene id of the WorldBuilder collection the active object belongs to, if any."""
    o = context.active_object
    for c in (o.users_collection if o else ()):
        if c.name.startswith("WorldBuilder "):
            return re.sub(r"\.\d{3}$", "", c.name[len("WorldBuilder "):])    # "scene.001" on a re-import
    return None


class WB_OT_review(Operator):
    """Open a scene in the WorldBuilder web viewer to remove, move, turn or resize objects. """ \
        """Uses the scene of the selected object, else the last one, else the newest. """ \
        """Press Save Edits there, then import it again"""
    bl_idname = "worldbuilder.review"
    bl_label = "Edit in Browser"

    def execute(self, context):
        server = _server(context)
        sid = _scene_of_selection(context) or _STATE.get("review") or _STATE.get("scene_id")
        try:
            if not sid:
                scenes = _get_json(f"{server}/api/scenes").get("scenes", [])
                if not scenes:
                    self.report({"ERROR"}, "The server has no finished scenes")
                    return {"CANCELLED"}
                sid = scenes[0]["scene_id"]
        except Exception as e:  # noqa: BLE001
            self.report({"ERROR"}, f"Server unreachable: {e}")
            return {"CANCELLED"}
        bpy.ops.wm.url_open(url=_viewer_url(server, sid))
        return {"FINISHED"}


class WB_OT_check_server(Operator):
    """Check that the WorldBuilder server (Server URL in the add-on preferences) and its detector are reachable"""
    bl_idname = "worldbuilder.check_server"
    bl_label = "Check Server"

    def execute(self, context):
        server = _server(context)
        try:
            _get_json(f"{server}/api/health", timeout=5)
            v = _get_json(f"{server}/api/vlm-status", timeout=5)
            local = f"local detector ready ({v.get('model')})" if v.get("ok") else \
                "local detector offline (start vlm_server/server.py, or use gemini)"
            self.report({"INFO"}, f"Server OK; {local}")
        except Exception as e:  # noqa: BLE001
            self.report({"ERROR"}, f"Server unreachable at {server}: {e}")
        return {"FINISHED"}


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

def _inset(layout, margin=0.06):
    """A column with a little space on both sides, so buttons don't run edge to edge."""
    split = layout.split(factor=margin)
    split.label(text="")
    inner = split.split(factor=1 - margin / (1 - margin))
    col = inner.column()
    inner.label(text="")
    return col


def _draw_progress(layout, pr):
    box = layout.box()
    col = box.column(align=True)
    col.label(text=f"Step {pr['step']} of {pr['steps']}: {pr['name']}", icon="TIME")
    if pr.get("total"):
        text = f"{pr['done']} / {pr['total']} objects"
        if hasattr(col, "progress"):        # Blender 4.0+
            col.progress(factor=pr.get("fraction") or 0.0, type="BAR", text=text)
        else:
            col.label(text=text)
    eta = _fmt_minutes(pr.get("eta_seconds"))
    elapsed = _fmt_clock(pr.get("elapsed_seconds"))
    col.label(text=f"{eta} left · {elapsed} elapsed" if eta else f"{elapsed} elapsed")


class WB_PT_panel(Panel):
    bl_label = "WorldBuilder"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "WorldBuilder"

    def draw_header_preset(self, context):
        op = self.layout.operator("wm.url_open", text="", icon="QUESTION", emboss=False)
        op.url = DOCS_URL

    def draw(self, context):
        settings = context.scene.worldbuilder
        col = _inset(self.layout)
        col.scale_y = 1.4
        col.operator(WB_OT_generate.bl_idname, icon="IMAGE_DATA")
        sub = col.column()
        sub.scale_y = 1 / 1.4
        sub.prop(settings, "review_first")
        col.separator(factor=0.6)
        col.operator(WB_OT_import_existing.bl_idname, icon="IMPORT")
        col.separator(factor=0.3)
        col.operator(WB_OT_review.bl_idname, icon="WORLD")
        col.separator(factor=0.3)
        col.operator(WB_OT_check_server.bl_idname, icon="URL")

        layout = self.layout
        if _STATE["review"]:
            box = layout.box()
            c = box.column(align=True)
            c.label(text="Open in your browser for review:", icon="WORLD")
            c.label(text=_STATE["review"])
            c.label(text="Fix it there, press Save Edits, then:")
            c.separator(factor=0.5)
            big = c.column()
            big.scale_y = 1.4
            big.operator(WB_OT_import_reviewed.bl_idname, icon="IMPORT")
            r = c.row(align=True)
            r.operator(WB_OT_review.bl_idname, text="Reopen Browser")
            r.operator(WB_OT_dismiss_review.bl_idname)
        if _STATE["progress"]:
            _draw_progress(layout, _STATE["progress"])
        elif _STATE["status"]:
            layout.label(text=_STATE["status"], icon="INFO")
        if _STATE["last_import"]:
            layout.label(text=f"Last import: {_STATE['last_import']}")


class WB_PT_options(Panel):
    bl_label = "Import Options"
    bl_parent_id = "WB_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "WorldBuilder"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        settings = context.scene.worldbuilder
        col = self.layout.column()
        col.prop(settings, "import_room")
        col.prop(settings, "full_detail")
        col.prop(settings, "use_metric_scale")


class WB_PT_help(Panel):
    bl_label = "Help"
    bl_parent_id = "WB_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "WorldBuilder"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        col = self.layout.column(align=True)
        for line in ("The server is the WorldBuilder web app",
                     "(python -m webapp.server). Set its",
                     "address in Preferences > Add-ons.",
                     "",
                     "Photos: JPG, PNG, WebP, BMP, TIFF.",
                     "A scene takes a few minutes.",
                     "",
                     "Scenes are the folders in the",
                     "server's outputs/ directory.",
                     "",
                     "Each scene imports under one empty:",
                     "select it to move or turn it all."):
            col.label(text=line)
        col.separator()
        col.label(text=f"Server: {_prefs(context).server_url}")
        op = col.operator("wm.url_open", text="Open the Guide", icon="HELP")
        op.url = DOCS_URL


classes = (WB_Preferences, WB_Settings, WB_OT_generate, WB_OT_import_reviewed, WB_OT_dismiss_review,
           WB_OT_import_existing, WB_OT_review, WB_OT_check_server, WB_PT_panel, WB_PT_options, WB_PT_help)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.worldbuilder = PointerProperty(type=WB_Settings)


def unregister():
    del bpy.types.Scene.worldbuilder
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
