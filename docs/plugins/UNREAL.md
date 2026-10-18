# Unreal Engine

> **Status: untested scaffold.** The plugin has not yet been compiled or run in Unreal. The
> Python part follows the same flow as the working Blender add-on. If you have Unreal 5.4+
> and want to help, trying the steps below and reporting what happens is very useful.

The plugin adds **Tools → WorldBuilder → Generate scene from photo…** to the editor. It uploads
the photo to your WorldBuilder server, waits for the scene, and imports `scene_lite.glb`
through Interchange, scaled to centimetres.

## Install

1. Copy `integrations/unreal/WorldBuilderBridge/` into `<YourProject>/Plugins/`.
2. Enable **Python Editor Script Plugin**, **Editor Scripting Utilities** and **Interchange**
   in the project (the `.uplugin` declares them, so Unreal prompts for them).
3. Rebuild the project; the C++ module only adds the menu entry.

To skip C++ entirely, delete `Source/` and use it as a content-only plugin. In the Output Log
(Python):

```python
import sys; sys.path.append(r"<project>/Plugins/WorldBuilderBridge/Content/Python")
import worldbuilder_bridge as wb
wb.generate_and_import(r"C:/photos/room.jpg")
```

The server address is `SERVER` at the top of `worldbuilder_bridge.py`
(default `http://127.0.0.1:5174`). If Unreal runs on another computer, start the server with
`python -m webapp.server --host 0.0.0.0`.

## Known Limits

- `generate_and_import` blocks the editor while the pipeline runs (a few minutes). An Editor
  Utility Widget with a timer is the planned fix.
- Y-up to Z-up and metres to centimetres are handled by Interchange; the bridge applies the
  scene's estimated metric scale on top.

## Help Wanted

- Compile on Win64 with Unreal 5.4 and 5.5 and fix what breaks.
- A non-blocking UI with progress.
- Import objects as they finish instead of waiting for the whole scene.
