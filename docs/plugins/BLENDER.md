# Blender

Generate a scene from a photo without leaving Blender, or import one you already made. Each
object arrives as its own named mesh with a baked texture; the floor, walls, ceiling and
built-in furniture are separate objects too, all under one empty scaled to metres.

Requires Blender 4.2 or newer, and a running WorldBuilder server ([Setup](../SETUP.md)) on this
computer or one Blender can reach.

## Install

1. Download `worldbuilder-<version>.zip` from the
   [releases page](https://github.com/AdamRolander/WorldBuilder/releases).
   Or build it yourself from a clone:
   `blender --command extension build --source-dir integrations/blender_worldbuilder`
2. In Blender: **Edit → Preferences → Get Extensions → ▾ (top right) → Install from Disk…**
   and pick the zip.
3. Under **Preferences → Add-ons → WorldBuilder**, set the **Server URL** if the server is not
   on this computer (default `http://127.0.0.1:5174`).

On Blender 3.6 to 4.1 the same zip installs as a legacy add-on through
**Preferences → Add-ons → Install**.

## Use

Start the server on the GPU machine:

```bash
conda activate worldbuilder-main
python -m webapp.server                  # this computer only
python -m webapp.server --host 0.0.0.0   # if Blender runs on another computer
```

Then in Blender, open the sidebar in the 3D Viewport (**N**) and the **WorldBuilder** tab:

| Button | What it does |
| --- | --- |
| **Generate scene from photo** | uploads a photo, waits for the pipeline (a few minutes), imports the result. Blender stays responsive while it runs. |
| **Import existing scene** | imports a scene the server already has; leave the id empty for the newest |
| **Review in browser** | opens the scene in the web viewer, where you can remove, move, turn and resize objects and save the edits |
| **Check server** | confirms the server and the detector are reachable |

Options when importing: the room (floor, walls, ceiling) on or off, lite meshes with baked
textures (default) or the dense originals, and scaling to metres.

After **Save edits** in the browser, **Import existing scene** brings in the corrected scene;
the server applies the edits before sending it.

## What You Get

- One collection named after the scene, with an empty at its root carrying the metric scale.
- `NNN_label` meshes for each object, with image textures.
- `room_floor`, `room_wall_*`, `room_ceiling`, `room_relief_*` (the photo's surfaces that
  weren't objects, as low relief) and `room_builtin_*` (solid boxes for counters and cabinets).

Scale is an estimate (from MoGe-2 when installed, otherwise from a typical camera height).
Check one known dimension, such as a door or a counter, before relying on it.

## Troubleshooting

- **"Server unreachable"**: the server isn't running, or it only listens on `127.0.0.1` and
  Blender is on another computer (start it with `--host 0.0.0.0`). **Check server** tells you
  which.
- **Import existing scene finds nothing**: the server lists scenes directly under `outputs/`.
  Scenes in sub-folders need `WORLDBUILDER_OUTPUTS=outputs/<folder>` when starting the server.

## For Developers

The add-on lives in `integrations/blender_worldbuilder/` and uses only Blender's Python
standard library (`urllib`, no wheels). HTTP runs in a worker thread under a modal timer;
only the glTF import runs on Blender's main thread.

Test it headless against a running server (no GPU work when importing an existing scene):

```bash
blender --background --python scripts/test_blender_addon.py -- \
    --scene <scene id> --render /tmp/wb.png
blender --command extension validate integrations/blender_worldbuilder
```

Bump `version` in `blender_manifest.toml` (and `bl_info`) whenever the add-on itself changes.
