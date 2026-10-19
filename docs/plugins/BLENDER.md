# Blender

Turn a photo into a scene from inside Blender, or import one you already made. Every object
arrives as its own named, textured mesh, with the room as separate pieces, all under one
empty.

You need Blender 4.2 or newer and a WorldBuilder install with a GPU ([Setup](../SETUP.md)).

## 1. Install the Add-On

1. Download `worldbuilder-<version>.zip` from the
   [releases page](https://github.com/AdamRolander/WorldBuilder/releases), or build it from your
   clone: `blender --command extension build --source-dir integrations/blender_worldbuilder`
2. In Blender: **Edit → Preferences → Get Extensions → ▾ (top right) → Install from Disk…**,
   then pick the zip.
3. Optional, in **Preferences → Add-ons → WorldBuilder**:
   - **Server URL**: leave the default unless the server runs on another computer.
   - **Detector**: leave empty, or type `qwen` (local) or `gemini` (cloud). It's a name, not a URL.

## 2. Start the Server

Two terminals on the GPU machine:

```bash
# terminal 1: the local detector (skip if you use Gemini)
conda activate worldbuilder-vlm
python vlm_server/server.py

# terminal 2: the server Blender talks to
conda activate worldbuilder-main
python -m webapp.server
```

They are separate because the detector needs a newer version of a library than the rest of
the pipeline. Gemini needs `GEMINI_API_KEY` in the `.env` file instead of terminal 1. Flask's
"development server" warning is expected; the server is meant for your own machine.

## 3. Use It

Press **N** in the 3D Viewport and open the **WorldBuilder** tab.

| Control | What it does |
| --- | --- |
| **Generate Scene From Photo** | Pick a photo (JPG, PNG, WebP, BMP or TIFF). The panel shows the step (1 of 4 to 4 of 4) and, during the slow step, objects done and time left. Blender stays usable meanwhile. |
| **Review in Browser First** | On (default): the finished scene opens in your browser. Fix it there, press **Save Edits**, then **Import Into Blender**. Off: it imports straight away. |
| **Import Existing Scene** | Choose from the scenes the server has already built, newest first, with their object counts and sizes. |
| **Edit in Browser** | Opens the selected object's scene (or the last one) in the browser viewer. Save there, then import it again. |
| **Check Server** | Confirms the server and the detector are reachable. |
| **Import Options** | Room on or off, **Full Detail** (the original dense meshes, often 10-20x larger), scale to metres. |
| **?** and **Help** | A short reminder of the above, and a link to this page. |

Each import goes into a collection named `WorldBuilder <scene>`, under one empty. Select that
empty to move, turn or scale the whole scene. Inside are `NNN_label` objects, the room
(`room_floor`, `room_wall_*`, `room_ceiling`), `room_relief_*` (surfaces from the photo that
aren't objects) and `room_builtin_*` (solid blocks for counters and cabinets).

Scale is an estimate. Check one known size, such as a door or a counter, before relying on it.

## Troubleshooting

- **Local detector offline**: terminal 1 isn't running yet, or is still loading (about a
  minute). Or use `gemini`.
- **Server unreachable**: terminal 2 isn't running, or Blender is on another computer (see below).
- **A scene is missing from Import Existing Scene**: the server lists the folders directly in
  `outputs/`. For scenes in a sub-folder, start the server with
  `WORLDBUILDER_OUTPUTS=outputs/<folder> python -m webapp.server`.

### Blender on Another Computer

Start the server with `python -m webapp.server --host 0.0.0.0` and set **Server URL** to
`http://<gpu machine's IP>:5174`. Both computers must be on the same network, and the server
has no login, so only do this on a network you trust. From anywhere else, an SSH tunnel is
safer: `ssh -L 5174:localhost:5174 <gpu machine>` and keep the default URL.

## For Developers

The add-on (`integrations/blender_worldbuilder/`) uses only Blender's bundled Python (`urllib`,
no wheels); HTTP runs in a worker thread, and only the glTF import runs on Blender's main
thread. To test against a running server:

```bash
blender --background --python scripts/test_blender_addon.py -- --scene <scene id> --render /tmp/wb.png
blender --command extension validate integrations/blender_worldbuilder
```

Bump `version` in `blender_manifest.toml` and `bl_info` whenever the add-on changes.
