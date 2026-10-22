# Plugins

WorldBuilder scenes can be opened in other tools. Each plugin is a small client of the
WorldBuilder server you run yourself; the models never leave that server.

| Plugin | Status | Install | Docs |
| --- | --- | --- | --- |
| **Blender** 4.2+ | Working. Passes Blender's extension validation and an end-to-end test on Blender 5.0.1 | zip from the [releases page](https://github.com/AdamRolander/WorldBuilder/releases) | [Blender](BLENDER.md) |
| **MuJoCo / MJX** | Working. Exporter plus a Gymnasium wrapper | part of this repo (`pip install -e ".[robotics]"`) | [MuJoCo](MUJOCO.md) |
| **Unreal Engine** 5.4+ | Scaffold, not yet compiled or tested | copy the plugin folder into your project | [Unreal](UNREAL.md) |
| Anything that reads glTF | Works today: download `scene_lite.glb` | | [Usage: Outputs](../USAGE.md#outputs) |

## How the Plugins Work

The pipeline needs a 24-32 GB NVIDIA card, two conda environments and model weights that are
licensed separately by Meta. None of that fits inside a Blender add-on or an Unreal plugin,
and the weights may not be redistributed anyway. So every plugin talks HTTP to the web app
(`python -m webapp.server`), which may be on the same machine or another one on your network:

```
photo ──► plugin (Blender, Unreal, ...) ──HTTP──► webapp/server.py ──► pipeline (GPU)
              ▲                                          │
              └──────── scene_lite.glb / MuJoCo zip ◄────┘
```

A side effect is that most improvements (placement, textures, the viewer's editor) reach
plugin users with a `git pull` on the server and no plugin update.

The endpoints a plugin uses:

| Endpoint | What it returns |
| --- | --- |
| `POST /api/upload` | starts a job for a photo: `{job_id, scene_id}` |
| `GET /api/jobs/<id>` | progress: `{status, stage, scene_id, error}` |
| `GET /api/scenes` | finished scenes, newest first |
| `GET /api/scenes/<id>` | manifest: objects, layout, file URLs |
| `GET /api/scenes/<id>/glb?lite=1` | the whole scene as one GLB with named nodes and baked textures; edits saved in the viewer are applied |
| `GET /api/scenes/<id>/mujoco[?mjx=1]` | a zip with a MuJoCo model of the scene |

The full list is in [Usage](../USAGE.md#web-app).

## Review in the Browser, Then Import

The browser viewer is the review step in front of every plugin. Open a scene there (Blender
has a **Review in browser** button), remove or move objects, and **Save edits**. Every GLB and
MuJoCo export the server hands out afterwards has the edits applied, so the next import into
Blender, Unreal or MuJoCo is the corrected scene.

## Where We'd Like Plugins Next

Rough effort for a first working version, given that the server and the GLB already exist.
Ideas and help on any of these are welcome; open an issue to claim one.

| Target | Audience | Effort | Notes |
| --- | --- | --- | --- |
| **Godot** (Asset Library) | indie game developers | small | Native glTF import and `HTTPRequest` in GDScript; MIT is accepted. A port of the Blender add-on's flow. |
| **Isaac Sim / Isaac Lab** | robotics, sim-to-real | small to medium | Imports MJCF and glTF already; a USD export with physics schemas would make it native. |
| **Unity** (UPM package via git URL, later the Asset Store) | game developers, XR | medium | glTF through glTFast; C# `UnityWebRequest` client. |
| **USD export** | film, Omniverse, Isaac | medium | `scene_lite.glb` to USD with named prims; no GPU needed to work on. |
| **ROS 2 / Gazebo** (SDF or URDF) | robotics | medium | Static world from the room, models from the objects; collision geometry exists for MuJoCo. |
| **Genesis** | robotics, RL | small | Loads MJCF; mainly a documented example. |
| **Houdini, Maya, Cinema 4D** | VFX and motion design | medium each | Each has a Python API and glTF import; a thin client per tool. |
| **SketchUp, Rhino** | architecture and interiors | medium | Ruby (SketchUp) or Python (Rhino 8) clients; interior designers are a natural audience. |
| **ComfyUI** (custom node) | generative AI users | small | A node that sends an image to the server and returns the GLB; large, active audience. |
| **Static web demo** | everyone | small | `viewer.html` and `scene_lite.glb` on GitHub Pages; no server, no GPU. |

Two rules apply to all of them: never bundle SAM weights or SAM code in a plugin, and point
users at their own server rather than one we host.
