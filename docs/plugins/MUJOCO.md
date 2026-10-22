# MuJoCo and MJX

Turn a reconstructed scene into a MuJoCo model you can drop a robot into. The exporter is part
of this repository (`src/mujoco_export.py`); the helpers for loading a scene with a robot and
wrapping it as a Gymnasium environment are in `integrations/mujoco/`. Exporting needs no GPU.

```bash
pip install -e ".[robotics]"            # mujoco, gymnasium (env wrapper), coacd (--collision coacd)

python -m src.mujoco_export outputs/<scene> --stabilize --check
#  → outputs/<scene>/mujoco/scene.xml, assets/*.obj|png, export_report.json

python -m mujoco.viewer --mjcf outputs/<scene>/mujoco/scene.xml       # look at it
```

or, from a machine that only has the server's URL:

```bash
curl -OJ "http://<server>:5174/api/scenes/<scene>/mujoco"            # zip: scene.xml + assets/
curl -OJ "http://<server>:5174/api/scenes/<scene>/mujoco?mjx=1"      # MJX variant
```

## What Is in the Model

| | |
| --- | --- |
| Frame | X forward (the photo's viewing direction), Y left, Z up; floor at `z = 0`; the photo's camera above the origin |
| Units | metres, via `layout.estimated_metric_scale` (MoGe-2 estimate; a camera-height prior without it); override with `--scale` |
| Objects | one body each, named `obj_<id>_<label>`; visual geom = decimated mesh with a baked texture; collision geoms in group 3 |
| Collision | `--collision hull` (default, one convex hull per object), `coacd` (convex decomposition: chairs, bowls and shelves keep their concavities), `box` |
| Movable | objects for which placement found a support (floor, another object, a counter) get a free joint; wall-mounted and hanging things and anything over `--static-above` metres are welded. `--dynamic all|none` overrides |
| Room | floor plane + four wall boxes (collision, invisible); textured shell quads and the photo relief of built-ins (visual only) |
| Support pads | invisible static boxes under objects that rest on something the model has no collision for (a countertop that exists only in the relief, or a welded object whose hull does not reach under them) |
| Cameras | `photo` (the input photo's pose and field of view) and `overview` |
| Geom groups | 2 visual, 3 collision, 4 structure/pads; toggle them in the viewer with the number keys |

`--stabilize` simulates two seconds and welds whatever drifts more than
5 cm, repeating until the scene is at rest; `export_report.json` lists what
was welded. Single-image reconstruction does not guarantee every object
rests on its supporter to the millimetre, and for learning environments a
scene that is still at `t = 0` matters more than every mug being movable.
Without the flag everything supported is free and `--check` tells you what
moves.

Measured on the validation kitchen (35 objects, 2026-10-08): 32 MB of
assets, 16 free bodies of which 6 were welded by `--stabilize`,
remaining drift ≤ 4 cm over 2 s.

## A Robot in the Room

```python
from worldbuilder_mujoco import load_scene, scene_objects, SceneEnv   # integrations/mujoco/ on sys.path

model = load_scene("outputs/k1_q/mujoco/scene.xml",
                   robot_xml="mujoco_menagerie/franka_emika_panda/panda.xml",
                   robot_pos=(0.6, 0.0, 0.0))          # metres, scene frame
print(scene_objects(model))                             # {'obj_014_fruit': {'id': ..., 'free': True, ...}, ...}

env = SceneEnv("outputs/k1_q/mujoco/scene.xml", robot_xml=..., camera="photo",
               reward_fn=lambda m, d: 0.0)              # your task goes here
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

`load_scene` attaches the robot's MJCF through `mujoco.MjSpec.attach` with
a name prefix, so nothing in the scene file needs editing. `SceneEnv` is a
deliberately small Gymnasium wrapper (actuator controls in, `qpos`/`qvel`
and optionally a camera image out); reward and termination are callbacks
because they are the task, not the scene.

## MJX

`--mjx` writes a variant meant for `mujoco.mjx`: box collision geoms (MJX
handles primitives far better than meshes), no relief, a 4 ms timestep and
conservative solver iterations. It loads and steps in MJX
(`mjx.put_model` → `jax.jit(mjx.step)`; verified on CPU JAX, 2026-10-08).
Not yet measured: throughput on a GPU and how many objects a batch of
thousands of environments tolerates; the contact count is what to watch.
For large batches weld the clutter (`--dynamic none`) and free only the
objects the task manipulates.

## Limits Worth Knowing

* **Scale is an estimate.** The photo has no absolute scale; the exporter
  uses a prior (camera 1.5 m above the floor). Check one known dimension
  (a door is ~2.03 m, a counter ~0.91 m) and pass `--scale`.
* **Masses are made up.** Density is uniform (`--density`, default
  250 kg/m³ over the collision volume). Set real masses for the objects
  your task touches.
* **Back sides are invented.** SAM 3D completes what the photo did not
  see; an object's hidden half is plausible, not measured.
* **The relief has no collision.** It is a visual shell of built-ins;
  pads carry what stands on it. A robot arm can pass through a cabinet
  front. Planned (plane-fitted collision for built-ins).
* **Rendering.** `--check --render out.png` needs a working MuJoCo GL
  backend; with the PyOpenGL version pinned by `pyrender` in the main env,
  `MUJOCO_GL=egl` fails to import. Leave `MUJOCO_GL` unset on a desktop,
  or use a separate env for headless rendering.

## Other Simulators

Isaac Sim / Isaac Lab and Genesis import MJCF directly, so `scene.xml` is
the interchange format for now; `scene_lite.glb` works anywhere glTF does
(Unity, Godot, three.js, Blender, Unreal). Native USD and URDF/SDF exports
are planned and are good no-GPU tasks.
