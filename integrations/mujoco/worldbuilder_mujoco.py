"""Use a WorldBuilder scene as a MuJoCo / Gymnasium environment.

Only needs ``mujoco`` (and ``gymnasium`` for the env class); it never
imports the WorldBuilder pipeline, so it runs on a laptop from an exported
``scene.xml`` (``python -m src.mujoco_export outputs/<scene> --stabilize``
or ``GET /api/scenes/<scene>/mujoco`` on a WorldBuilder server).

    from worldbuilder_mujoco import load_scene, SceneEnv

    # 1. just the room
    model = load_scene("k1_q/mujoco/scene.xml")

    # 2. the room with a robot from MuJoCo Menagerie standing in it
    model = load_scene("k1_q/mujoco/scene.xml",
                       robot_xml="mujoco_menagerie/franka_emika_panda/panda.xml",
                       robot_pos=(0.6, 0.0, 0.0))

    # 3. a Gymnasium env around it (reward/termination are yours to define)
    env = SceneEnv("k1_q/mujoco/scene.xml", robot_xml=..., camera="photo")
    obs, info = env.reset()

Frame: X forward (the direction the photo was taken in), Y left, Z up,
floor at z = 0, the photo's camera above the origin. Units: metres, under
the scale estimate recorded in the header of ``scene.xml``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import mujoco
import numpy as np


def load_scene(scene_xml: str | Path, robot_xml: Optional[str | Path] = None,
               robot_pos: Sequence[float] = (0.0, 0.0, 0.0),
               robot_quat: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
               robot_prefix: str = "robot/") -> mujoco.MjModel:
    """Compile a WorldBuilder scene, optionally with a robot attached.

    The robot's MJCF is attached at ``robot_pos`` (metres, scene frame) with
    every name prefixed by ``robot_prefix``, so its actuators, joints and
    sensors cannot collide with scene names.
    """
    spec = mujoco.MjSpec.from_file(str(scene_xml))
    if robot_xml is not None:
        robot = mujoco.MjSpec.from_file(str(robot_xml))
        frame = spec.worldbody.add_frame(pos=list(robot_pos), quat=list(robot_quat))
        spec.attach(robot, frame=frame, prefix=robot_prefix)
    return spec.compile()


def scene_objects(model: mujoco.MjModel) -> Dict[str, Dict]:
    """``{body name: {"id", "free", "label"}}`` for every WorldBuilder object."""
    out = {}
    for b in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if not name.startswith("obj_"):
            continue
        j = model.body_jntadr[b]
        free = j >= 0 and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        out[name] = {"id": b, "free": bool(free), "label": "_".join(name.split("_")[2:])}
    return out


try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:                                    # the loader above still works
    gym = None


if gym is not None:
    class SceneEnv(gym.Env):
        """Minimal Gymnasium wrapper: actions are the model's actuator
        controls, observations are ``qpos``/``qvel`` (and an RGB image from
        ``camera`` if given). Pass ``reward_fn(model, data) -> float`` and
        ``done_fn(model, data) -> bool`` to make it a task; by default the
        reward is 0 and episodes end only on ``max_steps``.
        """
        metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

        def __init__(self, scene_xml, robot_xml=None, robot_pos=(0.0, 0.0, 0.0), camera: Optional[str] = None,
                     image_size: Tuple[int, int] = (240, 320), frame_skip: int = 10, max_steps: int = 500,
                     reward_fn: Optional[Callable] = None, done_fn: Optional[Callable] = None,
                     settle_seconds: float = 0.5):
            self.model = load_scene(scene_xml, robot_xml, robot_pos)
            self.data = mujoco.MjData(self.model)
            self.camera, self.image_size = camera, image_size
            self.frame_skip, self.max_steps = frame_skip, max_steps
            self.reward_fn, self.done_fn = reward_fn, done_fn
            self._settle = int(settle_seconds / self.model.opt.timestep)
            self._renderer = None
            self._t = 0
            lo, hi = self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1]
            limited = self.model.actuator_ctrllimited.astype(bool)
            self.action_space = spaces.Box(np.where(limited, lo, -1.0).astype(np.float32),
                                           np.where(limited, hi, 1.0).astype(np.float32), dtype=np.float32)
            obs = {"qpos": spaces.Box(-np.inf, np.inf, (self.model.nq,), np.float64),
                   "qvel": spaces.Box(-np.inf, np.inf, (self.model.nv,), np.float64)}
            if camera:
                obs["image"] = spaces.Box(0, 255, (*image_size, 3), np.uint8)
            self.observation_space = spaces.Dict(obs)

        def _obs(self):
            o = {"qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy()}
            if self.camera:
                o["image"] = self.render()
            return o

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            mujoco.mj_resetData(self.model, self.data)
            for _ in range(self._settle):               # let objects find their rest pose
                mujoco.mj_step(self.model, self.data)
            self._t = 0
            return self._obs(), {}

        def step(self, action):
            if self.model.nu:
                self.data.ctrl[:] = np.asarray(action, dtype=np.float64)
            for _ in range(self.frame_skip):
                mujoco.mj_step(self.model, self.data)
            self._t += 1
            reward = float(self.reward_fn(self.model, self.data)) if self.reward_fn else 0.0
            terminated = bool(self.done_fn(self.model, self.data)) if self.done_fn else False
            return self._obs(), reward, terminated, self._t >= self.max_steps, {}

        def render(self):
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, *self.image_size)
            self._renderer.update_scene(self.data, camera=self.camera or -1)
            return self._renderer.render()

        def close(self):
            if self._renderer is not None:
                self._renderer.close()
                self._renderer = None


if __name__ == "__main__":
    import sys
    m = load_scene(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    objs = scene_objects(m)
    print(f"{len(objs)} objects ({sum(o['free'] for o in objs.values())} free), {m.nu} actuators, "
          f"{m.ngeom} geoms, nq={m.nq}")
