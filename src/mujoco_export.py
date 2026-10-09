"""Export a finished scene as a MuJoCo model (MJCF + mesh assets).

    python -m src.mujoco_export outputs/<scene>                 # -> outputs/<scene>/mujoco/scene.xml
    python -m src.mujoco_export outputs/<scene> --collision coacd
    python -m src.mujoco_export outputs/<scene> --mjx           # primitives only, for MJX / Brax
    python -m src.mujoco_export outputs/<scene> --check         # load, settle 2 s, report what moved
    python -m src.mujoco_export outputs/<scene> --stabilize     # weld whatever will not rest

What comes out
--------------
* **Frame and units.** MuJoCo is Z-up and metric. The scene is rotated to
  X forward / Y left / Z up (REP-103, what robot models expect), scaled to
  metres with ``layout.estimated_metric_scale`` (MoGe-2's metric estimate when
  ``scripts/estimate_metric_scale.py`` has run, else a camera-height prior;
  override with ``--scale``), and shifted so the floor is ``z = 0``
  and the photo's camera stands above the origin.
* **Objects.** One body per object, origin at the object's centre. Visual
  geom: the decimated, texture-baked mesh from ``scene_lite.glb``.
  Collision geoms: its convex hull (default), a CoACD convex decomposition
  (``--collision coacd``; concave things like chairs and bowls behave
  properly), or its bounding box (``--collision box`` / ``--mjx``).
* **What can move.** An object the placement stage found a support for
  (floor, another object, a counter) gets a free joint; wall-mounted and
  hanging things have no known support and are welded to the world, as is
  anything larger than ``--static-above`` metres. ``--dynamic none|all``
  overrides.
* **Room.** A floor plane and wall boxes for collision; the textured shell
  quads and the photo relief as visual-only meshes. Objects resting on an
  unmodelled surface (a countertop that exists only in the relief) get a
  static, invisible support pad under them so they do not fall through.
* **Cameras.** ``photo`` reproduces the input photo's viewpoint and field
  of view; ``overview`` looks down into the room.

The MJCF is plain text and meant to be edited: add a robot with
``<include>`` or ``mujoco.MjSpec.attach`` (see ``integrations/mujoco``).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.sax.saxutils import quoteattr

import numpy as np

# aligned frame (X left, Y up, Z forward) -> MuJoCo (X forward, Y left, Z up)
AXES = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


def _fmt(a) -> str:
    return " ".join(f"{float(x):.5g}" for x in np.ravel(a))


def _write_obj(path: Path, v: np.ndarray, f: np.ndarray, uv: Optional[np.ndarray] = None) -> None:
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in v]
    if uv is not None:
        lines += [f"vt {a:.6f} {b:.6f}" for a, b in uv]
        lines += [f"f {a}/{a} {b}/{b} {c}/{c}" for a, b, c in np.asarray(f) + 1]
    else:
        lines += [f"f {a} {b} {c}" for a, b, c in np.asarray(f) + 1]
    path.write_text("\n".join(lines) + "\n")


def _hulls(v: np.ndarray, f: np.ndarray, mode: str, max_hulls: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Convex collision pieces for one object (vertices in the body frame)."""
    import trimesh
    if mode == "coacd":
        try:
            import coacd
            coacd.set_log_level("error")
            parts = coacd.run_coacd(coacd.Mesh(np.asarray(v, np.float64), np.asarray(f, np.int32)),
                                    threshold=0.08, max_convex_hull=max_hulls, preprocess_resolution=40)
            parts = [(np.asarray(pv), np.asarray(pf)) for pv, pf in parts if len(pv) >= 4]
            if parts:
                return parts
        except Exception as e:                      # non-manifold input, missing wheel: fall back
            print(f"    coacd failed ({str(e)[:60]}); using the convex hull")
    try:
        hull = trimesh.Trimesh(vertices=v, faces=f, process=False).convex_hull
        return [(np.asarray(hull.vertices), np.asarray(hull.faces))]
    except Exception:
        return []


def export_mujoco(scene_dir: Path, out_dir: Optional[Path] = None, scale: Optional[float] = None,
                  collision: str = "hull", dynamic: str = "supported", mjx: bool = False,
                  density: float = 250.0, static_above: float = 1.2, max_hulls: int = 12,
                  include_relief: bool = True, verbose: bool = True,
                  force_static: Optional[set] = None) -> Path:
    """Write ``<out_dir>/scene.xml`` (+ ``assets/``) and return its path."""
    import trimesh
    log = print if verbose else (lambda *a, **k: None)
    scene_dir = Path(scene_dir)
    out_dir = Path(out_dir) if out_dir else scene_dir / "mujoco"
    assets = out_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    if mjx:
        collision = "box"

    rec = json.loads((scene_dir / "reconstruction_results.json").read_text())
    layout = json.loads((scene_dir / "3d_models" / "layout.json").read_text())
    lite = scene_dir / "scene_lite.glb"
    if not lite.exists():
        from src.scene_export import export_scene_lite
        log("  building scene_lite.glb (decimated meshes with baked textures)...")
        export_scene_lite(scene_dir, rec["objects"])
    glb = trimesh.load(str(lite), process=False)

    s = float(scale if scale else layout.get("estimated_metric_scale", 1.0))
    floor_y = float(layout.get("floor_y", layout.get("bounds", {}).get("min", [0, 0, 0])[1]))

    def to_mj(p):
        p = np.asarray(p, float) - np.array([0.0, floor_y, 0.0])
        return (p @ AXES.T) * s

    by_name = {f"{int(o['id']):03d}_{str(o['label']).replace(' ', '_')}": o for o in rec["objects"]}
    asset_xml: List[str] = []
    body_xml: List[str] = []
    pads_xml: List[str] = []
    report = {"objects": 0, "dynamic": 0, "static": 0, "collision_geoms": 0, "pads": 0}

    def add_visual(name: str, mesh, v_local: np.ndarray) -> str:
        """Register mesh (+ texture) assets; return the visual geom's attributes."""
        uv = getattr(mesh.visual, "uv", None)
        tex = None
        mat = getattr(mesh.visual, "material", None)
        if uv is not None and mat is not None:
            tex = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
        _write_obj(assets / f"{name}.obj", v_local, mesh.faces, uv if tex is not None else None)
        # visual meshes are open or flat; "shell" keeps the compiler from
        # rejecting them for having no volume (they carry no mass anyway)
        asset_xml.append(f'    <mesh name="{name}" file="{name}.obj" inertia="shell"/>')
        if tex is not None:
            tex.convert("RGB").save(assets / f"{name}.png")
            asset_xml.append(f'    <texture name="{name}" type="2d" file="{name}.png"/>')
            asset_xml.append(f'    <material name="{name}" texture="{name}" specular="0" shininess="0"/>')
            return f'mesh="{name}" material="{name}"'
        rgba = [0.7, 0.7, 0.7, 1.0]
        if getattr(mesh.visual, "kind", None) == "vertex":
            rgba = (np.asarray(mesh.visual.vertex_colors)[:, :3].mean(0) / 255).tolist() + [1.0]
        return f'mesh="{name}" rgba="{_fmt(rgba)}"'

    # ---- objects ---------------------------------------------------------
    force_static = force_static or set()
    sizes = {n: float(np.max(g.extents)) * s for n, g in glb.geometry.items() if not n.startswith("room")}
    id_to_name = {int(o["id"]): n for n, o in by_name.items()}

    def _is_dynamic(gname: str) -> bool:
        o = by_name.get(gname, {})
        if "obj_" + _safe(gname) in force_static:
            return False
        if dynamic in ("all", "none"):
            return dynamic == "all"
        return o.get("support") is not None and sizes.get(gname, 0.0) <= static_above

    for gname, geom in glb.geometry.items():
        if gname.startswith("room"):
            continue
        o = by_name.get(gname, {})
        name = "obj_" + _safe(gname)
        V = to_mj(geom.vertices)
        lo, hi = V.min(0), V.max(0)
        centre = 0.5 * (lo + hi)
        v_local = V - centre
        vis = add_visual(name, geom, v_local)
        support = o.get("support")
        is_dynamic = _is_dynamic(gname)
        lines = [f'    <body name="{name}" pos="{_fmt(centre)}">']
        if is_dynamic:
            lines.append(f'      <freejoint name="{name}_free"/>')
        lines.append(f'      <geom class="visual" {vis}/>')
        if collision == "box" or float(np.min(hi - lo)) < 0.015:     # rugs, boards: a hull has no volume
            half = np.maximum(0.5 * (hi - lo), 2e-3)
            lines.append(f'      <geom class="collision" type="box" size="{_fmt(half)}"/>')
            report["collision_geoms"] += 1
        else:
            for k, (pv, pf) in enumerate(_hulls(v_local, np.asarray(geom.faces), collision, max_hulls)):
                cname = f"{name}_c{k}"
                _write_obj(assets / f"{cname}.obj", pv, pf)
                asset_xml.append(f'    <mesh name="{cname}" file="{cname}.obj"/>')
                lines.append(f'      <geom class="collision" mesh="{cname}"/>')
                report["collision_geoms"] += 1
        lines.append("    </body>")
        body_xml += lines
        report["objects"] += 1
        report["dynamic" if is_dynamic else "static"] += 1
        # Something stands here on a thing that will not move: an unmodelled
        # surface (a countertop that exists only in the relief) or a welded
        # object whose convex hull may not reach under it. A thin invisible
        # pad makes the contact explicit. Objects on *movable* supporters get
        # no pad: it would hold them in mid-air once the supporter moves.
        sup_name = id_to_name.get(int(o["supported_by"])) if o.get("supported_by") else None
        on_static = support == "surface" or (support == "object" and sup_name and not _is_dynamic(sup_name))
        if is_dynamic and on_static:
            pad = np.array([0.5 * (hi[0] - lo[0]) + 0.03, 0.5 * (hi[1] - lo[1]) + 0.03, 0.01])
            pads_xml.append(f'    <geom name="pad_{name}" class="pad" type="box" size="{_fmt(pad)}" '
                            f'pos="{_fmt([centre[0], centre[1], lo[2] - 0.01])}"/>')
            report["pads"] += 1

    # ---- room ------------------------------------------------------------
    room_xml: List[str] = []
    if "bounds_min" in layout:
        mn, mx = to_mj(layout["bounds_min"]), to_mj(layout["bounds_max"])
        mn, mx = np.minimum(mn, mx), np.maximum(mn, mx)
    else:
        mn, mx = to_mj(layout["bounds"]["min"]), to_mj(layout["bounds"]["max"])
    c, half, t = 0.5 * (mn + mx), 0.5 * (mx - mn), 0.05
    room_xml.append(f'    <geom name="floor" class="structure" type="plane" pos="{_fmt([c[0], c[1], 0])}" '
                    f'size="{_fmt([half[0], half[1], 0.1])}"/>')
    for wname, pos, size in (
            ("wall_back", [mx[0] + t, c[1], c[2]], [t, half[1], half[2]]),
            ("wall_front", [mn[0] - t, c[1], c[2]], [t, half[1], half[2]]),
            ("wall_left", [c[0], mx[1] + t, c[2]], [half[0], t, half[2]]),
            ("wall_right", [c[0], mn[1] - t, c[2]], [half[0], t, half[2]])):
        room_xml.append(f'    <geom name="{wname}" class="structure" type="box" pos="{_fmt(pos)}" size="{_fmt(size)}"/>')
    for i, b in enumerate(layout.get("builtins") or []):
        # counters, cabinet runs, soffits: solid, static, invisible (their
        # textured faces come in as visual meshes below)
        bmn, bmx = to_mj(b["min"]), to_mj(b["max"])
        bmn, bmx = np.minimum(bmn, bmx), np.maximum(bmn, bmx)
        room_xml.append(f'    <geom name="builtin_{i:02d}" class="structure" type="box" pos="{_fmt(0.5 * (bmn + bmx))}" '
                        f'size="{_fmt(np.maximum(0.5 * (bmx - bmn), 1e-3))}"/>')
    report["builtin_boxes"] = len(layout.get("builtins") or [])
    for gname, geom in glb.geometry.items():
        if not gname.startswith("room"):
            continue
        if (gname == "room_relief" or gname.startswith("room_builtin")) and (not include_relief or mjx):
            continue
        name = _safe(gname)
        vis = add_visual(name, geom, to_mj(geom.vertices))
        room_xml.append(f'    <geom name="{name}" class="visual" {vis}/>')

    # ---- cameras ---------------------------------------------------------
    cam_xml: List[str] = []
    npz = scene_dir / "3d_models" / "scene_pointmap.npz"
    if npz.exists():
        z = np.load(npz)
        R, K = z["R_total"].astype(float), z["intrinsics"]
        right = AXES @ (R @ np.array([-1.0, 0, 0]))
        up = AXES @ (R @ np.array([0, 1.0, 0]))
        fovy = float(np.degrees(2 * np.arctan(0.5 / K[1, 1])))
        cam_xml.append(f'    <camera name="photo" pos="{_fmt(to_mj([0, 0, 0]))}" xyaxes="{_fmt(right)} {_fmt(up)}" '
                       f'fovy="{fovy:.3f}"/>')
    cam_xml.append(f'    <camera name="overview" pos="{_fmt([c[0], c[1], mx[2] + 1.6 * max(half[0], half[1])])}" '
                   f'xyaxes="0 -1 0 1 0 0" fovy="60"/>')

    extent = float(np.linalg.norm(mx - mn))
    option = ('  <option timestep="0.004" iterations="4" ls_iterations="8" solver="Newton">\n'
              '    <flag eulerdamp="disable"/>\n  </option>' if mjx else
              '  <option timestep="0.002" integrator="implicitfast"/>')
    xml = f"""<mujoco model={quoteattr(scene_dir.name)}>
  <!-- Generated by WorldBuilder (src/mujoco_export.py). Units: metres (scale x{s:.4g} from scene units,
       {'user-supplied' if scale else layout.get('metric_scale_source', 'camera-height prior')}). Frame: X forward, Y left, Z up; floor at z=0. -->
  <compiler angle="degree" meshdir="assets" texturedir="assets" autolimits="true" boundmass="0.005" boundinertia="1e-7"/>
{option}
  <statistic center="{_fmt(c)}" extent="{extent:.3f}"/>
  <visual>
    <headlight ambient="0.9 0.9 0.9" diffuse="0.25 0.25 0.25" specular="0 0 0"/>
    <global offwidth="1280" offheight="960"/>
  </visual>
  <default>
    <default class="visual">
      <geom type="mesh" contype="0" conaffinity="0" group="2" density="0"/>
    </default>
    <default class="collision">
      <geom type="mesh" group="3" density="{density:g}" friction="0.8 0.02 0.001" rgba="0.2 0.6 1 0.35"/>
    </default>
    <default class="structure">
      <geom group="4" rgba="0.8 0.8 0.8 0" friction="0.9 0.02 0.001"/>
    </default>
    <default class="pad">
      <geom group="4" rgba="1 0.5 0 0" friction="0.9 0.02 0.001"/>
    </default>
  </default>
  <asset>
{chr(10).join(asset_xml)}
  </asset>
  <worldbody>
    <light name="sun" pos="{_fmt([c[0], c[1], mx[2] + 2])}" dir="0 0 -1" diffuse="0.3 0.3 0.3" castshadow="false"/>
{chr(10).join(cam_xml)}
{chr(10).join(room_xml)}
{chr(10).join(pads_xml)}
{chr(10).join(body_xml)}
  </worldbody>
</mujoco>
"""
    path = out_dir / "scene.xml"
    path.write_text(xml)
    report.update(scale=s, collision=collision, mjx=mjx)
    (out_dir / "export_report.json").write_text(json.dumps(report, indent=2))
    log(f"✓ {path}: {report['objects']} objects ({report['dynamic']} free, {report['static']} welded), "
        f"{report['collision_geoms']} collision geoms, {report['pads']} support pads, scale ×{s:.3g}")
    return path


def check_model(xml_path: Path, seconds: float = 2.0, render: Optional[Path] = None, verbose: bool = True) -> Dict:
    """Load the model, let it settle, and report how far each free body
    moved. Small numbers mean the scene is physically consistent as placed."""
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    start = {j: data.qpos[model.jnt_qposadr[j]:model.jnt_qposadr[j] + 3].copy() for j in free}
    contacts0 = int(data.ncon)
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    moved = {}
    for j in free:
        d = float(np.linalg.norm(data.qpos[model.jnt_qposadr[j]:model.jnt_qposadr[j] + 3] - start[j]))
        moved[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j).removesuffix("_free")] = round(d, 4)
    out = {"bodies": int(model.nbody) - 1, "free_bodies": len(free), "geoms": int(model.ngeom),
           "contacts_at_start": contacts0, "seconds": seconds,
           "moved_over_5cm": {k: v for k, v in moved.items() if v > 0.05},
           "median_displacement_m": round(float(np.median(list(moved.values()))), 4) if moved else 0.0,
           "max_displacement_m": round(max(moved.values()), 4) if moved else 0.0,
           "finite": bool(np.isfinite(data.qpos).all())}
    if render is not None:
        from PIL import Image
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        tiles = []
        with mujoco.Renderer(model, 480, 640) as r:
            for cam in ("photo", "overview"):
                if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam) < 0:
                    continue
                r.update_scene(data, camera=cam)
                tiles.append(r.render())
        if tiles:
            Image.fromarray(np.concatenate(tiles, axis=1)).save(render)
            out["render"] = str(render)
    if verbose:
        print(json.dumps(out, indent=2))
    return out


def export_stable(scene_dir: Path, out_dir: Optional[Path] = None, rounds: int = 3, verbose: bool = True,
                  **kwargs) -> Tuple[Path, Dict]:
    """Export, simulate, weld whatever does not stay put, repeat.

    Single-image reconstruction does not guarantee that every object rests
    on something: a mesh can be a centimetre off its supporter's hull, or
    lean. For learning environments a scene that is still at t = 0 matters
    more than every mug being movable, so objects that drift more than 5 cm
    in two seconds are welded where they were placed. The report lists them.
    """
    welded: set = set()
    path = export_mujoco(scene_dir, out_dir, verbose=verbose, **kwargs)
    result = check_model(path, verbose=False)
    for _ in range(rounds):
        unstable = set(result["moved_over_5cm"])
        if not unstable:
            break
        welded |= unstable
        path = export_mujoco(scene_dir, out_dir, verbose=False, force_static=welded, **kwargs)
        result = check_model(path, verbose=False)
    result["welded_for_stability"] = sorted(welded)
    report_path = path.parent / "export_report.json"
    report = json.loads(report_path.read_text())
    report["stability"] = result
    report_path.write_text(json.dumps(report, indent=2))
    if verbose:
        print(f"  stabilised: {len(welded)} object(s) welded; {result['free_bodies']} remain free, "
              f"max drift {result['max_displacement_m']} m over {result['seconds']} s")
    return path, result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="WorldBuilder scene → MuJoCo MJCF")
    ap.add_argument("scene", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--scale", type=float, help="metres per scene unit (default: layout's estimate)")
    ap.add_argument("--collision", choices=["hull", "coacd", "box"], default="hull")
    ap.add_argument("--dynamic", choices=["supported", "all", "none"], default="supported")
    ap.add_argument("--mjx", action="store_true", help="primitive collisions and MJX-friendly options")
    ap.add_argument("--density", type=float, default=250.0, help="kg/m^3 for collision geoms")
    ap.add_argument("--static-above", type=float, default=1.2, help="objects larger than this (m) are welded")
    ap.add_argument("--no-relief", action="store_true")
    ap.add_argument("--check", action="store_true", help="load in MuJoCo, simulate 2 s, report displacements")
    ap.add_argument("--stabilize", action="store_true",
                    help="weld objects that drift more than 5 cm in 2 s, so the scene starts at rest")
    ap.add_argument("--render", type=Path, help="with --check: save a render from the scene cameras")
    a = ap.parse_args(argv)
    kwargs = dict(scale=a.scale, collision=a.collision, dynamic=a.dynamic, mjx=a.mjx, density=a.density,
                  static_above=a.static_above, include_relief=not a.no_relief)
    if a.stabilize:
        path, _ = export_stable(a.scene, a.out, **kwargs)
    else:
        path = export_mujoco(a.scene, a.out, **kwargs)
    if a.check or a.render:
        check_model(path, render=a.render)
    return 0


if __name__ == "__main__":
    sys.exit(main())
