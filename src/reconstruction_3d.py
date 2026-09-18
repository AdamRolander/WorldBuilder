"""Stage 3: per-object 3D reconstruction with SAM 3D Objects, plus scene assembly.

For each segmented instance we hand SAM 3D the image and the mask and get
back a textured mesh (as a GLB) and a pose (translation, quaternion, scale)
in the camera frame. Then ``src/room_layout.py`` recovers the room from the
scene point map and ``src/room_generator.py`` places the objects.

What changed from the demo-era version (docs/AUDIT.md §3):

* **One point map per scene, not per object.** SAM 3D runs MoGe on the whole
  image inside every call. We compute it once via the pipeline's own
  ``compute_pointmap`` and pass it back through the public ``pointmap=``
  argument for every object. This removes N-1 MoGe passes and gives the
  layout stage the scene geometry for free. Set
  ``WORLDBUILDER_SHARED_POINTMAP=0`` to get the old behaviour.
* **No GPU in the assembly stage.** Pose math is numpy (``src/geometry.py``);
  the three duplicated CUDA transform routines are gone.
* **Gravity + wall alignment** of the whole scene before snapping, and a
  textured room when the point map is available (box room as fallback).
* Per-object failures are recorded in the results (``status``) instead of
  silently vanishing, so the timing report can count them.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from src.geometry import world_vertices

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SAM3D_REPO = _PROJECT_ROOT / "sam3d_objects_repo"
if str(_SAM3D_REPO / "notebook") not in sys.path:
    sys.path.insert(0, str(_SAM3D_REPO / "notebook"))

# The upstream notebook helper does ``os.environ["CONDA_HOME"] =
# os.environ["CONDA_PREFIX"]`` at import time, so running from a non-activated
# interpreter (cron, systemd, an IDE) crashed with KeyError. Derive it.
if "CONDA_PREFIX" not in os.environ:
    os.environ["CONDA_PREFIX"] = str(Path(sys.executable).resolve().parent.parent)


def save_glb_as_ply(output: dict, output_path: Path):
    """Write SAM 3D's GLB mesh as a vertex-coloured PLY (model space)."""
    glb_mesh = output['glb']
    n_v, n_f = len(glb_mesh.vertices), len(glb_mesh.faces)
    glb_mesh.export(str(output_path), file_type='ply')
    mb = output_path.stat().st_size / (1024 * 1024)
    print(f"    ✓ {output_path.name}: {n_v:,} verts, {n_f:,} faces ({mb:.2f} MB)")
    return glb_mesh


class SAM3DReconstructor:
    """3D reconstruction using SAM 3D Objects (process-wide singleton model)."""

    _shared_inference = None
    _shared_config_path = None

    def __init__(self, sam3d_repo_path: Optional[str] = None, checkpoint_tag: str = "hf"):
        from inference import Inference  # sam3d_objects_repo/notebook/inference.py

        self.sam3d_path = Path(sam3d_repo_path) if sam3d_repo_path else _SAM3D_REPO
        config_path = str(self.sam3d_path / "checkpoints" / checkpoint_tag / "pipeline.yaml")
        if not Path(config_path).exists():
            raise FileNotFoundError(
                f"SAM 3D config not found: {config_path}\n"
                f"  Download the checkpoints per README §4 (the 'mv' step is easy to miss).")

        if (SAM3DReconstructor._shared_inference is not None
                and SAM3DReconstructor._shared_config_path == config_path):
            self.inference = SAM3DReconstructor._shared_inference
            print("\n✓ SAM 3D Objects (reusing cached model)")
        else:
            print("\nLoading SAM 3D Objects...")
            self.inference = Inference(config_path, compile=False)
            SAM3DReconstructor._shared_inference = self.inference
            SAM3DReconstructor._shared_config_path = config_path
            print("✓ SAM 3D Objects loaded")

    # -- VRAM management ----------------------------------------------------
    def _to_device(self, device: str):
        """Move every torch module hanging off the inference pipeline.

        ``Inference`` keeps everything under the *private* attribute
        ``_pipeline`` (an ``InferencePipelinePointMap`` whose ``models``
        ModuleDict, ``depth_model`` and ``pose_decoder`` hold the weights).
        The demo-era version skipped underscore attributes, so SAM 3D never
        actually left the GPU: ~13 GB stayed resident between images and the
        local VLM OOM'd on the second photo of every batch (the "works for one
        image, then 500s" symptom). Walk the pipeline's ``__dict__`` directly.
        """
        import gc

        import torch
        moved = 0
        seen = set()

        def _move(obj, depth=0):
            nonlocal moved
            if depth > 3 or id(obj) in seen:
                return
            seen.add(id(obj))
            if isinstance(obj, torch.nn.Module):
                obj.to(device)
                moved += 1
                return
            if isinstance(obj, dict):
                for v in obj.values():
                    _move(v, depth + 1)
                return
            if isinstance(obj, (list, tuple)):
                for v in obj:
                    _move(v, depth + 1)
                return
            d = getattr(obj, '__dict__', None)
            if d is None or callable(obj):
                return
            for v in list(d.values()):
                _move(v, depth + 1)

        _move(getattr(self.inference, '_pipeline', self.inference))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            print(f"  SAM 3D → {device} ({moved} modules); GPU free {free / 1024**3:.1f} / {total / 1024**3:.1f} GiB")

    # -- scene point map ----------------------------------------------------
    def compute_scene_pointmap(self, image_rgb: np.ndarray) -> Dict:
        """Run the pipeline's depth model once for the whole image.

        Returns ``{"pointmap": (H,W,3) float32 torch tensor in SAM 3D's
        camera-frame world, "intrinsics": (3,3) normalised, "valid": (H,W)}``.
        The tensor is exactly what ``InferencePipelinePointMap.run(pointmap=…)``
        expects (it skips MoGe and only interpolates if sizes differ).
        """
        import torch
        pipe = self.inference._pipeline
        rgba = np.concatenate([image_rgb[..., :3], np.full(image_rgb.shape[:2] + (1,), 255, np.uint8)], -1)
        # Temporarily disable the per-mask clipping so the cached map is raw.
        clip = getattr(pipe, "clip_pointmap_beyond_scale", None)
        pipe.clip_pointmap_beyond_scale = None
        try:
            with torch.no_grad():
                pm = pipe.compute_pointmap(rgba)
        finally:
            pipe.clip_pointmap_beyond_scale = clip
        points = pm["pointmap"].permute(1, 2, 0).contiguous()          # HxWx3, camera-frame world
        intr = pm.get("intrinsics")
        intr = intr.detach().float().cpu().numpy() if intr is not None else None
        if intr is not None and intr.ndim == 3:
            intr = intr[0]
        valid = torch.isfinite(points).all(-1) & (points.norm(dim=-1) > 1e-6)
        valid_np = valid.cpu().numpy()
        if intr is None:
            # Upstream only stores intrinsics when the depth model gave none;
            # MoGe gives them, so they are dropped. Recover from geometry.
            from src.room_layout import intrinsics_from_pointmap
            intr = intrinsics_from_pointmap(points.detach().float().cpu().numpy(), valid_np)
        return {"pointmap": points, "intrinsics": intr, "valid": valid_np}

    # -- per-object reconstruction -----------------------------------------
    def reconstruct_objects(self, image_path: str, segmentation_results: List[Dict],
                            output_dir: str = "outputs/3d_models", quality: str = 'high',
                            seed: int = 42, structural_masks: Optional[Dict[str, np.ndarray]] = None,
                            build_room: bool = True) -> List[Dict]:
        """Reconstruct every segmented instance, then assemble the scene.

        Returns one dict per *successful* object (``status == "ok"``); failed
        ones are appended with ``status`` set and no ``ply_path`` so callers
        can report them. Writes ``room.ply``, ``layout.json``,
        ``scene_pointmap.npz`` and ``scene_combined.ply`` next to the objects.
        """
        import torch
        from inference import load_image

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"\nLoading image: {image_path}")
        image = load_image(str(image_path))          # HxWx3 uint8 (RGB)
        H, W = image.shape[:2]

        use_shared = os.environ.get("WORLDBUILDER_SHARED_POINTMAP", "1") != "0"
        scene_pm = None
        if use_shared:
            try:
                t0 = time.time()
                scene_pm = self.compute_scene_pointmap(image)
                print(f"✓ Scene point map {tuple(scene_pm['pointmap'].shape[:2])} "
                      f"in {time.time() - t0:.1f}s (shared across {len(segmentation_results)} objects)")
            except Exception as e:
                print(f"⚠️  Shared point map failed ({str(e)[:120]}); falling back to per-object MoGe")
                scene_pm = None

        results: List[Dict] = []
        failed: List[Dict] = []
        print(f"\nReconstructing {len(segmentation_results)} objects...")
        print("=" * 60)
        for i, seg in enumerate(segmentation_results, 1):
            obj_id, label = seg['id'], seg['label']
            print(f"\n[{i}/{len(segmentation_results)}] {label}")
            t0 = time.time()
            try:
                mask = np.array(Image.open(seg['mask_path']).convert('L')) > 128
                if mask.shape != (H, W):
                    mask = np.array(Image.fromarray(mask.astype(np.uint8) * 255)
                                    .resize((W, H), Image.NEAREST)) > 128
                if mask.sum() < 100:
                    raise ValueError("mask too small")

                pm_arg = scene_pm["pointmap"] if scene_pm is not None else None
                try:
                    output = self.inference(image, mask, seed=seed, pointmap=pm_arg)
                except Exception as e:
                    if pm_arg is None:
                        raise
                    print(f"  ⚠️  shared-pointmap path failed ({str(e)[:80]}); retrying with per-object MoGe")
                    output = self.inference(image, mask, seed=seed)

                translation = output['translation'].cpu().numpy()[0]
                rotation = output['rotation'].cpu().numpy()[0]
                scale = output['scale'].cpu().numpy()[0]
                safe_label = label.replace(' ', '_').replace('/', '_')
                ply_path = output_path / f"{obj_id:03d}_{safe_label}.ply"
                save_glb_as_ply(output, ply_path)
                results.append({
                    'id': obj_id, 'label': label, 'status': 'ok',
                    'ply_path': str(ply_path),
                    'translation': translation.tolist(),
                    'rotation_quaternion': rotation.tolist(),
                    'scale': scale.tolist(),
                    'bbox': seg.get('bbox', []),
                    'confidence': seg.get('confidence', 0),
                    'mask_path': seg.get('mask_path'),
                    'seconds': round(time.time() - t0, 2),
                })
                print(f"  ✓ pose t={np.round(translation, 3).tolist()} "
                      f"s={np.round(scale, 3).tolist()} ({time.time() - t0:.1f}s)")
                del output
            except Exception as e:
                msg = str(e)[:150]
                print(f"  ❌ Failed: {msg}")
                failed.append({'id': obj_id, 'label': label, 'status': f'failed: {msg}',
                               'confidence': seg.get('confidence', 0)})
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print("\n" + "=" * 60)
        print(f"✓ Reconstructed {len(results)}/{len(segmentation_results)} objects")

        # ------------------------------------------------------------------
        # Scene assembly (CPU). Keep the model-space PLYs until the very end:
        # the layout/assembly steps only change poses.
        # ------------------------------------------------------------------
        if results:
            try:
                self._assemble(results, scene_pm, image, structural_masks, output_path,
                                        build_room=build_room)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"⚠️  Scene assembly failed: {e}")

        if results:
            self._save_combined_scene(results, output_path / "scene_combined.ply")
            self._bake_world_space_plys(results)

        return results + failed

    # -- assembly -------------------------------------------------------------
    def _assemble(self, results: List[Dict], scene_pm: Optional[Dict], image: np.ndarray,
                  structural_masks: Optional[Dict[str, np.ndarray]], output_path: Path,
                  build_room: bool = True):
        from src import room_layout as rl
        from src.room_generator import (
            assemble_scene,
            compute_per_object_aabb,
            compute_scene_bounds,
            make_box_room,
        )

        print("\nAssembling scene...")
        layout = None
        floor_y = None
        if scene_pm is not None:
            P = scene_pm["pointmap"].detach().float().cpu().numpy().astype(np.float64)
            valid = scene_pm["valid"]
            aabbs = compute_per_object_aabb(results)
            pairs = [(np.array(b['min']), np.array(b['max'])) for b in aabbs.values()]
            layout = rl.estimate_layout(P, valid, pairs, structural_masks)
            for n in layout.notes:
                print(f"  layout: {n}")
            # Rotate every object pose into the gravity/wall-aligned frame.
            R = layout.R_total
            for r in results:
                r.update(rl.rotate_pose(r, R))
            floor_y = layout.floor_y
            # Point cloud for debugging / future viewers (downsampled).
            step = max(1, int(np.sqrt(P.shape[0] * P.shape[1] / 120_000)))
            Pd = (P[::step, ::step] @ R.T).astype(np.float32)
            Cd = np.array(Image.fromarray(image).resize((Pd.shape[1], Pd.shape[0]), Image.BILINEAR))
            Vd = valid[::step, ::step]
            np.savez_compressed(output_path / "scene_pointmap.npz",
                                points=Pd[Vd], colors=Cd[Vd], intrinsics=scene_pm["intrinsics"],
                                R_total=R.astype(np.float32), image_hw=np.array(image.shape[:2]))

        diag = assemble_scene(results, floor_y=floor_y,
                              stretch_structural=os.environ.get("WORLDBUILDER_STRETCH_STRUCTURAL") == "1")

        if build_room:
            bounds = diag.get('bounds') or compute_scene_bounds(compute_per_object_aabb(results))
            room = None
            if layout is not None and scene_pm is not None:
                try:
                    # Objects poking through *detected* walls are nudged back
                    # in; the room only grows toward sides with no wall evidence.
                    from src.room_generator import clamp_to_room
                    aabbs = compute_per_object_aabb(results)
                    enforce = dict(layout.wall_sources)
                    floor_fitted = not str(layout.floor.source).startswith("fallback")
                    if floor_fitted:
                        enforce["y_min"] = "floor"
                        layout.bounds_min[1] = layout.floor_y
                    n_clamped = clamp_to_room(results, aabbs, layout.bounds_min, layout.bounds_max, enforce)
                    if n_clamped:
                        print(f"  nudged {n_clamped} object(s) back inside detected walls")
                        bounds = compute_scene_bounds(aabbs)
                    bmin, bmax = list(layout.bounds_min), list(layout.bounds_max)
                    for side, src in layout.wall_sources.items():
                        axis = 0 if side.startswith("x") else 2
                        if str(src).startswith("wall"):
                            continue
                        if side.endswith("min"):
                            bmin[axis] = min(bmin[axis], bounds['min'][axis] - 0.02)
                        else:
                            bmax[axis] = max(bmax[axis], bounds['max'][axis] + 0.02)
                    # A fitted floor is evidence; never lower it to a sunk object.
                    bmin[1] = layout.floor_y if floor_fitted else min(layout.floor_y, bounds['min'][1])
                    bmax[1] = max(layout.ceiling_y, bounds['max'][1] + 0.02)
                    layout.bounds_min, layout.bounds_max = bmin, bmax
                    P_cv = scene_pm["pointmap"].detach().float().cpu().numpy() @ rl.OPENCV_TO_WORLD.T
                    depth = P_cv[..., 2].astype(np.float64)
                    room, stats = rl.build_textured_room(layout, scene_pm["intrinsics"], image, depth)
                    for k, v in stats.items():
                        print(f"  room {k}: {v['visible_fraction']:.0%} textured from photo")
                    rl.save_layout(layout, output_path / "layout.json",
                                   extra={"plane_texture_stats": stats, "assembly": _jsonable(diag)})
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"⚠️  Textured room failed ({e}); using box room")
                    room = None
            if room is None and bounds is not None:
                room = make_box_room(bounds)
                (output_path / "layout.json").write_text(json.dumps(
                    {"source": "box-fallback", "bounds": bounds, "assembly": _jsonable(diag)}, indent=2))
            if room is not None:
                room_path = output_path / "room.ply"
                room.export(str(room_path), file_type='ply')
                print(f"✓ room.ply saved ({room_path.stat().st_size / 1024:.0f} KB)")
        return layout

    # -- exports -----------------------------------------------------------------
    @staticmethod
    def _world_mesh(result: Dict):
        import trimesh
        mesh = trimesh.load(result['ply_path'], process=False)
        wv = world_vertices(np.asarray(mesh.vertices), result)
        colors = mesh.visual.vertex_colors if mesh.visual.kind == 'vertex' else None
        return trimesh.Trimesh(vertices=wv, faces=mesh.faces, vertex_colors=colors, process=False)

    def _save_combined_scene(self, results: List[Dict], output_path: Path):
        import trimesh
        meshes = [self._world_mesh(r) for r in results
                  if r.get('ply_path') and Path(r['ply_path']).exists()]
        if not meshes:
            return
        combined = trimesh.util.concatenate(meshes)
        combined.export(str(output_path), file_type='ply')
        mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✓ scene_combined.ply: {len(combined.faces):,} triangles ({mb:.2f} MB)")

    def _bake_world_space_plys(self, results: List[Dict]):
        """Overwrite each object PLY with world-space vertices so viewers and
        DCC importers need no pose math. Runs last: everything above consumes
        model-space PLYs plus poses."""
        baked = 0
        for r in results:
            if not r.get('ply_path') or not Path(r['ply_path']).exists():
                continue
            self._world_mesh(r).export(str(r['ply_path']), file_type='ply')
            r['ply_space'] = 'world'
            baked += 1
        print(f"✓ Baked {baked}/{len(results)} PLYs to world space")


def _jsonable(d):
    if isinstance(d, dict):
        return {str(k): _jsonable(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_jsonable(v) for v in d]
    if isinstance(d, (np.floating, np.integer)):
        return d.item()
    if isinstance(d, np.ndarray):
        return d.tolist()
    return d
