"""Stage 4: turn stage 3's raw output into a placed scene. CPU only.

Input is the stage-3 cache (``src/stage3_cache.py``): model-space meshes,
raw camera-frame poses and the scene point map, plus the photo and the SAM 3
masks. Output is everything a viewer or DCC importer reads: world-space
PLYs, ``room.ply`` (+ ``room.glb`` with real textures), ``layout.json``,
``scene_pointmap.npz`` and ``scene_combined.ply``.

Because nothing here touches a model, ``scripts/replay_assembly.py`` can
rerun it on a finished scene in seconds.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from src import room_layout as rl
from src import stage3_cache
from src.geometry import world_vertices


def _jsonable(d):
    if isinstance(d, dict):
        return {str(k): _jsonable(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_jsonable(v) for v in d]
    if isinstance(d, (np.floating, np.integer)):
        return d.item()
    if isinstance(d, np.bool_):
        return bool(d)
    if isinstance(d, np.ndarray):
        return d.tolist()
    return d


def load_structural_masks(mask_dir: Path) -> Dict[str, np.ndarray]:
    out = {}
    for k in ("floor", "wall", "ceiling"):
        p = Path(mask_dir) / f"structural_{k}.png"
        if p.exists():
            out[k] = np.asarray(Image.open(p).convert("L")) > 127
    return out


def world_mesh(result: Dict):
    import trimesh
    mesh = trimesh.load(result.get('model_path') or result['ply_path'], process=False)
    wv = world_vertices(np.asarray(mesh.vertices), result)
    colors = mesh.visual.vertex_colors if mesh.visual.kind == 'vertex' else None
    return trimesh.Trimesh(vertices=wv, faces=mesh.faces, vertex_colors=colors, process=False)


def assemble_from_cache(models_dir: Path, image_rgb: np.ndarray,
                        structural_masks: Optional[Dict[str, np.ndarray]] = None,
                        build_room: bool = True, verbose: bool = True,
                        placement: Optional[str] = None) -> Tuple[List[Dict], List[Dict]]:
    """Run layout + placement + room + exports for one scene.

    Returns ``(results, failed)`` where ``results`` carry the final poses and
    ``ply_path`` points at the world-space PLY. ``placement`` selects the
    object-placement strategy: "evidence" (default, ``src/placement.py``),
    "legacy" (box/label heuristics in ``src/room_generator.py``) or "raw"
    (SAM 3D's poses untouched, for comparison).
    """
    from src.room_generator import (
        assemble_scene,
        clamp_to_room,
        clear_vertex_cache,
        compute_per_object_aabb,
        compute_scene_bounds,
        make_box_room,
    )
    log = print if verbose else (lambda *a, **k: None)
    models_dir = Path(models_dir)
    raw = stage3_cache.load_raw(models_dir)
    if raw is None:
        raise FileNotFoundError(f"no stage-3 cache in {models_dir} (scene predates the cache; rerun stage 3)")
    results = copy.deepcopy(raw["objects"])
    failed = raw.get("failed", [])
    pm = stage3_cache.load_pointmap(models_dir)
    clear_vertex_cache()
    if not results:
        return results, failed

    log("\nAssembling scene...")
    layout = None
    floor_y = None
    if pm is not None:
        P, valid = pm["points"], pm["valid"]
        aabbs = compute_per_object_aabb(results)
        pairs = [(np.array(b['min']), np.array(b['max'])) for b in aabbs.values()]
        layout = rl.estimate_layout(P, valid, pairs, structural_masks,
                                    metric_scale=stage3_cache.load_metric_scale(models_dir))
        for n in layout.notes:
            log(f"  layout: {n}")
        R = layout.R_total
        for r in results:
            r.update(rl.rotate_pose(r, R))
        floor_y = layout.floor_y
        step = max(1, int(np.sqrt(P.shape[0] * P.shape[1] / 120_000)))
        Pd = (P[::step, ::step] @ R.T).astype(np.float32)
        Cd = np.array(Image.fromarray(image_rgb).resize((Pd.shape[1], Pd.shape[0]), Image.BILINEAR))
        Vd = valid[::step, ::step]
        np.savez_compressed(models_dir / "scene_pointmap.npz",
                            points=Pd[Vd], colors=Cd[Vd], intrinsics=pm["intrinsics"],
                            R_total=R.astype(np.float32), image_hw=np.array(image_rgb.shape[:2]))

    mode = placement or os.environ.get("WORLDBUILDER_PLACEMENT", "evidence")
    if layout is None and mode == "evidence":
        mode = "legacy"                      # no point map, nothing to ask
    masks = {}
    for r in results:
        mp = r.get("mask_path")
        if mp and Path(mp).exists():
            masks[r["id"]] = np.asarray(Image.open(mp).convert("L")) > 127
    object_pixels = None                     # every pixel that belongs to some object
    if pm is not None and masks:
        object_pixels = np.zeros(pm["valid"].shape, np.int32)
        for rid in sorted(masks, key=lambda i: -int(masks[i].sum())):
            object_pixels[rl._resize_mask(masks[rid], pm["valid"].shape)] = rid
    if mode == "evidence":
        from src.placement import build_evidence, contained_parts, place_objects, structural_duplicates
        from src.room_generator import model_vertices
        structural = structural_duplicates(masks, structural_masks)
        parts = contained_parts({k: v for k, v in masks.items() if k not in structural},
                                {r["id"]: r["label"] for r in results})
        if parts:
            log("  same object twice (mask inside mask): " + ", ".join(
                f"{r['label']} #{r['id']}→#{parts[r['id']]}" for r in results if r["id"] in parts))
            failed = failed + [{"id": r["id"], "label": r["label"], "status": f"part or group of #{parts[r['id']]}"}
                               for r in results if r["id"] in parts]
            results[:] = [r for r in results if r["id"] not in parts]
            masks = {k: v for k, v in masks.items() if k not in parts}
        if structural:
            log("  structural, left to the room: " + ", ".join(
                f"{r['label']} #{r['id']} (is the {structural[r['id']]})" for r in results if r["id"] in structural))
            failed = failed + [{"id": r["id"], "label": r["label"], "status": f"structural: {structural[r['id']]}"}
                               for r in results if r["id"] in structural]
            results[:] = [r for r in results if r["id"] not in structural]
            masks = {k: v for k, v in masks.items() if k not in structural}
            object_pixels[np.isin(object_pixels, list(structural))] = 0
        ev = build_evidence(pm["points"], pm["valid"], pm["intrinsics"], layout.R_total,
                            floor_y=None if str(layout.floor.source).startswith("fallback") else layout.floor_y,
                            room_height=max(1e-6, layout.ceiling_y - layout.floor_y), masks=masks,
                            floor_mask=(structural_masks or {}).get("floor"))
        verts = {r["id"]: model_vertices(r) for r in results}
        verts = {k: v for k, v in verts.items() if v is not None}
        room_info = {"min": list(layout.bounds_min), "max": list(layout.bounds_max),
                     "sources": dict(layout.wall_sources),
                     "ceiling_y": layout.ceiling_y if layout.ceiling_source == "geometric" else None}
        hs, ws = pm["valid"].shape
        image_small = np.asarray(Image.fromarray(image_rgb).resize((ws, hs), Image.BILINEAR))
        diag = place_objects(results, verts, ev, verbose=verbose, room=room_info, image_small=image_small,
                             share=os.environ.get("WORLDBUILDER_SHARE_INSTANCES", "1") != "0")
        diag["structural"] = structural
        failed = failed + [{"id": i, "label": v["label"], "status": f"removed: {v['reason']}"}
                           for i, v in diag.get("removed", {}).items()]
    elif mode == "raw":
        diag = {"bounds": compute_scene_bounds(compute_per_object_aabb(results))}
        log("  placement skipped (raw SAM 3D poses)")
    else:
        diag = assemble_scene(results, floor_y=floor_y, verbose=verbose,
                              stretch_structural=os.environ.get("WORLDBUILDER_STRETCH_STRUCTURAL") == "1")
    diag["placement"] = mode

    if build_room:
        bounds = diag.get('bounds') or compute_scene_bounds(compute_per_object_aabb(results))
        room = None
        if layout is not None and pm is not None:
            try:
                aabbs = compute_per_object_aabb(results)
                enforce = dict(layout.wall_sources)
                floor_fitted = not str(layout.floor.source).startswith("fallback")
                if floor_fitted:
                    enforce["y_min"] = "floor"
                    layout.bounds_min[1] = layout.floor_y
                if mode != "evidence":          # evidence placement keeps objects inside itself
                    n_clamped = clamp_to_room(results, aabbs, layout.bounds_min, layout.bounds_max, enforce)
                    if n_clamped:
                        log(f"  nudged {n_clamped} object(s) back inside detected walls")
                bounds = compute_scene_bounds(aabbs, robust_percentile=0.0)
                bmin, bmax = list(layout.bounds_min), list(layout.bounds_max)
                for side, src in layout.wall_sources.items():
                    axis = 0 if side.startswith("x") else 2
                    if str(src).startswith("wall"):
                        continue
                    if side.endswith("min"):
                        bmin[axis] = min(bmin[axis], bounds['min'][axis] - 0.02)
                    else:
                        bmax[axis] = max(bmax[axis], bounds['max'][axis] + 0.02)
                bmin[1] = layout.floor_y if floor_fitted else min(layout.floor_y, bounds['min'][1])
                # An observed ceiling is evidence too; only a prior-placed one
                # moves up to make room for a tall object.
                bmax[1] = (layout.ceiling_y if layout.ceiling_source == "geometric"
                           else max(layout.ceiling_y, bounds['max'][1] + 0.02))
                layout.bounds_min, layout.bounds_max = bmin, bmax
                from scipy import ndimage

                from src import room_texture as rt
                depth = (pm["points"] @ rl.OPENCV_TO_WORLD.T)[..., 2]
                depth = np.where(pm["valid"], depth, np.nan)
                owner = object_pixels if object_pixels is not None else np.zeros(pm["valid"].shape, np.int32)
                blocked = ndimage.binary_dilation(owner > 0, iterations=2)
                room_meshes, stats = rt.build_shell(layout, pm["intrinsics"], image_rgb, depth, blocked)
                m = room_meshes.get("room_floor")
                for k, v in stats.items():
                    log(f"  room {k}: {v['visible_fraction']:.0%} seen in the photo, rest {v['fill']}")
                relief_stats = {}
                if os.environ.get("WORLDBUILDER_RELIEF", "1") != "0":
                    relief, relief_stats = rt.build_relief(pm["points"] @ layout.R_total.T, pm["valid"], owner,
                                                           layout, image_rgb)
                    if relief is not None:
                        room_meshes["room_relief"] = relief
                        log(f"  relief (built-ins from the point map): {relief_stats['faces']:,} triangles, "
                            f"{relief_stats['frame_fraction']:.0%} of the frame")
                builtins = []
                if relief_stats and os.environ.get("WORLDBUILDER_BUILTINS", "1") != "0" \
                        and getattr(rt.build_relief, "last", None) is not None:
                    from src.builtin_boxes import build_builtins
                    P_bg, keep_bg = rt.build_relief.last
                    density = max(m.visual.material.baseColorTexture.size) / float(
                        max(np.asarray(layout.bounds_max) - np.asarray(layout.bounds_min))) \
                        if "room_floor" in room_meshes else None
                    bi_meshes, builtins = build_builtins(P_bg.astype(np.float64), keep_bg, pm["valid"], layout,
                                                         pm["intrinsics"], image_rgb, depth, blocked, density)
                    room_meshes.update(bi_meshes)
                    if builtins:
                        log(f"  built-ins: {len(builtins)} solid boxes behind the relief "
                            f"({sum(1 for b in builtins if b['thin'])} thinned to slabs)")
                rt.export_room(room_meshes, models_dir / "room.glb")
                room = rt.shell_vertex_color_mesh(room_meshes)
                rl.save_layout(layout, models_dir / "layout.json",
                               extra={"plane_texture_stats": stats, "relief": relief_stats, "builtins": builtins,
                                      "assembly": _jsonable(diag)})
            except Exception as e:
                import traceback
                traceback.print_exc()
                log(f"⚠️  Textured room failed ({e}); using box room")
                room = None
        if room is None and bounds is not None:
            room = make_box_room(bounds)
            (models_dir / "layout.json").write_text(json.dumps(
                {"source": "box-fallback", "bounds": bounds, "assembly": _jsonable(diag)}, indent=2))
        if room is not None:
            room_path = models_dir / "room.ply"
            room.export(str(room_path), file_type='ply')
            log(f"✓ room.ply saved ({room_path.stat().st_size / 1024:.0f} KB)")

    export_world(results, models_dir, verbose=verbose)
    return results, failed


def export_world(results: List[Dict], models_dir: Path, verbose: bool = True):
    """World-space PLY per object (so viewers and importers need no pose
    math) and one merged ``scene_combined.ply``. Stale PLYs from a previous
    assembly of the same scene are removed."""
    import trimesh
    log = print if verbose else (lambda *a, **k: None)
    models_dir = Path(models_dir)
    keep = {"room.ply", "scene_combined.ply"}
    meshes = []
    for r in results:
        src = r.get('model_path')
        if not src or not Path(src).exists():
            continue
        m = world_mesh(r)
        # named by instance, not by mesh: shared instances reuse one model file
        out = models_dir / f"{int(r['id']):03d}_{str(r['label']).replace(' ', '_').replace('/', '_')}.ply"
        m.export(str(out), file_type='ply')
        r['ply_path'] = str(out)
        r['ply_space'] = 'world'
        keep.add(out.name)
        meshes.append(m)
    for p in models_dir.glob("*.ply"):
        if p.name not in keep:
            p.unlink()
    if meshes:
        combined = trimesh.util.concatenate(meshes)
        combined.export(str(models_dir / "scene_combined.ply"), file_type='ply')
        log(f"✓ {len(meshes)} world-space PLYs + scene_combined.ply ({len(combined.faces):,} triangles)")


def finalize_scene(output_path: Path, results: List[Dict], failed: List[Dict], metadata: Dict,
                   verbose: bool = True, lite: bool = True) -> None:
    """Write everything downstream consumers read: the results JSON, the
    viewer and the GLB. Shared by the pipeline and the replay script."""
    log = print if verbose else (lambda *a, **k: None)
    output_path = Path(output_path)
    (output_path / "reconstruction_results.json").write_text(json.dumps(
        {"objects": results, "failed": failed, "metadata": metadata}, indent=2))
    try:
        from src.viewer_generator import generate_viewer
        viewer_path = generate_viewer(output_path, results, room_file="3d_models/room.ply",
                                      layout_file="3d_models/layout.json", failed=failed)
        log(f"✓ Viewer: {viewer_path}  (serve with: python -m http.server -d {output_path})")
    except Exception as e:
        log(f"⚠️  Viewer generation failed: {e}")
    try:
        from src.scene_export import export_scene_glb, export_scene_lite
        glb = export_scene_glb(output_path, results)
        if glb:
            log(f"✓ GLB scene: {glb} ({glb.stat().st_size / 1e6:.0f} MB)")
        if lite and os.environ.get("WORLDBUILDER_LITE", "1") != "0":
            lite_glb = export_scene_lite(output_path, results)
            if lite_glb:
                log(f"✓ Lite GLB (decimated, baked textures): {lite_glb} ({lite_glb.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        import traceback
        traceback.print_exc()
        log(f"⚠️  GLB export failed: {e}")
