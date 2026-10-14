"""Stage 4: turn stage 3's raw output into a placed scene. CPU only.

Input is the stage-3 cache (``src/stage3_cache.py``): model-space meshes,
raw camera-frame poses and the scene point map, plus the photo and the SAM 3
masks. Output is everything a viewer or DCC importer reads: world-space
PLYs, ``room.ply`` (+ ``room.glb`` with real textures), ``layout.json``,
and ``scene_pointmap.npz``.

Because nothing here touches a model, ``scripts/replay_assembly.py`` can
rerun it on a finished scene in seconds.
"""
from __future__ import annotations

import copy
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from src import profiling, stage3_cache
from src import room_layout as rl
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
    for k in ("floor", "wall", "ceiling", "window"):
        p = Path(mask_dir) / f"structural_{k}.png"
        if p.exists():
            out[k] = np.asarray(Image.open(p).convert("L")) > 127
    return out


def ev_normals_up(pm: Dict, layout) -> np.ndarray:
    """Per-pixel upward component of the surface normal in the aligned frame."""
    N = rl.compute_normals(pm["points"], pm["valid"]) @ layout.R_total.T
    return np.nan_to_num(N[..., 1], nan=0.0)


def world_mesh(result: Dict):
    import trimesh
    mesh = trimesh.load(result.get('model_path') or result['ply_path'], process=False)
    wv = world_vertices(np.asarray(mesh.vertices), result)
    colors = mesh.visual.vertex_colors if mesh.visual.kind == 'vertex' else None
    return trimesh.Trimesh(vertices=wv, faces=mesh.faces, vertex_colors=colors, process=False)


def clean_window_mask(window: Optional[np.ndarray], points_aligned: np.ndarray, valid: np.ndarray,
                      layout, masks: Optional[Dict[int, np.ndarray]] = None, min_inside_frac: float = 0.04,
                      same_iou: float = 0.85) -> Optional[np.ndarray]:
    """Drop "window" regions that stand inside the room. SAM 3 calls the
    glass front of a display case or a cabinet a window too; a real window
    is in a wall, so what the photo saw there lies on or beyond one of the
    room's sides. A region whose points are well in front of every side is
    furniture glass, and the things behind it are objects, not the view.
    So is a region that *is* a detected object's mask (a glass-fronted case
    flush with the wall): a tree seen through a pane covers only part of it."""
    if window is None or not window.any():
        return window
    from scipy import ndimage
    h, w = valid.shape
    win = rl._resize_mask(window, (h, w))
    mn, mx = np.asarray(layout.bounds_min, float), np.asarray(layout.bounds_max, float)
    span = float(max(mx[0] - mn[0], mx[2] - mn[2]))
    lab, n = ndimage.label(win)
    keep = np.zeros((h, w), bool)
    for i in range(1, n + 1):
        comp = (lab == i) & valid
        if int(comp.sum()) < 10:
            continue
        P = np.median(points_aligned[comp], axis=0)
        inset = min(P[0] - mn[0], mx[0] - P[0], mx[2] - P[2])     # distance in front of the nearest side
        if inset > min_inside_frac * span:
            continue
        ci = lab == i
        if any(float((mm & ci).sum()) >= same_iou * float((mm | ci).sum())
               for mm in (rl._resize_mask(m, (h, w)) for m in (masks or {}).values())):
            continue
        keep |= ci
    return keep


def beyond_window(masks: Dict[int, np.ndarray], window: Optional[np.ndarray], points_aligned: np.ndarray,
                  valid: np.ndarray, layout, min_inside: float = 0.7, min_beyond: float = 0.6) -> List[int]:
    """Detections that are the view through a window: the trees, cars and
    buildings outside, a blind in the frame. Their mask lies mostly inside
    the window mask *and* the points the photo saw there lie on or beyond an
    observed wall. A plant on the sill is inside the window's outline too,
    but in front of the wall."""
    if window is None or not window.any():
        return []
    h, w = valid.shape
    win = rl._resize_mask(window, (h, w))
    mn, mx = np.asarray(layout.bounds_min, float), np.asarray(layout.bounds_max, float)
    tol = 0.02 * float(max(mx[0] - mn[0], mx[2] - mn[2]))
    out = []
    for rid, m in masks.items():
        mm = rl._resize_mask(m, (h, w)) & valid
        n = int(mm.sum())
        if n < 20 or float((mm & win).sum()) < min_inside * n:
            continue
        inside = float((mm & win).sum()) / n
        P = points_aligned[mm]
        beyond = np.zeros(len(P), bool)
        on_wall = np.zeros(len(P), bool)
        for side, src in layout.wall_sources.items():
            if not str(src).startswith("wall"):
                continue
            axis = 0 if side.startswith("x") else 2
            d = (mn[axis] - P[:, axis]) if side.endswith("min") else (P[:, axis] - mx[axis])   # >0 beyond
            beyond |= d > tol
            on_wall |= d > -tol
        # A depth map often reads window glass as flat, putting the view on
        # the wall plane: then the mask must lie almost wholly in the window.
        if float(beyond.mean()) >= min_beyond or (inside >= 0.9 and float(on_wall.mean()) >= min_beyond):
            out.append(int(rid))
    return out


def mark_glass(results: List[Dict], verts: Dict[int, np.ndarray], seg_json: Path) -> None:
    """Flag transparent objects so exports draw them translucent.

    Two pieces of evidence must agree: over the object's own mask the photo
    saw something well behind the placed mesh (``seen_through``, set by
    ``placement.enforce_visibility``), and the detector described the object
    as glass. Neither alone is enough: a misplaced object is also "seen
    through", and a glass-fronted oven is not transparent. Flat things
    against a wall (mirrors, framed glass) are excluded: a mirror shows a
    reflection, which also reads as depth behind it. Descriptions are copied
    onto the results either way.
    """
    desc = {}
    if seg_json.exists():
        try:
            desc = {int(s["id"]): s.get("description", "") for s in json.loads(seg_json.read_text())}
        except (ValueError, KeyError, TypeError):
            desc = {}
    for r in results:
        d = desc.get(int(r["id"]))
        if d:
            r["description"] = d
        # only what the object itself is made of: "metal handle on a glass
        # door" is not glass, "glass case with cakes" is
        head = re.split(r"\b(?:on|with|in|inside|of|under|behind|near|against|through)\b", (d or "").lower())[0]
        text = f"{r.get('label', '')} {head}".lower()
        if (r.get("seen_through") or 0.0) < 0.5 or "glass" not in text:
            continue
        v = verts.get(r["id"])
        if v is None:
            continue
        ext = np.sort(np.ptp(world_vertices(v, r), axis=0))
        if ext[0] < 0.15 * ext[2] and r.get("against_wall"):
            continue
        r["glass"] = True


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
    t_phase = time.time()
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

    profiling.add("assemble.layout", time.time() - t_phase)
    t_phase = time.time()
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
        from src.placement import (
            build_evidence,
            contained_parts,
            ground_masks,
            place_objects,
            structural_duplicates,
        )
        from src.room_generator import model_vertices
        if structural_masks and structural_masks.get("window") is not None:
            structural_masks = dict(structural_masks)
            structural_masks["window"] = clean_window_mask(structural_masks["window"],
                                                           pm["points"] @ layout.R_total.T, pm["valid"], layout,
                                                           masks)
        structural = structural_duplicates(masks, structural_masks)
        for rid in beyond_window(masks, (structural_masks or {}).get("window"), pm["points"] @ layout.R_total.T,
                                 pm["valid"], layout):
            structural.setdefault(rid, "view through a window")
        if layout.scene_type == "open" and "floor" not in (structural_masks or {}):
            P_al = pm["points"] @ layout.R_total.T
            N_al = rl.compute_normals(pm["points"], pm["valid"]) @ layout.R_total.T
            for rid, what in ground_masks({k: v for k, v in masks.items() if k not in structural}, P_al,
                                          np.nan_to_num(N_al), pm["valid"], layout.floor_y,
                                          layout.camera_height).items():
                structural[rid] = what
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
        for side in diag.get("disputed_walls", []):
            layout.wall_sources[side] = "extent (objects stand beyond the fitted plane)"
        diag["structural"] = structural
        mark_glass(results, verts, models_dir.parent / "segmentation_results.json")
        if any(r.get("glass") for r in results):
            log("  glass (the photo sees through it; exported translucent): " + ", ".join(
                f"{r['label']} #{r['id']}" for r in results if r.get("glass")))
        failed = failed + [{"id": i, "label": v["label"], "status": f"removed: {v['reason']}"}
                           for i, v in diag.get("removed", {}).items()]
    elif mode == "raw":
        diag = {"bounds": compute_scene_bounds(compute_per_object_aabb(results))}
        log("  placement skipped (raw SAM 3D poses)")
    else:
        diag = assemble_scene(results, floor_y=floor_y, verbose=verbose,
                              stretch_structural=os.environ.get("WORLDBUILDER_STRETCH_STRUCTURAL") == "1")
    diag["placement"] = mode
    profiling.add("assemble.placement", time.time() - t_phase)
    t_phase = time.time()

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
                glass_ids = [r["id"] for r in results if r.get("glass")]
                if glass_ids:
                    # the photo sees through glass: what is behind it (the tiled
                    # wall of a shower) belongs to the room's surfaces
                    owner = np.where(np.isin(owner, glass_ids), 0, owner)
                # grown past the mask edge: the soft edge and the shadow an
                # object casts are not the surface's material
                blocked = ndimage.binary_dilation(owner > 0, iterations=max(2, int(0.006 * owner.shape[0])))
                is_open = layout.scene_type == "open"
                shell_layout, sky_rgb, terrain = layout, None, False
                if is_open:
                    # Is the ground a plane (a hall, a paved yard) or terrain? A flat
                    # ground keeps its textured quad; uneven ground is meshed from the
                    # point map instead and the quad drops below it as a base.
                    P_al = pm["points"] @ layout.R_total.T
                    ground = pm["valid"] & (owner == 0)
                    fm = (structural_masks or {}).get("floor")
                    if fm is not None and rl._resize_mask(fm, pm["valid"].shape).mean() > 0.02:
                        ground &= rl._resize_mask(fm, pm["valid"].shape)
                    else:
                        ground &= (np.abs(P_al[..., 1] - layout.floor_y) < 0.5 * layout.camera_height) \
                            & (ev_normals_up(pm, layout) > 0.7)
                    # judged near the camera: far ground is noisy in depth, not rough
                    near = np.hypot(P_al[..., 0], P_al[..., 2]) < 5.0 * layout.camera_height
                    if (ground & near).sum() > 100:
                        ground &= near
                    if ground.sum() > 100:
                        gy = P_al[ground][:, 1]
                        lo, hi = np.percentile(gy, [3, 97])
                        rough = float(hi - lo) / max(layout.camera_height, 1e-6)
                        terrain = rough > 0.08
                        log(f"  ground: height varies by {rough:.0%} of the camera height: "
                            + ("meshed as terrain" if terrain else "flat, kept as a plane"))
                        if terrain:
                            shell_layout = copy.copy(layout)
                            shell_layout.bounds_min = list(layout.bounds_min)
                            shell_layout.bounds_min[1] = min(layout.floor_y, float(lo)) - 0.02 * layout.camera_height
                    sky = ~pm["valid"]
                    sky[sky.shape[0] // 2:] = False
                    if sky.sum() > 200:
                        small = np.asarray(Image.fromarray(image_rgb).resize(sky.shape[::-1], Image.BILINEAR))
                        sky_rgb = [int(c) for c in np.median(small[sky], axis=0)]
                # windows are painted where seen but never teach a wall its material
                win = (structural_masks or {}).get("window")
                exclude = None
                if win is not None and win.any():
                    exclude = ndimage.binary_dilation(rl._resize_mask(win, pm["valid"].shape),
                                                      iterations=max(1, int(0.004 * owner.shape[0])))
                room_meshes, stats = rt.build_shell(shell_layout, pm["intrinsics"], image_rgb, depth, blocked,
                                                    exclude=exclude)
                if terrain and stats.get("floor", {}).get("fill") in ("fallback", "median"):
                    # nothing of the lowered base was seen: give it the ground's own colour
                    small = np.asarray(Image.fromarray(image_rgb).resize(ground.shape[::-1], Image.BILINEAR))
                    rgb = np.median(small[ground], axis=0).astype(np.uint8)
                    room_meshes["room_floor"].visual.material.baseColorTexture = Image.fromarray(
                        np.tile(rgb, (8, 8, 1)))
                    stats["floor"]["fill"] = "ground colour"
                m = room_meshes.get("room_floor")
                for k, v in stats.items():
                    log(f"  room {k}: {v['visible_fraction']:.0%} seen in the photo, rest {v['fill']}")
                relief_stats = {}
                if os.environ.get("WORLDBUILDER_RELIEF", "1") != "0":
                    relief, relief_stats = rt.build_relief(
                        pm["points"] @ layout.R_total.T, pm["valid"], owner, layout, image_rgb,
                        terrain=terrain, max_view_angle_deg=84.0 if terrain else 80.0)
                    if relief:
                        room_meshes.update(relief)
                        log(f"  relief (built-ins from the point map): {relief_stats['faces']:,} triangles in "
                            f"{relief_stats['pieces']} pieces, {relief_stats['frame_fraction']:.0%} of the frame")
                builtins = []
                if relief_stats and not is_open and os.environ.get("WORLDBUILDER_BUILTINS", "1") != "0" \
                        and getattr(rt.build_relief, "last", None) is not None:
                    from src.builtin_boxes import build_builtins
                    P_bg, keep_bg = rt.build_relief.last
                    density = max(m.visual.material.baseColorTexture.size) / float(
                        max(np.asarray(layout.bounds_max) - np.asarray(layout.bounds_min))) \
                        if "room_floor" in room_meshes else None
                    bi_meshes, builtins = build_builtins(P_bg.astype(np.float64), keep_bg, pm["valid"], layout,
                                                         pm["intrinsics"], image_rgb, depth, blocked, density,
                                                         object_boxes=[(np.asarray(b["min"]), np.asarray(b["max"]))
                                                                       for b in aabbs.values()])
                    room_meshes.update(bi_meshes)
                    if builtins:
                        log(f"  built-ins: {len(builtins)} solid boxes behind the relief "
                            f"({sum(1 for b in builtins if b['thin'])} thinned to slabs)")
                rt.export_room(room_meshes, models_dir / "room.glb")
                room = rt.shell_vertex_color_mesh(room_meshes)
                rl.save_layout(layout, models_dir / "layout.json",
                               extra={"plane_texture_stats": stats, "relief": relief_stats, "builtins": builtins,
                                      "sky_rgb": sky_rgb, "terrain": terrain,
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

    profiling.add("assemble.room", time.time() - t_phase)
    t_phase = time.time()
    photo_ctx = None
    if (layout is not None and pm is not None and object_pixels is not None
            and os.environ.get("WORLDBUILDER_PHOTO_TEXTURE", "1") != "0"):
        from src.object_texture import make_context
        cam_z = np.where(pm["valid"], (pm["points"] @ rl.OPENCV_TO_WORLD.T)[..., 2], np.nan)
        photo_ctx = make_context(image_rgb, pm["intrinsics"], layout.R_total, cam_z, object_pixels)
    export_world(results, models_dir, verbose=verbose, photo_ctx=photo_ctx)
    profiling.add("assemble.export_world", time.time() - t_phase)
    return results, failed


def export_world(results: List[Dict], models_dir: Path, verbose: bool = True, photo_ctx=None):
    """World-space PLY per object (so viewers and importers need no pose
    math), optionally one merged ``scene_combined.ply``. Stale PLYs from a previous
    assembly of the same scene are removed. With ``photo_ctx`` the side of
    each object the photo saw is re-coloured from the photo
    (``src/object_texture.py``)."""
    import trimesh
    log = print if verbose else (lambda *a, **k: None)
    models_dir = Path(models_dir)
    keep = {"room.ply", "scene_combined.ply"}
    meshes = []
    painted: List[float] = []
    for r in results:
        src = r.get('model_path')
        if not src or not Path(src).exists():
            continue
        m = world_mesh(r)
        if photo_ctx is not None:
            try:
                from src.object_texture import repaint
                st = repaint(m, int(r['id']), photo_ctx)
                if st is not None:
                    r['photo_painted'] = st['painted']
                    painted.append(st['painted'])
            except Exception as e:                     # appearance only: never lose the object over it
                log(f"  ⚠️  photo texture skipped for {r['label']} #{r['id']}: {str(e)[:80]}")
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
    if meshes and os.environ.get("WORLDBUILDER_COMBINED") == "1":
        # One merged mesh of every object. Nothing in the project reads it and it
        # doubles the scene's size on disk, so it is opt-in.
        combined = trimesh.util.concatenate(meshes)
        combined.export(str(models_dir / "scene_combined.ply"), file_type='ply')
        log(f"✓ scene_combined.ply ({len(combined.faces):,} triangles)")
    elif (models_dir / "scene_combined.ply").exists():
        (models_dir / "scene_combined.ply").unlink()         # stale, from an earlier assembly
    if meshes:
        log(f"✓ {len(meshes)} world-space PLYs")
    if painted:
        log(f"✓ photo re-projected onto {sum(1 for x in painted if x > 0.02)}/{len(painted)} objects "
            f"(mean {100 * float(np.mean(painted)):.0f}% of each mesh's vertices)")


def finalize_scene(output_path: Path, results: List[Dict], failed: List[Dict], metadata: Dict,
                   verbose: bool = True, lite: bool = True) -> None:
    """Write everything downstream consumers read: the results JSON, the
    viewer and the GLB. Shared by the pipeline and the replay script."""
    log = print if verbose else (lambda *a, **k: None)
    output_path = Path(output_path)
    (output_path / "reconstruction_results.json").write_text(json.dumps(
        {"objects": results, "failed": failed, "metadata": metadata}, indent=2))
    want_lite = lite and os.environ.get("WORLDBUILDER_LITE", "1") != "0"
    for stale in ("scene_lite.edited.glb", "scene.edited.glb") + (() if want_lite else ("scene_lite.glb",)):
        if (output_path / stale).exists():       # built from an earlier assembly of this scene
            (output_path / stale).unlink()
    try:
        from src.scene_export import export_scene_glb, export_scene_lite
        t_phase = time.time()
        glb = export_scene_glb(output_path, results)
        profiling.add("finalize.glb", time.time() - t_phase)
        t_phase = time.time()
        if glb:
            log(f"✓ GLB scene: {glb} ({glb.stat().st_size / 1e6:.0f} MB)")
        if want_lite:
            lite_glb = export_scene_lite(output_path, results)
            if lite_glb:
                log(f"✓ Lite GLB (decimated, baked textures): {lite_glb} ({lite_glb.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        import traceback
        traceback.print_exc()
        log(f"⚠️  GLB export failed: {e}")
    profiling.add("finalize.lite_glb", time.time() - t_phase)
    t_phase = time.time()
    try:
        from src.viewer_generator import generate_viewer
        viewer_path = generate_viewer(output_path, results, room_file="3d_models/room.ply",
                                      layout_file="3d_models/layout.json", failed=failed)
        log(f"✓ Viewer: {viewer_path}  (serve with: python -m http.server -d {output_path})")
    except Exception as e:
        log(f"⚠️  Viewer generation failed: {e}")
    profiling.add("finalize.viewer", time.time() - t_phase)
    write_metrics(output_path, results, failed)


def write_metrics(output_path: Path, results: List[Dict], failed: List[Dict]) -> Optional[Path]:
    """``metrics.json``: time per phase, GPU memory peaks, seconds per object,
    output sizes. A replay keeps the GPU phases of the original run and
    replaces only the phases it re-ran."""
    prof = profiling.current()
    if prof is None:
        return None
    output_path = Path(output_path)
    path = output_path / "metrics.json"
    new = prof.to_dict()
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except ValueError:
            data = {}
    gpu_run = any(k in new["seconds"] for k in ("segment", "reconstruct"))
    if gpu_run or not data:
        data = new
    else:
        data.setdefault("seconds", {}).update(new["seconds"])
        data["replay_wall_sec"] = new["wall_sec"]
    data["objects"] = {"placed": len(results), "failed_or_removed": len(failed)}
    per_obj = profiling.summarize_objects(results)
    if per_obj:
        data["seconds_per_object"] = per_obj
    sizes = {}
    for name in ("scene.glb", "scene_lite.glb", "3d_models/room.glb"):
        f = output_path / name
        if f.exists():
            sizes[Path(name).name] = round(f.stat().st_size / 1e6, 2)
    data["size_mb"] = sizes
    path.write_text(json.dumps(data, indent=2))
    return path
