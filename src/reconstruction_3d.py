import sys
from pathlib import Path
import numpy as np
from PIL import Image
from typing import List, Dict
import json
import open3d as o3d

# Add SAM 3D Objects to path
sam3d_repo = Path(__file__).parent.parent / "sam3d_objects_repo"
sys.path.insert(0, str(sam3d_repo / "notebook"))

from inference import Inference, load_image, make_scene

def save_glb_as_ply(output: dict, output_path: Path, target_triangles: int = 100000):
    """Save the GLB mesh from SAM 3D output directly as PLY."""
    import trimesh
    
    glb_mesh = output['glb']
    
    print(f"    GLB mesh: {len(glb_mesh.vertices):,} vertices, {len(glb_mesh.faces):,} faces")
    print(f"    Visual kind: {glb_mesh.visual.kind}, has colors: {glb_mesh.visual.vertex_colors is not None if glb_mesh.visual.kind == 'vertex' else False}")
    
    # Export as PLY with vertex colors (skip simplification to preserve colors)
    glb_mesh.export(str(output_path), file_type='ply')
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"    ✓ Saved: {output_path.name} ({file_size_mb:.2f} MB)")
    
    return glb_mesh

def save_colored_mesh(output: dict, output_path: Path, format: str = 'obj', quality: str = 'high'):
    """
    Save mesh with colors from SAM 3D Objects output.
    """
    # Quality presets
    presets = {
        'medium': {
            'poisson_depth': 9,
            'density_threshold': 0.1,
            'normal_radius': 0.05,
            'normal_nn': 30,
            'target_triangles': 50000
        },
        'high': {
            'poisson_depth': 10,
            'density_threshold': 0.05,
            'normal_radius': 0.03,
            'normal_nn': 50,
            'target_triangles': 100000
        }
    }
    
    settings = presets.get(quality, presets['high'])
    
    # Extract Gaussian splat data
    gs = output['gs']
    positions = gs.get_xyz.cpu().numpy()
    colors = gs.get_features.cpu().numpy().squeeze()
    colors = np.clip(colors, 0, 1)
    
    print(f"    Creating mesh from {len(positions):,} points...")
    
    # Create point cloud with colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Remove outliers
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"    After outlier removal: {len(pcd.points):,} points")
    
    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=settings['normal_radius'],
            max_nn=settings['normal_nn']
        )
    )
    pcd.orient_normals_consistent_tangent_plane(30)
    
    # Poisson surface reconstruction
    print(f"    Poisson reconstruction (depth={settings['poisson_depth']})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=settings['poisson_depth'],
        width=0,
        scale=1.1,
        linear_fit=False
    )
    
    print(f"    Initial mesh: {len(mesh.triangles):,} triangles")
    
    # Remove low-density vertices
    densities = np.asarray(densities)
    vertices_to_remove = densities < np.quantile(densities, settings['density_threshold'])
    mesh.remove_vertices_by_mask(vertices_to_remove)
    print(f"    After cleanup: {len(mesh.triangles):,} triangles")
    
    # Fill holes
    if hasattr(mesh, 'fill_holes'):
        mesh.fill_holes()
    
    # Transfer colors from point cloud to mesh
    print(f"    Transferring colors...")
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    mesh_vertices = np.asarray(mesh.vertices)
    pcd_colors = np.asarray(pcd.colors)
    
    vertex_colors = np.zeros((len(mesh_vertices), 3))
    for i, vertex in enumerate(mesh_vertices):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(vertex, 1)
        vertex_colors[i] = pcd_colors[idx[0]]
    
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    
    # Simplify if needed
    if len(mesh.triangles) > settings['target_triangles']:
        print(f"    Simplifying to ~{settings['target_triangles']:,} triangles...")
        mesh = mesh.simplify_quadric_decimation(settings['target_triangles'])
        print(f"    Final mesh: {len(mesh.triangles):,} triangles")
    
    # Smooth
    mesh = mesh.filter_smooth_simple(number_of_iterations=2)
    mesh.compute_vertex_normals()
    
    # Cleanup
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    
    # Save
    if format == 'ply':
        o3d.io.write_triangle_mesh(str(output_path), mesh, write_vertex_colors=True)
    elif format == 'obj':
        o3d.io.write_triangle_mesh(str(output_path), mesh, write_vertex_colors=True)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"    ✓ Saved: {output_path.name} ({file_size_mb:.2f} MB)")
    
    return mesh


class SAM3DReconstructor:
    """3D reconstruction using SAM 3D Objects."""

    # Process-wide singleton inference handle. Loading the model once and
    # reusing it across pipeline runs is the only way to keep the long-lived
    # Flask worker from OOMing on the second image (the old per-call
    # __init__ left a copy of the model in VRAM each time).
    _shared_inference = None
    _shared_config_path = None

    def __init__(
        self,
        sam3d_repo_path: str = None,
        checkpoint_tag: str = "hf"
    ):
        """Initialize (or reuse) SAM 3D Objects model."""
        if sam3d_repo_path is None:
            sam3d_repo_path = Path(__file__).parent.parent / "sam3d_objects_repo"

        self.sam3d_path = Path(sam3d_repo_path)
        config_path = str(self.sam3d_path / "checkpoints" / checkpoint_tag / "pipeline.yaml")

        if (SAM3DReconstructor._shared_inference is not None
                and SAM3DReconstructor._shared_config_path == config_path):
            self.inference = SAM3DReconstructor._shared_inference
            print("\n✓ SAM 3D Objects (reusing cached model)")
        else:
            print(f"\nLoading SAM 3D Objects...")
            self.inference = Inference(config_path, compile=False)
            SAM3DReconstructor._shared_inference = self.inference
            SAM3DReconstructor._shared_config_path = config_path
            print("✓ SAM 3D Objects loaded")

    def _to_device(self, device: str):
        """Move the SAM 3D inference pipeline between cuda and cpu.

        Mirrors what vlm_server does for Qwen — needed so all three big
        models (Qwen, SAM 3, SAM 3D) don't try to coexist on the GPU at
        once. Inference's internal modules vary by checkpoint, so we walk
        any attribute that looks torch-like.
        """
        import gc, torch
        for name in dir(self.inference):
            if name.startswith('_'):
                continue
            try:
                obj = getattr(self.inference, name)
            except Exception:
                continue
            if isinstance(obj, torch.nn.Module):
                obj.to(device)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # def _save_scene_with_room(self, results: List[Dict],
    #                           room_mesh: 'trimesh.Trimesh',
    #                           output_path: Path):
    #     """Same as _save_combined_scene, but appends a pre-transformed room mesh."""
    #     import trimesh
    #     import torch
    #     from pytorch3d.transforms import quaternion_to_matrix
    #     from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform

    #     meshes = []
    #     for result in results:
    #         ply_path = result.get('ply_path')
    #         if not ply_path or not Path(ply_path).exists():
    #             continue
    #         mesh = trimesh.load(ply_path)
    #         v = mesh.vertices.copy()
    #         old_y, old_z = v[:, 1].copy(), v[:, 2].copy()
    #         v[:, 1] = -old_z
    #         v[:, 2] = old_y

    #         vt = torch.tensor(v, dtype=torch.float32, device='cuda').unsqueeze(0)
    #         quat  = torch.tensor([result['rotation_quaternion']], dtype=torch.float32, device='cuda')
    #         trans = torch.tensor([result['translation']],         dtype=torch.float32, device='cuda')
    #         scale = torch.tensor([result['scale']],               dtype=torch.float32, device='cuda')
    #         T = compose_transform(scale=scale,
    #                               rotation=quaternion_to_matrix(quat),
    #                               translation=trans)
    #         tv = T.transform_points(vt)[0].cpu().numpy()

    #         meshes.append(trimesh.Trimesh(
    #             vertices=tv, faces=mesh.faces,
    #             vertex_colors=mesh.visual.vertex_colors if mesh.visual.kind == 'vertex' else None,
    #             process=False))
    #     meshes.append(room_mesh)
    #     combined = trimesh.util.concatenate(meshes)
    #     combined.export(str(output_path), file_type='ply')
    #     mb = output_path.stat().st_size / (1024 * 1024)
    #     print(f"    Scene+room: {len(combined.faces):,} triangles ({mb:.2f} MB)")

    def _bake_world_space_plys(self, results: List[Dict]):
        """Overwrite each object's PLY with its transform baked into vertex positions.

        Same math as _save_combined_scene, applied per-object. After this, the
        individual PLY files on disk are in world space — the viewer can load
        them with no further math.
        """
        import trimesh
        import torch
        from pytorch3d.transforms import quaternion_to_matrix
        from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform

        print("\nBaking per-object PLYs to world space...")
        baked = 0
        for result in results:
            ply_path = result.get('ply_path')
            if not ply_path or not Path(ply_path).exists():
                continue

            mesh = trimesh.load(ply_path)
            v = mesh.vertices.copy()
            old_y = v[:, 1].copy()
            old_z = v[:, 2].copy()
            v[:, 1] = -old_z
            v[:, 2] = old_y

            vt = torch.tensor(v, dtype=torch.float32, device='cuda').unsqueeze(0)
            quat  = torch.tensor([result['rotation_quaternion']], dtype=torch.float32, device='cuda')
            trans = torch.tensor([result['translation']],         dtype=torch.float32, device='cuda')
            scale = torch.tensor([result['scale']],               dtype=torch.float32, device='cuda')
            T = compose_transform(scale=scale,
                                  rotation=quaternion_to_matrix(quat),
                                  translation=trans)
            world_v = T.transform_points(vt)[0].cpu().numpy()

            world_mesh = trimesh.Trimesh(
                vertices=world_v, faces=mesh.faces,
                vertex_colors=mesh.visual.vertex_colors if mesh.visual.kind == 'vertex' else None,
                process=False,
            )
            world_mesh.export(str(ply_path), file_type='ply')
            baked += 1
        print(f"✓ Baked {baked}/{len(results)} PLYs to world space")
        
    def _save_combined_scene(self, results: List[Dict], output_path: Path):
        """Combine individual object PLYs into a single scene file using SAM 3D's transform."""
        import trimesh
        import torch
        from pytorch3d.transforms import quaternion_to_matrix
        from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform
        
        meshes = []
        
        for result in results:
            ply_path = result.get('ply_path')
            if not ply_path or not Path(ply_path).exists():
                continue
                
            mesh = trimesh.load(ply_path)
            
            # GLB mesh has Y and Z swapped relative to Gaussian space
            # Convert: new_y = -old_z, new_z = old_y
            swapped_verts = mesh.vertices.copy()
            old_y = swapped_verts[:, 1].copy()
            old_z = swapped_verts[:, 2].copy()
            swapped_verts[:, 1] = -old_z
            swapped_verts[:, 2] = old_y
            
            # Apply pose transform using PyTorch3D (matches make_scene behavior)
            verts_t = torch.tensor(swapped_verts, dtype=torch.float32, device='cuda').unsqueeze(0)
            
            quat = torch.tensor([result['rotation_quaternion']], dtype=torch.float32, device='cuda')
            trans = torch.tensor([result['translation']], dtype=torch.float32, device='cuda')
            scale = torch.tensor([result['scale']], dtype=torch.float32, device='cuda')
            
            R_l2c = quaternion_to_matrix(quat)
            l2c_transform = compose_transform(scale=scale, rotation=R_l2c, translation=trans)
            transformed_verts = l2c_transform.transform_points(verts_t)[0].cpu().numpy()
            
            new_mesh = trimesh.Trimesh(
                vertices=transformed_verts,
                faces=mesh.faces,
                vertex_colors=mesh.visual.vertex_colors if mesh.visual.kind == 'vertex' else None,
                process=False
            )
            meshes.append(new_mesh)
        
        if meshes:
            combined = trimesh.util.concatenate(meshes)
            combined.export(str(output_path), file_type='ply')
            
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"    Combined scene: {len(combined.faces):,} triangles ({file_size_mb:.2f} MB)")
    
    def reconstruct_objects(
        self,
        image_path: str,
        segmentation_results: List[Dict],
        output_dir: str = "outputs/3d_models",
        quality: str = 'high'
    ) -> List[Dict]:
        """Generate 3D reconstructions with pose data."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nLoading image: {image_path}")
        image = load_image(str(image_path))
        
        results = []
        
        print(f"\nReconstructing {len(segmentation_results)} objects...")
        print("="*60)
        
        for i, seg in enumerate(segmentation_results, 1):
            obj_id = seg['id']
            label = seg['label']
            
            print(f"\n[{i}/{len(segmentation_results)}] {label}")
            
            try:
                # Load mask
                mask_path = seg['mask_path']
                mask_pil = Image.open(mask_path).convert('L')
                mask = np.array(mask_pil) > 128
                
                if mask.sum() < 100:
                    print(f"  ⚠️  Mask too small")
                    continue
                
                # Run inference
                print(f"  Running inference...")
                output = self.inference(image, mask, seed=42)
                
                # Extract pose
                translation = output['translation'].cpu().numpy()[0]
                rotation = output['rotation'].cpu().numpy()[0]
                scale = output['scale'].cpu().numpy()[0]
                
                print(f"  ✓ Pose extracted")
                print(f"    Translation: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]")
                print(f"    Rotation: [{rotation[0]:.3f}, {rotation[1]:.3f}, {rotation[2]:.3f}, {rotation[3]:.3f}]")
                print(f"    Scale: [{scale[0]:.3f}, {scale[1]:.3f}, {scale[2]:.3f}]")
                
                # Save colored meshes
                safe_label = label.replace(' ', '_').replace('/', '_')
                ply_path = output_path / f"{obj_id:03d}_{safe_label}.ply"
                save_glb_as_ply(output, ply_path)
                
                results.append({
                    'id': obj_id,
                    'label': label,
                    'ply_path': str(ply_path),
                    'translation': translation.tolist(),
                    'rotation_quaternion': rotation.tolist(),
                    'scale': scale.tolist(),
                    'bbox': seg.get('bbox', []),
                    'confidence': seg.get('confidence', 0)
                })
                
            except Exception as e:
                print(f"  ❌ Failed: {str(e)[:150]}")
                continue
        
        print("\n" + "="*60)
        print(f"✓ Reconstructed {len(results)}/{len(segmentation_results)} objects")
        
        # Post-process: snap to floor, save scene + standalone room
        # Post-process: upright → snap → resolve collisions → re-snap.
        if len(results) > 1:
            from src.room_generator import (
                compute_per_object_aabb, compute_scene_bounds,
                snap_ground_objects, upright_correct,
                resolve_xz_collisions, make_box_room,
                stretch_vertical_structural
            )
            try:
                print("\nComputing scene bounds...")
                aabbs = compute_per_object_aabb(results)
                bounds = compute_scene_bounds(aabbs)

                if bounds is not None:
                    print(f"  size: {[round(s, 3) for s in bounds['size']]}")
                    # 1. Upright on EVERYTHING — the 45° threshold still
                    #    protects genuinely sideways poses.
                    corrected = upright_correct(results, only_snapped=False)
                    print(f"  upright-corrected {corrected} objects")

                    # 2. Stretch tall structural elements (pillars, beams) to
                    #    floor-to-ceiling height before snap. SAM 3D returns
                    #    these too small for single-image-3D reasons.
                    # aabbs = compute_per_object_aabb(results)
                    # bounds = compute_scene_bounds(aabbs)
                    # stretched = stretch_vertical_structural(results, aabbs, bounds)
                    # if stretched:
                    #     print(f"  vertical-stretched {stretched} structural element(s)")

                    # 3. Snap (with class-based force-snap for floor furniture)
                    aabbs = compute_per_object_aabb(results)
                    bounds = compute_scene_bounds(aabbs)
                    snapped = snap_ground_objects(results, aabbs, bounds)
                    forced = sum(1 for r in results if r.get('snap_forced'))
                    print(f"  snapped {snapped}/{len(results)} to floor "
                          f"({forced} via class-based force-snap)")

                    # 3. Push overlapping AABBs apart in XZ
                    aabbs = compute_per_object_aabb(results)
                    pushes = resolve_xz_collisions(results, aabbs)
                    print(f"  collision resolution: {pushes} pair-pushes")

                    # 4. Re-snap (collision pushes don't change Y but AABBs
                    #    have shifted, so refresh state)
                    aabbs = compute_per_object_aabb(results)
                    bounds = compute_scene_bounds(aabbs)
                    snap_ground_objects(results, aabbs, bounds)
                    aabbs = compute_per_object_aabb(results)
                    bounds = compute_scene_bounds(aabbs)

                scene_path = output_path / "scene_combined.ply"
                self._save_combined_scene(results, scene_path)
                print(f"✓ scene_combined.ply saved (full export)")

                if bounds is not None:
                    room = make_box_room(bounds)
                    room_path = output_path / "room.ply"
                    room.export(str(room_path), file_type='ply')
                    kb = room_path.stat().st_size / 1024
                    print(f"✓ room.ply saved ({kb:.1f} KB)")
            except Exception as e:
                print(f"⚠️  Post-processing failed: {e}")

        # Bake per-object transforms so the viewer never has to. Runs even for
        # a single-object scene. Must happen AFTER _save_combined_scene and
        # AABB/room generation (those consume model-space PLYs + poses).
        if results:
            self._bake_world_space_plys(results)

        return results