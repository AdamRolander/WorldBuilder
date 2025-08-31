import blenderproc as bproc
import os
import shutil
import numpy as np
import sys
import json
import mathutils
import random
import glob
import bpy

# --- CONFIGURATION ---
BASE_ASSET_DIR = "PATH"
ASSET_TAGS_PATH = os.path.join(BASE_ASSET_DIR, "asset_tags.json")

# Placement & Camera Parameters
INNER_RADIUS = 2.0
OUTER_RADIUS_MIN = 5.5
OUTER_RADIUS_MAX = 7.5

CAMERA_LOCATIONS = 5
ZOOM_LEVELS = [35, 50, 85]
TOTAL_VIEWS = CAMERA_LOCATIONS * len(ZOOM_LEVELS)

# --- HELPER FUNCTIONS ---

def load_glb_objects(filepath):
    """Loads a GLB file and returns the new objects as a list of bproc MeshObjects."""
    print(f"Loading GLB from: {filepath}")
    
    bpy.ops.object.select_all(action='DESELECT')
    objects_before = set(bpy.context.scene.objects)
    
    try:
        bpy.ops.import_scene.gltf(filepath=filepath, loglevel=50)
        
        objects_after = set(bpy.context.scene.objects)
        new_objects = list(objects_after - objects_before)
        
        bproc_objects = []
        for blender_obj in new_objects:
            if blender_obj.type == 'MESH':
                try:
                    bproc_obj = bproc.types.MeshObject(blender_obj)
                    bproc_obj.set_cp("category_id", 0)
                    bproc_objects.append(bproc_obj)
                except Exception as e:
                    print(f"Failed to wrap {blender_obj.name}: {e}")
                    bpy.data.objects.remove(blender_obj, do_unlink=True)
            else:
                bpy.data.objects.remove(blender_obj, do_unlink=True)
        
        return bproc_objects
        
    except Exception as e:
        print(f"Error loading GLB {filepath}: {e}")
        return []

def compute_bounding_box_from_mask(mask):
    """Computes a 2D bounding box from a binary mask."""
    if np.sum(mask) == 0: return None, None, None, None, 0, 0, 0, 0
    y_coords, x_coords = np.where(mask > 0)
    if len(x_coords) == 0: return None, None, None, None, 0, 0, 0, 0
    min_x, max_x = int(np.min(x_coords)), int(np.max(x_coords))
    min_y, max_y = int(np.min(y_coords)), int(np.max(y_coords))
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    return min_x, min_y, max_x, max_y, width, height, center_x, center_y

def compute_apparent_size_metrics(mask, bbox_data):
    """Computes size metrics from a mask and its bounding box."""
    pixel_count = int(np.sum(mask))
    if bbox_data[0] is None: return {"pixel_count": pixel_count, "bbox_width": 0, "bbox_height": 0, "bbox_area": 0}
    _, _, _, _, width, height, center_x, center_y = bbox_data
    return {
        "pixel_count": pixel_count, "bbox_width": int(width), "bbox_height": int(height),
        "bbox_area": int(width * height), "bbox_center": [float(center_x), float(center_y)],
    }

# --- MAIN SCRIPT ---
# CHANGED: Now accepts scene_id, output_dir, and theme_name from the command line
if len(sys.argv) >= 4:
    scene_id = int(sys.argv[1])
    output_dir = sys.argv[2]
    chosen_theme = sys.argv[3]
else:
    # Fallback for manual testing
    import time
    scene_id = int(time.time() * 1000) % 100000
    output_dir = f"output_manual_{scene_id:05d}/"
    with open(ASSET_TAGS_PATH, 'r') as f:
        asset_tags = json.load(f)
    chosen_theme = random.choice(list(asset_tags.keys()))

print(f"Generating scene {scene_id:04d} (Theme: {chosen_theme}) -> {output_dir}")

bproc.init()
random.seed(scene_id)
np.random.seed(scene_id)

# --- 1. SCENE INITIALIZATION ---
THEMED_ASSET_DIR = os.path.join(BASE_ASSET_DIR, chosen_theme)

light = bproc.types.Light()
light.set_type("POINT")
light.set_energy(5000)
light.set_location([0, 0, 8])

ground = bproc.object.create_primitive("PLANE", scale=[25, 25, 1], location=[0, 0, -0.05])
ground.set_cp("category_id", 0)

# --- 2. PLACE ALL ASSETS FROM THEME ---
placed_bproc_objects = []
placed_object_info = []
category_id_counter = 1

all_asset_paths = glob.glob(os.path.join(THEMED_ASSET_DIR, "*.glb"))
if not all_asset_paths:
    print(f"Error: No .glb assets found in {THEMED_ASSET_DIR}. Exiting.")
    sys.exit(1)

# CHANGED: Randomly select a subset of 3 to 6 assets
if len(all_asset_paths) < 3:
    print(f"Warning: Not enough assets in '{chosen_theme}' to meet the minimum of 3. Using all {len(all_asset_paths)} assets.")
    selected_asset_paths = all_asset_paths
else:
    num_to_place = random.randint(3, min(7, len(all_asset_paths)))
    selected_asset_paths = random.sample(all_asset_paths, num_to_place)

print(f"Found {len(all_asset_paths)} total assets. Randomly selected {len(selected_asset_paths)} for this scene.")

num_assets = len(selected_asset_paths)
grid_size = int(np.ceil(np.sqrt(num_assets)))
spacing = 3.0 # Increased spacing for fewer objects

for i, asset_path in enumerate(selected_asset_paths): # Iterate over the selected subset
    row, col = i // grid_size, i % grid_size
    x = (col - (grid_size - 1) / 2.0) * spacing + random.uniform(-0.2, 0.2)
    y = (row - (grid_size - 1) / 2.0) * spacing + random.uniform(-0.2, 0.2)
    location = [x, y, 0.2]
    
    loaded_objs = load_glb_objects(asset_path)
    if not loaded_objs: continue

    try:
        if len(loaded_objs) > 1:
            bpy.context.view_layer.update()
            bbox = bproc.object.get_comprehending_bound_box(loaded_objs)
            min_coords, max_coords = np.min(np.array([c for c in bbox]), axis=0), np.max(np.array([c for c in bbox]), axis=0)
            dims = max_coords - min_coords
        else:
            dims = loaded_objs[0].blender_obj.dimensions
        object_dimensions = {"width": float(dims[0]), "depth": float(dims[1]), "height": float(dims[2])}
    except Exception as e:
        print(f"Warning: Could not calculate dimensions for {asset_path}. Error: {e}")
        object_dimensions = {"width": 0, "depth": 0, "height": 0}

    for obj in loaded_objs:
        obj.set_location(location)
        obj.set_rotation_euler([0, 0, random.uniform(0, 2 * np.pi)])
        obj.set_cp("category_id", category_id_counter)
        
    placed_bproc_objects.extend(loaded_objs)
    placed_object_info.append({
        "asset_path": os.path.basename(asset_path), "category_id": category_id_counter,
        "type": "placed_asset", "dimensions_xyz": object_dimensions
    })
    category_id_counter += 1

# --- 3. PHYSICS SIMULATION ---
print("Running physics simulation...")
ground.enable_rigidbody(active=False, collision_shape='BOX')
for obj in placed_bproc_objects:
    obj.enable_rigidbody(active=True, collision_shape='CONVEX_HULL', mass=random.uniform(2, 5))
    obj.set_cp("physics_restitution", 0.1); obj.set_cp("physics_friction", 0.8)
    obj.set_cp("physics_damping_linear", 0.9); obj.set_cp("physics_damping_angular", 0.9)
    if not obj.get_materials():
        mat = bproc.material.create('basic_material')
        mat.set_principled_shader_value("Base Color", [random.uniform(0.3, 0.9) for _ in range(3)] + [1.0])
        obj.add_material(mat)
bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=1.0, max_simulation_time=4.0, check_object_interval=0.5)

# --- 4. GATHER FINAL OBJECT DATA & RENDER ---
for info in placed_object_info:
    rep_obj = next((obj for obj in placed_bproc_objects if obj.get_cp("category_id") == info["category_id"]), None)
    if rep_obj:
        info["final_location"] = rep_obj.get_location().tolist()
        info["final_rotation"] = rep_obj.get_rotation_euler().tolist()

image_width, image_height = 640, 480
bproc.camera.set_resolution(image_width, image_height)
sensor_width_mm = 36
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.enable_segmentation_output(map_by=["category_id"], default_values={"category_id": 0})

final_metadata = {"scene_id": scene_id, "theme": chosen_theme, "scene_objects": placed_object_info, "frames": []}
frame_id_counter = 0
sorted_zoom_levels = sorted(ZOOM_LEVELS)
final_positions = np.array([obj.get_location()[:2] for obj in placed_bproc_objects])
scene_center = np.mean(final_positions, axis=0) if len(final_positions) > 0 else np.array([0, 0])
scene_bounds = np.max(np.abs(final_positions - scene_center)) if len(final_positions) > 0 else 1.0

for location_idx in range(CAMERA_LOCATIONS):
    cam_radius = random.uniform(OUTER_RADIUS_MIN, OUTER_RADIUS_MAX)
    if location_idx < 4:
        azimuth_angles = [90, 0, 270, 180]
        base_azimuth = azimuth_angles[location_idx]
        cam_azimuth = base_azimuth + random.uniform(-20, 20)
        cam_elevation = random.uniform(25, 55)
        cam_elevation_rad, cam_azimuth_rad = np.radians(cam_elevation), np.radians(cam_azimuth)
        cam_location = np.array([
            scene_center[0] + cam_radius * np.cos(cam_elevation_rad) * np.cos(cam_azimuth_rad),
            scene_center[1] + cam_radius * np.cos(cam_elevation_rad) * np.sin(cam_azimuth_rad),
            1.0 + cam_radius * np.sin(cam_elevation_rad)
        ])
        poi = np.array([
            scene_center[0] + random.uniform(-scene_bounds*0.2, scene_bounds*0.2),
            scene_center[1] + random.uniform(-scene_bounds*0.2, scene_bounds*0.2),
            random.uniform(0.8, 1.2)
        ])
    else:
        cam_location = np.array([scene_center[0] + random.uniform(-1.0, 1.0), scene_center[1] + random.uniform(-1.0, 1.0), cam_radius])
        poi = np.array([scene_center[0], scene_center[1], 0.5])

    for focal_length_mm in sorted_zoom_levels:
        print(f"\n=== RENDERING VIEW {frame_id_counter + 1}/{TOTAL_VIEWS} (Loc: {location_idx}, Frame: {frame_id_counter}, Zoom: {focal_length_mm}mm) ===")
        bproc.utility.reset_keyframes()
        bproc.camera.set_intrinsics_from_blender_params(lens=focal_length_mm, lens_unit="MILLIMETERS")
        fx = (focal_length_mm * image_width) / sensor_width_mm
        fy = fx; cx = image_width / 2; cy = image_height / 2
        cam_intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        forward = poi - cam_location; forward /= np.linalg.norm(forward)
        world_up = np.array([0, 0, 1])
        right = np.cross(forward, world_up); right /= np.linalg.norm(right)
        up = np.cross(right, forward); up /= np.linalg.norm(up)
        rotation_matrix = np.column_stack([right, up, -forward])
        cam_pose_matrix = np.eye(4); cam_pose_matrix[:3, :3] = rotation_matrix; cam_pose_matrix[:3, 3] = cam_location
        bproc.camera.add_camera_pose(cam_pose_matrix)
        data = bproc.renderer.render()
        if not data or "category_id_segmaps" not in data: continue
        seg_data = data["category_id_segmaps"][0]
        frame_dynamic_data = {}
        for info in placed_object_info:
            obj_id = info["category_id"]
            
            # Find the corresponding bproc object to get its final location after physics
            rep_obj = next((obj for obj in placed_bproc_objects if obj.get_cp("category_id") == obj_id), None)
            if not rep_obj: continue

            # --- THIS IS THE FIX ---
            # Calculate the true z_depth for this specific camera view
            world_pos_h = np.append(rep_obj.get_location(), 1)
            cam_pos = np.linalg.inv(cam_pose_matrix) @ world_pos_h
            true_z_depth = -cam_pos[2]
            # --- END OF FIX ---
            
            mask = (seg_data == obj_id).astype(np.uint8)
            bbox_data = compute_bounding_box_from_mask(mask)
            size_metrics = compute_apparent_size_metrics(mask, bbox_data)

            # Use the calculated true_z_depth instead of 0
            frame_dynamic_data[f"object_{obj_id}"] = {
                "z_depth": float(true_z_depth), 
                **size_metrics
            }
        final_metadata["frames"].append({
            "frame_id": frame_id_counter, "camera_pose": cam_pose_matrix.tolist(), "camera_intrinsics": cam_intrinsics.tolist(),
            "focal_length_mm": focal_length_mm, "objects_in_frame": frame_dynamic_data
        })
        frame_output_dir = os.path.join(output_dir, f"frame_{frame_id_counter:04d}")
        os.makedirs(frame_output_dir, exist_ok=True)
        bproc.writer.write_hdf5(frame_output_dir, data)
        frame_id_counter += 1
    
metadata_path = os.path.join(output_dir, "metadata.json")
with open(metadata_path, 'w') as f: json.dump(final_metadata, indent=2, fp=f)
print(f"Comprehensive metadata saved to {metadata_path}")
bproc.clean_up()
print(f"Scene {scene_id:04d} with {TOTAL_VIEWS} views generation complete!")