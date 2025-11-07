import blenderproc as bproc
import numpy as np
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Union
import random

def print_coordinate_analysis(pose_results):
    """Print detailed coordinate analysis to help debug positioning."""
    print("\n" + "="*70)
    print("COORDINATE ANALYSIS")
    print("="*70)
    
    for obj_data in pose_results:
        label = obj_data['label']
        model_coords = obj_data['position_m']
        blender_coords = convert_to_blender_coords(*model_coords)
        dims = obj_data.get('dimensions_m', [0.5, 0.5, 0.5])
        rotation_y = obj_data.get('rotation_y_deg', 0.0)
        
        print(f"\n{label} (ID: {obj_data['id']}):")
        print(f"  Model coords (X,Y,Z):    {model_coords}")
        print(f"  Blender coords (X,Y,Z):  {blender_coords}")
        print(f"  Rotation Y: {rotation_y:.1f}°")
        print(f"  Dimensions (L,W,H):      {dims}")
        print(f"  Is Reference: {obj_data.get('is_reference', False)}")
    
    print("\n" + "="*70 + "\n")

def load_results(results_input: Union[str, List[Dict], Path]) -> List[Dict]:
    """
    Load 3D pose results from JSON file or accept list directly.
    
    Args:
        results_input: Path to JSON file or list of result dictionaries
        
    Returns:
        List of result dictionaries
    """
    if isinstance(results_input, (str, Path)):
        with open(results_input, 'r') as f:
            return json.load(f)
    elif isinstance(results_input, list):
        return results_input
    else:
        raise ValueError("results_input must be a JSON file path or list of dictionaries")


def convert_to_blender_coords(x: float, y: float, z: float) -> tuple:
    """
    Convert from model's coordinate system to Blender's Z-up system.
    
    Testing different mappings:
    Option 1 (current): (X,Y,Z) → (X,Y,Z) - direct
    Option 2: (X,Y,Z) → (X,Z,-Y) - Y↔Z swap, negate Y
    Option 3: (X,Y,Z) → (X,-Z,Y) - Y↔Z swap, negate Z
    """
    
    # Try Option 2 first (common for Y-up to Z-up conversion)
    return (x, -z, y)
    
    # If that doesn't work, try Option 3:
    # return (x, z, -y)
    
    # Or go back to direct:
    # return (x, y, z)

def add_text_label(text: str, location: tuple, size: float = 0.1):
    """Add a text label in the 3D scene."""
    import bpy
    
    bpy.ops.object.text_add(location=location)
    text_obj = bpy.context.active_object
    text_obj.data.body = text
    text_obj.data.size = size
    text_obj.data.align_x = 'CENTER'
    text_obj.data.align_y = 'CENTER'
    
    # Make text face camera (billboard effect)
    constraint = text_obj.constraints.new('TRACK_TO')
    constraint.target = bpy.context.scene.camera
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'
    
    return text_obj

def generate_color_palette(n_objects: int, seed: int = 42) -> List[tuple]:
    """
    Generate distinct random colors for objects.
    
    Args:
        n_objects: Number of unique colors needed
        seed: Random seed for reproducibility
        
    Returns:
        List of (R, G, B, A) color tuples
    """
    random.seed(seed)
    np.random.seed(seed)
    
    colors = []
    for i in range(n_objects):
        # Generate vibrant colors (avoid too dark or too light)
        hue = i / n_objects
        saturation = 0.7 + np.random.rand() * 0.3
        value = 0.6 + np.random.rand() * 0.4
        
        # Convert HSV to RGB
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append((r, g, b, 1.0))
    
    # Shuffle to avoid similar colors being adjacent in the list
    random.shuffle(colors)
    return colors


def create_box_mesh(dimensions: List[float], location: tuple, color: tuple, label: str, rotation_y_deg: float = 0.0) -> bproc.types.MeshObject:
    """
    Create a rectangular prism (box) mesh at specified location with rotation.
    
    Args:
        dimensions: [length, width, height] in meters
        location: (x, y, z) in Blender coordinates
        color: (R, G, B, A) color tuple
        label: Object label for naming
        rotation_y_deg: Rotation around Y-axis in model coordinates (degrees)
        
    Returns:
        BlenderProc MeshObject
    """
    # Ensure dimensions are positive and reasonable
    dimensions = [max(0.1, d) for d in dimensions]
    
    # Create a primitive cube and scale it to match dimensions
    box = bproc.object.create_primitive('CUBE', scale=[d/2 for d in dimensions])
    box.set_location(location)
    box.set_name(f"{label}")
    
    # Apply rotation
    # Model's Y-rotation becomes Z-rotation in Blender (since we map Y→Z, Z→-Y)
    # The Y-axis rotation in model space rotates around the vertical axis
    rotation_z_rad = np.deg2rad(rotation_y_deg)
    box.set_rotation_euler([0, 0, rotation_z_rad])
    
    # Create and assign material with color
    mat = bproc.material.create(f"mat_{label}")
    mat.set_principled_shader_value("Base Color", color[:3] + (1.0,))
    mat.set_principled_shader_value("Roughness", 0.5)
    mat.set_principled_shader_value("Metallic", 0.1)
    box.add_material(mat)
    
    return box

def setup_camera(location: tuple = (0, 0, 0), 
                 rotation_euler: tuple = (0, 0, 0),
                 fov: float = 90.0,
                 resolution: tuple = (512, 512)) -> None:
    """
    Setup camera with specified pose and parameters.
    
    Args:
        location: Camera (x, y, z) position in Blender coordinates
        rotation_euler: Camera rotation in (x, y, z) Euler angles (radians)
        fov: Field of view in degrees
        resolution: (width, height) in pixels
    """
    import bpy
    
    # Convert location to Blender coordinates if needed
    cam_location = location
    
    # Set camera pose
    cam_pose = bproc.math.build_transformation_mat(cam_location, rotation_euler)
    bproc.camera.add_camera_pose(cam_pose)
    
    # Set camera parameters
    bproc.camera.set_resolution(resolution[0], resolution[1])
    
    # Set FOV using bpy directly
    fov_rad = np.deg2rad(fov)
    
    # Access camera through bpy
    cam = bpy.context.scene.camera
    if cam is not None:
        cam.data.angle = fov_rad
        cam.data.lens_unit = 'FOV'
        cam.data.clip_start = 0.1
        cam.data.clip_end = 1000

def setup_lighting() -> None:
    """Setup basic lighting for the scene."""
    # Add a sun light for general illumination
    light = bproc.types.Light()
    light.set_type("SUN")
    light.set_location([5, -5, 10])
    light.set_energy(1.5)
    
    # Add point lights for better visibility
    for loc in [[3, 3, 5], [-3, 3, 5], [0, -3, 5]]:
        point_light = bproc.types.Light()
        point_light.set_type("POINT")
        point_light.set_location(loc)
        point_light.set_energy(200)


def visualize_scene(results: Union[str, List[Dict], Path],
                   output_dir: str = "output",
                   image_name: str = "reconstruction",
                   resolution: tuple = (512, 512),
                   fov: float = 90.0,
                   camera_location: tuple = (0, 0, 0),
                   save_blend: bool = True) -> str:
    """
    Main visualization function. Creates BlenderProc scene and renders.
    
    Args:
        results: Path to JSON results file or list of result dictionaries
        output_dir: Directory to save outputs
        image_name: Base name for output files
        resolution: Image resolution (width, height)
        fov: Camera field of view in degrees
        camera_location: Camera position in model coordinates (X, Y, Z)
        save_blend: Whether to save .blend file
        
    Returns:
        Path to output HDF5 file
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("BlenderProc 3D Scene Reconstruction Visualization")
    print(f"{'='*60}")
    
    # Load results
    print("\n1. Loading reconstruction results...")
    pose_results = load_results(results)
    print(f"   ✓ Loaded {len(pose_results)} objects")

    print_coordinate_analysis(pose_results)
    
    # Initialize BlenderProc
    print("\n2. Initializing BlenderProc...")
    bproc.init()
    
    # Generate color palette
    print("\n3. Generating color palette...")
    colors = generate_color_palette(len(pose_results))
    
    # Create objects
    print("\n4. Creating 3D objects...")

    print("\n   COLOR LEGEND:")
    print("   " + "="*50)

    created_objects = []

    dimension_scale_factor = 1.0  # Scale factor for dimensions if needed
    
    for idx, obj_data in enumerate(pose_results):
        label = obj_data['label']
        obj_id = obj_data['id']
        color = colors[idx]
        
        # Print color info
        color_str = f"RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})"
        ref_marker = " (REFERENCE)" if obj_data.get('is_reference', False) else ""
        print(f"   [{idx+1}] {label} (ID: {obj_id}){ref_marker}: {color_str}")
        
        # Get position in model coordinates
        model_x, model_y, model_z = obj_data['position_m']
        
        # Convert to Blender coordinates
        blender_x, blender_y, blender_z = convert_to_blender_coords(model_x, model_y, model_z)
        
        # Get dimensions (default to small box if not available)
        if 'dimensions_m' in obj_data:
            dimensions = [d * dimension_scale_factor for d in obj_data['dimensions_m']]
        else:
            dimensions = [0.5, 0.5, 0.5]
        
        # Get rotation (default to 0 if not available)
        rotation_y = obj_data.get('rotation_y_deg', 0.0)
        
        # Create box with rotation
        box = create_box_mesh(
            dimensions=dimensions,
            location=(blender_x, blender_y, blender_z),
            color=colors[idx],
            label=f"{label}_{obj_id}",
            rotation_y_deg=rotation_y
        )
        created_objects.append(box)
        
        # Add text label above object
        label_height = max(dimensions) * 0.6
        label_location = (blender_x, blender_y, blender_z + label_height)
        add_text_label(f"{label}\n({obj_id})", label_location, size=0.15)

    print(f"   " + "="*50)
    print(f"   ✓ Created {len(created_objects)} objects\n")
    
    # Setup lighting
    print("\n5. Setting up lighting...")
    setup_lighting()
    print("   ✓ Lighting configured")
    
    # Setup camera
    print("\n6. Setting up camera with auto-framing...")

    # Calculate scene bounding box including object dimensions
    all_positions = []
    all_extents = []  # Store min/max extents of each object

    for obj_data in pose_results:
        pos = convert_to_blender_coords(*obj_data['position_m'])
        all_positions.append(pos)
        
        # Get object dimensions to calculate actual extents
        if 'dimensions_m' in obj_data:
            dims = np.array(obj_data['dimensions_m']) * dimension_scale_factor
        else:
            dims = np.array([0.5, 0.5, 0.5])
        
        # Calculate bounding box corners
        half_dims = dims / 2
        min_extent = np.array(pos) - half_dims
        max_extent = np.array(pos) + half_dims
        all_extents.append((min_extent, max_extent))

    if all_positions:
        # Calculate overall scene bounds
        all_mins = np.array([extent[0] for extent in all_extents])
        all_maxs = np.array([extent[1] for extent in all_extents])
        
        scene_min = all_mins.min(axis=0)
        scene_max = all_maxs.max(axis=0)
        scene_center = (scene_min + scene_max) / 2
        scene_size = scene_max - scene_min
        max_dimension = np.max(scene_size)
        
        # Position camera to view the scene
        # Increase distance multiplier to avoid clipping
        camera_distance = max(5.0, max_dimension)  # Increased from 2.5 to 3.0

        cam_x = scene_center[0]
        cam_y = scene_center[1] - camera_distance  # Back from scene
        cam_z = scene_center[2] + camera_distance * 0.5  # Elevated view (increased from 0.4)
        
        print(f"   Scene bounds: min={scene_min}, max={scene_max}")
        print(f"   Scene center: ({scene_center[0]:.3f}, {scene_center[1]:.3f}, {scene_center[2]:.3f})")
        print(f"   Scene size: ({scene_size[0]:.3f}, {scene_size[1]:.3f}, {scene_size[2]:.3f})")
        print(f"   Max dimension: {max_dimension:.3f}m")
        print(f"   Camera distance: {camera_distance:.3f}m")
        print(f"   Camera position: ({cam_x:.3f}, {cam_y:.3f}, {cam_z:.3f})")
    else:
        # Fallback to default camera position
        cam_x, cam_y, cam_z = convert_to_blender_coords(*camera_location)
        scene_center = np.array([0, 0, 0])
        print(f"   Using default camera position: ({cam_x:.3f}, {cam_y:.3f}, {cam_z:.3f})")
        
    print(f"   FOV: {fov}°")
    print(f"   Resolution: {resolution[0]}x{resolution[1]}")
    
    # Calculate rotation to look at scene center
    import math
    look_at = scene_center if all_positions else np.array([0, 0, 0])
    cam_pos = np.array([cam_x, cam_y, cam_z])
    
    # Vector from camera to target
    direction = look_at - cam_pos
    direction_normalized = direction / np.linalg.norm(direction)
    
    # Calculate rotation to look at target
    # This creates a "look at" rotation
    rotation_matrix = bproc.camera.rotation_from_forward_vec(direction_normalized, inplane_rot=0)
    cam_pose = bproc.math.build_transformation_mat(cam_pos, rotation_matrix)
    
    bproc.camera.add_camera_pose(cam_pose)
    
    # Set camera parameters
    bproc.camera.set_resolution(resolution[0], resolution[1])
    
    # Set FOV
    import bpy
    fov_rad = np.deg2rad(fov)
    cam = bpy.context.scene.camera
    if cam is not None:
        cam.data.angle = fov_rad
        cam.data.lens_unit = 'FOV'
        cam.data.clip_start = 0.1
        cam.data.clip_end = 1000
    
    print("   ✓ Camera configured with auto-framing")
    
    # Enable depth and normal rendering
    print("\n7. Configuring render outputs...")
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_normals_output()
    
    # Render
    print("\n8. Rendering scene...")
    data = bproc.renderer.render()
    print("   ✓ Render complete")
    
    # Save outputs
    print("\n9. Saving outputs...")
    hdf5_path = output_path / "0.hdf5"
    bproc.writer.write_hdf5(str(output_path), data)
    print(f"   ✓ HDF5 saved: {hdf5_path}")
    
    # Optionally save .blend file
    if save_blend:
        import bpy
        blend_path = output_path / f"{image_name}.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
        print(f"   ✓ Blend file saved: {blend_path}")
    
    print(f"\n{'='*60}")
    print("Visualization Complete!")
    print(f"{'='*60}")
    print(f"\nOutputs saved to: {output_path.absolute()}")
    print(f"  • HDF5 file: 0.hdf5")
    if save_blend:
        print(f"  • Blend file: {image_name}.blend")
    
    return str(hdf5_path)


def visualize_hdf5(hdf5_path: str, save_png: bool = True) -> None:
    """
    Visualize the HDF5 output using matplotlib.
    
    Args:
        hdf5_path: Path to HDF5 file
        save_png: Whether to save PNG images
    """
    import h5py
    import matplotlib.pyplot as plt
    
    print(f"\nVisualizing HDF5: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'r') as f:
        # Get image data
        colors = np.array(f['colors'])
        
        # Create figure
        fig, axes = plt.subplots(1, 1, figsize=(10, 10))
        
        # Show RGB image
        axes.imshow(colors[0])
        axes.set_title('Reconstructed Scene')
        axes.axis('off')
        
        plt.tight_layout()
        
        if save_png:
            output_path = Path(hdf5_path).parent / "reconstruction.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"   ✓ PNG saved: {output_path}")
        
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize 3D scene reconstruction using BlenderProc"
    )
    parser.add_argument(
        "results_json",
        type=str,
        help="Path to JSON file with 3D pose results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for rendered images"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[512, 512],
        help="Image resolution (width height)"
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=90.0,
        help="Camera field of view in degrees"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Display visualization with matplotlib after rendering"
    )
    
    args = parser.parse_args()
    
    # Run visualization
    hdf5_path = visualize_scene(
        results=args.results_json,
        output_dir=args.output_dir,
        resolution=tuple(args.resolution),
        fov=args.fov
    )
    
    # Optionally show visualization
    if args.visualize:
        visualize_hdf5(hdf5_path, save_png=True)