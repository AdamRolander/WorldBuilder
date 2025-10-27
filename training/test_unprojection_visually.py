import pandas as pd
import numpy as np
import math

def calculate_unprojection_simple(row, img_width=300, img_height=300):
    """Simple pinhole unprojection - what you use in inference."""
    fov_rad = row['field_of_view'] * (math.pi / 180.0)
    focal_length = (img_width / 2.0) / math.tan(fov_rad / 2.0)
    cx = img_width / 2.0
    cy = img_height / 2.0
    
    px = row['target_bbox_center_x'] * img_width
    py = row['target_bbox_center_y'] * img_height
    depth = row['dist_to_target']  # Using actual target distance
    
    K_inv = np.linalg.inv(np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ]))
    
    pixel_coords = np.array([px, py, 1])
    camera_coords = K_inv @ pixel_coords * depth
    
    return camera_coords[0], camera_coords[1], camera_coords[2]

# Load data
df = pd.read_csv("ai2thor_adjusted.csv")

print("=== Visual Validation of Unprojection ===\n")
print("Testing if unprojection gives reasonable spatial relationships...\n")

# Sample some data
sample_df = df.sample(min(10, len(df)), random_state=42)

for idx, row in sample_df.iterrows():
    unproj_x, unproj_y, unproj_z = calculate_unprojection_simple(row)
    
    # The key test: Does unprojection preserve RELATIVE positions?
    # If object is on left of image, X should be negative
    # If object is on right, X should be positive
    # If object is at top, Y should be negative (camera Y axis points down)
    # If object is at bottom, Y should be positive
    
    bbox_center_x = row['target_bbox_center_x']  # 0-1, where 0.5 is center
    bbox_center_y = row['target_bbox_center_y']
    
    expected_x_sign = "positive" if bbox_center_x > 0.5 else "negative"
    actual_x_sign = "positive" if unproj_x > 0 else "negative"
    
    expected_y_sign = "positive" if bbox_center_y > 0.5 else "negative"  
    actual_y_sign = "positive" if unproj_y > 0 else "negative"
    
    x_correct = (expected_x_sign == actual_x_sign)
    y_correct = (expected_y_sign == actual_y_sign)
    
    print(f"Sample {idx}:")
    print(f"  Bbox center: ({bbox_center_x:.2f}, {bbox_center_y:.2f})")
    print(f"  Unprojection: ({unproj_x:.2f}, {unproj_y:.2f}, {unproj_z:.2f})")
    print(f"  X direction: {'✓' if x_correct else '✗'} (expected {expected_x_sign}, got {actual_x_sign})")
    print(f"  Y direction: {'✓' if y_correct else '✗'} (expected {expected_y_sign}, got {actual_y_sign})")
    print(f"  Distance: {row['dist_to_target']:.2f}m")
    print()

# Test relative positioning
print("\n=== Testing Relative Positioning ===")
print("Checking if unprojection preserves left/right relationships...\n")

# Find two objects from the same scene
scenes = df['scene_name'].unique()
test_scene = scenes[0]
scene_df = df[df['scene_name'] == test_scene].head(5)

if len(scene_df) >= 2:
    for i in range(len(scene_df) - 1):
        obj1 = scene_df.iloc[i]
        obj2 = scene_df.iloc[i + 1]
        
        # If obj1 is left of obj2 in image
        if obj1['target_bbox_center_x'] < obj2['target_bbox_center_x']:
            unproj1_x, _, _ = calculate_unprojection_simple(obj1)
            unproj2_x, _, _ = calculate_unprojection_simple(obj2)
            
            relative_correct = unproj1_x < unproj2_x
            
            print(f"Object 1 bbox X: {obj1['target_bbox_center_x']:.2f} → unproj X: {unproj1_x:.2f}")
            print(f"Object 2 bbox X: {obj2['target_bbox_center_x']:.2f} → unproj X: {unproj2_x:.2f}")
            print(f"Left-right relationship preserved: {'✓' if relative_correct else '✗'}\n")

print("\n=== Conclusion ===")
print("If most directions and relationships are correct (✓),")
print("then unprojection IS useful and should be kept as a feature!")