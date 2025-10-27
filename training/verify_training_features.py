# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Load training data
# df = pd.read_csv("ai2thor_adjusted.csv")

# # Calculate unprojection features for a sample
# def calculate_unprojection_corrected(row, img_width=300, img_height=300):
#     fov_rad = row['field_of_view'] * (np.pi / 180.0)
#     focal_length = (img_width / 2.0) / np.tan(fov_rad / 2.0)
#     cx = img_width / 2.0
#     cy = img_height / 2.0
    
#     px = row['target_bbox_center_x'] * img_width
#     py = row['target_bbox_center_y'] * img_height
#     depth = row['dist_to_target']  # Use correct depth
    
#     K_inv = np.linalg.inv(np.array([
#         [focal_length, 0, cx],
#         [0, focal_length, cy],
#         [0, 0, 1]
#     ]))
    
#     pixel_coords = np.array([px, py, 1])
#     camera_coords = K_inv @ pixel_coords * depth
    
#     return camera_coords[0], camera_coords[1], camera_coords[2]

# # Sample 100 rows
# sample_df = df.sample(min(100, len(df)), random_state=42)

# print("=== Training Data Feature Verification ===\n")

# errors = []
# for idx, row in sample_df.iterrows():
#     unproj_x, unproj_y, unproj_z = calculate_unprojection_corrected(row)
    
#     error_x = abs(unproj_x - row['world_x'])
#     error_y = abs(unproj_y - row['world_y'])
#     error_z = abs(unproj_z - row['world_z'])
#     total_error = np.sqrt(error_x**2 + error_y**2 + error_z**2)
    
#     errors.append(total_error)
    
#     if idx < 5:  # Print first 5 for inspection
#         print(f"Sample {idx}:")
#         print(f"  Unprojection: ({unproj_x:.3f}, {unproj_y:.3f}, {unproj_z:.3f})")
#         print(f"  Ground truth: ({row['world_x']:.3f}, {row['world_y']:.3f}, {row['world_z']:.3f})")
#         print(f"  Error: {total_error:.3f}m\n")

# print(f"\nOverall Statistics:")
# print(f"  Mean error: {np.mean(errors):.3f}m")
# print(f"  Median error: {np.median(errors):.3f}m")
# print(f"  90th percentile: {np.percentile(errors, 90):.3f}m")
# print(f"  Max error: {np.max(errors):.3f}m")

# if np.mean(errors) < 0.1:
#     print("\n✓ Unprojection features match ground truth - training data is consistent!")
# else:
#     print(f"\n❌ Large unprojection errors - training data has coordinate issues!")

# import pandas as pd
# df = pd.read_csv("ai2thor_adjusted.csv")

# # Check a few samples
# for i in range(5):
#     row = df.iloc[i]
#     print(f"\nSample {i}:")
#     print(f"  dist_to_target: {row['dist_to_target']:.3f}m")
#     print(f"  dist_to_ref: {row['dist_to_ref']:.3f}m")
#     print(f"  world_z (ground truth): {row['world_z']:.3f}m")
#     print(f"  Should world_z ≈ dist_to_target? {abs(row['world_z'] - row['dist_to_target']) < 0.5}")

import pandas as pd
import numpy as np
import math

df = pd.read_csv("ai2thor_adjusted.csv")

def calculate_distance_from_coords(row):
    """Calculate Euclidean distance from camera to object using coordinates."""
    return np.sqrt(row['world_x']**2 + row['world_y']**2 + row['world_z']**2)

def calculate_unprojection_with_dist_to_target(row, img_width=300, img_height=300):
    """Unprojection using dist_to_target."""
    fov_rad = row['field_of_view'] * (math.pi / 180.0)
    focal_length = (img_width / 2.0) / math.tan(fov_rad / 2.0)
    cx = img_width / 2.0
    cy = img_height / 2.0
    
    px = row['target_bbox_center_x'] * img_width
    py = row['target_bbox_center_y'] * img_height
    depth = row['dist_to_target']
    
    K_inv = np.linalg.inv(np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ]))
    
    pixel_coords = np.array([px, py, 1])
    camera_coords = K_inv @ pixel_coords * depth
    
    return camera_coords[0], camera_coords[1], camera_coords[2]

print("=== Coordinate System Diagnosis ===\n")

sample_df = df.sample(min(20, len(df)), random_state=42)

errors_with_target = []
errors_with_ref = []

for idx, row in sample_df.iterrows():
    # Ground truth distance (from coordinates)
    gt_distance = calculate_distance_from_coords(row)
    
    # Unprojection using dist_to_target
    unproj_x, unproj_y, unproj_z = calculate_unprojection_with_dist_to_target(row)
    
    # Compare with ground truth
    error = np.sqrt(
        (unproj_x - row['world_x'])**2 + 
        (unproj_y - row['world_y'])**2 + 
        (unproj_z - row['world_z'])**2
    )
    
    errors_with_target.append(error)
    
    # Check if world_z matches the depth component
    z_error = abs(unproj_z - row['world_z'])
    
    if idx < 5:
        print(f"Sample {idx}:")
        print(f"  GT coords: ({row['world_x']:.3f}, {row['world_y']:.3f}, {row['world_z']:.3f})")
        print(f"  GT distance: {gt_distance:.3f}m")
        print(f"  dist_to_target: {row['dist_to_target']:.3f}m")
        print(f"  Unprojection: ({unproj_x:.3f}, {unproj_y:.3f}, {unproj_z:.3f})")
        print(f"  Total error: {error:.3f}m")
        print(f"  Z-only error: {z_error:.3f}m")
        print()

print("\nOverall Statistics (using dist_to_target):")
print(f"  Mean error: {np.mean(errors_with_target):.3f}m")
print(f"  Median error: {np.median(errors_with_target):.3f}m")
print(f"  90th percentile: {np.percentile(errors_with_target, 90):.3f}m")

if np.mean(errors_with_target) < 0.2:
    print("\n✓ Using dist_to_target for unprojection is correct!")
    print("Action: Change calculate_unprojection() in enhanced_attention.py to use dist_to_target")
else:
    print("\n❌ Unprojection formula doesn't match AI2-THOR coordinate transform")
    print("Action: May need to remove unprojection features entirely")