import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2

# --- CONFIGURATION ---
# Specify the category ID you want to isolate and view.
# Specify the path to the HDF5 file you want to render.
HDF5_FILE_PATH = "output/0.hdf5" 
# ---------------------

# Load the .hdf5 file
try:
    file = h5py.File(HDF5_FILE_PATH, "r")
    print(f"Successfully loaded HDF5 file: {HDF5_FILE_PATH}")
except FileNotFoundError:
    print(f"Error: {HDF5_FILE_PATH} not found. Make sure the pipeline ran successfully.")
    exit()

# Load render outputs
rgb = file["colors"][()]
depth = file["depth"][()]

# --- Data Processing ---
# Check if we have multiple frames and select the first one
if len(rgb.shape) == 4:
    rgb = rgb[0]
if len(depth.shape) == 3:
    depth = depth[0]

# Normalize RGB for display
rgb_vis = rgb.copy()
if rgb.dtype in [np.float32, np.float64]:
    rgb_vis = np.clip(rgb_vis, 0, 1)
elif rgb.max() > 1:
    rgb_vis = rgb_vis / 255.0

# Normalize depth for display
depth_vis = depth.copy()
if depth.max() > 0:
    depth_vis[depth_vis > 1000] = depth.min() 
    depth_vis = cv2.normalize(depth_vis, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

# --- Visualization ---
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# RGB Image
axs[0].imshow(rgb_vis)
axs[0].set_title(f"RGB Image\nShape: {rgb.shape}")
axs[0].axis('off')

# Depth Map
axs[1].imshow(depth_vis, cmap='plasma')
axs[1].set_title(f"Depth Map (Normalized)\nNon-zero pixels: {np.count_nonzero(depth)}")
axs[1].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Close the file
file.close()