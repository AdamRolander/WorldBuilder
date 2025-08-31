import torch
from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random

def calculate_bounding_box_from_mask(mask_tensor):
    """Calculates the bounding box [x, y, width, height] from a boolean mask tensor."""
    if not mask_tensor.any():
        return None
        
    coords = torch.nonzero(mask_tensor)
    y_min, x_min = coords.min(dim=0).values
    y_max, x_max = coords.max(dim=0).values
    
    width = (x_max - x_min + 1).item()
    height = (y_max - y_min + 1).item()
    
    return [x_min.item(), y_min.item(), width, height]


# --- 1. SETUP MODEL AND IMAGE ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading Mask2Former model...")
model_name = "facebook/mask2former-swin-large-coco-panoptic"
processor = AutoImageProcessor.from_pretrained(model_name)
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name).to(device)
print("Model loaded.")

# Load your local image
local_image_path = r"C:\Users\XRLab\CoordinateEstimation\V5\room_test.png"
img = Image.open(local_image_path).convert("RGB")


# --- 2. PERFORM PANOPTIC SEGMENTATION ---
print("Performing panoptic segmentation...")
with torch.no_grad():
    inputs = processor(images=img, return_tensors="pt").to(device)
    outputs = model(**inputs)
    panoptic_result = processor.post_process_panoptic_segmentation(outputs, target_sizes=[img.size[::-1]])[0]

print("Segmentation complete.")
segmentation_map = panoptic_result["segmentation"]
segments_info = panoptic_result["segments_info"]


# --- 3. GENERATE FINAL OUTPUT FOR NEXT SCRIPT (INCLUDING STUFF) ---
panoptic_segments = []
print("\n--- Generating Data for Coordinate Estimation (Things and Stuff) ---")

for segment in segments_info:
    # REMOVED the filter for "Thing" vs "Stuff"
    segment_id = segment['id']
    label_id = segment['label_id']
    
    mask_tensor = (segmentation_map == segment_id)
    bbox = calculate_bounding_box_from_mask(mask_tensor)
    mask_area = mask_tensor.sum().item()

    if bbox is not None and mask_area > 0: # Only add segments that are actually in the image
        panoptic_segments.append({
            'id': segment_id,
            'label_id': label_id,
            'bounding_box': bbox,
            'mask_area': mask_area
        })

print("\npanoptic_segments = [")
for seg in panoptic_segments:
    label = model.config.id2label[seg['label_id']]
    print(f"    {str(seg)}, # {label}")
print("]")


# --- 4. VISUALIZE THE RESULTS ---
print("\nGenerating visualization...")

# Create a color map for the segments
segment_colors = {}
for segment in panoptic_segments:
    segment_colors[segment['id']] = tuple(np.random.randint(60, 256, 3))

# Create a colorized mask
colorized_mask = Image.new("RGB", img.size)
mask_pixels = colorized_mask.load()
width, height = img.size

for y in range(height):
    for x in range(width):
        segment_id = segmentation_map[y, x].item()
        if segment_id in segment_colors:
            mask_pixels[x, y] = segment_colors[segment_id]

# Blend the original image with the colorized mask
visual_image = Image.blend(img, colorized_mask, alpha=0.4)

# Draw bounding boxes and labels
draw = ImageDraw.Draw(visual_image)
try:
    font = ImageFont.truetype("arial.ttf", 15)
except IOError:
    font = ImageFont.load_default()

for segment in panoptic_segments:
    bbox = segment['bounding_box']
    segment_id = segment['id']
    label_id = segment['label_id']
    label_text = model.config.id2label[label_id]
    
    # Bounding box coordinates
    x, y, w, h = bbox
    box_coords = [(x, y), (x + w, y + h)]
    
    # Draw the box with the segment's color
    draw.rectangle(box_coords, outline=segment_colors[segment_id], width=2)
    
    # Draw a filled rectangle for the text background
    text_bbox = draw.textbbox((x, y - 18), label_text, font=font)
    draw.rectangle(text_bbox, fill=segment_colors[segment_id])
    
    # Draw the text label
    draw.text((x, y - 18), label_text, fill="black", font=font)

# Display the final image
visual_image.show()

# Optionally, save the image to a file
visual_image.save("panoptic_visualization.png")
print("Saved visualization to 'panoptic_visualization.png'")