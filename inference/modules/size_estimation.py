"""
Object size estimation using Gemini API.
"""

import os
import json
from PIL import Image
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

def estimate_object_sizes(segments: List[Dict[str, Any]], image_path: str) -> Dict[int, List[float]]:
    """
    Call Gemini API to estimate real-world dimensions of objects.
    
    Args:
        segments: List of segment dictionaries with id, label, bbox
        image_path: Path to input image
        
    Returns:
        Dictionary mapping segment ID to [length, width, height] in meters
    """
    try:
        from google import genai
    except ImportError:
        print("ERROR: google-genai library not found. Install: pip install google-genai")
        return {}
    
    if 'GEMINI_API_KEY' not in os.environ:
        print("ERROR: GEMINI_API_KEY environment variable not set")
        return {}
    
    client = genai.Client()
    
    # Format object list
    object_list_str = "\n".join([
        f"- ID {seg['id']}: {seg['label']} (bbox: x={seg['bbox']['center_x_norm']:.3f}, "
        f"y={seg['bbox']['center_y_norm']:.3f}, w={seg['bbox']['width_norm']:.3f}, "
        f"h={seg['bbox']['height_norm']:.3f})"
        for seg in segments
    ])
    
    prompt = f"""You are an expert AI for 3D world reconstruction. Estimate the real-world dimensions (length, width, height) of objects in the image.

Objects in the image:
{object_list_str}

Provide JSON output in this exact format:
[
  {{
    "id": [OBJECT_ID],
    "label": "[OBJECT_LABEL]",
    "dimensions_meters": [[LENGTH], [WIDTH], [HEIGHT]],
    "justification": "[BRIEF REASONING]"
  }}
]

Dimensions must be in meters. Be conservative with estimates."""
    
    try:
        img = Image.open(image_path)
        contents = [prompt, img]
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents
        )
        
        # Parse JSON response
        json_start = response.text.find('[')
        json_end = response.text.rfind(']') + 1
        
        if json_start == -1 or json_end == 0:
            print("ERROR: No valid JSON in LLM response")
            return {}
        
        json_string = response.text[json_start:json_end]
        llm_output = json.loads(json_string)
        
        # Format output
        estimated_sizes = {}
        for item in llm_output:
            if 'id' in item and 'dimensions_meters' in item:
                estimated_sizes[item['id']] = item['dimensions_meters']
        
        print(f"Successfully estimated sizes for {len(estimated_sizes)} objects")
        return estimated_sizes
        
    except Exception as e:
        print(f"Error during LLM size estimation: {e}")
        return {}