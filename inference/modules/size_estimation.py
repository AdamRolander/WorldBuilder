"""
Object size estimation using Gemini API.
"""

import os
import json
from PIL import Image
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def estimate_object_sizes(segments: List[Dict[str, Any]], image_path: str) -> Dict[int, Dict]:
    """
    Call Gemini API to estimate real-world dimensions and rotation of objects.
    
    Args:
        segments: List of segment dictionaries with id, label, bbox
        image_path: Path to input image
        
    Returns:
        Dictionary mapping segment ID to dict with:
            - dimensions_meters: [length, width, height]
            - rotation_y_degrees: Y-axis rotation
            - confidence: Confidence score
            - justification: Reasoning
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

    prompt_path = Path(__file__).parent / "prompts" / "dimension_rotation.txt"
    
    try:
        with open(prompt_path, 'r') as f:
            prompt_template = f.read()
    except FileNotFoundError:
        print(f"ERROR: Prompt file not found at {prompt_path}")
        print("Falling back to default prompt...")
        prompt_template = """[Your fallback prompt here]
        
{object_list_str}"""

    prompt = prompt_template.replace("{object_list_str}", object_list_str)
    
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

        print("\n=== RAW GEMINI OUTPUT ===")
        print(json.dumps(llm_output, indent=2))
        print("=========================\n")
        
        # Format output
        estimated_data = {}
        for item in llm_output:
            if 'id' in item and 'dimensions_meters' in item:
                estimated_data[item['id']] = {
                    'dimensions_meters': item['dimensions_meters'],
                    'rests_on_id': item.get('rests_on_id', None),
                    'confidence': item.get('confidence', 0.5),
                    'justification': item.get('justification', '')
                }
                
                # Debug output
                obj_id = item['id']
                label = item.get('label', 'unknown')
                rot = item.get('rotation_y_degrees', 0)
                conf = item.get('confidence', 0)
                print(f"  {label} (ID {obj_id}): {rot:.0f}° (confidence: {conf:.2f})")

        print(f"Successfully estimated sizes and rotations for {len(estimated_data)} objects")
        return estimated_data
        
    except Exception as e:
        print(f"Error during LLM size estimation: {e}")
        return {}