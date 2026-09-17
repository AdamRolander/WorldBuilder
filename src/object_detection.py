import os
from typing import List, Dict
from pathlib import Path
from google import genai
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

class GeminiObjectDetector:
    """Detect objects in images using Gemini vision."""
    
    def __init__(self, api_key: str = None):
        """Initialize Gemini client."""
        if api_key is None:
            api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found")
        
        self.client = genai.Client(api_key=api_key)
        self.model = 'gemini-2.5-flash'  
    
    def detect_objects(self, image_path: str) -> List[Dict[str, str]]:
        """
        Detect all objects in the scene.
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of dicts with 'id', 'label', 'description'
        """
        image = Image.open(image_path).convert('RGB')
        
        prompt = """Analyze this scene and list ALL distinct objects visible.

For each object, provide:
- id: Sequential number starting from 1
- label: Simple object name using everyday language (e.g., "sofa", "lamp", "pillow", "playground"). Avoid technical jargon like "modular container unit" or "workstation framework" - use plain terms a layperson would say.
- description: Brief description focusing on the most distinctive features. Use simple everyday words.
- expected_instances: "single" if there is typically one of this object type in a scene like this (e.g., a sink in a bathroom, a TV in a living room), or "multiple" if several are expected (e.g., desks in a classroom, plates on a table)

IMPORTANT: Keep descriptions CONCISE and NATURAL. Avoid technical or niche terms. Use everyday language.

Format as JSON array:
[
  {
    "id": 1,
    "label": "dining table",
    "description": "wooden rectangular dining table with dark finish",
    "expected_instances": "single"
  },
  {
    "id": 2,
    "label": "chair",
    "description": "upholstered dining chair with grey fabric",
    "expected_instances": "multiple"
  }
]

Rules:
- Include ALL objects (furniture, decor, appliances, structures, etc.)
- If multiple similar objects exist (e.g., many chairs), list that object TYPE exactly ONCE, like "dining chair"
- Do NOT list duplicate objects as separate entries and do NOT include counts in the label
- Be specific but concise
- Focus on objects that would be 3D modeled, not walls/floors/ceilings/windows
- Return ONLY the JSON array, no other text"""

# incl. reflective objects/mirrors, humans != object -- maybe only animated humans/characters, repeated objects 

        import time
        response = None
        for attempt in range(5):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[prompt, image]
                )
                break
            except Exception as e:
                msg = str(e)
                transient = any(code in msg for code in ('503', '429', 'UNAVAILABLE', 'RESOURCE_EXHAUSTED'))
                if not transient or attempt == 4:
                    raise
                wait = 2 ** attempt  # 1, 2, 4, 8, 16s
                print(f"  ⚠️  Gemini transient error (attempt {attempt+1}/5): "
                      f"{msg[:80]}... retrying in {wait}s")
                time.sleep(wait)
        
        # Parse JSON response
        import json
        response_text = response.text.strip()
        
        # Extract JSON if wrapped in markdown code blocks
        if response_text.startswith('```'):
            response_text = response_text.split('```')[1]
            if response_text.startswith('json'):
                response_text = response_text[4:]
        
        objects = json.loads(response_text)
        
        print(f"\nDetected {len(objects)} objects:")
        for obj in objects:
            print(f"  {obj['id']}: {obj['label']} - {obj['description']}")
        
        return objects


if __name__ == "__main__":
    # Test the detector
    detector = GeminiObjectDetector()
    objects = detector.detect_objects("test_images/living_room.jpg")
    
    import json
    with open("outputs/detected_objects.json", "w") as f:
        json.dump(objects, f, indent=2)
