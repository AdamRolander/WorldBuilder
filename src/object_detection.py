"""Stage 1a: object detection with Gemini (cloud backend).

Kept as the reference detector: it is the most reliable lister of objects we
have and the local backend is benchmarked against it (``scripts/bench_detectors.py``).
Uses the shared prompt in ``src/prompts.py`` and Gemini's JSON response mode
so the output never needs fence-stripping or salvage parsing.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

from PIL import Image

from src.prompts import full_prompt

DEFAULT_MODEL = os.environ.get("WORLDBUILDER_GEMINI_MODEL", "gemini-2.5-flash")
TRANSIENT_MARKERS = ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "DEADLINE_EXCEEDED", "overloaded")


class GeminiObjectDetector:
    """Detect object types in an image with Gemini vision."""

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Put it in .env (project root) or set "
                "WORLDBUILDER_DETECTOR=qwen to use the local VLM instead.")
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def detect_objects(self, image_path: str, max_attempts: int = 5) -> List[Dict]:
        image = Image.open(image_path).convert("RGB")
        prompt = full_prompt() + f"\n\nThe image is {image.width}x{image.height} pixels."

        from google.genai import types
        config = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2)

        response = None
        for attempt in range(max_attempts):
            try:
                response = self.client.models.generate_content(
                    model=self.model, contents=[prompt, image], config=config)
                break
            except Exception as e:
                msg = str(e)
                transient = any(m in msg for m in TRANSIENT_MARKERS)
                if not transient or attempt == max_attempts - 1:
                    raise
                wait = 2 ** attempt
                print(f"  ⚠️  Gemini transient error (attempt {attempt + 1}/{max_attempts}): "
                      f"{msg[:80]}... retrying in {wait}s")
                time.sleep(wait)

        text = (response.text or "").strip()
        if text.startswith("```"):                      # belt and braces
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        objects = json.loads(text)
        if isinstance(objects, dict):                   # some models wrap the array
            for v in objects.values():
                if isinstance(v, list):
                    objects = v
                    break
        if not isinstance(objects, list):
            raise ValueError(f"Gemini returned unexpected JSON: {text[:200]}")

        print(f"\nDetected {len(objects)} objects:")
        for obj in objects:
            print(f"  {obj.get('id', '?')}: {obj.get('label')} - {obj.get('description', '')}")
        return objects


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "living_room.jpg"
    print(json.dumps(GeminiObjectDetector().detect_objects(path), indent=2))
