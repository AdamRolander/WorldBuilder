"""Client for the local VLM detection server.

Same interface as GeminiObjectDetector — drop-in swap. The server runs
in a separate conda env (worldbuilder-vlm); start it before running
the pipeline with WORLDBUILDER_DETECTOR=qwen.
"""
import os
from pathlib import Path
from typing import List, Dict
import requests


class LocalVLMObjectDetector:
    def __init__(self, server_url: str = None, timeout: int = 300):
        self.server_url = (server_url
                           or os.environ.get("WORLDBUILDER_VLM_URL")
                           or "http://127.0.0.1:8765")
        self.timeout = timeout

        try:
            r = requests.get(f"{self.server_url}/health", timeout=5)
            r.raise_for_status()
            info = r.json()
        except Exception as e:
            raise RuntimeError(
                f"VLM server unreachable at {self.server_url}.\n"
                f"  Start it in another terminal:\n"
                f"    conda activate worldbuilder-vlm\n"
                f"    python vlm_server/server.py\n"
                f"  Underlying error: {e}"
            )
        if not info.get("ok"):
            raise RuntimeError(f"VLM server reports model not loaded: {info}")
        print(f"\n✓ VLM server: {info.get('model')}")

    def detect_objects(self, image_path: str) -> List[Dict[str, str]]:
        image_path = str(Path(image_path).resolve())
        try:
            r = requests.post(
                f"{self.server_url}/detect",
                json={"image_path": image_path},
                timeout=self.timeout,
            )
            if r.status_code != 200:
                raise RuntimeError(f"VLM server error: {r.text[:500]}")
            payload = r.json()
            if "error" in payload:
                raise RuntimeError(f"VLM detect failed: {payload['error']}")
            objects = payload["objects"]

            print(f"\nDetected {len(objects)} objects:")
            for obj in objects:
                print(f"  {obj['id']}: {obj['label']} - {obj['description']}")
            return objects
        finally:
            # Free VRAM for SAM 3D, even if detection raised mid-flight.
            try:
                requests.post(f"{self.server_url}/unload", timeout=30)
                print("  ✓ VLM offloaded to CPU (VRAM freed for SAM 3D)")
            except Exception as e:
                print(f"  ⚠️  VLM unload failed (non-fatal): {e}")