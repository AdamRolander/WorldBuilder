# regen_viewer.py  — drop this in the project root
import json, sys
from pathlib import Path
from src.viewer_generator import generate_viewer

for out_dir in [Path(p) for p in sys.argv[1:]]:
    with open(out_dir / "reconstruction_results.json") as f:
        results = json.load(f)["objects"]
    p = generate_viewer(out_dir, results, room_file="room.ply")
    print(f"✓ {p}")