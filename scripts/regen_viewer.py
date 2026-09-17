"""Rebuild viewer.html (and scene.glb) for existing output directories after
viewer/exporter edits:  python scripts/regen_viewer.py outputs/scene_a outputs/scene_b"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.scene_export import export_scene_glb  # noqa: E402
from src.viewer_generator import generate_viewer  # noqa: E402

for out_dir in [Path(p) for p in sys.argv[1:]]:
    data = json.loads((out_dir / "reconstruction_results.json").read_text())
    p = generate_viewer(out_dir, data["objects"], room_file="3d_models/room.ply",
                        layout_file="3d_models/layout.json", failed=data.get("failed"))
    print(f"✓ {p}")
    g = export_scene_glb(out_dir, data["objects"])
    if g:
        print(f"✓ {g}")
