"""Job progress as the web app reports it to the Blender add-on (no models needed)."""
import json

import pytest

pytest.importorskip("flask")
pytest.importorskip("flask_cors")

from src import stage3_cache  # noqa: E402
from webapp import server  # noqa: E402


def test_progress_steps_and_eta(tmp_path):
    job = {"_t0": 0.0}
    assert server._progress(tmp_path, job)["step"] == 1
    (tmp_path / "detected_objects.json").write_text("[]")
    assert server._progress(tmp_path, job)["step"] == 2
    (tmp_path / "segmentation_results.json").write_text("[]")
    pr = server._progress(tmp_path, job)
    assert pr["step"] == 3 and pr["total"] is None          # SAM 3D still loading

    models = tmp_path / "3d_models"
    stage3_cache.cache_dir(models).mkdir(parents=True)
    stage3_cache.save_progress(models, 10, 40, 60.0)        # 6 s per object so far
    pr = server._progress(tmp_path, job)
    assert (pr["done"], pr["total"], pr["fraction"]) == (10, 40, 0.25)
    assert pr["eta_seconds"] == 30 * 6 + server.ASSEMBLY_SECONDS

    (stage3_cache.cache_dir(models) / "raw.json").write_text(json.dumps({"objects": []}))
    pr = server._progress(tmp_path, job)
    assert pr["step"] == 4 and pr["name"] == server.STEPS[3]
    (tmp_path / "viewer.html").write_text("")
    assert server._progress(tmp_path, job)["step"] == 4      # finished: stays on the last step
