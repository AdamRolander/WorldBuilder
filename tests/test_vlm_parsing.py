import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

# vlm_server/server.py imports flask/torch/transformers at module import. Stub
# them so the pure parsing helpers can be tested in the main env / CI.
for name in ("torch", "flask", "transformers"):
    if name not in sys.modules:
        mod = types.ModuleType(name)
        if name == "flask":
            mod.Flask = lambda *a, **k: types.SimpleNamespace(route=lambda *a, **k: (lambda f: f))
            mod.jsonify = lambda *a, **k: None
            mod.request = None
        if name == "transformers":
            mod.AutoModelForImageTextToText = object
            mod.AutoProcessor = object
        if name == "torch":
            mod.nn = types.SimpleNamespace(Module=object)
            mod.Tensor = object
        sys.modules[name] = mod

spec = importlib.util.spec_from_file_location(
    "vlm_server", Path(__file__).resolve().parent.parent / "vlm_server" / "server.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


def test_extract_json_handles_fences_and_truncation():
    assert server.extract_json('```json\n[{"label": "a"}]\n```') == [{"label": "a"}]
    assert server.extract_json("[]") == []
    truncated = '[{"id":1,"label":"sofa"},{"id":2,"label":"lamp"},{"id":3,"lab'
    assert [o["label"] for o in server.extract_json(truncated)] == ["sofa", "lamp"]
    with pytest.raises(json.JSONDecodeError):
        server.extract_json("no json here")


def test_dedupe_and_bbox_scaling():
    objs = [{"label": "Tool", "bbox_2d": [10, 10, 20, 20]}] * 50 + [{"label": "desk"}]
    out = server.dedupe_objects(objs)
    assert [o["label"] for o in out] == ["tool", "desk"]
    assert out[0]["expected_instances"] == "multiple"
    out = server.scale_bboxes(out, 2.0, 0.5)
    assert out[0]["bbox_2d"] == [20.0, 5.0, 40.0, 10.0]
