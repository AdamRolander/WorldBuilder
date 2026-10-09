"""The single detection prompt shared by every stage-1 backend.

Both the Gemini detector (``src/object_detection.py``) and the local VLM
server (``vlm_server/server.py``) used to carry their own copy of this text
and they had drifted apart (one said "indoor scene", one allowed
"structures"; only one gave "playground" as an example). Because the label
becomes SAM 3's text prompt in stage 2, the two backends must be steered
identically or their downstream behaviour differs for reasons that have
nothing to do with the model. Edit this file, not the backends.

``vlm_server`` runs in a different conda env and imports this module by
file path, so keep it dependency-free (stdlib only).
"""

# JSON schema the detectors are asked to emit. ``bbox_2d`` is optional: the
# local VLM (Qwen3-VL) is trained for grounding and returns absolute pixel
# boxes reliably; SAM 3 uses them as a geometric fallback prompt when the
# text prompt finds nothing. Gemini is asked for the same field but the
# pipeline works without it.
DETECTION_FIELDS = ("id", "label", "description", "expected_instances", "bbox_2d")

DETECTION_PROMPT = """You are helping build a 3D model of the scene in this photo. List every distinct TYPE of physical object that a 3D artist would model as a separate movable asset.

Return a JSON array. Each entry has:
- "id": sequential integer starting at 1
- "label": a short, common noun phrase of 1–3 words that a text-prompted segmentation model will recognise: "sofa", "office chair", "floor lamp", "3D printer". Prefer the everyday word over jargon ("cabinet" not "modular storage unit"). Never put counts in the label.
- "description": one short phrase (under 12 words) with the most distinctive visual features: colour, material, shape.
- "expected_instances": "single" if there is clearly exactly one of this type visible, otherwise "multiple".
- "bbox_2d": [x1, y1, x2, y2] pixel coordinates of the most prominent instance of this type, or null if you cannot give one.

Rules:
- One entry per object TYPE. If there are six identical chairs, output "chair" once with expected_instances "multiple".
- Include furniture, appliances, fixtures, decor, plants, tools, equipment, vehicles, and outdoor objects (trees, benches, play structures).
- Do NOT list walls, floors, ceilings, windows, doors, countertops, or other architecture — those are reconstructed separately.
- Do NOT list people or animals.
- Do NOT list parts of an object as separate objects (a chair's legs, a lamp's shade).
- Small clutter counts if it is clearly visible (mug, book, bottle), but stop at 25 entries, most prominent first.
- Do not repeat an entry. Output at most one entry per label.
- Output ONLY the JSON array. No prose, no markdown fences."""

EXAMPLE_OUTPUT = """[
  {"id": 1, "label": "dining table", "description": "long rectangular oak table", "expected_instances": "single", "bbox_2d": [212, 540, 1180, 900]},
  {"id": 2, "label": "dining chair", "description": "grey upholstered chair with wooden legs", "expected_instances": "multiple", "bbox_2d": [260, 600, 470, 960]}
]"""


def full_prompt() -> str:
    """Prompt plus a worked example (both backends use this)."""
    return f"{DETECTION_PROMPT}\n\nExample of the required format:\n{EXAMPLE_OUTPUT}"
