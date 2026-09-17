"""Qwen3-VL object detection server.

Runs in its own conda env (worldbuilder-vlm). Loads the model once,
serves /detect over HTTP. The main pipeline (in sam3d-objects env)
talks to it via src/local_vlm_detection.py.
"""
import argparse
import json
import re
from flask import Flask, request, jsonify
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

# Swap to "Qwen/Qwen2.5-VL-7B-Instruct" if Qwen3-VL has issues — the rest
# of this file works unchanged.
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
DEVICE = "cuda"

PROMPT = """Analyze this indoor scene and list ALL distinct objects visible.

For each object, provide:
- id: Sequential number starting from 1
- label: Simple object name using everyday language (e.g., "sofa", "lamp", "pillow"). Avoid technical jargon - use plain terms a layperson would say.
- description: Brief description focusing on the most distinctive features. Use simple everyday words.
- expected_instances: "single" if there is typically one of this object type in a scene like this (e.g., a sink in a bathroom, a TV in a living room), or "multiple" if several are expected (e.g., desks in a classroom, plates on a table)

IMPORTANT: Keep descriptions CONCISE and NATURAL. Avoid technical or niche terms.

Format as JSON array:
[
  {"id": 1, "label": "dining table", "description": "wooden rectangular dining table with dark finish", "expected_instances": "single"},
  {"id": 2, "label": "chair", "description": "upholstered dining chair with grey fabric", "expected_instances": "multiple"}
]

Rules:
- Include ALL objects (furniture, decor, appliances, etc.)
- If multiple similar objects exist (e.g., many chairs), list that object TYPE exactly ONCE, like "dining chair"
- Do NOT list duplicate objects as separate entries and do NOT include counts in the label
- Be specific but concise
- Focus on objects that would be 3D modeled, not walls/floors/ceilings/windows
- Return ONLY the JSON array, no other text, no markdown fences"""

import gc

app = Flask(__name__)
_model = None
_processor = None
_current_device = None  # 'cuda', 'cpu', or None


def load_model():
    """Load model into CPU RAM. /detect will move it to GPU on demand."""
    global _model, _processor, _current_device
    print(f"Loading {MODEL_NAME}...")
    _model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        attn_implementation="sdpa",
    )
    _model.eval()
    _processor = AutoProcessor.from_pretrained(MODEL_NAME)
    _current_device = "cpu"
    print(f"✓ Model loaded into CPU RAM (will move to {DEVICE} on demand)")


def _to_device(target: str):
    """Move model between cuda/cpu. Releases VRAM back to the GPU when going
    to CPU so other processes (SAM 3D) can use it.

    AutoModelForImageTextToText keeps submodules (vision tower, projection
    layers, sometimes a separate language model) that aren't always picked
    up by a bare _model.to(target). Walk the module tree explicitly and
    move every nn.Module + every tensor parameter/buffer we find.
    """
    global _model, _current_device
    if _current_device == target:
        return
    print(f"  Moving model: {_current_device} → {target}")

    # Top-level move first — handles the standard registered children.
    _model.to(target)

    # Now sweep every attribute for stragglers. Common offenders:
    # `vision_tower`, `visual`, `image_processor`'s patch embed, `lm_head`,
    # cached generation buffers. Don't assume names; introspect.
    seen = set()
    def _move(obj, depth=0):
        if depth > 3 or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, torch.nn.Module):
            obj.to(target)
            for child in obj.children():
                _move(child, depth + 1)
        elif isinstance(obj, torch.Tensor):
            try:
                obj.data = obj.data.to(target)
            except Exception:
                pass
    for name in dir(_model):
        if name.startswith('__'):
            continue
        try:
            _move(getattr(_model, name))
        except Exception:
            pass

    # The KV cache from past .generate() calls is a giant easy-to-miss
    # tensor pool living on the model. Nuke it.
    if hasattr(_model, 'past_key_values'):
        _model.past_key_values = None

    _current_device = target
    gc.collect()
    gc.collect()  # second pass catches cycles freed by the first
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Report what's left so we can see it in the log
        free, total = torch.cuda.mem_get_info()
        print(f"  GPU free after move: {free / 1024**3:.2f} GiB / {total / 1024**3:.2f} GiB")

def extract_json(text: str):
    """Tolerant JSON extractor — strips fences, finds the array, and salvages
    truncated output by trimming back to the last complete object."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)

    # Empty array is a valid response ("VLM saw no objects"). Handle it
    # before the regex, which requires at least one {...} to match.
    if re.fullmatch(r"\[\s*\]", text):
        return []

    m = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))

    # Truncated output: find the array start, walk back to the last complete
    # object boundary "}," and close the array there.
    start = text.find("[")
    if start == -1:
        raise json.JSONDecodeError("No JSON array found", text, 0)
    last_close = text.rfind("},")
    if last_close == -1:
        raise json.JSONDecodeError("No complete object in output", text, 0)
    salvaged = text[start:last_close + 1] + "]"
    return json.loads(salvaged)


@app.route("/health")
def health():
    return jsonify({
        "ok": _model is not None,
        "model": MODEL_NAME,
        "device": _current_device,
    })


@app.route("/unload", methods=["POST"])
def unload():
    """Move model to CPU and free VRAM. Call after detection so SAM 3D
    can use the GPU."""
    _to_device("cpu")
    return jsonify({"device": _current_device})


@app.route("/reload", methods=["POST"])
def reload_to_gpu():
    """Move model back to GPU. Called automatically by /detect; exposed
    here for explicit pre-warming if you ever want it."""
    _to_device(DEVICE)
    return jsonify({"device": _current_device})


@app.route("/detect", methods=["POST"])
def detect():
    _to_device(DEVICE)  # auto-reload if previously unloaded
    # Clear any prior KV cache before a new generation. Cheap insurance.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    data = request.get_json()
    image_path = data["image_path"]
    image = Image.open(image_path).convert("RGB")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ],
    }]

    chat_text = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _processor(
        text=[chat_text], images=[image],
        padding=True, return_tensors="pt",
    ).to(_current_device)

    last_raw = ""
    last_err = None
    # First attempt: greedy/deterministic. Retries: small temp to nudge
    # out of any malformed-JSON local minimum.
    for attempt in range(3):
        with torch.inference_mode():
            out_ids = _model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=(attempt > 0),
                temperature=0.3 if attempt > 0 else 1.0,
            )
        gen_ids = out_ids[:, inputs.input_ids.shape[1]:]
        last_raw = _processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        try:
            objects = extract_json(last_raw)
            if isinstance(objects, list) and all("label" in o for o in objects):
                print(f"  [attempt {attempt+1}] parsed {len(objects)} objects")
                # Free per-call tensors so they don't pile up across images.
                del out_ids, gen_ids, inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return jsonify({"objects": objects})
            last_err = "Schema check failed"
        except (json.JSONDecodeError, AttributeError) as e:
            last_err = str(e)
        print(f"  [attempt {attempt+1}] parse failed: {last_err}")
        print(f"  raw[:300]: {last_raw[:300]!r}")

    return jsonify({"error": f"Parse failed: {last_err}", "raw": last_raw}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    load_model()
    print(f"\nServer ready at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=False)