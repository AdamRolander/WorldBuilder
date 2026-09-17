"""Local VLM object-detection server (the "fully local" stage 1).

Runs in its own conda env (``worldbuilder-vlm``) because Qwen3-VL needs
transformers 5.x while SAM 3D pins 4.39. Loads the model once into CPU
RAM, moves it to the GPU only for the duration of a ``/detect`` call, and
hands VRAM back on ``/unload`` so SAM 3 / SAM 3D can use it.

    python vlm_server/server.py                       # Qwen3-VL-8B-Instruct
    python vlm_server/server.py --model Qwen/Qwen3-VL-32B-Instruct-AWQ

Fixes relative to the demo-era server (docs/AUDIT.md §1):

* the prompt is the shared one in ``src/prompts.py`` (was a drifted copy);
* ``repetition_penalty`` + a lower ``max_new_tokens`` stop the degenerate
  loops that produced 100+ copies of "tool" and 30-minute segmentation runs;
* decoded output is deduplicated by label before it leaves the server;
* the model is asked for a ``bbox_2d`` per object type (Qwen3-VL is trained
  for grounding); SAM 3 uses it as a fallback prompt;
* an empty list triggers a sampled retry instead of being returned;
* ``do_sample=False`` no longer passes a temperature (was a warning per call);
* ``--model`` on the command line instead of editing the file.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import re
import time
from pathlib import Path

import torch
from flask import Flask, jsonify, request
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# Import the shared prompt by path: this env does not have the project's
# other dependencies and `src` is not a package it can see.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("wb_prompts", _PROJECT_ROOT / "src" / "prompts.py")
_prompts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_prompts)
PROMPT = _prompts.full_prompt()

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEVICE = "cuda"
MAX_IMAGE_SIDE = 1536          # Qwen3-VL scales tokens with pixels; this caps latency
MAX_NEW_TOKENS = 2048          # 25 entries × ~60 tokens; 4096 only ever fed loops

app = Flask(__name__)
_model = None
_processor = None
_current_device = None
_model_name = DEFAULT_MODEL


def load_model(model_name: str = DEFAULT_MODEL):
    global _model, _processor, _current_device, _model_name
    _model_name = model_name
    print(f"Loading {model_name}...")
    t0 = time.time()
    _model = AutoModelForImageTextToText.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="sdpa")
    _model.eval()
    _processor = AutoProcessor.from_pretrained(model_name)
    _current_device = "cpu"
    print(f"✓ Model loaded into CPU RAM in {time.time() - t0:.0f}s (moves to {DEVICE} on demand)")


def _to_device(target: str):
    """Move the model between cuda/cpu and really release VRAM."""
    global _model, _current_device
    if _current_device == target:
        return
    print(f"  Moving model: {_current_device} → {target}")
    _model.to(target)
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
        if name.startswith("__"):
            continue
        try:
            _move(getattr(_model, name))
        except Exception:
            pass
    if hasattr(_model, "past_key_values"):
        _model.past_key_values = None
    _current_device = target
    gc.collect()
    gc.collect()  # second pass catches cycles freed by the first
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        print(f"  GPU free after move: {free / 1024**3:.2f} / {total / 1024**3:.2f} GiB")


# --------------------------------------------------------------------------
# Output parsing (pure functions; tested in tests/test_vlm_parsing.py)
# --------------------------------------------------------------------------

def extract_json(text: str):
    """Tolerant JSON extractor: strips fences, finds the array, salvages
    truncated output by cutting back to the last complete object."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    if re.fullmatch(r"\[\s*\]", text):
        return []
    m = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    start = text.find("[")
    if start == -1:
        raise json.JSONDecodeError("No JSON array found", text, 0)
    last_close = text.rfind("},")
    if last_close == -1:
        last_close = text.rfind("}")
        if last_close == -1:
            raise json.JSONDecodeError("No complete object in output", text, 0)
    salvaged = text[start:last_close + 1].rstrip(",") + "]"
    return json.loads(salvaged)


def dedupe_objects(objects, max_objects: int = 40):
    """Collapse repeated labels (the runaway-loop failure mode) server-side.
    The main pipeline does a stricter pass; this keeps the payload sane."""
    seen = {}
    out = []
    for o in objects:
        if not isinstance(o, dict):
            continue
        label = str(o.get("label", "")).strip().lower()
        if not label:
            continue
        if label in seen:
            seen[label]["expected_instances"] = "multiple"
            continue
        o = dict(o)
        o["label"] = label
        seen[label] = o
        out.append(o)
        if len(out) >= max_objects:
            break
    for i, o in enumerate(out, 1):
        o["id"] = i
    return out


def scale_bboxes(objects, scale_x: float, scale_y: float):
    """Map boxes from the resized image the model saw back to original pixels."""
    if scale_x == 1.0 and scale_y == 1.0:
        return objects
    for o in objects:
        b = o.get("bbox_2d")
        if isinstance(b, (list, tuple)) and len(b) == 4:
            try:
                o["bbox_2d"] = [float(b[0]) * scale_x, float(b[1]) * scale_y,
                                float(b[2]) * scale_x, float(b[3]) * scale_y]
            except (TypeError, ValueError):
                o["bbox_2d"] = None
    return objects


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.route("/health")
def health():
    return jsonify({"ok": _model is not None, "model": _model_name, "device": _current_device})


@app.route("/unload", methods=["POST"])
def unload():
    _to_device("cpu")
    return jsonify({"device": _current_device})


@app.route("/reload", methods=["POST"])
def reload_to_gpu():
    _to_device(DEVICE)
    return jsonify({"device": _current_device})


@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json(force=True) or {}
    image_path = data.get("image_path")
    if not image_path or not Path(image_path).exists():
        return jsonify({"error": f"image_path missing or not found: {image_path}"}), 400
    attempts = int(data.get("attempts", 3))

    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    if max(image.size) > MAX_IMAGE_SIDE:
        image = image.copy()
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    sx, sy = orig_w / image.width, orig_h / image.height

    _to_device(DEVICE)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    prompt = PROMPT + f"\n\nThe image is {image.width}x{image.height} pixels; bbox_2d uses those pixel coordinates."
    messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": prompt}]}]
    chat_text = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _processor(text=[chat_text], images=[image], padding=True, return_tensors="pt").to(_current_device)

    last_raw, last_err = "", None
    t0 = time.time()
    for attempt in range(attempts):
        gen_kwargs = dict(max_new_tokens=MAX_NEW_TOKENS, repetition_penalty=1.05)
        if attempt == 0:
            gen_kwargs.update(do_sample=False)
        else:
            gen_kwargs.update(do_sample=True, temperature=0.4, top_p=0.9, repetition_penalty=1.1)
        with torch.inference_mode():
            out_ids = _model.generate(**inputs, **gen_kwargs)
        gen_ids = out_ids[:, inputs.input_ids.shape[1]:]
        n_tokens = int(gen_ids.shape[1])
        last_raw = _processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        del out_ids, gen_ids
        try:
            objects = extract_json(last_raw)
            if not isinstance(objects, list):
                raise ValueError("not a list")
            objects = [o for o in objects if isinstance(o, dict) and o.get("label")]
            if not objects:
                last_err = "empty object list"
                print(f"  [attempt {attempt + 1}] empty list ({n_tokens} tokens) — retrying with sampling")
                continue
            objects = dedupe_objects(objects)
            objects = scale_bboxes(objects, sx, sy)
            print(f"  [attempt {attempt + 1}] parsed {len(objects)} objects "
                  f"({n_tokens} tokens, {time.time() - t0:.1f}s)")
            inputs = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return jsonify({"objects": objects, "model": _model_name,
                            "image_size": [orig_w, orig_h], "tokens": n_tokens})
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            last_err = str(e)
        print(f"  [attempt {attempt + 1}] parse failed: {last_err}")
        print(f"  raw[:300]: {last_raw[:300]!r}")

    inputs = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return jsonify({"error": f"Parse failed after {attempts} attempts: {last_err}", "raw": last_raw[:2000]}), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Any HF image-text-to-text model, e.g. Qwen/Qwen3-VL-8B-Instruct, "
                             "Qwen/Qwen2.5-VL-7B-Instruct, Qwen/Qwen3-VL-32B-Instruct-AWQ")
    args = parser.parse_args()
    load_model(args.model)
    print(f"\nServer ready at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=False)
