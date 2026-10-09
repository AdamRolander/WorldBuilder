"""WorldBuilder webapp server.

Run from the project root inside the `worldbuilder-main` conda env:

    python -m webapp.server

Exposes:
    GET  /                          index.html
    GET  /<file>                    static frontend files
    GET  /api/health                liveness
    GET  /api/vlm-status            proxies VLM server /health
    GET  /api/demos                 list of pre-baked scenes in outputs/
    POST /api/upload                accept image, queue pipeline job, return job_id
    GET  /api/jobs/<job_id>         job state (stage 1-5, status, scene_id)
    GET  /api/scenes/<scene>        scene manifest (objects, layout, files)
    GET  /api/scenes/<scene>/glb    the scene as one GLB (for Blender/Unreal/web)
    GET  /outputs/<scene>/...       static access to reconstruction outputs

Single in-process worker: only one pipeline job runs at a time (the GPU
can't be shared between SAM 3 and SAM 3D anyway). Job state is in memory
and pruned after ``JOB_TTL_SEC``; finished scenes on disk are always served.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests
from flask import Flask, abort, jsonify, request, send_from_directory
from flask_cors import CORS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_process_image = None   # imported lazily on the first job

STATIC_DIR = Path(__file__).resolve().parent / "static"
UPLOADS_DIR = PROJECT_ROOT / "webapp_uploads"
OUTPUTS_DIR = Path(os.environ.get("WORLDBUILDER_OUTPUTS", PROJECT_ROOT / "outputs")).resolve()
VLM_URL = os.environ.get("WORLDBUILDER_VLM_URL", "http://127.0.0.1:8765")
MAX_UPLOAD_MB = int(os.environ.get("WORLDBUILDER_MAX_UPLOAD_MB", "40"))
JOB_TTL_SEC = 6 * 3600

UPLOADS_DIR.mkdir(exist_ok=True, parents=True)
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
ALLOWED_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}

_jobs: dict = {}
_jobs_lock = threading.Lock()
_worker_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_child(base: Path, rel: str) -> Path:
    """Resolve ``rel`` under ``base`` or 404. ``str.startswith`` was the old
    check, which let ``outputs_og`` through as a child of ``outputs``."""
    candidate = (base / rel).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError:
        abort(403)
    if not candidate.exists():
        abort(404)
    return candidate


def _derive_stage(output_dir: Path) -> int:
    """Infer pipeline stage from artifacts on disk (frontend shows 1..5):
    1 detection · 2 segmentation · 3 reconstruction · 4 layout/placement · 5 viewer."""
    if not (output_dir / "detected_objects.json").exists():
        return 1
    if not (output_dir / "segmentation_results.json").exists():
        return 2
    if not (output_dir / "3d_models" / "room.ply").exists() and \
            not (output_dir / "reconstruction_results.json").exists():
        return 3
    if not (output_dir / "viewer.html").exists():
        return 4
    return 5


def _prune_jobs():
    cutoff = time.time() - JOB_TTL_SEC
    with _jobs_lock:
        for jid in [j for j, v in _jobs.items() if v.get("_t", 0) < cutoff
                    and v["status"] in ("complete", "failed")]:
            _jobs.pop(jid, None)


def _run_job(job_id: str, image_path: Path, scene_id: str, detector: str | None):
    global _process_image
    output_dir = OUTPUTS_DIR / scene_id
    with _worker_lock:
        try:
            if _process_image is None:
                from main import process_image as _pi
                _process_image = _pi
            with _jobs_lock:
                _jobs[job_id].update(status='running', stage=1)

            stop_poller = threading.Event()

            def poll_stage():
                while not stop_poller.is_set():
                    s = _derive_stage(output_dir)
                    with _jobs_lock:
                        if _jobs.get(job_id, {}).get('status') == 'running':
                            _jobs[job_id]['stage'] = s
                    time.sleep(1.0)

            t = threading.Thread(target=poll_stage, daemon=True)
            t.start()
            try:
                _process_image(str(image_path), str(output_dir), detector=detector)
            finally:
                stop_poller.set()
                t.join(timeout=2)
            with _jobs_lock:
                _jobs[job_id].update(status='complete', stage=5, finished_at=_now(), _t=time.time())
        except Exception as e:
            traceback.print_exc()
            msg = str(e)
            if 'No reconstructable objects' in msg or 'produced no masks' in msg:
                msg = "No reconstructable objects were found in the image. Try a clearer photo of a room."
            elif 'CUDA out of memory' in msg or 'OutOfMemoryError' in msg:
                msg = "GPU out of memory — restart the webapp server."
            elif 'VLM server unreachable' in msg:
                msg = "Local VLM server is not running (see README: vlm_server/server.py)."
            with _jobs_lock:
                _jobs[job_id].update(status='failed', error=msg[:500], finished_at=_now(), _t=time.time())
    _prune_jobs()


app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
CORS(app)


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/<path:filename>")
def static_file(filename):
    _safe_child(STATIC_DIR, filename)
    return send_from_directory(STATIC_DIR, filename)


@app.route("/outputs/<path:subpath>")
def serve_output(subpath):
    _safe_child(OUTPUTS_DIR, subpath)
    return send_from_directory(OUTPUTS_DIR, subpath)


@app.route("/api/health")
def health():
    return jsonify({"ok": True, "time": _now()})


@app.route("/api/vlm-status")
def vlm_status():
    try:
        r = requests.get(f"{VLM_URL}/health", timeout=2)
        r.raise_for_status()
        info = r.json()
        return jsonify({"ok": bool(info.get("ok")), "model": info.get("model"),
                        "device": info.get("device"), "url": VLM_URL})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:120], "url": VLM_URL})


@app.route("/api/demos")
def list_demos():
    demos = [{"scene_id": d.name} for d in sorted(OUTPUTS_DIR.iterdir())
             if d.is_dir() and (d / "viewer.html").exists()]
    return jsonify({"demos": demos})


@app.route("/api/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No 'image' field in form data"}), 400
    f = request.files["image"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported extension: {ext}"}), 400

    detector = request.form.get("detector") or os.environ.get("WORLDBUILDER_DETECTOR", "gemini")
    suffix = "q" if detector.lower() in ("qwen", "vlm", "local") else "g"
    job_id = uuid.uuid4().hex[:12]
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in Path(f.filename).stem)[:40] or "upload"
    scene_id = f"{stem}_{suffix}_{job_id[:6]}"
    saved_path = UPLOADS_DIR / f"{scene_id}{ext}"
    f.save(str(saved_path))

    with _jobs_lock:
        _jobs[job_id] = {"job_id": job_id, "scene_id": scene_id, "status": "queued", "stage": 0,
                         "error": None, "started_at": _now(), "finished_at": None,
                         "detector": detector, "_t": time.time()}
    threading.Thread(target=_run_job, args=(job_id, saved_path, scene_id, detector), daemon=True).start()
    return jsonify({"job_id": job_id, "scene_id": scene_id}), 202


@app.route("/api/jobs/<job_id>")
def job_status(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job:
        return jsonify({k: v for k, v in job.items() if not k.startswith("_")})
    # Demo seam: a scene id that already has a viewer reports complete.
    candidate = OUTPUTS_DIR / job_id
    if candidate.is_dir() and (candidate / "viewer.html").exists():
        return jsonify({"job_id": job_id, "scene_id": job_id, "status": "complete", "stage": 5,
                        "error": None, "started_at": None, "finished_at": None})
    return jsonify({"error": "Unknown job"}), 404


@app.route("/api/scenes/<scene_id>")
def scene_manifest(scene_id):
    d = _safe_child(OUTPUTS_DIR, scene_id)
    rec = d / "reconstruction_results.json"
    if not rec.exists():
        abort(404)
    data = json.loads(rec.read_text())
    layout_path = d / "3d_models" / "layout.json"
    layout = json.loads(layout_path.read_text()) if layout_path.exists() else None
    objects = [{"id": o["id"], "label": o["label"], "confidence": o.get("confidence"),
                "file": f"/outputs/{scene_id}/3d_models/{Path(o['ply_path']).name}",
                "supported_by": o.get("supported_by")} for o in data.get("objects", [])]
    return jsonify({"scene_id": scene_id, "objects": objects, "failed": data.get("failed", []),
                    "metadata": data.get("metadata", {}), "layout": layout,
                    "viewer": f"/outputs/{scene_id}/viewer.html",
                    "glb": f"/api/scenes/{scene_id}/glb" if (d / "scene.glb").exists() else None,
                    "room": f"/outputs/{scene_id}/3d_models/room.ply" if (d / "3d_models" / "room.ply").exists() else None})


@app.route("/api/scenes/<scene_id>/glb")
def scene_glb(scene_id):
    d = _safe_child(OUTPUTS_DIR, scene_id)
    glb = d / "scene.glb"
    if not glb.exists():
        # Build it on demand from the baked PLYs (fast, CPU only).
        try:
            from src.scene_export import export_scene_glb
            data = json.loads((d / "reconstruction_results.json").read_text())
            export_scene_glb(d, data["objects"])
        except Exception as e:
            return jsonify({"error": f"GLB export failed: {e}"}), 500
    return send_from_directory(d, "scene.glb", as_attachment=True,
                               download_name=f"{scene_id}.glb", mimetype="model/gltf-binary")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0", help="0.0.0.0 so a headset on the LAN can reach it")
    p.add_argument("--port", type=int, default=5174)
    args = p.parse_args()
    print(f"\n  Static dir: {STATIC_DIR}\n  Outputs:    {OUTPUTS_DIR}\n  VLM URL:    {VLM_URL}")
    print(f"\n  → http://{args.host}:{args.port}/\n")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)
