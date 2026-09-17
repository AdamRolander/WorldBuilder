"""WorldBuilder webapp server.

Run from the project root inside the `worldbuilder-main` conda env:

    python -m webapp.server

Exposes:
    GET  /                       index.html
    GET  /<file>                 static frontend files
    GET  /api/health             liveness
    GET  /api/vlm-status         proxies VLM server /health
    GET  /api/demos              list of pre-baked scenes in outputs/
    POST /api/upload             accept image, queue pipeline job, return job_id
    GET  /api/jobs/<job_id>      job state (stage 1-5, status, scene_id)
    GET  /outputs/<scene>/...    static access to reconstruction outputs

Single in-process worker — only one pipeline job runs at a time (the GPU
can't share between SAM 3 and SAM 3D anyway). Job state is in-memory; if
the server crashes mid-job, the user re-uploads.
"""
import os
import sys
import time
import uuid
import threading
import traceback
from pathlib import Path
from datetime import datetime

import requests
from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS

# Resolve project root so we can import the pipeline whether the server is
# launched as a module or as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Defer importing main.process_image until first job — startup is faster
# and a typo in the pipeline doesn't kill the whole server.
_process_image = None

# Config -------------------------------------------------------------------

STATIC_DIR  = Path(__file__).resolve().parent / "static"
UPLOADS_DIR = PROJECT_ROOT / "webapp_uploads"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
VLM_URL     = os.environ.get("WORLDBUILDER_VLM_URL", "http://127.0.0.1:8765")

UPLOADS_DIR.mkdir(exist_ok=True, parents=True)
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)

ALLOWED_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}

# Job state ----------------------------------------------------------------

# job_id -> { status, stage, scene_id, error, started_at, finished_at }
# status: 'queued' | 'running' | 'complete' | 'failed'
# stage:  1..5 mirrors the frontend's progress steps
_jobs = {}
_jobs_lock = threading.Lock()
_worker_lock = threading.Lock()   # ensures only one job runs at a time


def _derive_stage(output_dir: Path) -> int:
    """Infer current pipeline stage from which artifacts exist on disk.

    Frontend stages (matches app.jsx labels):
      1 VLM scene analysis        — running until detected_objects.json
      2 SAM 3 segmentation        — running until segmentation_results.json
      3 3D asset diffusion        — running until reconstruction_results.json
      4 Pose & quaternion         — same artifact, brief stage advertised
                                    while post-processing kicks off
      5 Heuristic refinement      — until viewer.html appears (= done)
    """
    if not output_dir.exists():
        return 1
    if not (output_dir / "detected_objects.json").exists():       return 1
    if not (output_dir / "segmentation_results.json").exists():   return 2
    if not (output_dir / "reconstruction_results.json").exists(): return 3
    if not (output_dir / "viewer.html").exists():                 return 5  # post-proc
    return 5


def _run_job(job_id: str, image_path: Path, scene_id: str):
    """Worker thread: run the pipeline, mutate _jobs as we go."""
    global _process_image
    output_dir = OUTPUTS_DIR / scene_id

    with _worker_lock:
        try:
            if _process_image is None:
                from main import process_image as _pi
                _process_image = _pi

            with _jobs_lock:
                _jobs[job_id]['status'] = 'running'
                _jobs[job_id]['stage']  = 1

            # Background stage poller — re-reads disk so the API endpoint
            # reflects real progress without us having to instrument the
            # pipeline with callbacks.
            stop_poller = threading.Event()
            def poll_stage():
                while not stop_poller.is_set():
                    s = _derive_stage(output_dir)
                    with _jobs_lock:
                        if _jobs[job_id]['status'] == 'running':
                            _jobs[job_id]['stage'] = s
                    time.sleep(1.0)
            t = threading.Thread(target=poll_stage, daemon=True)
            t.start()

            try:
                _process_image(str(image_path), str(output_dir))
            finally:
                stop_poller.set()
                t.join(timeout=2)

            with _jobs_lock:
                _jobs[job_id].update(
                    status='complete', stage=5,
                    finished_at=datetime.utcnow().isoformat(),
                )

        except Exception as e:
            traceback.print_exc()
            msg = str(e)
            # Friendly rewrite for the two failures most likely on demo day.
            if 'No complete object in output' in msg or 'objects": []' in msg:
                msg = "VLM found no objects in the image. Try a clearer indoor scene."
            elif 'CUDA out of memory' in msg or 'OutOfMemoryError' in msg:
                msg = "GPU out of memory — restart the webapp server."
            with _jobs_lock:
                _jobs[job_id].update(
                    status='failed', error=msg[:500],
                    finished_at=datetime.utcnow().isoformat(),
                )


# Flask app ----------------------------------------------------------------

app = Flask(__name__, static_folder=None)
CORS(app)


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/<path:filename>")
def static_file(filename):
    # Only serve files inside static/ — prevents directory traversal.
    candidate = (STATIC_DIR / filename).resolve()
    if not str(candidate).startswith(str(STATIC_DIR.resolve())):
        abort(403)
    if not candidate.exists():
        abort(404)
    return send_from_directory(STATIC_DIR, filename)


@app.route("/outputs/<path:subpath>")
def serve_output(subpath):
    candidate = (OUTPUTS_DIR / subpath).resolve()
    if not str(candidate).startswith(str(OUTPUTS_DIR.resolve())):
        abort(403)
    if not candidate.exists():
        abort(404)
    return send_from_directory(OUTPUTS_DIR, subpath)


@app.route("/api/health")
def health():
    return jsonify({"ok": True, "time": datetime.utcnow().isoformat()})


@app.route("/api/vlm-status")
def vlm_status():
    """Proxies the VLM server's /health. Returns ok=false if unreachable."""
    try:
        r = requests.get(f"{VLM_URL}/health", timeout=2)
        r.raise_for_status()
        info = r.json()
        return jsonify({
            "ok": bool(info.get("ok")),
            "model": info.get("model"),
            "device": info.get("device"),
            "url": VLM_URL,
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)[:120], "url": VLM_URL})


@app.route("/api/demos")
def list_demos():
    """List pre-completed scenes in outputs/ (anything with a viewer.html)."""
    demos = []
    if OUTPUTS_DIR.exists():
        for d in sorted(OUTPUTS_DIR.iterdir()):
            if d.is_dir() and (d / "viewer.html").exists():
                demos.append({"scene_id": d.name})
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

    # Honor the user's chosen detector. Webapp doesn't override.
    detector = os.environ.get("WORLDBUILDER_DETECTOR", "gemini").lower()
    suffix = "q" if detector in ("qwen", "vlm", "local") else "g"

    job_id = uuid.uuid4().hex[:12]
    stem = Path(f.filename).stem.replace(" ", "_")
    # Append job_id so two uploads of the same filename don't collide.
    scene_id = f"{stem}_{suffix}_{job_id[:6]}"
    saved_path = UPLOADS_DIR / f"{scene_id}{ext}"
    f.save(str(saved_path))

    with _jobs_lock:
        _jobs[job_id] = {
            "job_id":     job_id,
            "scene_id":   scene_id,
            "status":     "queued",
            "stage":      0,
            "error":      None,
            "started_at": datetime.utcnow().isoformat(),
            "finished_at": None,
        }

    threading.Thread(
        target=_run_job, args=(job_id, saved_path, scene_id), daemon=True
    ).start()

    return jsonify({"job_id": job_id, "scene_id": scene_id}), 202


@app.route("/api/jobs/<job_id>")
def job_status(job_id):
    # Special-case: if the job_id doesn't exist but matches an existing scene
    # directory with a viewer, return it as already-complete. This is the
    # seam for the demo fallback — frontend can pass a known scene_id as
    # job_id and immediately get back "complete".
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job:
        return jsonify(job)

    candidate = OUTPUTS_DIR / job_id
    if candidate.is_dir() and (candidate / "viewer.html").exists():
        return jsonify({
            "job_id": job_id, "scene_id": job_id,
            "status": "complete", "stage": 5,
            "error": None, "started_at": None, "finished_at": None,
        })
    return jsonify({"error": "Unknown job"}), 404


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0",
                   help="0.0.0.0 so the Quest browser on the LAN can reach it")
    p.add_argument("--port", type=int, default=5174)
    args = p.parse_args()
    print(f"\n  Static dir: {STATIC_DIR}")
    print(f"  Outputs:    {OUTPUTS_DIR}")
    print(f"  VLM URL:    {VLM_URL}")
    print(f"\n  → http://{args.host}:{args.port}/\n")
    app.run(host=args.host, port=args.port, threaded=True, debug=False)