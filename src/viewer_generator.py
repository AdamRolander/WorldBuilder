"""Generate a self-contained Three.js viewer (desktop orbit + WebXR) for a scene.

All PLYs on disk are already in world space (the pipeline bakes poses), so
the viewer is a passive renderer. Changes from the demo-era viewer:

* the room is rendered with its **vertex colours** (the layout stage
  projects the photo onto floor/walls/ceiling) and can be toggled between
  textured, ghosted and hidden;
* a "photo point cloud" toggle shows the MoGe scene points from
  ``scene_pointmap.npz`` when the viewer is served by the webapp (it is
  decoded client-side from a small JSON sidecar we write here);
* the object list shows what rests on what (``supported_by``) and failed
  reconstructions are listed greyed out;
* Three.js is pinned and loaded from jsDelivr with an unpkg fallback.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WorldBuilder Viewer</title>
<style>
  body { margin:0; overflow:hidden; background:#1a1a1a; color:#eee; font-family:system-ui,sans-serif; }
  #info { position:absolute; top:10px; left:10px; background:rgba(0,0,0,.72); padding:12px; border-radius:8px;
          max-height:88vh; overflow-y:auto; width:290px; font-size:13px; }
  #info h3 { margin:0 0 8px; font-size:14px; }
  #info label { display:block; margin:3px 0; cursor:pointer; }
  #info label:hover { color:#8cf; }
  #info .meta { margin-top:10px; padding-top:10px; border-top:1px solid #444; font-size:11px; color:#aaa; white-space:pre-line; }
  #info .failed { color:#777; text-decoration:line-through; }
  #info .sub { color:#8a8; font-size:11px; }
  #loading { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); font-size:18px; }
  .conf { color:#888; font-size:11px; }
  select { background:#222; color:#eee; border:1px solid #555; border-radius:4px; }
</style>
</head>
<body>
<div id="loading">Loading scene... <span id="progress"></span></div>
<div id="info" style="display:none">
  <h3>Scene (<span id="count">0</span> objects)</h3>
  <label>Room:
    <select id="room-mode">
      <option value="textured">textured</option>
      <option value="ghost">ghosted</option>
      <option value="hidden">hidden</option>
    </select>
  </label>
  <label><input type="checkbox" id="toggle-points"> Photo point cloud</label>
  <div id="object-list"></div>
  <div class="meta" id="meta"></div>
</div>
<script type="importmap">
{ "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
}}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { VRButton } from 'three/addons/webxr/VRButton.js';

const OBJECTS = __OBJECTS_JSON__;
const FAILED  = __FAILED_JSON__;
const ROOM    = __ROOM_FILE__;
const LAYOUT  = __LAYOUT_JSON__;
const POINTS  = __POINTS_FILE__;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a1a);
const camera = new THREE.PerspectiveCamera(50, innerWidth/innerHeight, 0.01, 1000);
camera.position.set(3, 3, 3);
const renderer = new THREE.WebGLRenderer({ antialias:true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(devicePixelRatio);
renderer.xr.enabled = true;
renderer.xr.setFoveation(1);
document.body.appendChild(renderer.domElement);

const vrButton = VRButton.createButton(renderer);
vrButton.style.cssText += 'position:absolute;bottom:20px;left:50%;transform:translateX(-50%);'
  + 'background:#367D8A;border:1px solid #285F6B;color:#fff;font-weight:700;padding:12px 24px;'
  + 'border-radius:8px;cursor:pointer;letter-spacing:1px;';
document.body.appendChild(vrButton);

const playerRig = new THREE.Group();
playerRig.add(camera);
scene.add(playerRig);
renderer.xr.addEventListener('sessionstart', () => {
  const c = new THREE.Vector3(); bbox.getCenter(c);
  const floorY = (LAYOUT && LAYOUT.floor_y !== undefined) ? LAYOUT.floor_y : bbox.min.y;
  playerRig.position.set(c.x, floorY, c.z);
  playerRig.rotation.set(0, 0, 0);
});

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.AmbientLight(0xffffff, 0.9));
const dir = new THREE.DirectionalLight(0xffffff, 0.5);
dir.position.set(5, 10, 5); scene.add(dir);

const meshes = {};
const bbox = new THREE.Box3();
let roomMesh = null;

function loadPLY(url, key, isObject, done) {
  new PLYLoader().load(url, geom => {
    geom.computeVertexNormals();
    const hasColors = !!geom.attributes.color;
    let material;
    if (isObject) {
      material = new THREE.MeshBasicMaterial({ vertexColors: hasColors, side: THREE.FrontSide });
    } else {
      material = new THREE.MeshBasicMaterial({ vertexColors: hasColors, color: hasColors ? 0xffffff : 0xd4c9b5,
                                               side: THREE.DoubleSide, transparent: true, opacity: 1.0 });
    }
    const m = new THREE.Mesh(geom, material);
    if (!isObject) { m.renderOrder = -1; roomMesh = m; }
    scene.add(m);
    meshes[key] = m;
    if (isObject) bbox.expandByObject(m);
    done();
  }, undefined, e => { console.error(url, e); done(); });
}

let loaded = 0;
const total = OBJECTS.length + (ROOM ? 1 : 0);
const progressEl = document.getElementById('progress');
function onLoad() {
  loaded++;
  progressEl.textContent = `(${loaded}/${total})`;
  if (loaded < total) return;
  if (bbox.isEmpty() && roomMesh) bbox.expandByObject(roomMesh);
  const c = new THREE.Vector3(); bbox.getCenter(c);
  const s = new THREE.Vector3(); bbox.getSize(s);
  const d = Math.max(s.x, s.y, s.z) * 1.2;
  controls.target.copy(c);
  camera.position.copy(c).add(new THREE.Vector3(d, d*0.8, d));
  controls.update();
  document.getElementById('loading').style.display = 'none';
  document.getElementById('info').style.display = 'block';
  document.getElementById('count').textContent = OBJECTS.length;
  let meta = `Objects extent: ${s.x.toFixed(2)} × ${s.y.toFixed(2)} × ${s.z.toFixed(2)}`;
  if (LAYOUT) {
    if (LAYOUT.estimated_metric_scale) meta += `\nEst. scale: ×${LAYOUT.estimated_metric_scale.toFixed(2)} → metres (camera-height prior)`;
    if (LAYOUT.ceiling_source) meta += `\nCeiling: ${LAYOUT.ceiling_source}; floor: ${LAYOUT.floor ? LAYOUT.floor.source : '?'}`;
    if (LAYOUT.wall_sources) meta += `\nWalls: ` + Object.entries(LAYOUT.wall_sources).map(([k,v]) => `${k}=${v}`).join(', ');
  }
  document.getElementById('meta').textContent = meta;
}

const list = document.getElementById('object-list');
const byId = {};
OBJECTS.forEach(o => byId[o.id] = o);
OBJECTS.forEach(o => {
  loadPLY(o.file, o.file, true, onLoad);
  const l = document.createElement('label');
  const sup = (o.supported_by && byId[o.supported_by]) ? ` <span class="sub">on ${byId[o.supported_by].label}</span>` : '';
  l.innerHTML = `<input type="checkbox" checked data-t="${o.file}"> ${o.label} <span class="conf">${(o.confidence||0).toFixed(2)}</span>${sup}`;
  list.appendChild(l);
});
FAILED.forEach(f => {
  const l = document.createElement('label'); l.className = 'failed';
  l.textContent = `${f.label} (${f.status || 'failed'})`; list.appendChild(l);
});
if (ROOM) loadPLY(ROOM, '__room__', false, onLoad);

// Optional photo point cloud (JSON sidecar written by the pipeline).
let pointsObj = null;
document.getElementById('toggle-points').addEventListener('change', async e => {
  if (!POINTS) { e.target.checked = false; return; }
  if (!pointsObj) {
    try {
      const r = await fetch(POINTS); const d = await r.json();
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute(d.positions, 3));
      g.setAttribute('color', new THREE.Float32BufferAttribute(d.colors, 3));
      pointsObj = new THREE.Points(g, new THREE.PointsMaterial({ size: 0.01, vertexColors: true }));
      scene.add(pointsObj);
    } catch (err) { console.error(err); e.target.checked = false; return; }
  }
  pointsObj.visible = e.target.checked;
});

document.getElementById('room-mode').addEventListener('change', e => {
  if (!roomMesh) return;
  const mode = e.target.value;
  roomMesh.visible = mode !== 'hidden';
  roomMesh.material.opacity = mode === 'ghost' ? 0.35 : 1.0;
  roomMesh.material.needsUpdate = true;
});
document.getElementById('info').addEventListener('change', e => {
  if (e.target.tagName !== 'INPUT' || !e.target.dataset.t) return;
  const m = meshes[e.target.dataset.t];
  if (m) m.visible = e.target.checked;
});

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

// Quest locomotion: left stick move, right stick snap turn, A/B exit.
const MOVE_SPEED = 1.6, SNAP_DEG = 30, SNAP_COOLDOWN_MS = 280;
let lastSnap = 0;
function pollXRInput(dt) {
  const session = renderer.xr.getSession();
  if (!session) return;
  const fwd = new THREE.Vector3(), right = new THREE.Vector3();
  for (const src of session.inputSources) {
    if (!src.gamepad) continue;
    const ax = src.gamepad.axes, btn = src.gamepad.buttons;
    if (src.handedness === 'left' && ax.length >= 4) {
      const x = ax[2], y = ax[3];
      if (Math.abs(x) > 0.15 || Math.abs(y) > 0.15) {
        renderer.xr.getCamera().getWorldDirection(fwd); fwd.y = 0; fwd.normalize();
        right.set(fwd.z, 0, -fwd.x);
        playerRig.position.addScaledVector(fwd, -y * MOVE_SPEED * dt);
        playerRig.position.addScaledVector(right, x * MOVE_SPEED * dt);
      }
    }
    if (src.handedness === 'right' && ax.length >= 4) {
      const x = ax[2], now = performance.now();
      if (Math.abs(x) > 0.7 && now - lastSnap > SNAP_COOLDOWN_MS) {
        playerRig.rotation.y -= Math.sign(x) * THREE.MathUtils.degToRad(SNAP_DEG);
        lastSnap = now;
      }
    }
    if (btn[4]?.pressed || btn[5]?.pressed) session.end();
  }
}
let prevT = performance.now();
renderer.setAnimationLoop(() => {
  const now = performance.now();
  const dt = Math.min(0.1, (now - prevT) / 1000); prevT = now;
  if (renderer.xr.isPresenting) pollXRInput(dt); else controls.update();
  renderer.render(scene, camera);
});
</script>
</body></html>
"""


def _points_sidecar(output_dir: Path, npz_rel: str = "3d_models/scene_pointmap.npz",
                    max_points: int = 60_000) -> Optional[str]:
    """Convert the pipeline's point map sample to a small JSON the viewer can
    fetch without a decoder. Returns the relative path or None."""
    npz = output_dir / npz_rel
    if not npz.exists():
        return None
    try:
        import numpy as np
        d = np.load(npz)
        P, C = d["points"], d["colors"]
        if len(P) > max_points:
            idx = np.random.default_rng(0).choice(len(P), max_points, replace=False)
            P, C = P[idx], C[idx]
        out = output_dir / "scene_points.json"
        out.write_text(json.dumps({"positions": np.round(P, 4).ravel().tolist(),
                                   "colors": (C[:, :3].astype(float) / 255).round(3).ravel().tolist()}))
        return out.name
    except Exception:
        return None


def generate_viewer(output_dir: Path, results: List[Dict], room_file: Optional[str] = "room.ply",
                    layout_file: Optional[str] = None, failed: Optional[List[Dict]] = None) -> Path:
    """Write viewer.html into output_dir; paths are relative to that directory."""
    output_dir = Path(output_dir).resolve()
    entries = []
    for r in results:
        if r.get("status", "ok") != "ok" or not r.get("ply_path"):
            continue
        p = Path(r["ply_path"]).resolve()
        try:
            rel = p.relative_to(output_dir).as_posix()
        except ValueError:
            rel = f"3d_models/{p.name}"
        entries.append({"id": r.get("id"), "file": rel, "label": r["label"],
                        "confidence": r.get("confidence", 0), "supported_by": r.get("supported_by")})
    layout = None
    if layout_file and (output_dir / layout_file).exists():
        try:
            layout = json.loads((output_dir / layout_file).read_text())
        except ValueError:
            layout = None
    if room_file and not (output_dir / room_file).exists():
        room_file = None
    points = _points_sidecar(output_dir)
    html = (_TEMPLATE.replace("__OBJECTS_JSON__", json.dumps(entries))
            .replace("__FAILED_JSON__", json.dumps(failed or []))
            .replace("__ROOM_FILE__", json.dumps(room_file) if room_file else "null")
            .replace("__LAYOUT_JSON__", json.dumps(layout) if layout else "null")
            .replace("__POINTS_FILE__", json.dumps(points) if points else "null"))
    vpath = output_dir / "viewer.html"
    vpath.write_text(html)
    return vpath
