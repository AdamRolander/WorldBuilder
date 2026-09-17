"""Generate a self-contained Three.js viewer for a reconstructed scene."""
import json
from pathlib import Path
from typing import List, Dict, Optional

_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>WorldBuilder Viewer</title>
<style>
  body { margin:0; overflow:hidden; background:#1a1a1a; color:#eee;
         font-family:system-ui,sans-serif; }
  #info { position:absolute; top:10px; left:10px; background:rgba(0,0,0,.7);
          padding:12px; border-radius:6px; max-height:88vh; overflow-y:auto;
          width:280px; font-size:13px; }
  #info h3 { margin:0 0 8px; font-size:14px; }
  #info label { display:block; margin:3px 0; cursor:pointer; }
  #info label:hover { color:#8cf; }
  #info .meta { margin-top:10px; padding-top:10px; border-top:1px solid #444;
                font-size:11px; color:#aaa; }
  #loading { position:absolute; top:50%; left:50%;
             transform:translate(-50%,-50%); font-size:18px; }
  .conf { color:#888; font-size:11px; }
</style>
</head>
<body>
<div id="loading">Loading scene... <span id="progress"></span></div>
<div id="info" style="display:none">
  <h3>Scene (<span id="count">0</span> objects)</h3>
  <label><input type="checkbox" id="toggle-room" checked> Room (floor/walls)</label>
  <div id="object-list"></div>
  <div class="meta" id="meta"></div>
</div>
<script type="importmap">
{ "imports": {
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
}}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { VRButton } from 'three/addons/webxr/VRButton.js';

const OBJECTS = __OBJECTS_JSON__;
const ROOM    = __ROOM_FILE__;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a1a);
const camera = new THREE.PerspectiveCamera(50, innerWidth/innerHeight, 0.01, 1000);
camera.position.set(3, 3, 3);
const renderer = new THREE.WebGLRenderer({ antialias:true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(devicePixelRatio);
renderer.xr.enabled = true;
renderer.xr.setFoveation(1); // aggressive fixed foveation — big Quest perf win
document.body.appendChild(renderer.domElement);

const vrButton = VRButton.createButton(renderer);
vrButton.style.cssText += 'position:absolute;bottom:20px;left:50%;'
  + 'transform:translateX(-50%);background:#367D8A;border:1px solid #285F6B;'
  + 'color:#fff;font-family:system-ui,sans-serif;font-weight:700;'
  + 'padding:12px 24px;border-radius:8px;cursor:pointer;letter-spacing:1px;';
document.body.appendChild(vrButton);

// Player rig: the camera lives inside this group, so moving the rig
// moves the user. WebXR overwrites the camera's local transform from
// the headset pose every frame, but the rig's transform is ours.
// This is what thumbstick locomotion translates and what we position
// at session start so the user spawns inside the room.
const playerRig = new THREE.Group();
playerRig.add(camera);
scene.add(playerRig);

// On session start, drop the user near the floor at the scene center.
// Standing reference space puts origin at the user's feet, so the
// headset adds the eye-height offset itself — don't add 1.6m here.
renderer.xr.addEventListener('sessionstart', () => {
  const c = new THREE.Vector3(); bbox.getCenter(c);
  playerRig.position.set(c.x, bbox.min.y, c.z);
  playerRig.rotation.set(0, 0, 0);
});

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.AmbientLight(0xffffff, 0.85));
const dir = new THREE.DirectionalLight(0xffffff, 0.6);
dir.position.set(5, 10, 5); scene.add(dir);

const meshes = {};
const bbox = new THREE.Box3();

// All PLYs — objects and room — are already in world space.
// The viewer is a passive renderer; no transforms.
function loadPLY(url, key, isObject, done) {
  new PLYLoader().load(url, geom => {
    geom.computeVertexNormals();
    const material = isObject
      ? new THREE.MeshBasicMaterial({
          vertexColors: true,
          side: THREE.FrontSide
        })
      : new THREE.MeshStandardMaterial({
          color: 0xd4c9b5, roughness: 0.9, metalness: 0,
          side: THREE.DoubleSide, transparent: true, opacity: 0.35
        });
    const m = new THREE.Mesh(geom, material);
    if (!isObject) m.renderOrder = -1;
    scene.add(m);
    meshes[key] = m;
    bbox.expandByObject(m);
    const tris = (geom.index ? geom.index.count : geom.attributes.position.count) / 3;
    console.log(`loaded ${key}: ${tris|0} triangles`);
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
  const c = new THREE.Vector3(); bbox.getCenter(c);
  const s = new THREE.Vector3(); bbox.getSize(s);
  const d = Math.max(s.x, s.y, s.z) * 1.2;
  controls.target.copy(c);
  camera.position.copy(c).add(new THREE.Vector3(d, d*0.8, d));
  controls.update();
  document.getElementById('loading').style.display = 'none';
  document.getElementById('info').style.display = 'block';
  document.getElementById('count').textContent = OBJECTS.length;
  document.getElementById('meta').textContent =
    `Bounds: ${s.x.toFixed(2)} × ${s.y.toFixed(2)} × ${s.z.toFixed(2)}`;
}

const list = document.getElementById('object-list');
OBJECTS.forEach(o => {
  loadPLY(o.file, o.file, true, onLoad);
  const l = document.createElement('label');
  l.innerHTML = `<input type="checkbox" checked data-t="${o.file}"> `
              + `${o.label} <span class="conf">${(o.confidence||0).toFixed(2)}</span>`;
  list.appendChild(l);
});
if (ROOM) loadPLY(ROOM, '__room__', false, onLoad);

document.getElementById('info').addEventListener('change', e => {
  if (e.target.tagName !== 'INPUT') return;
  if (e.target.id === 'toggle-room') {
    if (meshes['__room__']) meshes['__room__'].visible = e.target.checked;
  } else if (e.target.dataset.t) {
    const m = meshes[e.target.dataset.t];
    if (m) m.visible = e.target.checked;
  }
});

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
// Locomotion + exit input. Standard Quest button indices (Meta's
// xr-standard mapping): trigger=0, grip=1, stick-press=3, A/X=4, B/Y=5.
// Axes: [0,1] = (unused) touchpad, [2,3] = thumbstick X/Y.
const MOVE_SPEED  = 1.6;            // m/sec — comfortable walking pace
const SNAP_DEG    = 30;             // snap turn step
const SNAP_COOLDOWN_MS = 280;       // prevent retriggering on a held stick
let lastSnap = 0;

function pollXRInput(dt) {
  const session = renderer.xr.getSession();
  if (!session) return;
  const tmpFwd   = new THREE.Vector3();
  const tmpRight = new THREE.Vector3();

  for (const src of session.inputSources) {
    if (!src.gamepad) continue;
    const ax  = src.gamepad.axes;
    const btn = src.gamepad.buttons;

    // Left stick → smooth forward/strafe in headset's facing direction
    if (src.handedness === 'left' && ax.length >= 4) {
      const x = ax[2], y = ax[3];
      if (Math.abs(x) > 0.15 || Math.abs(y) > 0.15) {
        renderer.xr.getCamera().getWorldDirection(tmpFwd);
        tmpFwd.y = 0; tmpFwd.normalize();
        tmpRight.set(tmpFwd.z, 0, -tmpFwd.x);
        playerRig.position.addScaledVector(tmpFwd,   -y * MOVE_SPEED * dt);
        playerRig.position.addScaledVector(tmpRight,  x * MOVE_SPEED * dt);
      }
    }

    // Right stick → snap turn
    if (src.handedness === 'right' && ax.length >= 4) {
      const x = ax[2];
      const now = performance.now();
      if (Math.abs(x) > 0.7 && now - lastSnap > SNAP_COOLDOWN_MS) {
        playerRig.rotation.y -= Math.sign(x) * THREE.MathUtils.degToRad(SNAP_DEG);
        lastSnap = now;
      }
    }

    // A/X or B/Y on either controller → exit VR
    if (btn[4]?.pressed || btn[5]?.pressed) session.end();
  }
}

let prevT = performance.now();
renderer.setAnimationLoop(() => {
  const now = performance.now();
  const dt = Math.min(0.1, (now - prevT) / 1000);
  prevT = now;

  if (renderer.xr.isPresenting) pollXRInput(dt);
  else                          controls.update();

  renderer.render(scene, camera);
});
</script>
</body></html>
"""

def generate_viewer(output_dir: Path,
                    results: List[Dict],
                    room_file: Optional[str] = "room.ply") -> Path:
    """Write viewer.html into output_dir; paths relative to that directory.

    Expects object PLYs to be in world space on disk (done by the pipeline's
    _bake_world_space_plys step). Viewer does no transform math.
    """
    output_dir = Path(output_dir).resolve()
    entries = []
    for r in results:
        # Always resolve before relative_to — ply_path is absolute when the
        # webapp generated the scene, relative when main.py was run directly.
        # Mixing the two trips ValueError and falls back to bare filenames.
        p = Path(r['ply_path']).resolve()
        try:
            rel = p.relative_to(output_dir).as_posix()
        except ValueError:
            rel = p.name
        entries.append({
            'file': rel,
            'label': r['label'],
            'confidence': r.get('confidence', 0),
        })
    html = _TEMPLATE.replace('__OBJECTS_JSON__', json.dumps(entries))
    html = html.replace('__ROOM_FILE__',
                        json.dumps(room_file) if room_file else 'null')
    vpath = output_dir / 'viewer.html'
    vpath.write_text(html)
    return vpath