"""Generate a self-contained Three.js viewer (desktop orbit + WebXR) for a scene.

All PLYs on disk are already in world space (the pipeline bakes poses), so
the viewer is a passive renderer. Changes from the demo-era viewer:

* the room is loaded from ``room.glb`` (textured quads + the photo relief
  of built-ins, see ``src/room_texture.py``) with the vertex-coloured
  ``room.ply`` as fallback, and can be toggled between textured, ghosted
  and hidden; the relief has its own toggle;
* a "photo point cloud" toggle shows the MoGe scene points from
  ``scene_pointmap.npz`` when the viewer is served by the webapp (it is
  decoded client-side from a small JSON sidecar we write here);
* the object list shows what rests on what (``supported_by``) and failed
  reconstructions are listed greyed out;
* Three.js is pinned and loaded from jsDelivr with an unpkg fallback.
* objects, built-ins and the room box are editable (move, turn, resize,
  trim, copy, drop onto what is below, push against a wall); the edits are
  saved to edits.json (src/scene_edits.py) through the web app's API,
  into the scene folder picked once (Chrome, Edge), or as a download.
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
  .panel { position:absolute; background:rgba(0,0,0,.74); padding:12px; border-radius:8px; font-size:13px; }
  #info { top:10px; left:10px; max-height:88vh; overflow-y:auto; width:290px; }
  #info h3, #edit h3, #roomedit h3 { margin:0 0 8px; font-size:14px; }
  .handle { cursor:move; user-select:none; display:flex; align-items:center; justify-content:space-between; gap:8px; }
  .handle button.tiny { padding:0 7px; line-height:18px; font-size:14px; }
  #info.collapsed > :not(h3) { display:none !important; }
  #info.collapsed h3 { margin:0; }
  #info h4 { margin:10px 0 4px; font-size:12px; color:#aaa; font-weight:600; text-transform:uppercase; letter-spacing:.04em; }
  .row { display:flex; align-items:center; gap:6px; padding:2px 4px; border-radius:4px; cursor:pointer; }
  .row:hover, .row.hot { background:rgba(140,200,255,.18); }
  .row.sel { background:rgba(255,200,80,.25); }
  .row .name { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .row.edited .name::after { content:" •"; color:#fc5; }
  #info label { display:block; margin:3px 0; cursor:pointer; }
  #info .roomline { display:flex; align-items:center; gap:8px; }
  #info .meta { margin-top:10px; padding-top:10px; border-top:1px solid #444; font-size:11px; color:#aaa; white-space:pre-line; }
  #info .failed { color:#777; text-decoration:line-through; padding:2px 4px; }
  .sub { color:#8a8; font-size:11px; }
  .conf { color:#888; font-size:11px; }
  #loading { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); font-size:18px; }
  select, input[type=number], button { background:#222; color:#eee; border:1px solid #555; border-radius:4px; font:inherit; }
  button { padding:4px 10px; cursor:pointer; }
  button:hover { border-color:#8cf; }
  button.primary { background:#367D8A; border-color:#285F6B; font-weight:600; }
  button.on { background:#6b5a1e; border-color:#fc5; }
  #edit, #roomedit { top:10px; right:10px; width:340px; display:none; }
  .grid { display:grid; grid-template-columns:70px repeat(3, 1fr); gap:5px 6px; align-items:center; margin-bottom:6px; }
  .grid .lbl { font-size:12px; }
  input[type=number] { width:100%; box-sizing:border-box; padding:4px 5px; -moz-appearance:textfield; appearance:textfield; }
  input[type=number]::-webkit-inner-spin-button, input[type=number]::-webkit-outer-spin-button { -webkit-appearance:none; margin:0; }
  input[type=number]:hover { border-color:#777; }
  input[type=number]:focus { border-color:#8cf; outline:none; background:#1b2830; }
  input[type=number]::placeholder { color:#999; font-style:italic; }
  .num { display:flex; align-items:stretch; gap:3px; min-width:0; }
  .num input { flex:1; min-width:0; }
  .spin { display:flex; flex-direction:column; gap:3px; }
  .spin button { flex:1; padding:0 4px; min-height:11px; font-size:8px; line-height:1; border-radius:3px; color:#bbb; }
  .spin button:hover { background:#367D8A; border-color:#8cf; color:#fff; }
  .spin button:active { background:#8cf; color:#000; }
  .axis { color:#aaa; font-size:11px; text-align:center; font-weight:600; }
  .axis.x { color:#ff6b6b; } .axis.y { color:#6ddc6d; } .axis.z { color:#6fa8ff; }
  #edit .mode { cursor:pointer; border-radius:3px; padding:1px 2px; }
  #edit .mode:hover { background:rgba(255,255,255,.08); }
  #edit .mode.on { background:rgba(102,204,255,.22); }
  .hint { color:#888; font-size:11px; margin-top:6px; }
  .btns { display:flex; gap:6px; flex-wrap:wrap; margin-top:8px; }
  .dim { color:#aaa; font-size:11px; margin:4px 0; }
  #bar { bottom:12px; right:12px; display:none; gap:8px; align-items:center; }
  #status { color:#aaa; font-size:12px; }
  #status.dirty { color:#fc5; }
  #prompt { top:10px; left:50%; transform:translateX(-50%); display:none; color:#fc5; font-weight:600; }
</style>
</head>
<body>
<div id="loading">Loading scene... <span id="progress"></span></div>
<div id="info" class="panel" style="display:none">
  <h3 class="handle" title="Drag to move; double-click to put it back"><span>Scene (<span id="count">0</span> objects)</span><button id="btn-collapse" class="tiny" type="button" title="Collapse or expand the list">&minus;</button></h3>
  <div class="roomline">
    <label>Room:
      <select id="room-mode">
        <option value="textured" selected>textured</option>
        <option value="ghost">ghosted</option>
        <option value="hidden">hidden</option>
      </select>
    </label>
    <button id="btn-room" type="button" style="display:none" title="Move the walls, floor and ceiling">Resize</button>
  </div>
  <label><input type="checkbox" id="toggle-points"> Photo point cloud</label>
  <h4>Objects</h4>
  <div id="object-list"></div>
  <div id="builtin-section" style="display:none">
    <h4><label style="display:inline"><input type="checkbox" id="toggle-relief" checked> Built-ins</label></h4>
    <div id="builtin-list"></div>
  </div>
  <div class="meta" id="meta"></div>
</div>
<div id="edit" class="panel">
  <h3 class="handle" title="Drag to move; double-click to put it back"><span id="edit-title">Object</span></h3>
  <div class="dim" id="edit-dim"></div>
  <div class="grid">
    <span></span><span class="axis x">X</span><span class="axis y">Y (up)</span><span class="axis z">Z</span>
    <span class="mode lbl on" data-mode="p" title="Show which way +X, +Y and +Z move the object">Position</span><input type="number" id="px" step="0.01"><input type="number" id="py" step="0.01"><input type="number" id="pz" step="0.01">
    <span class="mode lbl" data-mode="r" title="Show which way a positive angle turns the object">Rotation°</span><input type="number" id="rx" step="1"><input type="number" id="ry" step="1"><input type="number" id="rz" step="1">
    <span class="mode lbl" data-mode="s" title="Show the scale axes">Scale</span><input type="number" id="sx" step="0.05" min="0.01"><input type="number" id="sy" step="0.05" min="0.01"><input type="number" id="sz" step="0.05" min="0.01">
    <span class="lbl trim" title="Percent of the object cut away from its −X side, its bottom and its −Z side (in the object's own frame)">Trim − %</span><input type="number" class="trim" id="t0x" step="1" min="0" max="95"><input type="number" class="trim" id="t0y" step="1" min="0" max="95"><input type="number" class="trim" id="t0z" step="1" min="0" max="95">
    <span class="lbl trim" title="Percent of the object cut away from its +X side, its top and its +Z side (in the object's own frame)">Trim + %</span><input type="number" class="trim" id="t1x" step="1" min="0" max="95"><input type="number" class="trim" id="t1y" step="1" min="0" max="95"><input type="number" class="trim" id="t1z" step="1" min="0" max="95">
  </div>
  <label><input type="checkbox" id="uniform" checked> uniform scale</label>
  <div class="btns">
    <button id="btn-floor" title="Move down or up until the lowest point touches the floor">To floor</button>
    <button id="btn-down" title="Lower until it rests on the nearest object, built-in or floor below">Send down</button>
    <button id="btn-wall" title="Then click a wall (or any surface) to push the selection against it">To wall&hellip;</button>
    <button id="btn-dup" title="Copy the selected objects (Ctrl+C, Ctrl+V)">Duplicate</button>
    <button id="btn-reset">Reset</button>
    <button id="btn-remove" title="Delete key">Remove</button>
    <button id="btn-close">Close</button>
  </div>
  <div class="hint" id="edit-hint">Shift-click to select several. Typing a value sets it on every selected object; &#9650;&#9660; and the arrow keys move, turn or scale them together (Shift: 5&times; the step). Ctrl+C / Ctrl+V copy, Delete removes, Ctrl+S saves, Esc clears.</div>
</div>
<div id="roomedit" class="panel">
  <h3 class="handle" title="Drag to move; double-click to put it back"><span>Room</span></h3>
  <div class="dim" id="room-dim"></div>
  <div class="grid">
    <span></span><span class="axis x">X</span><span class="axis y">Y (up)</span><span class="axis z">Z</span>
    <span class="lbl" title="Positions of the −X wall, the floor and the −Z wall">Low side</span><input type="number" id="rlx" step="0.05"><input type="number" id="rly" step="0.05"><input type="number" id="rlz" step="0.05">
    <span class="lbl" title="Positions of the +X wall, the ceiling and the +Z wall">High side</span><input type="number" id="rhx" step="0.05"><input type="number" id="rhy" step="0.05"><input type="number" id="rhz" step="0.05">
    <span class="lbl" title="Width, height and depth; X and Z change both walls equally, Y moves the ceiling">Size</span><input type="number" id="rsx" step="0.05"><input type="number" id="rsy" step="0.05"><input type="number" id="rsz" step="0.05">
  </div>
  <div class="btns"><button id="room-reset">Reset room</button><button id="room-close">Close</button></div>
  <div class="hint">Hover a value to see which side it moves. Where a wall moves in, its photo texture is cut off; where it moves out, the texture stretches. Objects stay where they are.</div>
</div>
<div id="prompt" class="panel"></div>
<div id="bar" class="panel">
  <span id="status"></span>
  <button id="btn-save" class="primary" title="Ctrl+S">Save edits</button>
  <a id="lnk-glb" style="display:none"><button>Download GLB</button></a>
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
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRButton } from 'three/addons/webxr/VRButton.js';

const OBJECTS = __OBJECTS_JSON__;
const FAILED  = __FAILED_JSON__;
const ROOM    = __ROOM_FILE__;
const LAYOUT  = __LAYOUT_JSON__;
const POINTS  = __POINTS_FILE__;
const ROOM_GLB = __ROOM_GLB__;
const LITE_GLB = __LITE_GLB__;
const params = new URLSearchParams(location.search);
// One decimated, texture-baked file (tens of MB) by default; ?full=1 loads
// the original dense meshes (hundreds of MB) one PLY at a time.
const USE_LITE = !!LITE_GLB && params.get('full') !== '1';
const UNIT = (LAYOUT && LAYOUT.estimated_metric_scale) ? LAYOUT.estimated_metric_scale : 1.0;
const UNIT_NAME = (LAYOUT && LAYOUT.estimated_metric_scale) ? 'm' : 'units';
// Typed positions are shown relative to the floor (Y) and the photo's camera (X, Z).
const ORIGIN = [0, (LAYOUT && typeof LAYOUT.floor_y === 'number') ? LAYOUT.floor_y : 0, 0];
// The room box as the pipeline fitted it; the room editor moves its sides.
const BOX0 = (LAYOUT && LAYOUT.bounds_min && LAYOUT.bounds_max && LAYOUT.scene_type !== 'open') ? [LAYOUT.bounds_min, LAYOUT.bounds_max] : null;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a1a);
if (LAYOUT && LAYOUT.scene_type === 'open' && LAYOUT.sky_rgb)      // outdoors: the photo's sky colour
  scene.background = new THREE.Color(LAYOUT.sky_rgb[0] / 255, LAYOUT.sky_rgb[1] / 255, LAYOUT.sky_rgb[2] / 255);
const camera = new THREE.PerspectiveCamera(50, innerWidth/innerHeight, 0.01, 1000);
camera.position.set(3, 3, 3);
const renderer = new THREE.WebGLRenderer({ antialias:true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(devicePixelRatio);
renderer.localClippingEnabled = true;          // trimmed objects
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
  playerRig.position.set(c.x, floorY(), c.z);
  playerRig.rotation.set(0, 0, 0);
});

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.AmbientLight(0xffffff, 0.9));
const dir = new THREE.DirectionalLight(0xffffff, 0.5);
dir.position.set(5, 10, 5); scene.add(dir);

// Everything that can be selected and edited, by key: an object's id (a number)
// or a built-in's key ('builtin_3', 'relief_0'). Each has a pivot Group at its
// original bounding-box centre (home) that the edit is applied to, and its
// original points in scene coordinates (orig) for boxes, drops and trims.
const items = {};
const bbox = new THREE.Box3();
let roomMesh = null;              // THREE.Mesh (vertex-coloured PLY) or THREE.Group (textured GLB nodes)
const builtins = {};              // group key -> { nodes: [Object3D], names: [node names], row }
const hiddenNodes = new Set();
const shellMeshes = [];           // floor, walls, ceiling: { mesh, name }
const K = k => (typeof k === 'number' || !/^\d+$/.test(k)) ? k : parseInt(k);
const esc = t => String(t).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const V = new THREE.Vector3();

function floorY() {
  const f = (LAYOUT && typeof LAYOUT.floor_y === 'number') ? LAYOUT.floor_y : bbox.min.y;
  return f + roomEdit.min[1];
}

function unlit(o, isRoom, doubleSided) {
  const map = o.material.map || null;
  if (map) map.anisotropy = renderer.capabilities.getMaxAnisotropy();
  o.material = new THREE.MeshBasicMaterial({
    map, color: 0xffffff, vertexColors: !map && !!o.geometry.attributes.color,
    transparent: isRoom, opacity: 1.0, side: doubleSided ? THREE.DoubleSide : THREE.FrontSide });
  if (isRoom) o.renderOrder = -1;
}

// The node's vertices in scene coordinates (at most ~30k of them).
function originalPoints(node) {
  node.updateMatrixWorld(true);
  let n = 0;
  node.traverse(o => { if (o.isMesh) n += o.geometry.attributes.position.count; });
  const stride = Math.max(1, Math.ceil(n / 30000)), out = [];
  node.traverse(o => {
    if (!o.isMesh) return;
    const pa = o.geometry.attributes.position;
    for (let i = 0; i < pa.count; i += stride) { V.fromBufferAttribute(pa, i).applyMatrix4(o.matrixWorld); out.push(V.x, V.y, V.z); }
  });
  return new Float32Array(out);
}

// Wrap an object in a pivot at its bounding-box centre so that typed
// rotations and scales act about the object, not about the scene origin.
function registerObject(id, node) {
  const box = new THREE.Box3().setFromObject(node);
  if (box.isEmpty()) return;
  const orig = originalPoints(node);
  const c = box.getCenter(new THREE.Vector3());
  const pivot = new THREE.Group();
  pivot.position.copy(c);
  node.position.sub(c);
  pivot.add(node);
  pivot.userData.key = id;
  scene.add(pivot);
  bbox.union(box);
  items[id] = Object.assign(items[id] || {}, { kind: 'object', pivot, node, orig, home: c.clone(),
    size: box.getSize(new THREE.Vector3()), lo: box.min.clone() });
  const info = OBJECTS.find(e => e.id === id);
  if (info && info.glass) {       // transparent in the photo: see the room through it
    node.traverse(o => { if (o.isMesh) { o.material.transparent = true; o.material.opacity = 0.45; o.material.depthWrite = false; } });
  }
}

function registerRoomNode(o) {
  const nm = o.name || (o.parent ? o.parent.name : '');
  const m = nm.match(/room_(relief|builtin)_(\d+)/);
  if (!m) return;
  const key = m[1] + '_' + m[2];
  (builtins[key] = builtins[key] || { kind: m[1], idx: parseInt(m[2]), nodes: [], names: [] });
  builtins[key].nodes.push(o);
  builtins[key].names.push(nm);                // node name as exported
}

function adoptRoom(group) {
  group.traverse(o => {
    if (!o.isMesh) return;
    const nm = o.name + ' ' + (o.parent ? o.parent.name : '');
    const isRelief = nm.includes('room_relief') || nm.includes('room_builtin');
    unlit(o, true, isRelief);
    if (isRelief) registerRoomNode(o);
    const shell = nm.match(/room_(floor|ceiling|wall_[xz]_(min|max))/);
    if (shell && !isRelief) shellMeshes.push({ mesh: o, name: shell[0] });
  });
}

// Built-ins (relief pieces and solid blocks) get pivots like objects, so they
// can be selected, moved, turned and resized.
function registerBuiltins() {
  scene.updateMatrixWorld(true);
  const title = b => b.kind === 'relief' ? `${LAYOUT && LAYOUT.scene_type === 'open' ? 'Ground / backdrop' : 'Surface'} ${b.idx + 1}` : `Solid block ${b.idx + 1}`;
  for (const [k, b] of Object.entries(builtins)) {
    b.title = title(b);
    const box = new THREE.Box3();
    b.nodes.forEach(n => box.expandByObject(n));
    if (box.isEmpty()) continue;
    const pts = b.nodes.map(n => originalPoints(n));
    const orig = new Float32Array(pts.reduce((s, p) => s + p.length, 0));
    pts.reduce((off, p) => { orig.set(p, off); return off + p.length; }, 0);
    const c = box.getCenter(new THREE.Vector3());
    const pivot = new THREE.Group();
    pivot.position.copy(c); pivot.userData.key = k;
    scene.add(pivot); pivot.updateMatrixWorld(true);
    b.nodes.forEach(n => pivot.attach(n));
    const m = b.names[0].match(/room_(builtin|relief)_\d+/);
    items[k] = { kind: 'builtin', pivot, orig, home: c.clone(), size: box.getSize(new THREE.Vector3()), lo: box.min.clone(),
                 label: b.title, prefix: m ? m[0] : b.names[0], builtin: b };
  }
}

function loadSceneGLB(url, done) {
  new GLTFLoader().load(url, gltf => {
    const room = new THREE.Group();
    const top = [];
    gltf.scene.updateMatrixWorld(true);
    gltf.scene.traverse(o => { if (o.isMesh) top.push(o); });
    for (const o of top) {
      const nm = o.name || (o.parent ? o.parent.name : '');
      const world = o.matrixWorld.clone();
      o.removeFromParent();
      o.matrix.copy(world); o.matrix.decompose(o.position, o.quaternion, o.scale);
      const m = nm.match(/^(\d{3})_/);
      if (m && !nm.startsWith('room')) {
        unlit(o, false, false);
        registerObject(parseInt(m[1]), o);
      } else {
        room.add(o);
      }
    }
    adoptRoom(room);
    roomMesh = room; scene.add(room);
    done();
  }, xhr => { if (xhr.total) progressEl.textContent = `(${(xhr.loaded / 1e6).toFixed(0)} / ${(xhr.total / 1e6).toFixed(0)} MB)`; },
  e => { console.error(url, e); loadFull(); });
}

function loadRoomGLB(url, done) {
  new GLTFLoader().load(url, gltf => {
    adoptRoom(gltf.scene);
    roomMesh = gltf.scene; scene.add(gltf.scene);
    done();
  }, undefined, e => { console.error(url, e); if (ROOM) loadPLY(ROOM, null, done); else done(); });
}
// "Room" means the shell (floor, walls, ceiling). Built-ins (relief pieces and
// solid blocks) are furniture-like and keep their own checkboxes, so ghosting
// or hiding the room leaves them as they are.
function isShell(o) {
  const nm = o.name + ' ' + (o.parent ? o.parent.name : '');
  return !(nm.includes('room_relief') || nm.includes('room_builtin'));
}
function setRoomMode(mode) {
  if (!roomMesh) return;
  const op = mode === 'ghost' ? 0.35 : 1.0;
  roomMesh.traverse(o => {
    if (!o.isMesh || !isShell(o)) return;
    o.visible = mode !== 'hidden';
    o.material.opacity = op; o.material.depthWrite = mode !== 'ghost'; o.material.needsUpdate = true;
  });
}

function loadPLY(url, id, done) {
  new PLYLoader().load(url, geom => {
    geom.computeVertexNormals();
    const hasColors = !!geom.attributes.color;
    if (id !== null) {
      const m = new THREE.Mesh(geom, new THREE.MeshBasicMaterial({ vertexColors: hasColors, side: THREE.FrontSide }));
      registerObject(id, m);
    } else {
      const m = new THREE.Mesh(geom, new THREE.MeshBasicMaterial({ vertexColors: hasColors, color: hasColors ? 0xffffff : 0xd4c9b5,
                                               side: THREE.DoubleSide, transparent: true, opacity: 1.0 }));
      m.renderOrder = -1; roomMesh = m; scene.add(m);
    }
    done();
  }, undefined, e => { console.error(url, e); done(); });
}

let loaded = 0, total = 1;
const progressEl = document.getElementById('progress');
function onLoad() {
  loaded++;
  if (total > 1) progressEl.textContent = `(${loaded}/${total})`;
  if (loaded < total) return;
  if (bbox.isEmpty() && roomMesh) bbox.expandByObject(roomMesh);
  const c = new THREE.Vector3(); bbox.getCenter(c);
  const s = new THREE.Vector3(); bbox.getSize(s);
  const d = Math.max(s.x, s.y, s.z) * 1.2;
  controls.target.copy(c);
  // Start where the photo was taken: the scene frame has the camera at the
  // origin looking down +Z. Step back a little so the whole room is in frame.
  camera.position.set(0, 0, Math.min(0, bbox.min.z) - 0.15 * d);
  controls.update();
  document.getElementById('loading').style.display = 'none';
  document.getElementById('info').style.display = 'block';
  document.getElementById('bar').style.display = 'flex';
  updateCount();
  let meta = `Objects extent: ${(s.x * UNIT).toFixed(2)} × ${(s.y * UNIT).toFixed(2)} × ${(s.z * UNIT).toFixed(2)} ${UNIT_NAME}`;
  meta += USE_LITE ? `\nMeshes: lite (decimated, baked textures). Add ?full=1 for the dense originals.` : `\nMeshes: full resolution`;
  if (LAYOUT) {
    if (LAYOUT.estimated_metric_scale) meta += `\nScale: ×${LAYOUT.estimated_metric_scale.toFixed(2)} → metres (${LAYOUT.metric_scale_source || 'camera-height prior'})`;
    if (LAYOUT.ceiling_source) meta += `\nCeiling: ${LAYOUT.ceiling_source}; floor: ${LAYOUT.floor ? LAYOUT.floor.source : '?'}`;
    if (LAYOUT.wall_sources) meta += `\nWalls: ` + Object.entries(LAYOUT.wall_sources).map(([k,v]) => `${k}=${v}`).join(', ');
  }
  document.getElementById('meta').textContent = meta;
  registerBuiltins();
  buildBuiltinList();
  if (BOX0 && shellMeshes.length) F('btn-room').style.display = 'inline-block';
  loadEdits().then(() => { const s = K(params.get('select') || ''); if (items[s]) { select(s); setHot(s); } });
}

// ---------------------------------------------------------------- lists
const F = id => document.getElementById(id);
const list = document.getElementById('object-list');
const byId = {};
OBJECTS.forEach(o => byId[o.id] = o);
function updateCount() { F('count').textContent = OBJECTS.length; }
function addRow(o, after) {
  const row = document.createElement('div'); row.className = 'row'; row.dataset.id = o.id;
  const sup = (o.supported_by && byId[o.supported_by]) ? ` <span class="sub">on ${esc(byId[o.supported_by].label)}</span>` : '';
  const tag = o.copyOf !== undefined ? ` <span class="sub">copy</span>` : sup;
  const conf = o.copyOf !== undefined ? '' : (o.confidence || 0).toFixed(2);
  row.innerHTML = `<input type="checkbox" checked><span class="name">${esc(o.label)}${tag}</span><span class="conf">${conf}</span>`;
  if (after) after.after(row); else list.appendChild(row);
  items[o.id] = Object.assign(items[o.id] || {}, { row, label: o.label });
  row.addEventListener('mouseenter', () => setHot(o.id));
  row.addEventListener('mouseleave', () => setHot(null));
  row.querySelector('input').addEventListener('change', e => { edit(o.id).hidden = !e.target.checked; applyEdit(o.id); markDirty(); });
  row.querySelector('.name').addEventListener('click', ev => select(o.id, ev.shiftKey));
}
OBJECTS.forEach(o => addRow(o));
FAILED.forEach(f => {
  const l = document.createElement('div'); l.className = 'failed';
  l.textContent = `${f.label} (${f.status || 'failed'})`; list.appendChild(l);
});

function buildBuiltinList() {
  const keys = Object.keys(builtins).sort((a, b) => builtins[a].kind.localeCompare(builtins[b].kind) || builtins[a].idx - builtins[b].idx);
  if (!keys.length) return;
  document.getElementById('builtin-section').style.display = 'block';
  const el = document.getElementById('builtin-list');
  keys.forEach(k => {
    const b = builtins[k];
    const row = document.createElement('div'); row.className = 'row';
    row.innerHTML = `<input type="checkbox" checked><span class="name">${esc(b.title)}</span><span class="conf">${b.kind}</span>`;
    el.appendChild(row); b.row = row;
    if (items[k]) items[k].row = row;
    row.title = 'Click to select it (move, turn or resize); untick to hide it';
    row.addEventListener('mouseenter', () => setHot(k));
    row.addEventListener('mouseleave', () => setHot(null));
    row.querySelector('input').addEventListener('change', e => { setBuiltin(k, e.target.checked); markDirty(); });
    row.querySelector('.name').addEventListener('click', ev => select(k, ev.shiftKey));
  });
}
function setBuiltin(k, on) {
  const b = builtins[k]; if (!b) return;
  b.nodes.forEach(n => n.visible = on);
  b.names.forEach(n => on ? hiddenNodes.delete(n) : hiddenNodes.add(n));
  if (b.row) b.row.querySelector('input').checked = on;
  if (selected.has(k)) updateSelection();
}
document.getElementById('toggle-relief').addEventListener('change', e => {
  Object.keys(builtins).forEach(k => setBuiltin(k, e.target.checked)); markDirty();
});

// The scene list can be folded away and both panels dragged by their titles.
function draggable(panel, key) {
  const handle = panel.querySelector('.handle'), store = 'wb-panel-' + key;
  function place(x, y) {
    panel.style.left = Math.max(0, Math.min(innerWidth - 80, x)) + 'px';
    panel.style.top = Math.max(0, Math.min(innerHeight - 30, y)) + 'px';
    panel.style.right = 'auto'; panel.style.bottom = 'auto';
  }
  try { const p = JSON.parse(localStorage.getItem(store)); if (p) place(p.x, p.y); } catch (err) { /* no storage */ }
  handle.addEventListener('pointerdown', ev => {
    if (ev.button !== 0 || ev.target.closest('button, input, select')) return;
    const r = panel.getBoundingClientRect(), dx = ev.clientX - r.left, dy = ev.clientY - r.top;
    handle.setPointerCapture(ev.pointerId);
    const move = m => place(m.clientX - dx, m.clientY - dy);
    const up = () => {
      handle.removeEventListener('pointermove', move); handle.removeEventListener('pointerup', up);
      const r2 = panel.getBoundingClientRect();
      try { localStorage.setItem(store, JSON.stringify({ x: r2.left, y: r2.top })); } catch (err) { /* no storage */ }
    };
    handle.addEventListener('pointermove', move); handle.addEventListener('pointerup', up);
  });
  handle.addEventListener('dblclick', ev => {
    if (ev.target.closest('button')) return;
    panel.style.left = panel.style.top = panel.style.right = panel.style.bottom = '';
    try { localStorage.removeItem(store); } catch (err) { /* no storage */ }
  });
}
draggable(F('info'), 'info'); draggable(F('edit'), 'edit'); draggable(F('roomedit'), 'room');
function setCollapsed(on) {
  F('info').classList.toggle('collapsed', on);
  F('btn-collapse').innerHTML = on ? '+' : '&minus;';
  try { localStorage.setItem('wb-info-collapsed', on ? '1' : ''); } catch (err) { /* no storage */ }
}
F('btn-collapse').addEventListener('click', () => setCollapsed(!F('info').classList.contains('collapsed')));
try { if (localStorage.getItem('wb-info-collapsed')) setCollapsed(true); } catch (err) { /* no storage */ }

// ------------------------------------------------------------ highlight
const HOT = new THREE.Color(1.0, 0.82, 0.45), WHITE = new THREE.Color(1, 1, 1);
function tint(nodes, on) {
  nodes.forEach(n => n.traverse(o => { if (o.isMesh) o.material.color.copy(on ? HOT : WHITE); }));
}
let hotId = null, selId = null;
const hotBox3 = new THREE.Box3();
const hotBox = new THREE.Box3Helper(hotBox3, 0xffc850); hotBox.visible = false; scene.add(hotBox);
const selBox3 = new THREE.Box3();
const selBox = new THREE.Box3Helper(selBox3, 0x66ccff); selBox.visible = false; scene.add(selBox);
const selected = new Set();      // every selected key; selId is the one whose values the panel shows
function shown(o) { for (; o; o = o.parent) if (!o.visible) return false; return true; }
function setHot(k) {
  if (hotId === k) return;
  const old = hotId !== null ? items[hotId] : null;
  if (old && old.pivot) { tint([old.pivot], false); if (old.row) old.row.classList.remove('hot'); }
  hotId = k;
  const it = k !== null ? items[k] : null;
  if (it && it.pivot) {
    tint([it.pivot], true);
    if (it.row) { it.row.classList.add('hot'); it.row.scrollIntoView({ block: 'nearest' }); }
    hotBox3.makeEmpty(); worldBox(k, hotBox3); hotBox.visible = !hotBox3.isEmpty() && shown(it.pivot.children[0] || it.pivot);
  } else hotBox.visible = false;
}

const ray = new THREE.Raycaster(), ptr = new THREE.Vector2();
// The first visible thing under the pointer: { key, hit } for an object or a
// built-in, { room: true, hit } for the floor, a wall or the ceiling, or null.
function pick(ev, withRoom, exclude) {
  ptr.set((ev.clientX / innerWidth) * 2 - 1, -(ev.clientY / innerHeight) * 2 + 1);
  ray.setFromCamera(ptr, camera);
  const targets = Object.entries(items).filter(([k, i]) => i.pivot && !(exclude && exclude.has(K(k)))).map(([, i]) => i.pivot);
  if (withRoom && roomMesh) targets.push(roomMesh);
  for (const hit of ray.intersectObjects(targets, true)) {
    if (!shown(hit.object)) continue;
    for (let o = hit.object; o; o = o.parent) if (o.userData && o.userData.key !== undefined) return { key: o.userData.key, hit };
    return { room: true, hit };
  }
  return null;
}
let downAt = null, lastMove = 0;
renderer.domElement.addEventListener('pointerdown', e => { downAt = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener('pointermove', e => {
  if (e.buttons || performance.now() - lastMove < 60) return;
  lastMove = performance.now();
  const p = pick(e);
  setHot(p ? p.key : null);
});
renderer.domElement.addEventListener('pointerup', e => {
  if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 4) return;   // a drag, not a click
  if (wallMode) { toWall(e); return; }
  const p = pick(e);
  if (p) select(p.key, e.shiftKey);
});

// ---------------------------------------------------------------- edits
const edits = {};     // key -> {translate:[3], rotate_deg:[3], scale:[3], hidden, crop: [[lo],[hi]] | null}
const FAR = 1e4;      // an untrimmed side of the crop box
function edit(k) { return edits[k] = edits[k] || { translate: [0,0,0], rotate_deg: [0,0,0], scale: [1,1,1], hidden: false, crop: null }; }
function isIdentity(e) { return !e.hidden && !e.crop && e.translate.every(v => Math.abs(v) < 1e-9) && e.rotate_deg.every(v => Math.abs(v) < 1e-9) && e.scale.every(v => Math.abs(v - 1) < 1e-9); }
// original scene coordinates -> where the edit puts them
function toWorld(k) {
  const it = items[k]; it.pivot.updateMatrixWorld(true);
  return it.pivot.matrixWorld.clone().multiply(new THREE.Matrix4().makeTranslation(-it.home.x, -it.home.y, -it.home.z));
}
function worldPoints(k) {
  const it = items[k], c = edit(k).crop, M = toWorld(k), P = it.orig, out = [];
  for (let i = 0; i < P.length; i += 3) {
    if (c && (P[i] < c[0][0] || P[i] > c[1][0] || P[i+1] < c[0][1] || P[i+1] > c[1][1] || P[i+2] < c[0][2] || P[i+2] > c[1][2])) continue;
    V.set(P[i], P[i+1], P[i+2]).applyMatrix4(M); out.push(V.x, V.y, V.z);
  }
  return out;
}
function worldBox(k, box) {
  const p = worldPoints(k);
  for (let i = 0; i < p.length; i += 3) box.expandByPoint(V.set(p[i], p[i+1], p[i+2]));
  return box;
}
// Trims are clipping planes in the object's original frame, carried along by its edit.
function updateClip(k) {
  const it = items[k];
  if (it.kind !== 'object') return;
  const c = edit(k).crop, planes = [];
  if (c) {
    const M = toWorld(k);
    for (let i = 0; i < 3; i++) {
      const n = AXES[i];
      if (c[0][i] > -FAR / 2) planes.push(new THREE.Plane(n.clone(), -c[0][i]).applyMatrix4(M));
      if (c[1][i] < FAR / 2) planes.push(new THREE.Plane(n.clone().negate(), c[1][i]).applyMatrix4(M));
    }
  }
  const changed = (it.nPlanes || 0) !== planes.length;
  it.nPlanes = planes.length;
  it.pivot.traverse(o => { if (o.isMesh) { o.material.clippingPlanes = planes.length ? planes : null; if (changed) o.material.needsUpdate = true; } });
}
function applyEdit(k) {
  k = K(k);
  const it = items[k]; if (!it || !it.pivot) return;
  const e = edit(k), r = THREE.MathUtils.degToRad;
  it.pivot.position.set(it.home.x + e.translate[0], it.home.y + e.translate[1], it.home.z + e.translate[2]);
  it.pivot.rotation.set(r(e.rotate_deg[0]), r(e.rotate_deg[1]), r(e.rotate_deg[2]), 'XYZ');
  it.pivot.scale.set(e.scale[0], e.scale[1], e.scale[2]);
  if (it.kind === 'object') {
    it.pivot.visible = !e.hidden;
    if (it.row) it.row.querySelector('input').checked = !e.hidden;
  }
  if (it.row) it.row.classList.toggle('edited', !isIdentity(e) || it.copyOf !== undefined);
  updateClip(k);
  if (selected.has(k)) updateSelection();
  if (hotId === k) { hotBox3.makeEmpty(); worldBox(k, hotBox3); }
}
const fields = { p: ['px','py','pz'], r: ['rx','ry','rz'], s: ['sx','sy','sz'], t: ['t0x','t0y','t0z','t1x','t1y','t1z'] };
const ALL_FIELDS = [...fields.p, ...fields.r, ...fields.s, ...fields.t];
// ------------------------------------------------------------ axis hints
// A small gizmo on the selection: arrows for +X/+Y/+Z (position), arcs with the
// direction a positive angle turns (rotation), or double arrows (scale). Click
// Position / Rotation° / Scale to switch; the axis being typed in is highlighted.
const AXES = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)];
const AXIS_RGB = [0xff6b6b, 0x6ddc6d, 0x6fa8ff], AXIS_NAME = ['X', 'Y', 'Z'];
const gizmo = new THREE.Group(); gizmo.visible = false; scene.add(gizmo);
let gizmoMode = 'p', gizmoAxis = -1, gizmoSize = 1;
function overlay(o, opacity) {        // always drawn on top, so it shows through the object
  o.traverse(m => { if (m.material) { m.material.depthTest = false; m.material.transparent = true; m.material.opacity = opacity; } m.renderOrder = 999; });
  return o;
}
function textSprite(text, color, h) {
  const c = document.createElement('canvas'); c.width = 160; c.height = 64;
  const g = c.getContext('2d');
  g.font = 'bold 40px system-ui, sans-serif'; g.textAlign = 'center'; g.textBaseline = 'middle';
  g.lineWidth = 7; g.strokeStyle = 'rgba(0,0,0,.85)'; g.strokeText(text, 80, 32);
  g.fillStyle = '#' + new THREE.Color(color).getHexString(); g.fillText(text, 80, 32);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c) }));
  sp.scale.set(h * 2.5, h, 1);
  return sp;
}
function buildGizmo() {
  gizmo.traverse(o => { if (o.geometry) o.geometry.dispose(); if (o.material) { if (o.material.map) o.material.map.dispose(); o.material.dispose(); } });
  gizmo.clear();
  const L = Math.max(0.6 * gizmoSize, 0.2), head = 0.16 * L, txt = 0.12 * L;
  AXES.forEach((a, i) => {
    if (gizmoMode === 'r' && gizmoAxis >= 0 && gizmoAxis !== i) return;     // one ring at a time while typing an angle
    const col = AXIS_RGB[i], op = (gizmoAxis < 0 || gizmoAxis === i) ? 1.0 : 0.3;
    const parts = new THREE.Group();
    if (gizmoMode === 'p') {
      parts.add(new THREE.ArrowHelper(a, new THREE.Vector3(), L, col, head, head * 0.55));
      const neg = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), a.clone().multiplyScalar(-0.45 * L)]),
                                 new THREE.LineBasicMaterial({ color: col }));
      parts.add(neg);
      const plus = textSprite('+' + AXIS_NAME[i], col, txt); plus.position.copy(a).multiplyScalar(L + 1.2 * txt); parts.add(plus);
      const minus = textSprite('−' + AXIS_NAME[i], col, 0.75 * txt); minus.position.copy(a).multiplyScalar(-0.45 * L - txt); parts.add(minus);
    } else if (gizmoMode === 'r') {
      // positive angle = counter-clockwise seen from the + end of the axis (right-hand rule)
      const u = AXES[(i + 1) % 3], v = AXES[(i + 2) % 3], r = 0.75 * L, end = 1.6 * Math.PI, pts = [];
      for (let k = 0; k <= 48; k++) { const t = end * k / 48; pts.push(u.clone().multiplyScalar(r * Math.cos(t)).add(v.clone().multiplyScalar(r * Math.sin(t)))); }
      parts.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: col })));
      const tangent = u.clone().multiplyScalar(-Math.sin(end)).add(v.clone().multiplyScalar(Math.cos(end))).normalize();
      const cone = new THREE.Mesh(new THREE.ConeGeometry(head * 0.45, head, 12), new THREE.MeshBasicMaterial({ color: col }));
      cone.position.copy(pts[pts.length - 1]);
      cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), tangent);
      parts.add(cone);
      const lbl = textSprite('+' + AXIS_NAME[i], col, txt); lbl.position.copy(pts[pts.length - 1]).addScaledVector(tangent, 1.5 * txt); parts.add(lbl);
      parts.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([a.clone().multiplyScalar(-0.3 * L), a.clone().multiplyScalar(0.3 * L)]),
                               new THREE.LineBasicMaterial({ color: col })));
    } else {
      [1, -1].forEach(sg => parts.add(new THREE.ArrowHelper(a.clone().multiplyScalar(sg), new THREE.Vector3(), 0.8 * L, col, head * 0.8, head * 0.45)));
      const lbl = textSprite(AXIS_NAME[i], col, txt); lbl.position.copy(a).multiplyScalar(0.8 * L + txt); parts.add(lbl);
    }
    gizmo.add(overlay(parts, op));
  });
}
function setGizmoMode(mode, axis) {
  if (!'prs'.includes(mode)) return;
  gizmoMode = mode; gizmoAxis = axis === undefined ? -1 : axis;
  document.querySelectorAll('#edit .mode').forEach(el => el.classList.toggle('on', el.dataset.mode === mode));
  buildGizmo();
}
document.querySelectorAll('#edit .mode').forEach(el => el.addEventListener('click', () => setGizmoMode(el.dataset.mode)));

// ------------------------------------------------------------ selection
function updateSelection() {
  selBox3.makeEmpty();
  selected.forEach(k => { const it = items[k]; if (it && it.pivot && shown(it.pivot.children[0] || it.pivot)) worldBox(k, selBox3); });
  const any = !selBox3.isEmpty();
  selBox.visible = gizmo.visible = any;
  if (!any) return;
  const size = selBox3.getSize(new THREE.Vector3());
  selBox3.getCenter(gizmo.position);
  const s = Math.max(size.x, size.y, size.z);
  if (Math.abs(s - gizmoSize) > 1e-6 || !gizmo.children.length) { gizmoSize = s; buildGizmo(); }
}
// shift (additive): add the object, or take it out again if it is already selected
function select(k, additive) {
  if (k !== null) k = K(k);
  if (!additive) { selected.forEach(i => items[i] && items[i].row && items[i].row.classList.remove('sel')); selected.clear(); selId = null; }
  if (k !== null && items[k] && items[k].pivot) {
    if (additive && selected.has(k)) {
      selected.delete(k); if (items[k].row) items[k].row.classList.remove('sel');
      if (selId === k) selId = selected.size ? [...selected].pop() : null;
    } else { selected.add(k); if (items[k].row) items[k].row.classList.add('sel'); selId = k; }
  }
  if (selId !== null) closeRoom();
  F('edit').style.display = selId !== null ? 'block' : 'none';
  updateSelection();
  if (selId === null) { cancelWall(); return; }
  const it = items[selId];
  F('edit-title').textContent = selected.size > 1 ? `${selected.size} selected` : (it.kind === 'object' ? `${it.label}${it.copyOf !== undefined ? ' (copy)' : ''} #${selId}` : it.label);
  refreshPanel();
}
const DIGITS = { p: 2, r: 2, s: 2, t: 1 };
function fieldValue(k, f) {
  const it = items[k], e = edit(k), i = 'xyz'.indexOf(f[f.length - 1]);
  if (f[0] === 'p') return (it.home.getComponent(i) + e.translate[i] - ORIGIN[i]) * UNIT;
  if (f[0] === 'r') return e.rotate_deg[i];
  if (f[0] === 's') return e.scale[i];
  const lo = it.lo.getComponent(i), sz = Math.max(it.size.getComponent(i), 1e-9);
  if (!e.crop) return 0;
  return f[1] === '0' ? Math.max(0, (e.crop[0][i] - lo) / sz * 100) : Math.max(0, (lo + sz - e.crop[1][i]) / sz * 100);
}
function setTrim(k, side, i, pct) {
  const it = items[k], e = edit(k), lo = it.lo.getComponent(i), sz = it.size.getComponent(i);
  pct = Math.max(0, Math.min(95, pct));
  if (!e.crop) e.crop = [[-FAR, -FAR, -FAR], [FAR, FAR, FAR]];
  const other = side === '0' ? fieldValue(k, 't1' + 'xyz'[i]) : fieldValue(k, 't0' + 'xyz'[i]);
  pct = Math.min(pct, 98 - other);                       // keep at least 2% of the object
  if (side === '0') e.crop[0][i] = pct > 1e-6 ? lo + pct / 100 * sz : -FAR;
  else e.crop[1][i] = pct > 1e-6 ? lo + sz - pct / 100 * sz : FAR;
  if (e.crop[0].every(v => v <= -FAR / 2) && e.crop[1].every(v => v >= FAR / 2)) e.crop = null;
}
function setFieldValue(k, f, v) {
  const it = items[k], e = edit(k), i = 'xyz'.indexOf(f[f.length - 1]);
  if (f[0] === 'p') e.translate[i] = v / UNIT + ORIGIN[i] - it.home.getComponent(i);
  else if (f[0] === 'r') e.rotate_deg[i] = v;
  else if (f[0] === 's') { v = Math.max(0.01, v); if (F('uniform').checked) e.scale = [v, v, v]; else e.scale[i] = v; }
  else if (it.kind === 'object') setTrim(k, f[1], i, v);
}
function refreshPanel() {
  const ids = [...selected].filter(k => items[k]), it = items[selId];
  const objs = ids.filter(k => items[k].kind === 'object');
  document.querySelectorAll('#edit .trim').forEach(el => el.style.display = objs.length ? '' : 'none');
  F('btn-dup').disabled = !objs.length;
  ALL_FIELDS.forEach(f => {
    const vals = (f[0] === 't' ? objs : ids).map(k => fieldValue(k, f));
    const same = vals.length && vals.every(v => Math.abs(v - vals[0]) < 5e-3);
    F(f).value = same ? (f[0] === 'p' || f[0] === 's' ? vals[0].toFixed(DIGITS[f[0]]) : +vals[0].toFixed(DIGITS[f[0]])) : '';
    F(f).placeholder = same ? '' : 'mixed';
  });
  const e = edit(selId);
  F('edit-dim').textContent = ids.length > 1
    ? `Fields left blank differ between the selected objects.`
    : `size ${[0, 1, 2].map(i => (it.size.getComponent(i) * e.scale[i] * UNIT).toFixed(2)).join(' × ')} ${UNIT_NAME}. Position: centre, in ${UNIT_NAME}; Y is height above the floor.`;
}
function pivotPos(k) { const it = items[k], e = edit(k); return it.home.clone().add(new THREE.Vector3(...e.translate)); }
function setPivotPos(k, p) { edit(k).translate = p.clone().sub(items[k].home).toArray(); }
function groupCentre() { updateSelection(); return selBox3.getCenter(new THREE.Vector3()); }
function rotateGroup(ids, k, deg) {
  const c = groupCentre(), q = new THREE.Quaternion().setFromAxisAngle(AXES[k], THREE.MathUtils.degToRad(deg)), r = THREE.MathUtils.degToRad;
  ids.forEach(id => {
    const e = edit(id);
    setPivotPos(id, pivotPos(id).sub(c).applyQuaternion(q).add(c));
    if ([0, 1, 2].every(j => j === k || Math.abs(e.rotate_deg[j]) < 1e-9)) { e.rotate_deg[k] += deg; return; }   // stays a one-axis turn
    const qo = new THREE.Quaternion().setFromEuler(new THREE.Euler(r(e.rotate_deg[0]), r(e.rotate_deg[1]), r(e.rotate_deg[2]), 'XYZ'));
    const eu = new THREE.Euler().setFromQuaternion(q.clone().multiply(qo), 'XYZ');
    e.rotate_deg = [eu.x, eu.y, eu.z].map(v => +THREE.MathUtils.radToDeg(v).toFixed(3));
  });
}
function scaleGroup(ids, ratios) {
  const c = groupCentre();
  ids.forEach(id => {
    const e = edit(id), p = pivotPos(id);
    setPivotPos(id, new THREE.Vector3(...[0, 1, 2].map(j => c.getComponent(j) + (p.getComponent(j) - c.getComponent(j)) * ratios[j])));
    e.scale = e.scale.map((v, j) => Math.max(0.01, v * ratios[j]));
  });
}
function changed(ids) { ids.forEach(applyEdit); updateSelection(); refreshPanel(); markDirty(); }
// A typed value is set on every selected object (each about its own centre).
function typed(f) {
  if (selId === null) return;
  const v = parseFloat(F(f).value);
  if (!Number.isFinite(v)) { refreshPanel(); return; }
  const ids = [...selected];
  ids.forEach(k => setFieldValue(k, f, v));
  changed(ids);
}
// A step (▲▼, arrow keys) moves, turns or scales the selection together.
const STEP = { p: 0.01, r: 1, s: 0.05, t: 1 };
function stepField(f, dir, big) {
  if (selId === null) return;
  const d = STEP[f[0]] * (big ? 5 : 1) * dir, i = 'xyz'.indexOf(f[f.length - 1]), ids = [...selected];
  if (f[0] === 'p') ids.forEach(k => edit(k).translate[i] += d / UNIT);
  else if (f[0] === 'r') { if (ids.length === 1) edit(ids[0]).rotate_deg[i] += d; else rotateGroup(ids, i, d); }
  else if (f[0] === 's') {
    const e = edit(selId), cur = e.scale[i], ratio = Math.max(0.01, cur + d) / cur;
    const ratios = F('uniform').checked ? [ratio, ratio, ratio] : [0, 1, 2].map(j => j === i ? ratio : 1);
    if (ids.length === 1) e.scale = e.scale.map((v, j) => Math.max(0.01, v * ratios[j])); else scaleGroup(ids, ratios);
  } else ids.forEach(k => { if (items[k].kind === 'object') setTrim(k, f[1], i, fieldValue(k, f) + d); });
  changed(ids);
}
// Number fields with their own ▲▼ (bigger than the browser's, held down to repeat).
function stepper(input, onStep) {
  const wrap = document.createElement('span');
  wrap.className = 'num ' + input.className;
  input.replaceWith(wrap); wrap.appendChild(input);
  const spin = document.createElement('span'); spin.className = 'spin';
  [['▲', 1], ['▼', -1]].forEach(([t, d]) => {
    const b = document.createElement('button');
    b.type = 'button'; b.textContent = t; b.tabIndex = -1; b.title = (d > 0 ? 'Up' : 'Down') + ' (Shift: 5× the step; hold to repeat)';
    let timer = null;
    const stop = () => { clearTimeout(timer); clearInterval(timer); timer = null; };
    b.addEventListener('pointerdown', ev => {
      ev.preventDefault(); stop(); onStep(d, ev.shiftKey);
      timer = setTimeout(() => { timer = setInterval(() => onStep(d, ev.shiftKey), 70); }, 400);
    });
    ['pointerup', 'pointerleave', 'pointercancel'].forEach(x => b.addEventListener(x, stop));
    spin.appendChild(b);
  });
  wrap.appendChild(spin);
  input.addEventListener('keydown', ev => {
    if (ev.key !== 'ArrowUp' && ev.key !== 'ArrowDown') return;
    ev.preventDefault(); onStep(ev.key === 'ArrowUp' ? 1 : -1, ev.shiftKey);
  });
  input.addEventListener('wheel', ev => { if (document.activeElement === input) ev.preventDefault(); }, { passive: false });
}
ALL_FIELDS.forEach(f => {
  const axis = () => 'xyz'.indexOf(f[f.length - 1]);
  stepper(F(f), (d, big) => { setGizmoMode(f[0], axis()); stepField(f, d, big); });
  F(f).addEventListener('change', () => typed(f));
  ['focus', 'pointerdown'].forEach(ev => F(f).addEventListener(ev, () => setGizmoMode(f[0], axis())));
});
F('btn-reset').addEventListener('click', () => {
  if (selId === null) return;
  const ids = [...selected];
  ids.forEach(k => { const keep = items[k].copyOf !== undefined; delete edits[k]; if (keep) edit(k); applyEdit(k); });
  updateSelection(); refreshPanel(); markDirty();
});
function removeSelected() {
  if (selId === null) return;
  [...selected].forEach(k => {
    const it = items[k];
    if (it.kind === 'builtin') setBuiltin(k, false);
    else if (it.copyOf !== undefined) deleteCopy(k);
    else { edit(k).hidden = true; applyEdit(k); }
  });
  markDirty(); select(null);
}
F('btn-remove').addEventListener('click', removeSelected);
F('btn-close').addEventListener('click', () => select(null));
function moveSelection(delta) {
  [...selected].forEach(k => { const e = edit(k); e.translate = e.translate.map((v, i) => v + delta.getComponent(i)); applyEdit(k); });
  updateSelection(); refreshPanel(); markDirty();
}
F('btn-floor').addEventListener('click', () => {
  if (selId === null) return;
  updateSelection();
  moveSelection(new THREE.Vector3(0, floorY() - selBox3.min.y, 0));   // lowest point of the selection to the floor
});
// Send down: drop straight down onto whatever is below (an object, a built-in,
// or the floor). Rays go down from the lowest points of the selection, one per
// cell of its footprint; the shortest gap is the drop. Rays start a little above
// the bottom, so something that sank slightly into its support comes back up.
F('btn-down').addEventListener('click', () => {
  if (selId === null) return;
  updateSelection();
  const keys = [...selected], size = selBox3.getSize(new THREE.Vector3()), N = 12, low = new Map();
  const cell = Math.max(size.x, size.z) / N || 1;
  keys.forEach(k => { const p = worldPoints(k);
    for (let i = 0; i < p.length; i += 3) {
      const c = Math.floor((p[i] - selBox3.min.x) / cell) + ',' + Math.floor((p[i+2] - selBox3.min.z) / cell);
      const q = low.get(c); if (!q || p[i+1] < q[1]) low.set(c, [p[i], p[i+1], p[i+2]]);
    } });
  const exclude = new Set(keys), lift = 0.03 * size.y + 1e-4, down = new THREE.Vector3(0, -1, 0);
  const targets = Object.entries(items).filter(([k, i]) => i.pivot && !exclude.has(K(k))).map(([, i]) => i.pivot);
  let drop = Infinity;
  for (const q of low.values()) {
    let d = q[1] - floorY();
    ray.set(new THREE.Vector3(q[0], q[1] + lift, q[2]), down);
    const hit = ray.intersectObjects(targets, true).find(h => shown(h.object));
    if (hit) d = Math.min(d, hit.distance - lift);
    drop = Math.min(drop, d);
  }
  if (Number.isFinite(drop)) moveSelection(new THREE.Vector3(0, -drop, 0));
});
// To wall: the next click picks a surface; the selection moves along that
// surface's normal until its nearest point touches it.
let wallMode = false;
function cancelWall() { wallMode = false; F('btn-wall').classList.remove('on'); F('prompt').style.display = 'none'; }
F('btn-wall').addEventListener('click', () => {
  if (selId === null) return;
  if (wallMode) { cancelWall(); return; }
  wallMode = true; F('btn-wall').classList.add('on');
  F('prompt').textContent = 'Click a wall or any surface to push the selection against it (Esc cancels)';
  F('prompt').style.display = 'block';
});
function toWall(ev) {
  const p = pick(ev, true, selected);
  cancelWall();
  if (!p || !p.hit.face) return;
  const n = p.hit.face.normal.clone().transformDirection(p.hit.object.matrixWorld);
  if (n.dot(ray.ray.direction) > 0) n.negate();            // the side facing the camera
  let gap = Infinity;
  [...selected].forEach(k => { const w = worldPoints(k);
    for (let i = 0; i < w.length; i += 3) gap = Math.min(gap, n.x * (w[i] - p.hit.point.x) + n.y * (w[i+1] - p.hit.point.y) + n.z * (w[i+2] - p.hit.point.z)); });
  if (Number.isFinite(gap)) moveSelection(n.multiplyScalar(-gap));
}

// ------------------------------------------------------------ copies
// A copy is a new object made from its original's mesh (a copy of a copy is a
// copy of the original), saved with "source" in edits.json.
function nextId() {
  const ids = Object.keys(items).filter(k => /^\d+$/.test(k)).map(Number).concat(OBJECTS.map(o => o.id));
  return Math.max(0, ...ids) + 1;
}
function makeCopy(root, id, e) {
  const src = items[root];
  if (!src || !src.node || items[id] && items[id].pivot) return false;
  const node = src.node.clone(true);
  node.traverse(o => { if (o.isMesh) { o.material = o.material.clone(); o.material.color.copy(WHITE); o.material.clippingPlanes = null; } });
  const pivot = new THREE.Group();
  pivot.position.copy(src.home); pivot.add(node); pivot.userData.key = id;
  scene.add(pivot);
  const o = { id, label: src.label, copyOf: root };
  OBJECTS.push(o); byId[id] = o;
  let after = src.row;
  while (after && after.nextElementSibling && items[K(after.nextElementSibling.dataset.id)] && items[K(after.nextElementSibling.dataset.id)].copyOf === root) after = after.nextElementSibling;
  addRow(o, after);
  Object.assign(items[id], { kind: 'object', pivot, node, orig: src.orig, home: src.home.clone(), size: src.size.clone(), lo: src.lo.clone(), copyOf: root });
  edits[id] = { translate: [...e.translate], rotate_deg: [...e.rotate_deg], scale: [...e.scale], hidden: !!e.hidden,
                crop: e.crop ? e.crop.map(a => [...a]) : null };
  applyEdit(id); updateCount();
  return true;
}
function deleteCopy(id) {
  const it = items[id];
  if (hotId === id) setHot(null);
  scene.remove(it.pivot);
  it.pivot.traverse(o => { if (o.isMesh) o.material.dispose(); });
  if (it.row) it.row.remove();
  delete items[id]; delete edits[id]; delete byId[id];
  OBJECTS.splice(OBJECTS.findIndex(o => o.id === id), 1);
  selected.delete(id); updateCount();
}
// Copies land beside their originals, one selection-width to the right on screen.
function duplicate(keys) {
  const ids = keys.filter(k => items[k] && items[k].kind === 'object');
  if (!ids.length) return;
  const box = new THREE.Box3(); ids.forEach(k => worldBox(k, box));
  const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion); right.y = 0;
  if (right.lengthSq() < 1e-6) right.set(1, 0, 0);
  right.normalize();
  const size = box.getSize(new THREE.Vector3()), off = right.multiplyScalar(1.05 * (Math.abs(right.x) * size.x + Math.abs(right.z) * size.z));
  const made = [];
  for (const k of ids) {
    const id = nextId();
    if (id > 999) { F('status').textContent = 'no more object ids (999)'; break; }
    const e = edit(k), root = items[k].copyOf !== undefined ? items[k].copyOf : k;
    if (makeCopy(root, id, { ...e, hidden: false, translate: e.translate.map((v, i) => v + off.getComponent(i)) })) made.push(id);
  }
  if (!made.length) return;
  select(null); made.forEach(id => select(id, true)); markDirty();
}
F('btn-dup').addEventListener('click', () => duplicate([...selected]));
let clipboard = [];

// ------------------------------------------------------------ room size
// The room's six sides move with the room editor. Each floor, wall and ceiling
// vertex is mapped from the old box to the new one; texture coordinates follow
// the photo where the room shrinks and stretch where it grows.
const roomEdit = { min: [0, 0, 0], max: [0, 0, 0] };
function fitAffine(P, UV) {             // least squares uv = [x y z 1] · A, A is 4×2
  const n = P.length / 3, M = Array.from({ length: 4 }, () => [0, 0, 0, 0]), B = Array.from({ length: 4 }, () => [0, 0]);
  for (let i = 0; i < n; i++) {
    const x = [P[3*i], P[3*i+1], P[3*i+2], 1];
    for (let a = 0; a < 4; a++) { for (let b = 0; b < 4; b++) M[a][b] += x[a] * x[b]; B[a][0] += x[a] * UV[2*i]; B[a][1] += x[a] * UV[2*i+1]; }
  }
  const lam = 1e-9 * (M[0][0] + M[1][1] + M[2][2] + M[3][3] + 1);
  for (let a = 0; a < 4; a++) M[a][a] += lam;        // flat pieces: one coordinate is constant
  for (let c = 0; c < 4; c++) {                       // Gaussian elimination with pivoting
    let p = c; for (let r = c + 1; r < 4; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]]; [B[c], B[p]] = [B[p], B[c]];
    for (let r = 0; r < 4; r++) { if (r === c) continue; const f = M[r][c] / M[c][c];
      for (let j = c; j < 4; j++) M[r][j] -= f * M[c][j]; B[r][0] -= f * B[c][0]; B[r][1] -= f * B[c][1]; }
  }
  const A = B.map((b, r) => [b[0] / M[r][r], b[1] / M[r][r]]);
  for (let i = 0; i < n; i++) {
    const x = [P[3*i], P[3*i+1], P[3*i+2], 1];
    for (let j = 0; j < 2; j++) if (Math.abs(x.reduce((s, v, a) => s + v * A[a][j], 0) - UV[2*i+j]) > 1e-3) return null;
  }
  return A;
}
function roomBox() { return [BOX0[0].map((v, i) => v + roomEdit.min[i]), BOX0[1].map((v, i) => v + roomEdit.max[i])]; }
function applyRoom() {
  if (!BOX0) return;
  const [lo0, hi0] = BOX0, [lo1, hi1] = roomBox();
  for (const s of shellMeshes) {
    const g = s.mesh.geometry, pa = g.attributes.position, uv = g.attributes.uv;
    if (!s.P0) {
      s.mesh.updateMatrixWorld(true);
      s.mw = s.mesh.matrixWorld.clone(); s.inv = s.mw.clone().invert();
      s.P0 = new Float32Array(pa.count * 3);
      for (let i = 0; i < pa.count; i++) { V.fromBufferAttribute(pa, i).applyMatrix4(s.mw); s.P0.set([V.x, V.y, V.z], 3 * i); }
      s.A = uv ? fitAffine(s.P0, uv.array) : null;
    }
    for (let i = 0; i < pa.count; i++) {
      const q = [0, 1, 2].map(a => { const span = hi0[a] - lo0[a]; return lo1[a] + (s.P0[3*i+a] - lo0[a]) * (Math.abs(span) > 1e-9 ? (hi1[a] - lo1[a]) / span : 1); });
      V.set(q[0], q[1], q[2]).applyMatrix4(s.inv); pa.setXYZ(i, V.x, V.y, V.z);
      if (s.A) {
        const c = [0, 1, 2].map(a => Math.min(Math.max(q[a], lo0[a]), hi0[a])).concat([1]);
        uv.setXY(i, c.reduce((t, v, a) => t + v * s.A[a][0], 0), c.reduce((t, v, a) => t + v * s.A[a][1], 0));
      }
    }
    pa.needsUpdate = true; if (s.A) uv.needsUpdate = true;
    g.computeBoundingBox(); g.computeBoundingSphere();
  }
}
const ROOM_FIELDS = ['rlx', 'rly', 'rlz', 'rhx', 'rhy', 'rhz', 'rsx', 'rsy', 'rsz'];
const SIDE = { l: ['wall_x_min', 'floor', 'wall_z_min'], h: ['wall_x_max', 'ceiling', 'wall_z_max'] };
function roomValue(f) {
  const [lo, hi] = roomBox(), i = 'xyz'.indexOf(f[2]);
  if (f[1] === 'l') return (lo[i] - ORIGIN[i]) * UNIT;
  if (f[1] === 'h') return (hi[i] - ORIGIN[i]) * UNIT;
  return (hi[i] - lo[i]) * UNIT;
}
// d: a change in display units; absolute: set the side (or size) to d instead
function moveRoom(f, d, absolute) {
  const i = 'xyz'.indexOf(f[2]), [lo, hi] = roomBox(), MIN = 0.1;
  if (absolute) d = d - roomValue(f);
  d /= UNIT;
  if (f[1] === 'l') roomEdit.min[i] = Math.min(lo[i] + d, hi[i] - MIN) - BOX0[0][i];
  else if (f[1] === 'h') roomEdit.max[i] = Math.max(hi[i] + d, lo[i] + MIN) - BOX0[1][i];
  else {
    const grow = Math.max(d, MIN - (hi[i] - lo[i]));
    if (i === 1) roomEdit.max[i] += grow; else { roomEdit.min[i] -= grow / 2; roomEdit.max[i] += grow / 2; }
  }
  applyRoom(); refreshRoom(); markDirty();
}
function refreshRoom() {
  ROOM_FIELDS.forEach(f => F(f).value = roomValue(f).toFixed(2));
  const [lo, hi] = roomBox();
  F('room-dim').textContent = `${[0, 1, 2].map(i => ((hi[i] - lo[i]) * UNIT).toFixed(2)).join(' × ')} ${UNIT_NAME} (was ${[0, 1, 2].map(i => ((BOX0[1][i] - BOX0[0][i]) * UNIT).toFixed(2)).join(' × ')}). X and Z are measured from the photo's camera, Y from the floor.`;
}
let roomHot = [];
function hotSides(f) {
  shellMeshes.forEach(s => { if (roomHot.includes(s)) s.mesh.material.color.copy(WHITE); });
  roomHot = [];
  if (!f) return;
  const i = 'xyz'.indexOf(f[2]), names = f[1] === 's' ? [SIDE.l[i], SIDE.h[i]] : [SIDE[f[1]][i]];
  roomHot = shellMeshes.filter(s => names.some(n => s.name === 'room_' + n));
  roomHot.forEach(s => s.mesh.material.color.copy(HOT));
}
ROOM_FIELDS.forEach(f => {
  stepper(F(f), (d, big) => moveRoom(f, 0.05 * (big ? 5 : 1) * d, false));
  F(f).addEventListener('change', () => { const v = parseFloat(F(f).value); if (Number.isFinite(v)) moveRoom(f, v, true); else refreshRoom(); });
  const wrap = F(f).parentElement;
  wrap.addEventListener('mouseenter', () => hotSides(f));
  wrap.addEventListener('mouseleave', () => hotSides(document.activeElement && ROOM_FIELDS.includes(document.activeElement.id) ? document.activeElement.id : null));
  F(f).addEventListener('focus', () => hotSides(f));
  F(f).addEventListener('blur', () => hotSides(null));
});
function closeRoom() { F('roomedit').style.display = 'none'; hotSides(null); }
F('btn-room').addEventListener('click', () => {
  if (F('roomedit').style.display === 'block') { closeRoom(); return; }
  select(null); refreshRoom(); F('roomedit').style.display = 'block';
});
F('room-close').addEventListener('click', closeRoom);
F('room-reset').addEventListener('click', () => { roomEdit.min = [0, 0, 0]; roomEdit.max = [0, 0, 0]; applyRoom(); refreshRoom(); markDirty(); });

// ------------------------------------------------------------ keys
addEventListener('keydown', e => {
  const mod = e.ctrlKey || e.metaKey, key = e.key.toLowerCase();
  if (mod && key === 's') {                 // the browser's "save page" is not useful here
    e.preventDefault();
    if (document.activeElement && document.activeElement.blur) document.activeElement.blur();   // commit a value being typed
    if (F('bar').offsetParent !== null) save();
    return;
  }
  if (e.target && ['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
  if (e.key === 'Escape') { if (wallMode) cancelWall(); else { select(null); closeRoom(); } }
  else if (mod && key === 'c' && !String(getSelection())) {
    clipboard = [...selected].filter(k => items[k] && items[k].kind === 'object');
    if (clipboard.length) F('status').textContent = `copied ${clipboard.length}: Ctrl+V to paste`;
  } else if (mod && key === 'v' && clipboard.length) { e.preventDefault(); duplicate(clipboard.filter(k => items[k])); }
  else if ((e.key === 'Delete' || e.key === 'Backspace') && selId !== null) { e.preventDefault(); removeSelected(); }
});

// ------------------------------------------------------------ saving
// Edits live in edits.json next to the scene. Served by the WorldBuilder web
// app (python -m webapp.server) they are saved through its API; from any other
// server, Chrome and Edge write the file into the scene folder once it has been
// picked (remembered for this page); other browsers download it.
const sceneMatch = location.pathname.match(/\/outputs\/(.+)\/viewer\.html$/);
const API = sceneMatch ? `/api/scenes/${sceneMatch[1]}` : null;
let dirty = false;
function markDirty() { dirty = true; F('status').textContent = 'unsaved edits'; F('status').className = 'dirty'; }
function saved(msg) { dirty = false; F('status').textContent = msg; F('status').className = ''; }
function collect() {
  const out = { version: 1, objects: {}, nodes: {}, hidden_nodes: [...hiddenNodes] };
  for (const [key, e] of Object.entries(edits)) {
    const k = K(key), it = items[k];
    if (!it || !it.home) continue;
    const t = { translate: e.translate, rotate_deg: e.rotate_deg, scale: e.scale, pivot: it.home.toArray() };
    if (it.kind === 'builtin') { if (!isIdentity(e)) out.nodes[it.prefix] = t; continue; }
    if (isIdentity(e) && it.copyOf === undefined) continue;
    const o = Object.assign(t, { hidden: e.hidden });
    if (e.crop) o.crop = e.crop;
    if (it.copyOf !== undefined) o.source = it.copyOf;
    out.objects[k] = o;
  }
  if (roomEdit.min.some(v => Math.abs(v) > 1e-9) || roomEdit.max.some(v => Math.abs(v) > 1e-9)) out.room = { min: [...roomEdit.min], max: [...roomEdit.max] };
  return out;
}
async function loadEdits() {
  try {
    const r = await fetch('edits.json', { cache: 'no-store' });
    if (r.ok) {
      const d = await r.json();
      const objs = Object.entries(d.objects || {});
      for (const [id, e] of objs) {
        if (e.source !== undefined || !items[id] || !items[id].pivot) continue;
        edits[id] = { translate: e.translate, rotate_deg: e.rotate_deg, scale: e.scale, hidden: !!e.hidden, crop: e.crop || null };
        applyEdit(id);
      }
      for (const [id, e] of objs) if (e.source !== undefined && items[e.source] && items[e.source].copyOf === undefined) makeCopy(e.source, K(id), e);
      for (const [prefix, e] of Object.entries(d.nodes || {})) {
        const k = Object.keys(items).find(k => items[k].prefix === prefix);
        if (k) { edits[k] = { translate: e.translate, rotate_deg: e.rotate_deg, scale: e.scale, hidden: false, crop: null }; applyEdit(k); }
      }
      if (d.room && BOX0) { roomEdit.min = [...d.room.min]; roomEdit.max = [...d.room.max]; applyRoom(); }
      const hidden = new Set(d.hidden_nodes || []);
      Object.entries(builtins).forEach(([k, b]) => { if (b.names.some(n => hidden.has(n))) setBuiltin(k, false); });
      F('status').textContent = 'edits loaded';
    }
  } catch (err) { /* no edits yet */ }
  if (API) {
    try { const h = await fetch('/api/health'); if (h.ok) { F('lnk-glb').href = `${API}/glb?lite=1`; F('lnk-glb').style.display = 'inline'; } } catch (err) {}
  }
}
// The picked scene folder is kept in IndexedDB, per page, so it is asked for once.
function folderStore(mode, fn) {
  return new Promise((resolve, reject) => {
    const rq = indexedDB.open('worldbuilder-viewer', 1);
    rq.onupgradeneeded = () => rq.result.createObjectStore('folders');
    rq.onerror = () => reject(rq.error);
    rq.onsuccess = () => {
      const tx = rq.result.transaction('folders', mode), r = fn(tx.objectStore('folders'));
      tx.oncomplete = () => resolve(r.result); tx.onerror = () => reject(tx.error);
    };
  });
}
async function saveToFolder(text) {
  const key = location.origin + location.pathname;
  let folder = null;
  try { folder = await folderStore('readonly', s => s.get(key)); } catch (err) { /* no IndexedDB */ }
  if (folder && (await folder.queryPermission({ mode: 'readwrite' })) !== 'granted'
      && (await folder.requestPermission({ mode: 'readwrite' })) !== 'granted') folder = null;
  if (!folder) {
    F('status').textContent = 'pick this scene\'s folder to save into';
    folder = await showDirectoryPicker({ id: 'worldbuilder-scene', mode: 'readwrite' });
    const here = decodeURIComponent(location.pathname.split('/').slice(-2, -1)[0] || '');
    if (folder.name !== here && !confirm(`This page is in a folder named "${here}", but you picked "${folder.name}". Save edits.json there anyway?`))
      throw new DOMException('wrong folder', 'AbortError');
    try { await folderStore('readwrite', s => s.put(folder, key)); } catch (err) { /* asked again next time */ }
  }
  const fh = await folder.getFileHandle('edits.json', { create: true });
  const w = await fh.createWritable(); await w.write(text); await w.close();
  return folder.name;
}
async function save() {
  const data = collect(), text = JSON.stringify(data, null, 1);
  if (API) {
    try {
      const r = await fetch(`${API}/edits`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
      if (r.ok) { saved('saved to the scene folder'); return; }
    } catch (err) { /* not the web app: try the folder */ }
  }
  if (window.showDirectoryPicker) {
    try { saved(`saved to ${await saveToFolder(text)}/edits.json`); return; }
    catch (err) { if (err.name === 'AbortError') { F('status').textContent = 'not saved'; return; } console.error(err); }
  }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
  a.download = 'edits.json'; a.click();
  saved('downloaded: put edits.json in the scene folder');
}
F('btn-save').addEventListener('click', save);
addEventListener('beforeunload', e => { if (dirty) { e.preventDefault(); e.returnValue = ''; } });

// ---------------------------------------------------------------- load
function loadFull() {
  total = OBJECTS.length + ((ROOM || ROOM_GLB) ? 1 : 0); loaded = 0;
  OBJECTS.forEach(o => loadPLY(o.file, o.id, onLoad));
  if (ROOM_GLB) loadRoomGLB(ROOM_GLB, onLoad);
  else if (ROOM) loadPLY(ROOM, null, onLoad);
  if (!total) { total = 1; onLoad(); }
}
if (USE_LITE) loadSceneGLB(LITE_GLB, onLoad); else loadFull();

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
document.getElementById('room-mode').addEventListener('change', e => setRoomMode(e.target.value));

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
// For scripted checks (tests drive the viewer headlessly through this).
if (params.has('debug')) window.wb = { items, edits, select, collect, duplicate, roomEdit, applyRoom, moveRoom, worldBox, floorY, typed, stepField, F };
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
                    layout_file: Optional[str] = None, failed: Optional[List[Dict]] = None,
                    expect_lite: bool = False) -> Path:
    """Write viewer.html into output_dir; paths are relative to that directory.
    expect_lite: scene_lite.glb is about to be written (the viewer
    falls back to the PLYs if it turns out to be missing)."""
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
                        "confidence": r.get("confidence", 0), "supported_by": r.get("supported_by"),
                        "glass": bool(r.get("glass"))})
    layout = None
    if layout_file and (output_dir / layout_file).exists():
        try:
            layout = json.loads((output_dir / layout_file).read_text())
        except ValueError:
            layout = None
    if room_file and not (output_dir / room_file).exists():
        room_file = None
    room_glb = None
    if room_file and (output_dir / room_file).with_suffix(".glb").exists():
        room_glb = str(Path(room_file).with_suffix(".glb").as_posix())
    points = _points_sidecar(output_dir)
    lite = "scene_lite.glb" if (output_dir / "scene_lite.glb").exists() or expect_lite else None
    html = (_TEMPLATE.replace("__OBJECTS_JSON__", json.dumps(entries))
            .replace("__FAILED_JSON__", json.dumps(failed or []))
            .replace("__ROOM_FILE__", json.dumps(room_file) if room_file else "null")
            .replace("__ROOM_GLB__", json.dumps(room_glb) if room_glb else "null")
            .replace("__LITE_GLB__", json.dumps(lite) if lite else "null")
            .replace("__LAYOUT_JSON__", json.dumps(layout) if layout else "null")
            .replace("__POINTS_FILE__", json.dumps(points) if points else "null"))
    vpath = output_dir / "viewer.html"
    vpath.write_text(html)
    return vpath
