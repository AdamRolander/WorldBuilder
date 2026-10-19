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
  #info h3, #edit h3 { margin:0 0 8px; font-size:14px; }
  #info h4 { margin:10px 0 4px; font-size:12px; color:#aaa; font-weight:600; text-transform:uppercase; letter-spacing:.04em; }
  .row { display:flex; align-items:center; gap:6px; padding:2px 4px; border-radius:4px; cursor:pointer; }
  .row:hover, .row.hot { background:rgba(140,200,255,.18); }
  .row.sel { background:rgba(255,200,80,.25); }
  .row .name { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .row.edited .name::after { content:" •"; color:#fc5; }
  #info label { display:block; margin:3px 0; cursor:pointer; }
  #info .meta { margin-top:10px; padding-top:10px; border-top:1px solid #444; font-size:11px; color:#aaa; white-space:pre-line; }
  #info .failed { color:#777; text-decoration:line-through; padding:2px 4px; }
  .sub { color:#8a8; font-size:11px; }
  .conf { color:#888; font-size:11px; }
  #loading { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); font-size:18px; }
  select, input[type=number], button { background:#222; color:#eee; border:1px solid #555; border-radius:4px; font:inherit; }
  button { padding:4px 10px; cursor:pointer; }
  button:hover { border-color:#8cf; }
  button.primary { background:#367D8A; border-color:#285F6B; font-weight:600; }
  #edit { top:10px; right:10px; width:300px; display:none; }
  #edit .grid { display:grid; grid-template-columns:62px repeat(3, 1fr); gap:4px; align-items:center; margin-bottom:6px; }
  #edit input[type=number] { width:100%; box-sizing:border-box; padding:3px 4px; }
  #edit .axis { color:#aaa; font-size:11px; text-align:center; font-weight:600; }
  #edit .axis.x { color:#ff6b6b; } #edit .axis.y { color:#6ddc6d; } #edit .axis.z { color:#6fa8ff; }
  #edit .mode { cursor:pointer; border-radius:3px; padding:1px 2px; }
  #edit .mode:hover { background:rgba(255,255,255,.08); }
  #edit .mode.on { background:rgba(102,204,255,.22); }
  #edit .hint { color:#888; font-size:11px; margin-top:6px; }
  #edit .btns { display:flex; gap:6px; flex-wrap:wrap; margin-top:8px; }
  #edit .dim { color:#aaa; font-size:11px; margin:4px 0; }
  #bar { bottom:12px; right:12px; display:none; gap:8px; align-items:center; }
  #status { color:#aaa; font-size:12px; }
  #status.dirty { color:#fc5; }
</style>
</head>
<body>
<div id="loading">Loading scene... <span id="progress"></span></div>
<div id="info" class="panel" style="display:none">
  <h3>Scene (<span id="count">0</span> objects)</h3>
  <label>Room:
    <select id="room-mode">
      <option value="textured" selected>textured</option>
      <option value="ghost">ghosted</option>
      <option value="hidden">hidden</option>
    </select>
  </label>
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
  <h3 id="edit-title">Object</h3>
  <div class="dim" id="edit-dim"></div>
  <div class="grid">
    <span></span><span class="axis x">X</span><span class="axis y">Y (up)</span><span class="axis z">Z</span>
    <span class="mode on" data-mode="p" title="Show which way +X, +Y and +Z move the object">Position</span><input type="number" id="px" step="0.01"><input type="number" id="py" step="0.01"><input type="number" id="pz" step="0.01">
    <span class="mode" data-mode="r" title="Show which way a positive angle turns the object">Rotation°</span><input type="number" id="rx" step="5"><input type="number" id="ry" step="5"><input type="number" id="rz" step="5">
    <span class="mode" data-mode="s" title="Show the scale axes">Scale</span><input type="number" id="sx" step="0.05" min="0.01"><input type="number" id="sy" step="0.05" min="0.01"><input type="number" id="sz" step="0.05" min="0.01">
  </div>
  <label><input type="checkbox" id="uniform" checked> uniform scale</label>
  <div class="btns">
    <button id="btn-floor" title="Move down or up until the lowest point touches the floor">To floor</button>
    <button id="btn-reset">Reset</button>
    <button id="btn-remove">Remove</button>
    <button id="btn-close">Close</button>
  </div>
  <div class="hint">Shift-click objects to select several and change them together. Esc clears.</div>
</div>
<div id="bar" class="panel">
  <span id="status"></span>
  <button id="btn-save" class="primary">Save edits</button>
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

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a1a);
if (LAYOUT && LAYOUT.scene_type === 'open' && LAYOUT.sky_rgb)      // outdoors: the photo's sky colour
  scene.background = new THREE.Color(LAYOUT.sky_rgb[0] / 255, LAYOUT.sky_rgb[1] / 255, LAYOUT.sky_rgb[2] / 255);
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

// id -> { pivot: Group (edits are applied to it), home: Vector3 (bbox centre), size: Vector3, row: element }
const items = {};
const bbox = new THREE.Box3();
let roomMesh = null;              // THREE.Mesh (vertex-coloured PLY) or THREE.Group (textured GLB nodes)
const builtins = {};              // group key -> { nodes: [Object3D], names: [node names], row }
const hiddenNodes = new Set();

function unlit(o, isRoom, doubleSided) {
  const map = o.material.map || null;
  if (map) map.anisotropy = renderer.capabilities.getMaxAnisotropy();
  o.material = new THREE.MeshBasicMaterial({
    map, color: 0xffffff, vertexColors: !map && !!o.geometry.attributes.color,
    transparent: isRoom, opacity: 1.0, side: doubleSided ? THREE.DoubleSide : THREE.FrontSide });
  if (isRoom) o.renderOrder = -1;
}

// Wrap an object in a pivot at its bounding-box centre so that typed
// rotations and scales act about the object, not about the scene origin.
function registerObject(id, node) {
  const box = new THREE.Box3().setFromObject(node);
  if (box.isEmpty()) return;
  const c = box.getCenter(new THREE.Vector3());
  const pivot = new THREE.Group();
  pivot.position.copy(c);
  node.position.sub(c);
  pivot.add(node);
  scene.add(pivot);
  bbox.union(box);
  items[id] = Object.assign(items[id] || {}, { pivot, home: c.clone(), size: box.getSize(new THREE.Vector3()), minY: box.min.y });
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
  });
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
  document.getElementById('count').textContent = OBJECTS.length;
  let meta = `Objects extent: ${(s.x * UNIT).toFixed(2)} × ${(s.y * UNIT).toFixed(2)} × ${(s.z * UNIT).toFixed(2)} ${UNIT_NAME}`;
  meta += USE_LITE ? `\nMeshes: lite (decimated, baked textures). Add ?full=1 for the dense originals.` : `\nMeshes: full resolution`;
  if (LAYOUT) {
    if (LAYOUT.estimated_metric_scale) meta += `\nScale: ×${LAYOUT.estimated_metric_scale.toFixed(2)} → metres (${LAYOUT.metric_scale_source || 'camera-height prior'})`;
    if (LAYOUT.ceiling_source) meta += `\nCeiling: ${LAYOUT.ceiling_source}; floor: ${LAYOUT.floor ? LAYOUT.floor.source : '?'}`;
    if (LAYOUT.wall_sources) meta += `\nWalls: ` + Object.entries(LAYOUT.wall_sources).map(([k,v]) => `${k}=${v}`).join(', ');
  }
  document.getElementById('meta').textContent = meta;
  buildBuiltinList();
  loadEdits().then(() => { const s = parseInt(params.get('select')); if (items[s]) { select(s); setHot(s); } });
}

// ---------------------------------------------------------------- lists
const list = document.getElementById('object-list');
const byId = {};
OBJECTS.forEach(o => byId[o.id] = o);
OBJECTS.forEach(o => {
  const row = document.createElement('div'); row.className = 'row'; row.dataset.id = o.id;
  const sup = (o.supported_by && byId[o.supported_by]) ? ` <span class="sub">on ${byId[o.supported_by].label}</span>` : '';
  row.innerHTML = `<input type="checkbox" checked><span class="name">${o.label}${sup}</span><span class="conf">${(o.confidence||0).toFixed(2)}</span>`;
  list.appendChild(row);
  items[o.id] = Object.assign(items[o.id] || {}, { row, label: o.label });
  row.addEventListener('mouseenter', () => setHot(o.id));
  row.addEventListener('mouseleave', () => setHot(null));
  row.querySelector('input').addEventListener('change', e => { edit(o.id).hidden = !e.target.checked; applyEdit(o.id); markDirty(); });
  row.querySelector('.name').addEventListener('click', ev => select(o.id, ev.shiftKey));
});
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
    const terrain = LAYOUT && LAYOUT.scene_type === 'open';
    const title = b.kind === 'relief' ? `${terrain ? 'Ground / backdrop' : 'Surface'} ${b.idx + 1}` : `Solid block ${b.idx + 1}`;
    row.innerHTML = `<input type="checkbox" checked><span class="name">${title}</span><span class="conf">${b.kind}</span>`;
    el.appendChild(row); b.row = row;
    row.title = 'Hover to highlight in the scene; shift-click a built-in in the scene to hide it';
    row.addEventListener('mouseenter', () => setHotBuiltin(k));
    row.addEventListener('mouseleave', () => setHotBuiltin(null));
    row.querySelector('input').addEventListener('change', e => { setBuiltin(k, e.target.checked); markDirty(); });
  });
}
function setBuiltin(k, on) {
  const b = builtins[k]; if (!b) return;
  b.nodes.forEach(n => n.visible = on);
  b.names.forEach(n => on ? hiddenNodes.delete(n) : hiddenNodes.add(n));
  if (b.row) b.row.querySelector('input').checked = on;
}
document.getElementById('toggle-relief').addEventListener('change', e => {
  Object.keys(builtins).forEach(k => setBuiltin(k, e.target.checked)); markDirty();
});

// ------------------------------------------------------------ highlight
const HOT = new THREE.Color(1.0, 0.82, 0.45), WHITE = new THREE.Color(1, 1, 1);
function tint(nodes, on) {
  nodes.forEach(n => n.traverse(o => { if (o.isMesh) o.material.color.copy(on ? HOT : WHITE); }));
}
let hotId = null, selId = null, hotBuiltin = null;
const hotBox = new THREE.BoxHelper(undefined, 0xffc850); hotBox.visible = false; scene.add(hotBox);
const selBox3 = new THREE.Box3();
const selBox = new THREE.Box3Helper(selBox3, 0x66ccff); selBox.visible = false; scene.add(selBox);
const selected = new Set();      // every selected id; selId is the one whose values the panel shows
function setHot(id) {
  if (hotId === id) return;
  if (hotId !== null && items[hotId] && items[hotId].pivot) { tint([items[hotId].pivot], false); items[hotId].row.classList.remove('hot'); }
  hotId = id;
  const it = id !== null ? items[id] : null;
  if (it && it.pivot) { tint([it.pivot], true); it.row.classList.add('hot'); hotBox.setFromObject(it.pivot); hotBox.visible = it.pivot.visible; }
  else hotBox.visible = false;
}
// Built-ins highlight on hover too (both ways: list row <-> scene), so a
// relief piece or a solid block can be found and switched off from the scene.
function setHotBuiltin(key) {
  if (hotBuiltin === key) return;
  if (hotBuiltin !== null && builtins[hotBuiltin]) { tint(builtins[hotBuiltin].nodes, false); if (builtins[hotBuiltin].row) builtins[hotBuiltin].row.classList.remove('hot'); }
  hotBuiltin = key;
  const b = key !== null ? builtins[key] : null;
  if (b) { tint(b.nodes, true); if (b.row) { b.row.classList.add('hot'); b.row.scrollIntoView({ block: 'nearest' }); } }
}

const ray = new THREE.Raycaster(), ptr = new THREE.Vector2();
// Returns { id } for an object, { builtin: key } for a built-in, or null.
function pick(ev) {
  ptr.set((ev.clientX / innerWidth) * 2 - 1, -(ev.clientY / innerHeight) * 2 + 1);
  ray.setFromCamera(ptr, camera);
  const targets = Object.values(items).filter(i => i.pivot && i.pivot.visible).map(i => i.pivot);
  const bnodes = [];
  for (const [k, b] of Object.entries(builtins)) b.nodes.forEach(n => { if (n.visible) { n.userData.builtinKey = k; bnodes.push(n); } });
  const hit = ray.intersectObjects(targets.concat(bnodes), true)[0];
  if (!hit) return null;
  let o = hit.object;
  while (o) {
    if (o.userData && o.userData.builtinKey) return { builtin: o.userData.builtinKey };
    for (const [id, it] of Object.entries(items)) if (o === it.pivot) return { id: parseInt(id) };
    o = o.parent;
  }
  return null;
}
let downAt = null, lastMove = 0;
renderer.domElement.addEventListener('pointerdown', e => { downAt = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener('pointermove', e => {
  if (e.buttons || performance.now() - lastMove < 60) return;
  lastMove = performance.now();
  const p = pick(e);
  const id = (p && p.id !== undefined) ? p.id : null;
  setHot(id); setHotBuiltin(p && p.builtin ? p.builtin : null);
  if (id !== null && items[id].row) items[id].row.scrollIntoView({ block: 'nearest' });
});
renderer.domElement.addEventListener('pointerup', e => {
  if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 4) return;   // a drag, not a click
  const p = pick(e);
  if (p && p.id !== undefined) select(p.id, e.shiftKey);
  else if (p && p.builtin && e.shiftKey) { setBuiltin(p.builtin, false); markDirty(); }   // shift-click hides a built-in
});

// ---------------------------------------------------------------- edits
const edits = {};     // id -> {translate:[3], rotate_deg:[3], scale:[3], hidden}
function edit(id) { return edits[id] = edits[id] || { translate: [0,0,0], rotate_deg: [0,0,0], scale: [1,1,1], hidden: false }; }
function isIdentity(e) { return !e.hidden && e.translate.every(v => Math.abs(v) < 1e-9) && e.rotate_deg.every(v => Math.abs(v) < 1e-9) && e.scale.every(v => Math.abs(v - 1) < 1e-9); }
function applyEdit(id) {
  const it = items[id]; if (!it || !it.pivot) return;
  const e = edit(id), r = THREE.MathUtils.degToRad;
  it.pivot.position.set(it.home.x + e.translate[0], it.home.y + e.translate[1], it.home.z + e.translate[2]);
  it.pivot.rotation.set(r(e.rotate_deg[0]), r(e.rotate_deg[1]), r(e.rotate_deg[2]), 'XYZ');
  it.pivot.scale.set(e.scale[0], e.scale[1], e.scale[2]);
  it.pivot.visible = !e.hidden;
  it.row.querySelector('input').checked = !e.hidden;
  it.row.classList.toggle('edited', !isIdentity(e));
  if (selected.has(id)) updateSelection();
  if (hotId === id) hotBox.setFromObject(it.pivot);
}
const F = id => document.getElementById(id);
const fields = { p: ['px','py','pz'], r: ['rx','ry','rz'], s: ['sx','sy','sz'] };
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
  gizmoMode = mode; gizmoAxis = axis === undefined ? -1 : axis;
  document.querySelectorAll('#edit .mode').forEach(el => el.classList.toggle('on', el.dataset.mode === mode));
  buildGizmo();
}
document.querySelectorAll('#edit .mode').forEach(el => el.addEventListener('click', () => setGizmoMode(el.dataset.mode)));

// ------------------------------------------------------------ selection
function updateSelection() {
  selBox3.makeEmpty();
  selected.forEach(id => { const pv = items[id] && items[id].pivot; if (pv && pv.visible) selBox3.expandByObject(pv); });
  const any = !selBox3.isEmpty();
  selBox.visible = gizmo.visible = any;
  if (!any) return;
  const size = selBox3.getSize(new THREE.Vector3());
  selBox3.getCenter(gizmo.position);
  const s = Math.max(size.x, size.y, size.z);
  if (Math.abs(s - gizmoSize) > 1e-6 || !gizmo.children.length) { gizmoSize = s; buildGizmo(); }
}
// shift (additive): add the object, or take it out again if it is already selected
function select(id, additive) {
  if (!additive) { selected.forEach(i => items[i] && items[i].row.classList.remove('sel')); selected.clear(); selId = null; }
  if (id !== null && items[id] && items[id].pivot) {
    if (additive && selected.has(id)) {
      selected.delete(id); items[id].row.classList.remove('sel');
      if (selId === id) selId = selected.size ? [...selected].pop() : null;
    } else { selected.add(id); items[id].row.classList.add('sel'); selId = id; }
  }
  F('edit').style.display = selId !== null ? 'block' : 'none';
  updateSelection();
  if (selId === null) return;
  const it = items[selId];
  F('edit-title').textContent = selected.size > 1 ? `${selected.size} objects selected` : `${it.label} #${selId}`;
  refreshPanel();
}
addEventListener('keydown', e => { if (e.key === 'Escape' && !(e.target && e.target.tagName === 'INPUT')) select(null); });
function refreshPanel() {
  const it = items[selId], e = edit(selId);
  fields.p.forEach((f, i) => F(f).value = ((it.home.getComponent(i) + e.translate[i] - ORIGIN[i]) * UNIT).toFixed(2));
  fields.r.forEach((f, i) => F(f).value = +e.rotate_deg[i].toFixed(2));
  fields.s.forEach((f, i) => F(f).value = e.scale[i].toFixed(2));
  F('edit-dim').textContent = selected.size > 1
    ? `Values shown are ${it.label} #${selId}'s. A change moves every selected object by the same amount; turning and scaling act about the middle of the group.`
    : `size ${fields.s.map((_, i) => (it.size.getComponent(i) * e.scale[i] * UNIT).toFixed(2)).join(' × ')} ${UNIT_NAME}. Position: centre, in ${UNIT_NAME}; Y is height above the floor.`;
}
function pivotPos(id) { const it = items[id], e = edit(id); return it.home.clone().add(new THREE.Vector3(...e.translate)); }
function setPivotPos(id, p) { edit(id).translate = p.clone().sub(items[id].home).toArray(); }
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
function fromPanel(changed) {
  if (selId === null) return;
  const it = items[selId], e = edit(selId), ids = [...selected];
  const num = (f, d) => { const v = parseFloat(F(f).value); return Number.isFinite(v) ? v : d; };
  const kind = changed[0], k = 'xyz'.indexOf(changed[1]);
  if (kind === 'p') {
    const d = num(changed, 0) / UNIT + ORIGIN[k] - it.home.getComponent(k) - e.translate[k];
    ids.forEach(id => edit(id).translate[k] += d);
  } else if (kind === 'r') {
    const d = num(changed, 0) - e.rotate_deg[k];
    if (ids.length === 1) e.rotate_deg[k] += d; else rotateGroup(ids, k, d);
  } else {
    const v = Math.max(0.01, num(changed, 1)), ratio = v / e.scale[k];
    if (ids.length === 1) {
      if (F('uniform').checked) e.scale = [v, v, v]; else e.scale[k] = v;
    } else scaleGroup(ids, F('uniform').checked ? [ratio, ratio, ratio] : [0, 1, 2].map(j => j === k ? ratio : 1));
  }
  ids.forEach(applyEdit); updateSelection(); refreshPanel(); markDirty();
}
[...fields.p, ...fields.r, ...fields.s].forEach(f => {
  F(f).addEventListener('change', () => fromPanel(f));
  ['focus', 'pointerdown'].forEach(ev => F(f).addEventListener(ev, () => setGizmoMode(f[0], 'xyz'.indexOf(f[1]))));
});
F('btn-reset').addEventListener('click', () => {
  if (selId === null) return;
  selected.forEach(id => { delete edits[id]; applyEdit(id); }); updateSelection(); refreshPanel(); markDirty();
});
F('btn-remove').addEventListener('click', () => {
  if (selId === null) return;
  selected.forEach(id => { edit(id).hidden = true; applyEdit(id); }); markDirty(); select(null);
});
F('btn-close').addEventListener('click', () => select(null));
F('btn-floor').addEventListener('click', () => {
  if (selId === null || !LAYOUT || LAYOUT.floor_y === undefined) return;
  const d = LAYOUT.floor_y - groupCentre().y + selBox3.getSize(new THREE.Vector3()).y / 2;   // lowest point of the selection to the floor
  selected.forEach(id => { edit(id).translate[1] += d; applyEdit(id); }); updateSelection(); refreshPanel(); markDirty();
});

// Edits live in edits.json next to the scene. Served by the WorldBuilder
// web app they are saved through its API (and applied to every GLB it hands
// out); opened from a plain file server they are downloaded instead.
const sceneMatch = location.pathname.match(/\/outputs\/([^\/]+)\/viewer\.html$/);
const API = sceneMatch ? `/api/scenes/${sceneMatch[1]}` : null;
let dirty = false;
function markDirty() { dirty = true; F('status').textContent = 'unsaved edits'; F('status').className = 'dirty'; }
function collect() {
  const out = { version: 1, objects: {}, hidden_nodes: [...hiddenNodes] };
  for (const [id, e] of Object.entries(edits)) {
    if (isIdentity(e) || !items[id] || !items[id].home) continue;
    out.objects[id] = { translate: e.translate, rotate_deg: e.rotate_deg, scale: e.scale, hidden: e.hidden, pivot: items[id].home.toArray() };
  }
  return out;
}
async function loadEdits() {
  try {
    const r = await fetch('edits.json', { cache: 'no-store' });
    if (r.ok) {
      const d = await r.json();
      for (const [id, e] of Object.entries(d.objects || {})) {
        if (!items[id] || !items[id].pivot) continue;
        edits[id] = { translate: e.translate, rotate_deg: e.rotate_deg, scale: e.scale, hidden: !!e.hidden };
        applyEdit(id);
      }
      const hidden = new Set(d.hidden_nodes || []);
      Object.entries(builtins).forEach(([k, b]) => { if (b.names.some(n => hidden.has(n))) setBuiltin(k, false); });
      F('status').textContent = 'edits loaded';
    }
  } catch (err) { /* no edits yet */ }
  if (API) {
    try { const h = await fetch('/api/health'); if (h.ok) { F('lnk-glb').href = `${API}/glb?lite=1`; F('lnk-glb').style.display = 'inline'; } } catch (err) {}
  }
}
F('btn-save').addEventListener('click', async () => {
  const data = collect();
  if (API) {
    try {
      const r = await fetch(`${API}/edits`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
      if (r.ok) { dirty = false; F('status').textContent = 'saved'; F('status').className = ''; return; }
    } catch (err) { /* fall through to a download */ }
  }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(data, null, 1)], { type: 'application/json' }));
  a.download = 'edits.json'; a.click();
  dirty = false; F('status').textContent = 'downloaded: put edits.json in the scene folder'; F('status').className = '';
});
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
