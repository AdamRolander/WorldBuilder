# DCC integrations: Blender extension and Unreal plugin

Both integrations follow one rule: **the DCC add-on is a thin HTTP client;
the models stay in the WorldBuilder server.** Reasons:

* SAM 3 / SAM 3D weights are gated and under Meta's SAM License — they cannot
  be redistributed inside an extension or a Fab listing (see the
  licence notes in `LICENSE`).
* The pipeline needs a 32 GB GPU and two conda environments; no add-on
  sandbox will host that.
* One server can serve Blender, Unreal, the web app and a future
  RoomBuilder backend through the same API.

The server-side seams added for this are `GET /api/scenes/<id>` (manifest)
and `GET /api/scenes/<id>/glb` (whole scene as one glTF binary with named
nodes and vertex colours), produced by `src/scene_export.py`.

```
photo ──► Blender add-on / Unreal plugin ──HTTP──► webapp/server.py ──► pipeline
                     ▲                                        │
                     └──────────── scene.glb ◄────────────────┘
```

---

## Blender — `integrations/blender_worldbuilder/`

Status: **scaffolded, untested in Blender** (no Blender on the GPU box yet).
Files: `blender_manifest.toml` (Extensions-platform manifest),
`__init__.py` (add-on: preferences, modal operator, sidebar panel).

### Try it locally

```bash
# 1. GPU box: start the server (and the local VLM if you want fully local)
conda activate worldbuilder-main && python -m webapp.server
# 2. Zip the add-on folder and install it in Blender 4.2+:
cd integrations && zip -r worldbuilder-0.1.0.zip blender_worldbuilder
# Blender: Edit > Preferences > Get Extensions > ▾ > Install from Disk…
# Then: 3D Viewport > N sidebar > WorldBuilder > "Generate scene from photo"
```

Legacy add-on install (Blender 3.6–4.1) works with the same zip via
Preferences > Add-ons > Install.

### Publishing on extensions.blender.org — what they require

(From the platform's manifest schema and moderation rules; the docs site
blocked automated fetching, so verify against
<https://docs.blender.org/manual/en/latest/advanced/extensions/> before
submitting.)

| requirement | status |
| --- | --- |
| `blender_manifest.toml` with `schema_version, id, version, name, tagline, maintainer, type, blender_version_min, license` | done |
| GPL-compatible license, SPDX identifier (`SPDX:MIT` ok) | done |
| Declared `[permissions]` (`network`, `files`) with a one-line justification each | done |
| No bundled binaries without wheels; wheels must be declared and platform-tagged | not needed (stdlib only) |
| Must not download executable code at runtime | complies (downloads a GLB, data) |
| Works offline or fails gracefully when the network is missing | error path reports "server unreachable" |
| Account on extensions.blender.org, `blender --command extension validate` passes, moderation review (days–weeks) | **todo** (DCC-2) |
| Extensions must be self-contained: the add-on cannot *require* a paid or external service to be useful | Grey area — it requires *your own* server. State this prominently in the listing; free/open server code satisfies reviewers in comparable cases (render-farm and LLM-bridge add-ons exist on the platform). |

Nuances worth knowing:

* Blender's Python is 3.11 and has no `requests`; the add-on uses `urllib`
  and a hand-rolled multipart encoder for that reason.
* Long blocking calls freeze the UI, so the operator is *modal* with a
  timer and runs HTTP in a thread; only `bpy.ops.import_scene.gltf` runs on
  the main thread.
* glTF import puts every node under the scene root; the add-on moves the
  new objects into a collection named after the scene id.
* Units: scene coordinates are scale-invariant. `layout.json` carries
  `estimated_metric_scale` (camera-height prior). Apply it as a collection
  scale if you want metres (DCC-3 adds a checkbox).

---

## Unreal Engine — `integrations/unreal/WorldBuilderBridge/`

Status: **scaffolded, uncompiled** (no UE install on this machine).
Files: `WorldBuilderBridge.uplugin`, a minimal editor module
(`Source/WorldBuilderBridge/...`) that adds *Tools ▸ WorldBuilder ▸ Generate
scene from photo…*, and `Content/Python/worldbuilder_bridge.py` which does
the work through the Python Editor Script Plugin and Interchange (glTF).

### Try it locally

1. Copy `WorldBuilderBridge/` into `<YourProject>/Plugins/`.
2. Enable *Python Editor Script Plugin*, *Editor Scripting Utilities*,
   *Interchange* in the project (the `.uplugin` declares them as
   dependencies so UE will prompt).
3. Rebuild the project (the C++ module is tiny) or, to skip C++ entirely,
   delete `Source/` and use it as a content-only plugin: in the Output Log
   (Python) run

   ```python
   import sys; sys.path.append(r"<project>/Plugins/WorldBuilderBridge/Content/Python")
   import worldbuilder_bridge as wb; wb.generate_and_import(r"C:/photos/room.jpg")
   ```

### Publishing on Fab — what Epic requires

From the [Fab publishing docs](https://dev.epicgames.com/documentation/en-us/fab/publishing-assets-for-sale-or-free-download-in-fab)
(fetched 2026-09-17) and community reports:

| requirement | status |
| --- | --- |
| Epic publisher account (Fab seller onboarding, tax/payout even for free listings) | **todo** (DCC-5) |
| *Code plugins* must contain ≥1 code module and be submitted per supported engine version; a new `.uplugin`/source zip per version | module scaffolded; target 5.4 + 5.5 first |
| Epic compiles the plugin with their toolchain; must build warning-clean on Win64 (Linux/Mac if listed). Test with *Package…* in the Plugins window | **todo** |
| Fab Terms of Service + Epic Content Guidelines (no bundled third-party assets you don't own the rights to) | complies: no models/weights bundled |
| Documentation URL, support URL, screenshots/video | docs URL set; media **todo** |
| Free listings allowed | yes |
| Review turnaround | reports range from days to several weeks |

Nuances:

* Interchange's glTF importer handles vertex colours (`COLOR_0`) but
  materials will be plain; a *Vertex Color* material instance is worth
  shipping in `Content/` (DCC-6).
* Y-up → Z-up and metres → centimetres are handled by Interchange; the
  Python applies `estimated_metric_scale` on top.
* `generate_and_import` blocks the editor for the pipeline's duration.
  The proper version is an Editor Utility Widget with a timer (DCC-4).

---

## Status after the October 2026 pass

* **Blender**: `blender --command extension validate` passes;
  `blender --command extension build --source-dir integrations/blender_worldbuilder`
  produces the installable zip. `scripts/test_blender_addon.py` runs the
  whole flow inside `blender --background` (register → upload → poll →
  download → import → render) and passed on Blender 5.0.1 (VALIDATION
  §4.4). New in add-on 0.2.0: lite GLB by default (tens of MB, baked
  textures), "Import existing scene", metric-scale checkbox, every room
  surface and the relief as separate named objects under one scene empty.
  Not yet exercised: the UI panel by a human, and Blender 4.2 LTS.
* **Unreal**: unchanged scaffold, untested. When you try it, import
  `GET /api/scenes/<id>/glb?lite=1` rather than the full GLB: the lite file
  has ordinary textured materials, which Interchange imports without the
  vertex-colour material the full file needs (`DCC-6`).
* **MuJoCo / MJX**: new, see `integrations/mujoco/README.md`.
* **Server endpoints for clients**: `GET /api/scenes` (newest first),
  `GET /api/scenes/<id>/glb?lite=1`, `GET /api/scenes/<id>/mujoco[?mjx=1]`.

### Testing the Blender add-on this weekend (10 minutes)

```bash
# terminal 1 (worldbuilder-vlm env), only for the local detector
python vlm_server/server.py
# terminal 2 (worldbuilder-main env)
python -m webapp.server
# terminal 3: automated check first
blender --background --python scripts/test_blender_addon.py -- \
    --scene <an existing scene id> --render /tmp/wb.png        # no GPU work, ~5 s
# then by hand: build and install the extension
blender --command extension build --source-dir integrations/blender_worldbuilder --output-dir /tmp
#   Blender > Edit > Preferences > Get Extensions > ▾ > Install from Disk… > /tmp/worldbuilder-0.2.0.zip
#   3D Viewport > N > WorldBuilder > "Import existing scene" (leave the id empty for the newest)
#   then "Generate scene from photo"
```

The server reads scenes from `outputs/` directly under the project root
(`WORLDBUILDER_OUTPUTS` overrides); scenes in sub-folders such as
`outputs/10-08-validation/` are not listed unless you point it there.

## Roadmap items (see ROADMAP.md)

* DCC-1 Test the Blender add-on end to end (needs Blender on a machine that can reach the server).
* DCC-2 Register on extensions.blender.org; run `blender --command extension validate`; submit.
* DCC-3 Metric-scale option + material presets in both bridges.
* DCC-4 Non-blocking Unreal UI (Editor Utility Widget) with progress.
* DCC-5 Fab publisher onboarding; build the plugin on 5.4/5.5 Win64.
* DCC-6 Ship a vertex-colour material and a LOD/decimation option (SAM 3D meshes are dense).
* DCC-7 Streaming: import objects as they finish instead of waiting for the whole scene (needs `/api/jobs/<id>/objects`).
