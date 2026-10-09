"""One HTML page comparing two runs of the same photos, scene by scene.

    python scripts/make_gallery.py outputs/10-08-validation --before outputs/9-17-validation
    python -m http.server -d outputs 8000      # then open http://localhost:8000/10-08-validation/

For every scene directory it shows the contact sheet (``renders.png`` from
``scripts/render_scene.py``), the mask-vs-mesh overlay and summary numbers
(``audit.png`` / ``audit.json`` from ``scripts/audit_scene.py``), what was
removed and why, and a link to the interactive viewer. Missing pieces are
skipped, so it works on a single run too. Serve the *parent* of both runs
so the relative image links resolve.
"""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path


def _summary(scene: Path) -> dict:
    p = scene / "audit.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text()).get("summary", {})
    except ValueError:
        return {}


def _removed(scene: Path) -> list:
    p = scene / "reconstruction_results.json"
    if not p.exists():
        return []
    return [f"{f.get('label')}: {f.get('status')}" for f in json.loads(p.read_text()).get("failed", [])]


def _block(title: str, scene: Path, root: Path) -> str:
    if not scene.exists():
        return ""
    rel = lambda p: html.escape(os.path.relpath(p, root))        # noqa: E731
    s = _summary(scene)
    nums = (f"{s.get('objects', '?')} objects · silhouette IoU {s.get('mean_reproj_iou', '?')} "
            f"· {s.get('low_iou(<0.3)', '?')} below 0.3 · offset {s.get('mean_offset', '?')}") if s else ""
    out = [f"<h3>{html.escape(title)} <small>{nums}</small></h3>"]
    for img in ("renders.png", "audit.png"):
        if (scene / img).exists():
            out.append(f'<a href="{rel(scene / img)}"><img src="{rel(scene / img)}" class="{img[:-4]}"></a>')
    links = [f'<a href="{rel(scene / "viewer.html")}">interactive viewer</a>'] if (scene / "viewer.html").exists() else []
    for f, label in (("scene_lite.glb", "lite GLB"), ("mujoco/scene.xml", "MuJoCo model")):
        if (scene / f).exists():
            links.append(f'<a href="{rel(scene / f)}">{label}</a>')
    out.append("<p>" + " · ".join(links) + "</p>")
    rem = _removed(scene)
    if rem:
        out.append("<details><summary>" + f"{len(rem)} removed / failed</summary><ul>"
                   + "".join(f"<li>{html.escape(r)}</li>" for r in rem) + "</ul></details>")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--before", type=Path)
    a = ap.parse_args()
    root = a.run.resolve()
    parts = ["""<!doctype html><meta charset="utf-8"><title>WorldBuilder runs</title>
<style>
 body{font:14px/1.5 system-ui,sans-serif;margin:24px;background:#16181c;color:#e6e6e6}
 h2{margin-top:40px;border-top:1px solid #333;padding-top:20px} h3 small{font-weight:400;color:#9aa}
 img{max-width:100%;display:block;margin:6px 0;border-radius:4px} img.audit{max-width:420px}
 a{color:#8cf} details{color:#bbb}
</style>
<h1>WorldBuilder: """ + html.escape(a.run.name) + ("" if not a.before else " vs " + html.escape(a.before.name)) + """</h1>
<p>Contact sheets: input photo · scene from the photo's camera · orbit · top · side.
Overlays: SAM 3 masks in green, placed meshes in red.</p>"""]
    for scene in sorted(p for p in a.run.iterdir() if (p / "reconstruction_results.json").exists()):
        parts.append(f"<h2>{html.escape(scene.name)}</h2>")
        if a.before:
            parts.append(_block("before", a.before.resolve() / scene.name, root))
        parts.append(_block("after" if a.before else "result", scene, root))
    out = a.run / "index.html"
    out.write_text("\n".join(parts))
    print(f"✓ {out}")


if __name__ == "__main__":
    main()
