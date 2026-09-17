"""Detector-agnostic clean-up of the stage-1 object list.

Every backend (Gemini, local Qwen, anything added later) produces a list of
``{id, label, description, expected_instances, bbox_2d?}`` dicts. This module
makes that list safe for stage 2 regardless of how well the model followed
the prompt. Observed failure modes it handles, all taken from real runs in
``outputs*/``:

* the local VLM looping and emitting the same label 100+ times
  (``ucsd-basement-prototype-lab_q``: 116 entries, 107 of them "tool");
* counts baked into labels — ``"glass storage jars (2)"``,
  ``"bare trees (several)"``;
* people, which reconstruct as uncanny static blobs;
* architecture (walls, floor, ceiling, window), which SAM 3D cannot
  reconstruct as objects and which the layout stage handles instead;
* labels that differ only by case/whitespace/plural, which cost a SAM 3
  query each and then get merged by cross-label NMS anyway.

Pure Python, no model access, fully unit-tested (``tests/test_postprocess.py``).
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional

EXCLUDE_KEYWORDS = (
    'person', 'people', 'human', 'man', 'woman', 'boy', 'girl', 'child',
    'kid', 'worker', 'employee', 'operator', 'staff', 'figure', 'customer',
    'student', 'teacher', 'crowd', 'pedestrian', 'passenger',
)

# Architecture that the structural-layout stage owns. Matched on whole
# tokens so "wall clock" and "floor lamp" survive but "wall" and "floor" don't.
STRUCTURAL_LABELS = frozenset({
    'wall', 'walls', 'floor', 'flooring', 'ceiling', 'window', 'windows',
    'door', 'doorway', 'doors', 'countertop', 'counter top', 'backsplash',
    'baseboard', 'skirting', 'crown molding', 'tile', 'tiles', 'floor tile',
    'wall tile', 'wallpaper', 'paint', 'room', 'path', 'ground', 'sky',
    'grass', 'lawn', 'road', 'pavement', 'sidewalk',
})

# Parts of objects that VLMs sometimes list on their own (the prompt forbids
# it, the local model ignores that in cluttered kitchens). SAM 3D reconstructs
# a "handle" as a floating blob and the support graph then chains
# drawer→drawer→handle. Whole-label match after normalisation.
PART_LABELS = frozenset({
    'handle', 'handles', 'knob', 'knobs', 'door handle', 'door knob', 'drawer handle',
    'cabinet handle', 'cabinet door', 'cabinet doors', 'drawer', 'drawers', 'hinge',
    'leg', 'legs', 'table leg', 'chair leg', 'armrest', 'backrest', 'wheel', 'wheels',
    'caster', 'button', 'switch', 'light switch', 'outlet', 'power outlet', 'socket',
    'electrical plate', 'wall plate', 'vent', 'air vent', 'grille', 'grout', 'seam',
    'cord', 'cable', 'wire', 'shadow', 'reflection', 'light', 'sunlight', 'glare',
})

_COUNT_PAREN = re.compile(r"\s*\((?:\d+|several|many|multiple|various|x\d+)\)\s*$", re.I)
_LEADING_COUNT = re.compile(r"^\s*(?:\d+|two|three|four|five|six|several|many|multiple|various)\s+", re.I)
_TRAILING_QTY = re.compile(r"\s*x\s*\d+\s*$", re.I)

MAX_OBJECTS_DEFAULT = 40


def normalize_label(label: str) -> str:
    """Lower-case, strip counts/quantities and extra whitespace.

    >>> normalize_label("Glass Storage Jars (2)")
    'glass storage jars'
    >>> normalize_label("3 wooden cutting boards")
    'wooden cutting boards'
    """
    s = str(label or "").strip().lower()
    s = _COUNT_PAREN.sub("", s)
    s = _TRAILING_QTY.sub("", s)
    s = _LEADING_COUNT.sub("", s)
    s = re.sub(r"[\"'`]", "", s)
    s = re.sub(r"\s+", " ", s).strip(" -_.,;:")
    return s


def _tokens(label: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", label.lower())


def _singular(token: str) -> str:
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("es") and token[-3] in "sxz":
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def dedupe_key(label: str) -> str:
    """Key under which near-identical labels collapse ("Chairs" == "chair")."""
    return " ".join(_singular(t) for t in _tokens(normalize_label(label)))


def is_person(label: str) -> bool:
    toks = set(_tokens(label))
    return any(k in toks for k in EXCLUDE_KEYWORDS)


def is_structural(label: str) -> bool:
    return normalize_label(label) in STRUCTURAL_LABELS


def is_part(label: str) -> bool:
    return normalize_label(label) in PART_LABELS


def _clean_bbox(bbox, image_size: Optional[tuple] = None):
    """Validate an [x1, y1, x2, y2] box; return None if unusable."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    if image_size is not None:
        w, h = image_size
        x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
        y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
    return [x1, y1, x2, y2]


def clean_detections(objects: Iterable[Dict],
                     max_objects: int = MAX_OBJECTS_DEFAULT,
                     image_size: Optional[tuple] = None,
                     drop_structural: bool = True,
                     verbose: bool = True) -> List[Dict]:
    """Normalise, filter and dedupe a raw detection list.

    Returns new dicts (inputs are not mutated) with contiguous ids, a
    normalised ``label``, ``expected_instances`` in {"single", "multiple"},
    and ``bbox_2d`` either a valid float list or ``None``.
    """
    kept: List[Dict] = []
    seen: Dict[str, Dict] = {}
    dropped_people, dropped_structural, dropped_dupes, dropped_empty = [], [], [], 0
    dropped_parts: List[str] = []

    for raw in objects or []:
        if not isinstance(raw, dict):
            dropped_empty += 1
            continue
        label = normalize_label(raw.get("label", ""))
        if not label:
            dropped_empty += 1
            continue
        if is_person(label):
            dropped_people.append(label)
            continue
        if drop_structural and is_structural(label):
            dropped_structural.append(label)
            continue
        if is_part(label):
            dropped_parts.append(label)
            continue

        key = dedupe_key(label)
        exp = str(raw.get("expected_instances", "multiple")).strip().lower()
        exp = "single" if exp in ("single", "one", "1") else "multiple"
        entry = {
            "label": label,
            "description": str(raw.get("description", "") or "").strip(),
            "expected_instances": exp,
            "bbox_2d": _clean_bbox(raw.get("bbox_2d"), image_size),
        }
        if key in seen:
            dropped_dupes.append(label)
            prev = seen[key]
            # A repeated label is itself evidence of multiple instances.
            prev["expected_instances"] = "multiple"
            if not prev["description"] and entry["description"]:
                prev["description"] = entry["description"]
            if prev["bbox_2d"] is None and entry["bbox_2d"] is not None:
                prev["bbox_2d"] = entry["bbox_2d"]
            continue
        seen[key] = entry
        kept.append(entry)

    if max_objects and len(kept) > max_objects:
        if verbose:
            print(f"  Capping detections at {max_objects} (had {len(kept)})")
        kept = kept[:max_objects]

    for i, o in enumerate(kept, 1):
        o["id"] = i

    if verbose:
        if dropped_people:
            print(f"  Filtered {len(dropped_people)} person label(s): {sorted(set(dropped_people))}")
        if dropped_structural:
            print(f"  Deferred {len(dropped_structural)} structural label(s) to layout stage: "
                  f"{sorted(set(dropped_structural))}")
        if dropped_parts:
            print(f"  Dropped {len(dropped_parts)} part-of-object label(s): {sorted(set(dropped_parts))}")
        if dropped_dupes:
            print(f"  Merged {len(dropped_dupes)} duplicate label(s)")
        if dropped_empty:
            print(f"  Dropped {dropped_empty} malformed entr{'y' if dropped_empty == 1 else 'ies'}")
    return kept


def filter_excluded(objects: List[Dict]) -> List[Dict]:
    """Backwards-compatible name used by older callers."""
    return clean_detections(objects)
