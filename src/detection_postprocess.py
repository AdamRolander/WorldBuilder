"""Detector-agnostic post-processing of detected objects.

Currently filters humans (the pipeline reconstructs them as static blobs,
which is uncanny and rarely useful for scene composition).
"""
from typing import List, Dict

EXCLUDE_KEYWORDS = (
    'person', 'people', 'human', 'man', 'woman', 'worker',
    'employee', 'operator', 'staff', 'figure',
)


def filter_excluded(objects: List[Dict]) -> List[Dict]:
    kept = []
    dropped = []
    for o in objects:
        label = o.get('label', '').lower()
        if any(k in label for k in EXCLUDE_KEYWORDS):
            dropped.append(o['label'])
            continue
        kept.append(o)
    # Renumber ids contiguously
    for i, o in enumerate(kept, 1):
        o['id'] = i
    if dropped:
        print(f"  Filtered {len(dropped)} excluded object(s): {dropped}")
    return kept