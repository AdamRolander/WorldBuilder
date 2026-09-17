"""Shared pytest config. Tests here run on CPU only and never load SAM 3 /
SAM 3D weights; GPU-backed end-to-end checks live in scripts/ and are
documented in docs/VALIDATION.md."""
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
