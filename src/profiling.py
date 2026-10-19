"""Wall-clock and GPU-memory accounting per pipeline phase.

One ``Profile`` is active per scene. Code brackets work with
``with phase("segment"):``; nested phases are recorded under dotted names
(``assemble.placement``). While a profile is active a background thread
samples the *device-wide* memory in use, so the peak per phase includes
other processes on the card (the local VLM server lives in one) and not
only what this process allocated.

The result is written to ``<scene>/metrics.json`` by
``scene_assembly.finalize_scene`` and read by ``scripts/benchmark.py``.
Everything degrades to plain timing when torch or a GPU is absent, so the
CPU replay path produces the same file with only the stage-4 phases.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Dict, List, Optional


def _device_used_gb() -> Optional[float]:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return (total - free) / 1024 ** 3
    except Exception:
        return None


def _process_peak_gb(reset: bool = False) -> Optional[float]:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        if reset:
            torch.cuda.reset_peak_memory_stats()
        return peak
    except Exception:
        return None


class Profile:
    def __init__(self, sample_gpu: bool = True, interval: float = 0.25):
        self.t0 = time.time()
        self.seconds: Dict[str, float] = {}
        self.device_peak_gb: Dict[str, float] = {}
        self.process_peak_gb: Dict[str, float] = {}
        self.marks: Dict[str, float] = {}
        self._stack: List[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.baseline_device_gb = _device_used_gb() if sample_gpu else None
        if self.baseline_device_gb is not None:
            self._thread = threading.Thread(target=self._sample, args=(interval,), daemon=True)
            self._thread.start()

    def _sample(self, interval: float):
        while not self._stop.wait(interval):
            used = _device_used_gb()
            if used is None:
                continue
            with self._lock:
                for name in self._stack:
                    if used > self.device_peak_gb.get(name, 0.0):
                        self.device_peak_gb[name] = used
                if used > self.device_peak_gb.get("total", 0.0):
                    self.device_peak_gb["total"] = used

    @contextmanager
    def phase(self, name: str):
        with self._lock:
            full = ".".join([self._stack[-1], name]) if self._stack else name
            self._stack.append(full)
        top_level = len(self._stack) == 1
        if top_level:
            _process_peak_gb(reset=True)
        t = time.time()
        try:
            yield
        finally:
            dt = time.time() - t
            with self._lock:
                self._stack.remove(full)
                self.seconds[full] = self.seconds.get(full, 0.0) + dt
            if top_level:
                peak = _process_peak_gb()
                if peak:
                    self.process_peak_gb[full] = max(self.process_peak_gb.get(full, 0.0), peak)

    def mark(self, name: str, once: bool = True):
        """Seconds since the profile started, e.g. when the first object was done."""
        if once and name in self.marks:
            return
        self.marks[name] = time.time() - self.t0

    def stop(self):
        self._stop.set()

    def to_dict(self) -> Dict:
        r = lambda d, n=2: {k: round(v, n) for k, v in d.items()}   # noqa: E731
        out = {"seconds": r(self.seconds), "marks_sec": r(self.marks),
               "wall_sec": round(time.time() - self.t0, 2)}
        if self.device_peak_gb:
            out["vram_device_peak_gb"] = r(self.device_peak_gb)
            out["vram_device_baseline_gb"] = round(self.baseline_device_gb or 0.0, 2)
        if self.process_peak_gb:
            out["vram_process_peak_gb"] = r(self.process_peak_gb)
        return out


_active: Optional[Profile] = None


def start(sample_gpu: bool = True) -> Profile:
    """Begin a new profile (replacing any active one) and return it."""
    global _active
    if _active is not None:
        _active.stop()
    _active = Profile(sample_gpu=sample_gpu)
    return _active


def current() -> Optional[Profile]:
    return _active


@contextmanager
def phase(name: str):
    """Record a phase on the active profile; a no-op when none is active."""
    if _active is None:
        yield
        return
    with _active.phase(name):
        yield


def mark(name: str):
    if _active is not None:
        _active.mark(name)


def add(name: str, seconds: float):
    """Record an already-measured duration (for code that is awkward to indent)."""
    if _active is not None:
        with _active._lock:
            _active.seconds[name] = _active.seconds.get(name, 0.0) + float(seconds)


def summarize_objects(results: List[Dict]) -> Dict:
    """Distribution of SAM 3D's seconds per object."""
    secs = sorted(float(r["seconds"]) for r in results if r.get("seconds") is not None)
    if not secs:
        return {}

    def q(p):
        return round(secs[min(len(secs) - 1, int(p * len(secs)))], 2)
    return {"n": len(secs), "mean": round(sum(secs) / len(secs), 2), "min": secs[0],
            "p50": q(0.5), "p90": q(0.9), "max": secs[-1]}
