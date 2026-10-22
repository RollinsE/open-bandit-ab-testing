# -*- coding: utf-8 -*-
"""Checkpoints: save each finished piece of slow work so an interrupted build can continue.

How checkpointing works:

1. What is saved. Every model scored in the marathon (stage 3), every bootstrap
   run of the off-policy benchmark (stage 4) and every time-split run is saved
   to ``builds/<run_id>/checkpoints/`` the moment it finishes. Stages 1, 2, 5
   and 6 take seconds and are simply re-run.
2. Safe writes. Each file is written under a temporary name and then renamed,
   so a run cut off mid-write never leaves a half-written checkpoint behind.
3. Continuing. ``--continue-build`` reuses the same build folder and the
   settings in its run manifest, so every checkpoint is still valid. Finished
   work is loaded instead of recomputed; the log says how much was reused.
4. Failures are not saved. A model that fails is tried again on the next
   continue, in case the failure was transient.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.utils import _json_default


def _safe(part: str) -> str:
    """File-system-safe version of a key part ('Baseline: same CTR' -> 'Baseline_same_CTR')."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(part)) or "_"


class CheckpointStore:
    """Small key-value store of JSON and NumPy files under one folder.

    Keys are tuples of strings, e.g. ``("ope", "men", "run_003")``, mapped to
    ``<root>/ope/men/run_003.json`` (and ``.npy`` for arrays).
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, key: tuple, suffix: str) -> Path:
        parts = [_safe(p) for p in key]
        return self.root.joinpath(*parts[:-1], parts[-1] + suffix)

    @staticmethod
    def _atomic_write(path: Path, write) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        write(tmp)
        os.replace(tmp, path)

    # -------------------------------------------------------------- JSON
    def has(self, *key: str) -> bool:
        return self._path(key, ".json").exists()

    def save_json(self, payload: Any, *key: str) -> Path:
        path = self._path(key, ".json")
        self._atomic_write(path, lambda tmp: tmp.write_text(
            json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"))
        return path

    def load_json(self, *key: str) -> Optional[Any]:
        path = self._path(key, ".json")
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    # ------------------------------------------------------------- arrays
    def save_array(self, array: np.ndarray, *key: str) -> Path:
        path = self._path(key, ".npy")

        def write(tmp: Path) -> None:
            with open(tmp, "wb") as fh:  # file handle, so numpy does not append another .npy
                np.save(fh, array)

        self._atomic_write(path, write)
        return path

    def load_array(self, *key: str) -> Optional[np.ndarray]:
        path = self._path(key, ".npy")
        return np.load(path) if path.exists() else None

    def count(self, *key: str) -> int:
        """Number of JSON checkpoints directly under ``key``."""
        folder = self.root.joinpath(*[_safe(p) for p in key])
        return len(list(folder.glob("*.json"))) if folder.exists() else 0


class TaskCache:
    """Checkpoint view for one batch of tasks (one campaign's marathon, say).

    ``load(name)`` returns a finished task's saved result or ``None``;
    ``save(name, result, array)`` stores a result as soon as it arrives.
    A ``TaskCache`` with no store does nothing, so code can always call it.
    """

    def __init__(self, store: Optional[CheckpointStore], *prefix: str):
        self.store, self.prefix = store, prefix

    def load(self, name: str, with_array: bool = False) -> Optional[dict]:
        if self.store is None or not self.store.has(*self.prefix, name):
            return None
        result = self.store.load_json(*self.prefix, name)
        if with_array:
            array = self.store.load_array(*self.prefix, name)
            if array is None:  # json written but array missing: treat as not done
                return None
            result["array"] = array
        return result

    def save(self, name: str, result: dict, array: Optional[np.ndarray] = None) -> None:
        if self.store is None:
            return
        if array is not None:  # array first, so a json file always has its array
            self.store.save_array(array, *self.prefix, name)
        self.store.save_json(result, *self.prefix, name)


class BuildLockedError(RuntimeError):
    """Raised when another live session is already working on a build."""


class BuildLock:
    """Stops two sessions working on the same build at once.

    The lock file records who holds it and is refreshed every ``refresh`` seconds
    by a background thread. A session that dies (a Colab disconnect, say) stops
    refreshing it, so after ``stale_after`` seconds the lock is treated as
    abandoned and a new session may take it over.
    """

    def __init__(self, build_dir: str | Path, refresh: float = 30.0, stale_after: Optional[float] = None):
        import socket
        import threading
        self.path = Path(build_dir) / ".lock"
        if stale_after is None:  # OBD_LOCK_STALE_SECONDS shortens the wait in tests
            stale_after = float(os.environ.get("OBD_LOCK_STALE_SECONDS", 120))
        self.refresh, self.stale_after = min(refresh, stale_after / 3), stale_after
        self.owner = {"pid": os.getpid(), "host": socket.gethostname()}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._keep_fresh, daemon=True)

    def _write(self) -> None:
        import time
        payload = {**self.owner, "updated": time.time()}
        CheckpointStore._atomic_write(self.path, lambda tmp: tmp.write_text(json.dumps(payload)))

    def _keep_fresh(self) -> None:
        while not self._stop.wait(self.refresh):
            try:
                self._write()
            except OSError:
                pass  # a missed refresh is harmless; the next one will land

    def age_of_existing(self) -> Optional[float]:
        """Seconds since another holder last refreshed the lock, or ``None`` if there is none."""
        import time
        try:
            held = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if held.get("pid") == self.owner["pid"] and held.get("host") == self.owner["host"]:
            return None
        return time.time() - float(held.get("updated", 0))

    def acquire(self) -> "BuildLock":
        age = self.age_of_existing()
        if age is not None and age < self.stale_after:
            wait = int(self.stale_after - age) + 1
            raise BuildLockedError(
                f"Another session is working on this build (last active {int(age)}s ago). "
                f"If that session has died, wait {wait}s and try again.")
        self._write()
        self._thread.start()
        return self

    def release(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)
        try:
            self.path.unlink()
        except OSError:
            pass
