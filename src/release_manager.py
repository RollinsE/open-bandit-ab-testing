# -*- coding: utf-8 -*-
"""Build folders, release promotion and the release pointer.

A run always writes to a new ``builds/<run_id>`` folder. Promoting copies a
finished build to ``releases/<name>`` and records which build that was in
``releases/release_pointer.json``, so the latest good results have a fixed path.
"""

import shutil
from pathlib import Path

from src.utils import get_timestamp, write_json


def prepare_build_directory(output_root: str | Path, run_id: str) -> Path:
    """Create and return ``<output_root>/builds/<run_id>``.

    Two runs started in the same second get ``<run_id>_2``, ``<run_id>_3`` and so
    on, so a build is never written over.
    """
    root = Path(output_root)
    build_dir = root / 'builds' / run_id
    suffix = 2
    while build_dir.exists():
        build_dir = root / 'builds' / f"{run_id}_{suffix}"
        suffix += 1
    build_dir.mkdir(parents=True)
    return build_dir


def promote_release(build_dir: str | Path, output_root: str | Path, release_name: str = 'latest') -> Path:
    """Copy a finished build to ``<output_root>/releases/<release_name>``, replacing any old copy.

    The release holds the results (manifest, logs, artifacts, reports). Checkpoints
    and the build lock are working files for resuming a run, so they stay in the build.
    """
    build_dir = Path(build_dir)
    output_root = Path(output_root)
    release_dir = output_root / 'releases' / release_name
    if release_dir.exists():
        shutil.rmtree(release_dir)
    release_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(build_dir, release_dir, ignore=shutil.ignore_patterns('checkpoints', '.lock', '*.tmp'))
    return release_dir


def write_release_pointer(output_root: str | Path, release_name: str, run_id: str) -> Path:
    """Record which build a release was promoted from."""
    pointer_path = Path(output_root) / 'releases' / 'release_pointer.json'
    return write_json(pointer_path, {
        "release_name": release_name,
        "run_id": run_id,
        "promoted_at": get_timestamp(),
    })
