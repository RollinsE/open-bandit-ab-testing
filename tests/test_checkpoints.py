import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline import main
from src.checkpoints import BuildLock, BuildLockedError, CheckpointStore, TaskCache


def test_store_round_trip_and_no_temp_files(tmp_path):
    store = CheckpointStore(tmp_path)
    store.save_json({"a": 1, "x": np.float64(0.5)}, "ope", "men", "run_000")
    store.save_array(np.arange(5.0), "marathon", "men", "Baseline: same CTR")
    assert store.load_json("ope", "men", "run_000") == {"a": 1, "x": 0.5}
    np.testing.assert_array_equal(store.load_array("marathon", "men", "Baseline: same CTR"), np.arange(5.0))
    assert store.count("ope", "men") == 1
    assert not list(tmp_path.rglob("*.tmp"))


def test_task_cache_needs_both_files_when_an_array_is_expected(tmp_path):
    store = CheckpointStore(tmp_path)
    cache = TaskCache(store, "marathon", "men")
    cache.save("Random Forest", {"name": "Random Forest"}, np.ones(3))
    assert cache.load("Random Forest", with_array=True)["array"].sum() == 3
    store.save_json({"name": "half written"}, "marathon", "men", "Orphan")  # json without its array
    assert cache.load("Orphan", with_array=True) is None
    assert TaskCache(None).load("anything") is None  # no store: always a miss


def test_lock_refuses_a_live_session_and_takes_over_a_dead_one(tmp_path):
    (tmp_path / ".lock").write_text(json.dumps({"pid": -1, "host": "elsewhere", "updated": time.time()}))
    with pytest.raises(BuildLockedError):
        BuildLock(tmp_path, stale_after=60).acquire()
    (tmp_path / ".lock").write_text(json.dumps({"pid": -1, "host": "elsewhere", "updated": time.time() - 600}))
    lock = BuildLock(tmp_path, stale_after=60).acquire()
    lock.release()
    assert not (tmp_path / ".lock").exists()


def _artifacts(build: Path) -> dict:
    # Timings ("Seconds") naturally differ between sessions; every result must match.
    return {name: pd.read_csv(build / "artifacts" / name).drop(columns=["Seconds"], errors="ignore")
            for name in ("model_performance.csv", "ope_estimates_per_run.csv", "estimator_selection.csv")}


def test_continue_build_reuses_checkpoints_and_matches_a_full_run(tmp_path):
    args = ["--output-dir", str(tmp_path), "--campaigns", "men", "--model-families", "logistic_regression",
            "hist_gradient_boosting", "--model-trials", "2", "--n-runs", "3", "--n-sim", "5000", "--n-jobs", "1"]
    build = Path(main(args))
    expected = _artifacts(build)

    # Simulate an interruption: drop some checkpoints and every output, mark the build unfinished.
    ckpt = build / "checkpoints"
    removed = sorted((ckpt / "marathon" / "men").glob("Hist*"))
    for path in removed + sorted(ckpt.glob("ope/men/*/run_002.json")) + [ckpt / "marathon" / "men" / "_result.json"]:
        path.unlink()
    for path in (build / "artifacts").glob("*"):
        path.unlink()
    manifest = json.loads((build / "run_manifest.json").read_text())
    manifest["status"] = "running"
    (build / "run_manifest.json").write_text(json.dumps(manifest))

    continued = Path(main(["--output-dir", str(tmp_path), "--continue-build", "latest", "--n-jobs", "1"]))
    assert continued == build
    for name, frame in _artifacts(build).items():
        sort = ["campaign", "Model"] if "Model" in frame else list(frame.columns[:3])
        pd.testing.assert_frame_equal(frame.sort_values(sort).reset_index(drop=True),
                                      expected[name].sort_values(sort).reset_index(drop=True))
    log = next((build / "logs").glob("*.log")).read_text()
    assert "loaded from checkpoints" in log and "Continuing build" in log
    assert json.loads((build / "run_manifest.json").read_text())["status"] == "complete"


def test_continue_latest_with_nothing_unfinished_is_an_error(tmp_path):
    main(["--output-dir", str(tmp_path), "--campaigns", "men", "--to-stage", "2", "--n-jobs", "1"])
    with pytest.raises(SystemExit):
        main(["--output-dir", str(tmp_path), "--continue-build", "latest"])


def test_continue_auto_starts_new_then_continues_unfinished(tmp_path):
    args = ["--output-dir", str(tmp_path), "--campaigns", "men", "--to-stage", "2", "--n-jobs", "1",
            "--continue-build", "auto"]
    first = Path(main(args))  # nothing to continue: a new build
    assert json.loads((first / "run_manifest.json").read_text())["status"] == "complete"
    second = Path(main(args))  # the first is complete, so another new build
    assert second != first

    manifest = json.loads((second / "run_manifest.json").read_text())
    manifest["status"] = "running"
    (second / "run_manifest.json").write_text(json.dumps(manifest))
    assert Path(main(args)) == second  # an unfinished build is continued in place
