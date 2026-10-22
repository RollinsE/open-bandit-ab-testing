import json
from pathlib import Path

import pandas as pd
import pytest

from pipeline import main


def test_full_pipeline_runs_on_bundled_sample(tmp_path):
    build_dir = main([
        "--output-dir", str(tmp_path), "--campaigns", "men",
        "--n-runs", "2", "--n-sim", "5000", "--model-trials", "2",
        "--model-families", "logistic_regression", "hist_gradient_boosting",
        "--n-jobs", "1", "--promote-release",
    ])
    build_dir = Path(build_dir)
    for name in ("dataset_summary.csv", "ab_test_results.csv", "model_performance.csv",
                 "reward_model_winners.json", "ope_summary.csv", "ope_time_split.csv",
                 "estimator_selection.csv", "ab_vs_ope_comparison.csv"):
        assert (build_dir / "artifacts" / name).exists()
    assert (build_dir / "reports" / "report.html").exists()
    assert (build_dir / "run_manifest.json").exists()
    log_text = next((build_dir / "logs").glob("open_bandit_pipeline_*.log")).read_text(encoding="utf-8")
    assert "Stage 6 complete: Reporting" in log_text
    assert "Model marathon men: winner" in log_text
    leaderboard = pd.read_csv(build_dir / "artifacts" / "model_performance.csv")
    assert leaderboard["winner"].sum() == 1
    winner = leaderboard[leaderboard["winner"]].iloc[0]
    assert winner["Role"] == "candidate"
    assert winner["Within margin of best"]  # default rule: simplest model within the margin of the best
    ope = pd.read_csv(build_dir / "artifacts" / "ope_summary.csv")
    assert set(ope["base_model"]) == {winner["Model"]}
    selection = pd.read_csv(build_dir / "artifacts" / "estimator_selection.csv")
    assert list(selection["split"]) == ["time"]  # one campaign: no campaign split, time split only
    release = tmp_path / "releases" / "latest"
    assert (release / "reports" / "report.md").exists()
    assert not (release / "checkpoints").exists() and not (release / ".lock").exists()
    release_log = next((release / "logs").glob("*.log")).read_text(encoding="utf-8")
    assert "Pipeline completed successfully" in release_log
    ab = pd.read_csv(build_dir / "artifacts" / "ab_test_results.csv")
    assert list(ab["campaign"]) == ["men"]

    # The README figure script rebuilds every figure from a finished build's artifacts.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "export_figures", Path(__file__).resolve().parents[1] / "scripts" / "export_figures.py")
    export = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export)
    assert set(export.build_figures(release)) == set(export.FIGURES)


def test_report_only_rebuild_from_resume_build(tmp_path):
    first = Path(main(["--output-dir", str(tmp_path), "--campaigns", "women", "--n-runs", "2",
                       "--n-sim", "5000", "--model-trials", "0", "--model-families", "logistic_regression",
                       "--n-jobs", "1"]))
    second = Path(main(["--output-dir", str(tmp_path), "--campaigns", "women", "--only-stage", "6",
                        "--resume-build", str(first), "--promote-release"]))
    assert second != first
    report = (second / "reports" / "report.md").read_text(encoding="utf-8")
    assert "Off-policy evaluation" in report and "Reward model marathon" in report
    # The new build is self-contained: every result table of the earlier build is carried over.
    tables = {p.name for p in (first / "artifacts").glob("*.csv")}
    assert tables <= {p.name for p in (second / "artifacts").glob("*.csv")}
    release = json.loads((tmp_path / "releases" / "latest" / "run_manifest.json").read_text())
    assert release["status"] == "complete"


def test_ope_only_rerun_reuses_marathon_winner(tmp_path):
    first = Path(main(["--output-dir", str(tmp_path), "--campaigns", "men", "--n-runs", "2", "--n-sim", "5000",
                       "--model-trials", "0", "--model-families", "logistic_regression", "--n-jobs", "1"]))
    assert (first / "artifacts" / "ab_vs_ope_comparison.csv").exists()
    second = Path(main(["--output-dir", str(tmp_path), "--campaigns", "men", "--only-stage", "4",
                        "--resume-build", str(first), "--n-runs", "2", "--n-sim", "5000", "--n-jobs", "1"]))
    assert (second / "artifacts" / "ope_summary.csv").exists()
    # Upstream results are carried over; nothing downstream of the re-run stage is copied in stale.
    assert (second / "artifacts" / "model_performance.csv").exists()
    assert not (second / "artifacts" / "ab_vs_ope_comparison.csv").exists()


def test_resumed_run_uses_the_same_data_as_the_build_it_resumes(tmp_path):
    first = Path(main(["--output-dir", str(tmp_path), "--campaigns", "men", "--n-runs", "2", "--n-sim", "5000",
                       "--model-trials", "0", "--model-families", "logistic_regression", "--n-jobs", "1"]))
    # No --data-dir: the earlier build's data folder is used.
    second = Path(main(["--output-dir", str(tmp_path), "--campaigns", "men", "--only-stage", "6",
                        "--resume-build", str(first)]))
    data_dirs = [json.loads((b / "run_manifest.json").read_text())["data_directory"] for b in (first, second)]
    assert data_dirs[0] == data_dirs[1]

    # Results from one dataset are never combined with another: pretend the first build saw other data.
    summary = pd.read_csv(first / "artifacts" / "dataset_summary.csv")
    summary["impressions"] *= 100
    summary.to_csv(first / "artifacts" / "dataset_summary.csv", index=False)
    with pytest.raises(SystemExit):
        main(["--output-dir", str(tmp_path), "--campaigns", "men", "--only-stage", "6", "--resume-build", str(first)])
    stopped = sorted((tmp_path / "builds").iterdir())[-1]
    assert json.loads((stopped / "run_manifest.json").read_text())["status"] == "invalid"
    with pytest.raises(SystemExit):  # nothing left to continue: the stopped build is not picked up
        main(["--output-dir", str(tmp_path), "--continue-build", "latest"])
