#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Main pipeline orchestrator with build, stage selection and release support.

Stages:

1. Data loading and validation (always runs; it is fast and later stages need it)
2. A/B test: Thompson Sampling vs random, per campaign
3. Reward model marathon: click models compete; the simplest model within a margin of the best wins
4. Off-policy evaluation benchmark, using each campaign's winning click model,
   then fair estimator selection (chosen on one slice of data, scored on another)
5. A/B vs off-policy comparison
6. Reporting: interactive Plotly figures, report.md and report.html

Every run writes to ``<output-dir>/builds/<run_id>/`` with a run manifest, a
full log file and its artifacts. ``--promote-release`` copies a successful build
to ``<output-dir>/releases/<release-name>``.

Slow work (each marathon model, each bootstrap run) is checkpointed as it
finishes. If a run is interrupted, ``--continue-build latest`` (or a build
folder) carries on in the same build with the same settings, skipping
everything already done, and still produces one build and one report.
``--continue-build auto`` does that when an unfinished build exists and
otherwise starts a new one, so the same command works before and after a
disconnect.
"""

import argparse
import json
import sys
import time
from pathlib import Path

separator = "=" * 80

sys.path.insert(0, str(Path(__file__).parent))

from src.ab_testing import execute_ab_test
from src.checkpoints import BuildLock, BuildLockedError, CheckpointStore
from src.comparison import compare_ab_and_ope
from src.config import apply_output_directory, detect_runtime, get_config, validate_runtime_configuration
from src.constants import HEARTBEAT_SECONDS, LOG_FILE_PREFIX, MODEL_FAMILIES, PROJECT_NAME, STAGES, OPEConstants
from src.data_pipeline import execute_data_pipeline
from src.estimator_selection import execute_estimator_selection
from src.model_training import execute_modeling_pipeline
from src.off_policy_evaluation import execute_ope_benchmark
from src.release_manager import prepare_build_directory, promote_release, write_release_pointer
from src.reporting import build_reports
from src.stage_resume import (
    ResumeArtifactError,
    load_artifact,
    load_json_artifact,
    require_resume_for_skipped_dependencies,
    resolve_stage_plan,
    validate_resume_build,
)
from src.utils import format_duration, get_timestamp, setup_logging, write_json


def _stage_start(stage_no: int, title: str, log) -> None:
    message = f"Stage {stage_no}: {title}"
    print(f"\n{separator}")
    print(message.upper())
    print(separator)
    log.info(message)


def _stage_done(stage_no: int, title: str, log, **summary) -> None:
    clean_summary = {k: v for k, v in summary.items() if v is not None}
    suffix = ""
    if clean_summary:
        suffix = ": " + ", ".join(f"{k}={v}" for k, v in clean_summary.items())
    message = f"Stage {stage_no} complete: {title}{suffix}"
    print(message)
    log.info(message)


def _artifact_message(config: dict, log) -> None:
    print("\nArtifacts written:")
    for label, key in (("build", "output_dir"), ("artifacts", "artifact_dir"),
                       ("reports", "report_dir"), ("logs", "log_dir")):
        value = config.get(key)
        if value:
            print(f"  - {label}: {value}")
            log.info("%s: %s", label, value)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"{PROJECT_NAME} pipeline")
    parser.add_argument("--mode", choices=["colab", "production", "local"], default=None,
                        help="Runtime mode (auto-detected if not specified)")
    parser.add_argument("--output-dir", default="outputs",
                        help="Output root directory for builds and releases")
    parser.add_argument("--data-dir", default=None,
                        help="Folder of the full Open Bandit Dataset (defaults to the bundled sample)")
    parser.add_argument("--campaigns", nargs="+", choices=["all", "men", "women"], default=None,
                        help="Campaigns to analyse (default: all three)")
    parser.add_argument("--n-runs", type=int, default=OPEConstants.N_RUNS,
                        help="Bootstrap repetitions for the off-policy benchmark")
    parser.add_argument("--n-sim", type=int, default=OPEConstants.N_SIM_ACTION_DIST,
                        help="Monte Carlo draws for Thompson Sampling's action distribution")
    parser.add_argument("--model-families", nargs="+", default=None, choices=list(MODEL_FAMILIES),
                        help="Model families that enter the reward model marathon (default: every installed one "
                             "except sklearn's slow gradient_boosting)")
    parser.add_argument("--model-trials", type=int, default=OPEConstants.MODEL_TRIALS,
                        help="Tuning trials per tuned model family in the marathon (0 = no tuning)")
    parser.add_argument("--tune-top", type=int, default=OPEConstants.TUNE_TOP,
                        help="Tune only the best N families on their default settings (-1 = all, 0 = none)")
    parser.add_argument("--winner-rule", choices=["simplest", "best"], default=OPEConstants.WINNER_RULE,
                        help="'simplest': simplest model within the margin of the best (fast stage 4); "
                             "'best': lowest log loss")
    parser.add_argument("--winner-tolerance", type=float, default=OPEConstants.WINNER_TOLERANCE,
                        help="Practical margin for the 'simplest' rule, as a fraction of the best log loss "
                             "(default 0.0025 = 0.25%%). The margin used is the wider of this and one standard "
                             "error; 0 leaves only the standard error")
    parser.add_argument("--ope-fit-max-rows", type=int, default=OPEConstants.OPE_FIT_MAX_ROWS,
                        help="Stage 4: train the click model on at most this many rows per fold (a click-stratified "
                             "sample); predictions still cover every row. 0 = every row (much slower)")
    parser.add_argument("--no-ensemble", action="store_true",
                        help="Leave the average-of-three-families model out of the marathon")
    parser.add_argument("--marathon-max-rows", type=int, default=OPEConstants.MARATHON_MAX_ROWS,
                        help="Row cap for the marathon (stratified sample); 0 = use every row")
    parser.add_argument("--tune-max-rows", type=int, default=OPEConstants.TUNE_MAX_ROWS,
                        help="Sample size for each tuning search inside the marathon; 0 = every training row")
    parser.add_argument("--no-time-split", action="store_true",
                        help="Skip the time-split estimator selection (it re-runs the benchmark on each half)")
    parser.add_argument("--expected-control-share", type=float, default=None,
                        help="Intended share of impressions for the control (random) arm, e.g. 0.5. Turns on the "
                             "sample ratio check; off by default because the dataset's split is not documented")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Two-sided significance level for the A/B test")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Parallel jobs for the marathon and the off-policy benchmark (-1 = all cores)")
    parser.add_argument("--random-state", type=int, default=12345, help="Random seed")
    parser.add_argument("--promote-release", action="store_true",
                        help="Promote successful build to releases/<release-name>")
    parser.add_argument("--release-name", default="latest",
                        help="Release name to promote on success")
    parser.add_argument("--heartbeat-minutes", type=float, default=HEARTBEAT_SECONDS / 60,
                        help="During long steps, log a 'still running' line after this many quiet minutes")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    parser.add_argument("--show-plots", action="store_true",
                        help="Show figures inline in a notebook. CLI runs are file-first by default.")
    parser.add_argument("--continue-build", default=None, metavar="BUILD|latest|auto",
                        help="Continue an interrupted build in place: a build folder, 'latest' for the newest "
                             "unfinished build in --output-dir, or 'auto' to continue it if there is one and "
                             "otherwise start a new build with the other options given. Its settings come from its run manifest; only "
                             "--data-dir, --n-jobs, --log-level, --heartbeat-minutes, --show-plots and "
                             "--expected-control-share can change.")
    parser.add_argument("--resume-build", default=None,
                        help="Read skipped stages' outputs from an existing build. Outputs still go to a new build.")
    parser.add_argument("--from-stage", type=int, default=1, choices=sorted(STAGES),
                        help="First stage to execute. Stage 1 always runs.")
    parser.add_argument("--to-stage", type=int, default=max(STAGES), choices=sorted(STAGES),
                        help="Last stage to execute.")
    parser.add_argument("--only-stage", type=int, default=None, choices=sorted(STAGES),
                        help="Run one stage only (plus stage 1).")
    return parser


SETTING_KEYS = ("alpha", "n_runs", "n_sim", "model_families", "model_trials", "marathon_max_rows",
                "tune_max_rows", "tune_top", "winner_rule", "winner_tolerance", "ope_fit_max_rows", "ensemble",
                "time_split", "random_state",
                "expected_control_share")


def write_run_manifest(config: dict, args: argparse.Namespace, stage_plan, resume_build) -> None:
    """Record what this build runs with. ``--continue-build`` reads the settings back from here."""
    write_json(config["output_files"]["run_manifest"], {
        "project": PROJECT_NAME,
        "runtime": config["runtime"],
        "run_id": config["run_id"],
        "generated_at": get_timestamp(),
        "status": "running",
        "sessions": [{"started_at": get_timestamp(), "runtime": config["runtime"]}],
        "data_directory": config["data_directory"],
        "using_bundled_sample": config["using_bundled_sample"],
        "campaigns": config["campaigns"],
        "output_root": config["output_root"],
        "output_dir": config["output_dir"],
        "settings": {key: config[key] for key in SETTING_KEYS},
        "stage_plan": {"from_stage": stage_plan.from_stage, "to_stage": stage_plan.to_stage,
                       "stages": stage_plan.stages},
        "resume_build": str(resume_build) if resume_build else None,
        "release_name": args.release_name if args.promote_release else None,
    })


def update_run_manifest(path: str | Path, **changes) -> dict:
    """Merge ``changes`` into an existing manifest; ``new_session`` appends to its session list."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    session = changes.pop("new_session", None)
    if session:
        manifest.setdefault("sessions", []).append(session)
    manifest.update(changes)
    write_json(path, manifest)
    return manifest


def find_build_to_continue(target: str, output_root: Path) -> Path:
    """Resolve ``--continue-build``: a build folder, or 'latest' for the newest unfinished build."""
    if target != "latest":
        build = Path(target)
        if not (build / "run_manifest.json").exists():
            raise ResumeArtifactError(f"Not a build folder (no run_manifest.json): {build}")
        return build
    builds = sorted(p for p in (output_root / "builds").glob("*") if (p / "run_manifest.json").exists())
    unfinished = [p for p in builds
                  if json.loads((p / "run_manifest.json").read_text(encoding="utf-8")).get("status")
                  not in ("complete", "invalid")]
    if not unfinished:
        raise ResumeArtifactError(f"No unfinished build to continue in {output_root / 'builds'}")
    return unfinished[-1]


# Settings added after a build was started take these values when that build is
# continued, so it carries on with the behaviour it began with.
LEGACY_SETTINGS = {"tune_top": -1, "winner_rule": "best", "winner_tolerance": 0.0, "ope_fit_max_rows": 0,
                   "ensemble": True}


def apply_manifest_to_args(args: argparse.Namespace, manifest: dict) -> argparse.Namespace:
    """Make a continued run use exactly the settings its build started with."""
    settings = {**LEGACY_SETTINGS, **manifest["settings"]}
    for key in SETTING_KEYS:
        if key == "expected_control_share" and args.expected_control_share is not None:
            continue  # stage 2 is recomputed every session, so this one may change
        if key == "time_split":
            args.no_time_split = not settings.get("time_split", True)
        elif key == "ensemble":
            args.no_ensemble = not settings.get("ensemble", True)
        elif key in settings:
            setattr(args, key, settings[key])
    args.campaigns = manifest["campaigns"]
    args.from_stage = manifest["stage_plan"]["from_stage"]
    args.to_stage = manifest["stage_plan"]["to_stage"]
    args.only_stage = None
    args.resume_build = manifest.get("resume_build")
    args.promote_release = bool(manifest.get("release_name"))
    args.release_name = manifest.get("release_name") or args.release_name
    args.mode = args.mode or manifest.get("runtime")
    if not args.data_dir:
        args.data_dir = manifest.get("data_directory")
    return args


def resume_data_directory(resume_build: Path) -> str | None:
    """The data folder an earlier build used, if it still exists, so a resumed run reads the same data."""
    manifest_path = resume_build / "run_manifest.json"
    if not manifest_path.exists():
        return None
    data_dir = json.loads(manifest_path.read_text(encoding="utf-8")).get("data_directory")
    return data_dir if data_dir and Path(data_dir).is_dir() else None


def check_same_data(resume_build: Path | None, summary, data_directory: str) -> None:
    """Refuse to combine an earlier build's results with a different dataset.

    Stage 1 always runs, so its row and click counts are compared with the
    earlier build's for every split both contain. Mixing, say, results from the
    full dataset with the bundled sample would give a report that contradicts
    itself.
    """
    if resume_build is None:
        return
    earlier_path = resume_build / "artifacts" / "dataset_summary.csv"
    if not earlier_path.exists():
        return
    import pandas as pd
    keys = ["policy", "campaign"]
    earlier = pd.read_csv(earlier_path).set_index(keys)[["impressions", "clicks"]]
    current = summary.set_index(keys)[["impressions", "clicks"]]
    shared = current.index.intersection(earlier.index)
    if len(shared) and not current.loc[shared].astype(int).equals(earlier.loc[shared].astype(int)):
        raise ResumeArtifactError(
            f"The data in {data_directory} is not the data {resume_build.name} was built from "
            f"({int(current['impressions'].sum()):,} rows here against {int(earlier['impressions'].sum()):,} "
            "for the same splits there). Pass --data-dir with the dataset that build used.")


def carry_over(source: Path | None, config: dict, filenames: tuple) -> None:
    """Copy outputs reused from ``--resume-build`` into this build, so every build is self-contained."""
    if source is None:
        return
    import shutil
    for name in filenames:
        src = source / "artifacts" / name
        if src.exists():
            shutil.copy2(src, Path(config["artifact_dir"]) / name)


def main(args: argparse.Namespace | None = None):
    parser = create_parser()
    if args is None:
        args = parser.parse_args()
    elif not isinstance(args, argparse.Namespace):
        args = parser.parse_args(args)

    output_root = Path(args.output_dir)
    continuing = None
    try:
        if args.continue_build == "auto":
            try:
                continuing = find_build_to_continue("latest", output_root)
            except ResumeArtifactError:
                continuing = None  # nothing unfinished: start a new build with the options given
        elif args.continue_build:
            continuing = find_build_to_continue(args.continue_build, output_root)
        if continuing:
            manifest = json.loads((continuing / "run_manifest.json").read_text(encoding="utf-8"))
            args = apply_manifest_to_args(args, manifest)
            output_root = Path(manifest.get("output_root") or output_root)
        stage_plan = resolve_stage_plan(args.only_stage, args.from_stage, args.to_stage)
        resume_build = validate_resume_build(args.resume_build)
        require_resume_for_skipped_dependencies(stage_plan, str(resume_build) if resume_build else None)
        if resume_build and not continuing and not args.data_dir:
            args.data_dir = resume_data_directory(resume_build)
    except (ValueError, ResumeArtifactError) as exc:
        parser.error(str(exc))

    runtime = args.mode or detect_runtime()
    if continuing:
        build_dir, run_id = continuing, continuing.name
    else:
        build_dir = prepare_build_directory(output_root, get_timestamp())
        run_id = build_dir.name  # carries the _2 suffix if two runs share a timestamp
    try:
        lock = BuildLock(build_dir).acquire()
    except BuildLockedError as exc:
        parser.error(str(exc))

    log_dir = build_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{LOG_FILE_PREFIX}_{run_id}.log"  # a continued build appends to the same log
    LOG = setup_logging(args.log_level, str(log_file))

    print(f"{PROJECT_NAME} pipeline run" + (" (continuing)" if continuing else ""))
    print("-" * 80)
    print(f"  runtime: {runtime}")
    print(f"  build:   {build_dir}")
    print(f"  stages:  {stage_plan.from_stage}-{stage_plan.to_stage}")
    if resume_build:
        print(f"  resume:  {resume_build}")
    if continuing:
        LOG.info("=" * 60)
        LOG.info("Continuing build %s with the settings in its run manifest; finished work is loaded "
                 "from checkpoints", run_id)
    LOG.info(f"Starting {PROJECT_NAME} pipeline")
    LOG.info(f"Runtime mode: {runtime}")
    LOG.info(f"Output root: {output_root}")
    LOG.info(f"Build directory: {build_dir}")
    LOG.info("Stage plan: %s-%s", stage_plan.from_stage, stage_plan.to_stage)
    if resume_build:
        LOG.info("Resume build: %s", resume_build)

    try:
        config = get_config(runtime, args.data_dir)
        config = apply_output_directory(config, str(build_dir))
        config.update(
            output_root=str(output_root), run_id=run_id,
            campaigns=args.campaigns or config["campaigns"],
            alpha=args.alpha, n_runs=args.n_runs, n_sim=args.n_sim,
            expected_control_share=args.expected_control_share,
            model_families=args.model_families, model_trials=args.model_trials,
            marathon_max_rows=args.marathon_max_rows, tune_max_rows=args.tune_max_rows,
            tune_top=args.tune_top, winner_rule=args.winner_rule, winner_tolerance=args.winner_tolerance,
            ope_fit_max_rows=args.ope_fit_max_rows,
            ensemble=not args.no_ensemble,
            time_split=not args.no_time_split,
            heartbeat_seconds=max(10.0, 60 * args.heartbeat_minutes),
            n_jobs=args.n_jobs, random_state=args.random_state,
            enable_inline_display=bool(args.show_plots),
        )
        config["checkpoints"] = CheckpointStore(config["checkpoint_dir"])
        LOG.info("Data directory: %s", config["data_directory"])
        validate_runtime_configuration(config)
        if continuing:
            update_run_manifest(config["output_files"]["run_manifest"], status="running",
                                data_directory=config["data_directory"],
                                new_session={"started_at": get_timestamp(), "runtime": runtime})
        else:
            write_run_manifest(config, args, stage_plan, resume_build)
        session_start = time.monotonic()

        ab_results = power_table = leaderboard = winners = ope_summary = selection = comparison = None

        _stage_start(1, STAGES[1], LOG)
        datasets, summary = execute_data_pipeline(config)
        _stage_done(1, STAGES[1], LOG, splits=len(datasets), rows=int(summary["impressions"].sum()))
        check_same_data(resume_build, summary, config["data_directory"])

        def reused(stage: int) -> bool:
            """Reuse --resume-build outputs only for stages before the first one re-run here.

            A skipped stage after the re-run ones would be stale (built from the old
            results), so it is left out of this build instead of being copied in.
            """
            return resume_build is not None and stage < stage_plan.from_stage

        if stage_plan.includes(2):
            _stage_start(2, STAGES[2], LOG)
            ab_results, power_table = execute_ab_test(datasets, config)
            _stage_done(2, STAGES[2], LOG, campaigns=len(ab_results),
                        significant=int(ab_results["significant"].sum()))
        elif reused(2):
            ab_results = load_artifact(resume_build, "ab_test_results.csv")
            power_table = load_artifact(resume_build, "ab_test_power.csv")
            carry_over(resume_build, config, ("ab_test_results.csv", "ab_test_power.csv"))

        if stage_plan.includes(3):
            _stage_start(3, STAGES[3], LOG)
            leaderboard, winners = execute_modeling_pipeline(datasets, config)
            _stage_done(3, STAGES[3], LOG, models=int(leaderboard["Model"].nunique()),
                        winners=", ".join(f"{c}={w['name']}" for c, w in winners.items()))
        elif reused(3):
            leaderboard = load_artifact(resume_build, "model_performance.csv")
            winners = load_json_artifact(resume_build, "reward_model_winners.json")
            carry_over(resume_build, config, ("model_performance.csv", "reward_model_winners.json",
                                              "model_diagnostics.json"))

        if stage_plan.includes(4):
            if winners is None or any(c not in winners for c in config["campaigns"]):
                raise RuntimeError("Stage 4 needs the stage 3 winners for every campaign. Include stage 3 or use --resume-build.")
            _stage_start(4, STAGES[4], LOG)
            _, ope_summary = execute_ope_benchmark(datasets, winners, config)
            selection = execute_estimator_selection(datasets, winners, ope_summary, config)
            _stage_done(4, STAGES[4], LOG, estimates=len(ope_summary), runs=config["n_runs"],
                        selections=len(selection))
        elif reused(4):
            ope_summary = load_artifact(resume_build, "ope_summary.csv")
            selection = load_artifact(resume_build, "estimator_selection.csv")
            carry_over(resume_build, config, ("ope_summary.csv", "ope_estimates_per_run.csv",
                                              "estimator_selection.csv", "ope_time_split.csv"))

        if stage_plan.includes(5):
            if ab_results is None or ope_summary is None:
                raise RuntimeError("Stage 5 requires stage 2 and 4 outputs. Include them or use --resume-build.")
            _stage_start(5, STAGES[5], LOG)
            comparison = compare_ab_and_ope(ab_results, ope_summary, config, selection)
            _stage_done(5, STAGES[5], LOG, inside_ab_interval=f"{int(comparison['within_ab_ci'].sum())}/{len(comparison)}")
        elif reused(5):
            comparison = load_artifact(resume_build, "ab_vs_ope_comparison.csv")
            carry_over(resume_build, config, ("ab_vs_ope_comparison.csv",))
            if comparison is not None and "selected_by_campaign_split" not in comparison.columns:
                comparison["selected_by_campaign_split"] = False

        if stage_plan.includes(6):
            _stage_start(6, STAGES[6], LOG)
            written = build_reports(summary, ab_results, power_table, leaderboard, ope_summary,
                                    selection, comparison, config)
            _stage_done(6, STAGES[6], LOG, files=len(written))

        # Mark the build complete before promoting it, so the release's manifest says so too.
        update_run_manifest(config["output_files"]["run_manifest"], status="complete",
                            completed_at=get_timestamp())
        if args.promote_release:
            release_dir = promote_release(build_dir, output_root, args.release_name)
            pointer = write_release_pointer(output_root, args.release_name, run_id)
            print(f"Release promoted: {release_dir}")
            LOG.info(f"Promoted build to release: {release_dir}")
            LOG.info(f"Updated release pointer: {pointer}")
        _artifact_message(config, LOG)
        print(f"\n{separator}")
        print("Pipeline completed successfully")
        LOG.info("Pipeline completed successfully (this session: %s)",
                 format_duration(time.monotonic() - session_start))
        if args.promote_release:
            # Refresh the release's copy of the log so it includes these final lines.
            import shutil
            for handler in LOG.handlers:
                handler.flush()
            shutil.copy2(log_file, release_dir / "logs" / log_file.name)
        return build_dir

    except ResumeArtifactError as exc:
        # A usage problem (wrong data or build folder), not a crash: no traceback needed.
        LOG.error(f"Pipeline stopped: {exc}")
        manifest_path = build_dir / "run_manifest.json"
        if manifest_path.exists() and not continuing:
            # Mark it so --continue-build never picks this build up again.
            update_run_manifest(manifest_path, status="invalid", error=str(exc))
        sys.exit(1)
    except Exception as exc:
        LOG.error(f"Pipeline failed: {exc}", exc_info=True)
        LOG.error("Pipeline terminated. Finished work is checkpointed; after fixing the cause, run again "
                  "with --continue-build %s", build_dir)
        sys.exit(1)
    finally:
        lock.release()


if __name__ == "__main__":
    main()
