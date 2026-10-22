# -*- coding: utf-8 -*-
"""Stage 4: off-policy evaluation (OPE) benchmark.

How the benchmark works:

1. Question. Using only the random recommender's logs, estimate the CTR that
   Thompson Sampling would have achieved. This is what OPE is for: judging a
   new policy before giving it any live traffic.
2. Target policy. Thompson Sampling's chance of showing each item in each slot
   comes from 1,000,000 Monte Carlo draws of its production Beta prior.
3. Reward model. Model-based estimators need a click model q(x, a, slot). The
   winner of stage 3's model marathon is used, one per slot, with 3-fold
   cross-fitting so no row is scored by a model that saw it. Each fold's
   models train on at most ``--ope-fit-max-rows`` rows (default 200,000, the
   marathon's own sample size), which bounds the run time whichever model won.
4. Estimators. DM, IPW, SNIPW, DR, SNDR, Switch-DR and DR-os. The last two
   choose their weight cap with SLOPE++.
5. Ground truth. Because the dataset came from an A/B test, Thompson Sampling's
   real CTR is known from its own logs. Each estimate is scored by its relative
   error against it, over bootstrap resamples of the random logs.
"""

import time
from typing import Dict, List

import numpy as np
import pandas as pd
from joblib import delayed

from src.constants import CONTROL_POLICY, HEARTBEAT_SECONDS, TREATMENT_POLICY, OPEConstants
from src.checkpoints import TaskCache
from src.model_evaluation import estimator_from_winner
from src.obp_lite import (
    BernoulliTS,
    DirectMethod,
    DoublyRobust,
    DoublyRobustWithShrinkageTuning,
    InverseProbabilityWeighting,
    OffPolicyEvaluation,
    RegressionModel,
    SelfNormalizedDoublyRobust,
    SelfNormalizedInverseProbabilityWeighting,
    SwitchDoublyRobustTuning,
)
from src.utils import Heartbeat, Progress, format_duration, get_logger, iter_parallel, memory_safe_n_jobs

def make_estimators(lambdas: List[float] = OPEConstants.LAMBDAS) -> list:
    """Return a fresh list of the seven compared estimators."""
    return [
        DirectMethod(),
        InverseProbabilityWeighting(),
        SelfNormalizedInverseProbabilityWeighting(),
        DoublyRobust(),
        SelfNormalizedDoublyRobust(),
        SwitchDoublyRobustTuning(lambdas=lambdas),
        DoublyRobustWithShrinkageTuning(lambdas=lambdas),
    ]


def compute_target_action_dist(n_actions: int, len_list: int, campaign: str,
                               n_sim: int, random_state: int) -> np.ndarray:
    """Thompson Sampling's probability of each item in each slot, shape (n_actions, len_list)."""
    policy = BernoulliTS(n_actions=n_actions, len_list=len_list, is_zozotown_prior=True,
                         campaign=campaign, random_state=random_state)
    return policy.compute_batch_action_dist(n_sim=int(n_sim))[0]


def ope_fit_max_rows(config: dict) -> int | None:
    """The cap on click-model training rows per fold in stage 4 (None = no cap)."""
    return config.get("ope_fit_max_rows", OPEConstants.OPE_FIT_MAX_ROWS) or None


def estimate_job_bytes(n_rows: int, n_actions: int, len_list: int, dim_context: int) -> float:
    """Peak memory of one parallel bootstrap run, from a measurement at full size.

    Measured on a 1.37M-row copy of the "all" campaign: a worker peaked at about
    2.6 GB, of which about 0.5 GB is the data it receives and the rest scales
    with q-hat (rows x items x slots, float32). The formula reproduces that with
    a little margin, so parallel jobs are capped by what they really use.
    """
    q_hat = n_rows * n_actions * len_list * 4
    return 0.15e9 + 2 * n_rows * dim_context * 8 + 1.7 * q_hat


def _one_bootstrap_run(run: int, obd, action_dist_single: np.ndarray, base_model,
                       n_folds: int, random_state: int, rows: np.ndarray | None = None,
                       fit_max_rows: int | None = None) -> Dict[str, float]:
    """Estimate the target value on one bootstrap resample of the logs (or of ``rows`` only)."""
    bf = obd.sample_bootstrap_bandit_feedback(random_state=run, rows=rows)
    reg = RegressionModel(
        n_actions=obd.n_actions, len_list=obd.len_list, action_context=obd.action_context,
        base_model=base_model, fit_max_rows=fit_max_rows or None,
    )
    q_hat = reg.fit_predict(context=bf["context"], action=bf["action"], reward=bf["reward"],
                            position=bf["position"], n_folds=n_folds, random_state=random_state)
    action_dist = np.broadcast_to(action_dist_single, (bf["n_rounds"],) + action_dist_single.shape)
    ope = OffPolicyEvaluation(bandit_feedback=bf, ope_estimators=make_estimators())
    return ope.estimate_policy_values(action_dist, q_hat)


def _timed_bootstrap_run(run: int, *args, **kwargs) -> tuple[int, Dict[str, float], float]:
    """``_one_bootstrap_run`` plus its run number and duration, so results can be logged as they finish."""
    started = time.monotonic()
    estimates = _one_bootstrap_run(run, *args, **kwargs)
    return run, estimates, time.monotonic() - started


def run_bootstraps(label: str, n_runs: int, n_jobs: int, obd, action_dist: np.ndarray, click_model,
                   seed: int, rows: np.ndarray | None = None, heartbeat_seconds: float = HEARTBEAT_SECONDS,
                   cache: TaskCache | None = None, fit_max_rows: int | None = None):
    """Run the bootstrap repetitions in parallel, logging each one as it finishes with an ETA.

    Runs saved in ``cache`` by an earlier session are loaded instead of re-run;
    each new run is saved the moment it finishes. Yields ``(run, estimates)``.
    """
    cache = cache or TaskCache(None)
    done, pending = {}, []
    for b in range(n_runs):
        saved = cache.load(f"run_{b:03d}")
        if saved is None:
            pending.append(b)
        else:
            done[b] = saved["estimates"]
    if done:
        get_logger().info("%s: %d of %d runs loaded from checkpoints; %d to run",
                          label, len(done), n_runs, len(pending))
    yield from done.items()

    progress = Progress(label, n_runs, already_done=len(done))
    tasks = [delayed(_timed_bootstrap_run)(b, obd, action_dist, click_model, OPEConstants.N_FOLDS, seed, rows,
                                           fit_max_rows)
             for b in pending]
    with Heartbeat(progress, heartbeat_seconds):
        for run, estimates, seconds in iter_parallel(n_jobs, tasks):
            cache.save(f"run_{run:03d}", {"run": run, "estimates": estimates, "seconds": seconds})
            progress.update(f"run {run + 1} took {format_duration(seconds)}")
            yield run, estimates


def execute_ope_benchmark(datasets: Dict, winners: Dict[str, dict], config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run stage 4 for every campaign with that campaign's winning reward model.

    Args:
        datasets: Output of ``execute_data_pipeline``.
        winners: Output of ``execute_modeling_pipeline``: campaign -> winning model.
        config: Run configuration. Reads ``n_runs``, ``n_sim``, ``n_jobs`` and
            ``random_state``.

    Returns:
        ``(per_run, summary)``. ``per_run`` has one row per estimate; ``summary``
        has the mean estimate and mean relative error per estimator. Both are
        written to the build's artifacts.
    """
    LOG = get_logger()
    n_runs = config.get("n_runs", OPEConstants.N_RUNS)
    n_sim = config.get("n_sim", OPEConstants.N_SIM_ACTION_DIST)
    n_jobs = config.get("n_jobs", -1)
    seed = config.get("random_state", 12345)

    records = []
    for campaign in config["campaigns"]:
        logs = datasets[(CONTROL_POLICY, campaign)]
        ground_truth = float(datasets[(TREATMENT_POLICY, campaign)].reward.mean())
        action_dist = compute_target_action_dist(logs.n_actions, logs.len_list, campaign, n_sim, seed)
        LOG.info("OPE %s: target action distribution from %s draws; true BTS CTR %.4f%%",
                 campaign, f"{int(n_sim):,}", 100 * ground_truth)

        winner = winners[campaign]
        click_model = estimator_from_winner(winner, seed, winner.get("n_item_features", logs.action_context.shape[1]))
        jobs = memory_safe_n_jobs(n_jobs, estimate_job_bytes(logs.n_rounds, logs.n_actions, logs.len_list,
                                                             logs.dim_context), headroom=0.8)
        LOG.info("OPE %s: %d bootstrap runs with the marathon winner, %s (%d parallel jobs)",
                 campaign, n_runs, winner["name"], jobs)
        runs = run_bootstraps(f"OPE {campaign} bootstrap runs", n_runs, jobs, logs, action_dist, click_model, seed,
                              heartbeat_seconds=config.get("heartbeat_seconds", HEARTBEAT_SECONDS),
                              cache=TaskCache(config.get("checkpoints"), "ope", campaign, winner["name"]),
                              fit_max_rows=ope_fit_max_rows(config))
        for b, estimates in runs:
            for estimator_name, value in estimates.items():
                records.append({
                    "campaign": campaign, "base_model": winner["name"], "run": b,
                    "estimator": estimator_name, "estimate": value, "ground_truth": ground_truth,
                    "control_ctr": float(logs.reward.mean()),
                })

    # Runs finish in any order; sort so the saved table reads run by run.
    per_run = pd.DataFrame(records).sort_values(["campaign", "run", "estimator"], ignore_index=True)
    per_run["relative_error"] = (per_run["estimate"] - per_run["ground_truth"]).abs() / per_run["ground_truth"]
    summary = (
        per_run.groupby(["campaign", "base_model", "estimator"], as_index=False)
        .agg(mean_estimate=("estimate", "mean"), std_estimate=("estimate", "std"),
             ground_truth=("ground_truth", "first"), control_ctr=("control_ctr", "first"),
             relative_error_mean=("relative_error", "mean"), relative_error_std=("relative_error", "std"),
             n_runs=("run", "nunique"))
        .sort_values(["campaign", "base_model", "relative_error_mean"])
    )
    per_run.to_csv(config["output_files"]["ope_estimates_per_run"], index=False)
    summary.to_csv(config["output_files"]["ope_summary"], index=False)

    for campaign, grp in summary.groupby("campaign"):
        best = grp.iloc[grp["relative_error_mean"].argmin()]
        LOG.info("OPE %s: lowest error %s with %s (%.1f%%)", campaign, best["estimator"],
                 best["base_model"], 100 * best["relative_error_mean"])
    LOG.info("Wrote OPE summary: %s", config["output_files"]["ope_summary"])
    return per_run, summary
