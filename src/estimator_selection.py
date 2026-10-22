# -*- coding: utf-8 -*-
"""Fair estimator selection: choose an estimator on one slice of data, score it on another.

Picking the off-policy estimator that lands closest to Thompson Sampling's true
CTR, and then reporting that same error, flatters the winner: part of its win
is luck. This module answers the question that matters in practice instead:
"if I had used this rule to choose an estimator, how wrong would I have been
on data it never saw?"

How the selection works:

1. Campaign split (leave one campaign out). For each campaign, the estimator
   with the lowest mean error on the *other* campaigns is chosen and scored on
   the held-out one. Uses the stage 4 benchmark results, so it costs nothing.
   Caveat: the "all" campaign shares many items with "men" and "women", so it
   is not a fully independent test.
2. Time split. Each campaign's logging period is cut at the median timestamp of
   the random arm. Every estimator is run on the first half (scored against
   Thompson Sampling's CTR over the same days) and on the second half. The one
   with the lowest error in the first half is chosen and scored on the second.
   Both arms ran in parallel, so the truth for each half is well defined.
3. Context. Each test result is shown next to the best estimator in hindsight
   ("oracle", optimistic by construction) and the median estimator, so the
   cost of having to choose is visible.
"""

from typing import Dict

import numpy as np
import pandas as pd

from src.constants import CONTROL_POLICY, HEARTBEAT_SECONDS, TREATMENT_POLICY, OPEConstants
from src.checkpoints import TaskCache
from src.model_evaluation import estimator_from_winner
from src.off_policy_evaluation import compute_target_action_dist, estimate_job_bytes, ope_fit_max_rows, run_bootstraps
from src.utils import get_logger, memory_safe_n_jobs


# =============================================================================
# SELECTION RULE
# =============================================================================

def _choose_and_score(selection_errors: pd.Series, test_errors: pd.Series) -> dict:
    """Pick the estimator with the lowest selection error; report its test error with context.

    Args:
        selection_errors: Mean relative error per estimator on the selection data.
        test_errors: Mean relative error per estimator on the held-out data.
    """
    common = selection_errors.index.intersection(test_errors.index)
    selection_errors, test_errors = selection_errors[common], test_errors[common]
    selected = selection_errors.idxmin()
    oracle = test_errors.idxmin()
    return {
        "selected_estimator": selected,
        "selection_error": float(selection_errors[selected]),
        "test_error": float(test_errors[selected]),
        "oracle_estimator": oracle,
        "oracle_error": float(test_errors[oracle]),
        "median_error": float(test_errors.median()),
        "selected_rank": int(test_errors.rank(method="min")[selected]),
        "n_estimators": int(len(test_errors)),
    }


def select_by_campaign(ope_summary: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-campaign-out selection from the stage 4 benchmark summary.

    Returns:
        One row per held-out campaign; empty when fewer than two campaigns ran.
    """
    errors = ope_summary.pivot_table(index="estimator", columns="campaign", values="relative_error_mean")
    rows = []
    if errors.shape[1] < 2:
        return pd.DataFrame(rows)
    for held_out in errors.columns:
        others = [c for c in errors.columns if c != held_out]
        result = _choose_and_score(errors[others].mean(axis=1), errors[held_out])
        held = ope_summary[(ope_summary["campaign"] == held_out) &
                           (ope_summary["estimator"] == result["selected_estimator"])].iloc[0]
        rows.append({
            "split": "campaign", "held_out": held_out, "selected_on": " + ".join(others),
            **result,
            "test_estimate": float(held["mean_estimate"]), "test_truth": float(held["ground_truth"]),
            "selected_lift": float(held["mean_estimate"] / held["control_ctr"] - 1),
        })
    return pd.DataFrame(rows)


# =============================================================================
# TIME SPLIT
# =============================================================================

def split_by_time(control, treatment) -> dict:
    """Cut both arms at the control arm's median timestamp.

    Returns:
        ``{"cutoff", "control_rows": (first, second), "treatment_ctr": (first, second),
        "control_ctr": (first, second)}``.
    """
    # Timestamps are parsed to UTC at load time; drop the zone to get fast datetime64 arrays.
    t_control = control.data["timestamp"].dt.tz_localize(None).to_numpy()
    t_treatment = treatment.data["timestamp"].dt.tz_localize(None).to_numpy()
    cutoff = np.sort(t_control)[len(t_control) // 2]
    first = np.flatnonzero(t_control < cutoff)
    second = np.flatnonzero(t_control >= cutoff)
    treat_first = t_treatment < cutoff
    return {
        "cutoff": pd.Timestamp(cutoff).isoformat() + "+00:00",
        "control_rows": (first, second),
        "control_ctr": (float(control.reward[first].mean()), float(control.reward[second].mean())),
        "treatment_ctr": (float(treatment.reward[treat_first].mean()), float(treatment.reward[~treat_first].mean())),
    }


def run_time_split(datasets: Dict, winners: Dict[str, dict], config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the benchmark separately on each half of every campaign's logging period.

    Returns:
        ``(per_period, selection)``: mean estimate and error per campaign, period
        and estimator; and one selection row per campaign.
    """
    LOG = get_logger()
    n_runs = config.get("n_runs", OPEConstants.N_RUNS)
    n_sim = config.get("n_sim", OPEConstants.N_SIM_ACTION_DIST)
    seed = config.get("random_state", 12345)
    records, selection = [], []

    for campaign in config["campaigns"]:
        control = datasets[(CONTROL_POLICY, campaign)]
        treatment = datasets[(TREATMENT_POLICY, campaign)]
        split = split_by_time(control, treatment)
        action_dist = compute_target_action_dist(control.n_actions, control.len_list, campaign, n_sim, seed)
        winner = winners[campaign]
        click_model = estimator_from_winner(winner, seed, winner.get("n_item_features", control.action_context.shape[1]))
        jobs = memory_safe_n_jobs(config.get("n_jobs", -1),
                                  estimate_job_bytes(control.n_rounds // 2, control.n_actions, control.len_list,
                                                     control.dim_context), headroom=0.8)
        LOG.info("Time split %s: cutoff %s; %s / %s control rows; true treatment CTR %.4f%% / %.4f%%",
                 campaign, split["cutoff"], f"{len(split['control_rows'][0]):,}",
                 f"{len(split['control_rows'][1]):,}", 100 * split["treatment_ctr"][0], 100 * split["treatment_ctr"][1])

        for period, label in enumerate(("first half", "second half")):
            rows = split["control_rows"][period]
            truth = split["treatment_ctr"][period]
            runs = run_bootstraps(f"Time split {campaign}, {label}", n_runs, jobs, control, action_dist,
                                  click_model, seed, rows,
                                  heartbeat_seconds=config.get("heartbeat_seconds", HEARTBEAT_SECONDS),
                                  cache=TaskCache(config.get("checkpoints"), "time_split", campaign,
                                                  winner["name"], label),
                                  fit_max_rows=ope_fit_max_rows(config))
            for b, estimates in runs:
                for estimator, value in estimates.items():
                    records.append({"campaign": campaign, "period": label, "run": b, "estimator": estimator,
                                    "estimate": value, "ground_truth": truth,
                                    "control_ctr": split["control_ctr"][period]})

        per = pd.DataFrame([r for r in records if r["campaign"] == campaign])
        per["relative_error"] = (per["estimate"] - per["ground_truth"]).abs() / per["ground_truth"]
        errors = per.groupby(["period", "estimator"])["relative_error"].mean().unstack("period")
        result = _choose_and_score(errors["first half"], errors["second half"])
        second = per[(per["period"] == "second half") & (per["estimator"] == result["selected_estimator"])]
        estimate = float(second["estimate"].mean())
        selection.append({
            "split": "time", "held_out": campaign, "selected_on": f"{campaign}, before {split['cutoff'][:16]}",
            **result, "test_estimate": estimate, "test_truth": split["treatment_ctr"][1],
            "selected_lift": estimate / split["control_ctr"][1] - 1,
        })
        LOG.info("Time split %s: chose %s on the first half; second-half error %.1f%% (best in hindsight: %s, %.1f%%)",
                 campaign, result["selected_estimator"], 100 * result["test_error"],
                 result["oracle_estimator"], 100 * result["oracle_error"])

    per_period = pd.DataFrame(records)
    per_period["relative_error"] = (per_period["estimate"] - per_period["ground_truth"]).abs() / per_period["ground_truth"]
    summary = (per_period.groupby(["campaign", "period", "estimator"], as_index=False)
               .agg(mean_estimate=("estimate", "mean"), ground_truth=("ground_truth", "first"),
                    control_ctr=("control_ctr", "first"), relative_error_mean=("relative_error", "mean"),
                    relative_error_std=("relative_error", "std")))
    return summary, pd.DataFrame(selection)


# =============================================================================
# STAGE ENTRY POINT
# =============================================================================

def execute_estimator_selection(datasets: Dict, winners: Dict[str, dict], ope_summary: pd.DataFrame,
                                config: dict) -> pd.DataFrame:
    """Run both selection splits and write ``estimator_selection.csv`` and ``ope_time_split.csv``.

    Returns:
        One row per split and held-out slice.
    """
    LOG = get_logger()
    files = config["output_files"]
    parts = []

    by_campaign = select_by_campaign(ope_summary)
    if by_campaign.empty:
        LOG.info("Campaign split skipped: it needs at least two campaigns")
    else:
        for _, r in by_campaign.iterrows():
            LOG.info("Campaign split: chose %s on %s; error on %s %.1f%% (best in hindsight: %s, %.1f%%)",
                     r["selected_estimator"], r["selected_on"], r["held_out"], 100 * r["test_error"],
                     r["oracle_estimator"], 100 * r["oracle_error"])
        parts.append(by_campaign)

    if config.get("time_split", True):
        per_period, by_time = run_time_split(datasets, winners, config)
        per_period.to_csv(files["ope_time_split"], index=False)
        LOG.info("Wrote time-split benchmark: %s", files["ope_time_split"])
        parts.append(by_time)

    selection = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    selection.to_csv(files["estimator_selection"], index=False)
    LOG.info("Wrote estimator selection: %s", files["estimator_selection"])
    return selection
