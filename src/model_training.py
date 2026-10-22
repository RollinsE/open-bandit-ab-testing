# -*- coding: utf-8 -*-
"""Stage 3: the reward-model marathon. Train the click models, pick the best one, save the outputs.

The training and scoring itself lives in src/model_evaluation.py.

For each campaign the marathon runs on the random arm's logs, which are the
logs the off-policy estimators use. The winner for each campaign is written to
``artifacts/reward_model_winners.json`` and is the click model stage 4 uses.

On the full dataset the marathon runs on a stratified sample of at most
``--marathon-max-rows`` rows (default 200,000), which keeps the click rate
intact. Tuning seven model families on 1.4M rows would take hours, and at
the dataset's click rate 200,000 rows still hold several hundred clicks to
learn from. The winner is then trained on every row in stage 4.

The marathon runs in two rounds: every family with its default settings, then
tuning for only the best ``--tune-top`` families (default 2). The winner is,
by default, the simplest model within a margin of the lowest log loss
(``--winner-rule simplest``): the wider of 0.25% of the best log loss
(``--winner-tolerance``) and one standard error. The click model is retrained
in every bootstrap run of stage 4, so a model that is only negligibly better
but several times slower would cost hours. ``--winner-rule best`` takes the lowest log loss.

Each model is checkpointed as soon as it is scored, and each campaign's result
once its marathon ends, so ``--continue-build`` never repeats finished work.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.constants import CONTROL_POLICY, DEFAULT_MODEL_FAMILIES, HEARTBEAT_SECONDS, LOGGER_NAME, OPEConstants
from src.checkpoints import TaskCache
from src.model_evaluation import train_and_evaluate
from src.utils import write_json

LOG = logging.getLogger(LOGGER_NAME)


def build_features(obd) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Features for the logged rows: ``[user context, item context of the shown item]``.

    Returns:
        ``(X, clicks, slot, n_item_features)``.
    """
    X = np.c_[obd.context, obd.action_context[obd.action]]
    return X, obd.reward.astype(int), obd.position, obd.action_context.shape[1]


def cap_rows(X: np.ndarray, y: np.ndarray, slot: np.ndarray, max_rows: int | None, seed: int):
    """Stratified sample of at most ``max_rows`` rows, keeping the click rate and the slot mix."""
    if not max_rows or len(y) <= max_rows:
        return X, y, slot
    strata = y * 10 + slot  # clicks within each slot
    rows, _ = train_test_split(np.arange(len(y)), train_size=int(max_rows), stratify=strata, random_state=seed)
    rows = np.sort(rows)
    return X[rows], y[rows], slot[rows]


def format_model_ranking_table(results: pd.DataFrame, campaign: str) -> str:
    """Leaderboard for the console, best first."""
    rows = [f"\nModel marathon - {campaign} (lower log loss is better)",
            f"  {'Model':<34}{'Type':<10}{'Log loss':>10}{'AUC':>8}{'Skill':>9}"]
    for name, r in results.iterrows():
        rows.append(f"  {name:<34}{r['Type']:<10}{r['Log loss']:>10.5f}{r['ROC AUC']:>8.3f}{r['Skill vs baseline']:>+9.1%}")
    return "\n".join(rows)


def execute_modeling_pipeline(datasets: dict, config: dict) -> tuple[pd.DataFrame, dict]:
    """Run stage 3 for every campaign.

    Args:
        datasets: Output of ``execute_data_pipeline``.
        config: Run configuration. Reads ``model_trials``, ``model_families``,
            ``n_jobs`` and ``random_state``.

    Returns:
        ``(leaderboard, winners)``. ``leaderboard`` has one row per campaign and
        model; ``winners`` maps campaign to the winning model's description.
    """
    params = {
        "random_state": config.get("random_state", 12345),
        "n_splits_cv": OPEConstants.N_FOLDS,
        "model_trials": config.get("model_trials", OPEConstants.MODEL_TRIALS),
        "model_families": config.get("model_families") or list(DEFAULT_MODEL_FAMILIES),
        "n_jobs": config.get("n_jobs", -1),
        "heartbeat_seconds": config.get("heartbeat_seconds", HEARTBEAT_SECONDS),
        "tune_max_rows": config.get("tune_max_rows", OPEConstants.TUNE_MAX_ROWS),
        "tune_top": config.get("tune_top", OPEConstants.TUNE_TOP),
        "winner_rule": config.get("winner_rule", OPEConstants.WINNER_RULE),
        "winner_tolerance": config.get("winner_tolerance", OPEConstants.WINNER_TOLERANCE),
        "ensemble": config.get("ensemble", True),
    }
    store = config.get("checkpoints")
    tables, winners, diagnostics = [], {}, {}
    for campaign in config["campaigns"]:
        saved = store.load_json("marathon", campaign, "_result") if store else None
        if saved is not None:
            table = pd.DataFrame(saved["table"])
            winners[campaign], diagnostics[campaign] = saved["winner"], saved["diagnostics"]
            tables.append(table)
            LOG.info("Model marathon %s: finished in an earlier session; winner %s (loaded from checkpoint)",
                     campaign, saved["winner"]["name"])
            continue

        obd = datasets[(CONTROL_POLICY, campaign)]
        X, y, slot, n_item = build_features(obd)
        max_rows = config.get("marathon_max_rows", OPEConstants.MARATHON_MAX_ROWS)
        if max_rows and len(y) > max_rows:
            LOG.info("Model marathon %s: using a stratified sample of %s of %s rows",
                     campaign, f"{int(max_rows):,}", f"{len(y):,}")
            X, y, slot = cap_rows(X, y, slot, max_rows, params["random_state"])
        LOG.info("Model marathon %s: training on the %s arm's logs", campaign, CONTROL_POLICY)
        results, winner, diag = train_and_evaluate(
            X, y, slot, n_item,
            {**params, "progress_label": f"Model marathon {campaign}",
             "cache": TaskCache(store, "marathon", campaign)},
        )
        winner["n_item_features"] = n_item
        winners[campaign] = winner
        diagnostics[campaign] = diag
        print(format_model_ranking_table(results, campaign))
        LOG.info("Model marathon %s: winner %s (log loss %.5f, %+.1f%% skill vs baseline; rule: %s)",
                 campaign, winner["name"], diag["best_log_loss"], 100 * diag["skill_vs_baseline"],
                 diag["winner_rule"])
        table = results.reset_index()
        table.insert(0, "campaign", campaign)
        table["winner"] = table["Model"] == winner["name"]
        tables.append(table)
        if store:
            store.save_json({"table": table.to_dict(orient="records"), "winner": winner, "diagnostics": diag},
                            "marathon", campaign, "_result")

    leaderboard = pd.concat(tables, ignore_index=True)
    files = config["output_files"]
    leaderboard.to_csv(files["model_performance"], index=False)
    write_json(files["reward_model_winners"], winners)
    write_json(files["model_diagnostics"], diagnostics)
    LOG.info("Wrote model leaderboard: %s", files["model_performance"])
    LOG.info("Wrote reward model winners: %s", files["reward_model_winners"])
    return leaderboard, winners
