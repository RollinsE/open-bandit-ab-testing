# -*- coding: utf-8 -*-
"""Reward-model training and evaluation: the model marathon.

The doubly robust and direct-method estimators need a click model: the chance
that a user clicks item a in slot k, given their features. Several kinds of
model compete for that job. The winner is the simplest model whose log loss
is within a margin of the best (see ``models_within_margin``), or the lowest
log loss with ``--winner-rule best``.

How the models are trained and scored:

1. One scoring method. Every model, tuned or not, single or ensemble, is scored
   the same way: log loss on its pooled out-of-fold predictions of the clicks
   that were actually logged. Lower is better. Brier score and ROC AUC are
   reported alongside.
2. Baselines. A "same CTR in every slot" baseline and an "item features only"
   baseline are scored too, so a log loss has something to be compared with.
   "Skill vs baseline" is how much a model improves on the first.
3. Nested tuning. Hyperparameters are tuned inside each training fold, so the
   score of a tuned model is not flattered by tuning on its own test data. The
   defaults are always one of the trials, so tuning cannot make a model worse
   on its own inner folds. On large data the search runs on a stratified sample
   of each training fold (``--tune-max-rows``); the chosen settings are then
   trained on the whole fold.
4. Ensemble make-up. The ensemble averages the best model from each of the top
   three model families, so its members are different kinds of model.
5. One model per slot. As in the original Open Bandit Pipeline, each candidate
   is fitted separately for each of the three slots.
6. No peeking. The winner is chosen on click prediction alone. Thompson
   Sampling's true CTR is never used to choose, so the off-policy benchmark
   that follows is a fair test of the winner.
7. Simplicity. Every bootstrap run in stage 4 refits the winner, so among
   models that are practically as good as the best, the cheapest to train
   wins (``SIMPLICITY_ORDER``).
"""
from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from joblib import delayed
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.constants import HEARTBEAT_SECONDS, LOGGER_NAME, SIMPLICITY_ORDER, OPEConstants
from src.checkpoints import TaskCache
from src.utils import Heartbeat, Progress, format_duration, iter_parallel

LOG = logging.getLogger(LOGGER_NAME)

SAME_CTR_BASELINE = "Baseline: same CTR in every slot"
ITEM_BASELINE = "Baseline: item features only"
ENSEMBLE_NAME = "HybridEnsemble"
EPS = 1e-6


# --------------------------------------------------------------------- models
class AveragedClassifier(BaseEstimator, ClassifierMixin):
    """Soft-voting ensemble: the mean of its members' click probabilities.

    Written as a scikit-learn estimator so it can be cloned and fitted per slot
    like any single model.
    """

    def __init__(self, estimators: list | None = None):
        self.estimators = estimators

    def fit(self, X, y):
        self.estimators_ = [clone(est).fit(X, y) for est in self.estimators]
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X):
        return np.mean([est.predict_proba(X) for est in self.estimators_], axis=0)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(float)


@dataclass(frozen=True)
class ModelSpec:
    """One competitor: how to build it and which parameters may be tuned."""

    name: str
    family: str
    build: Callable[[dict[str, Any]], Any]      # params -> unfitted classifier
    defaults: dict[str, Any] = field(default_factory=dict)
    space: dict[str, tuple] | None = None       # name -> ("int"|"float", low, high, log)
    role: str = "candidate"                     # or "baseline"


def candidate_specs(seed: int, n_item_features: int, families: list[str] | None = None,
                    quiet: bool = False) -> list[ModelSpec]:
    """Baselines plus every model family whose library is installed.

    Args:
        seed: Random seed for every model.
        n_item_features: Number of item feature columns, which sit at the end
            of the feature matrix (used by the item-only baseline).
        families: Optional subset of family names to run (baselines always run).
        quiet: Do not log which optional libraries are missing (used when only
            rebuilding a winner).
    """
    note = LOG.debug if quiet else LOG.info
    item_cols = list(range(-n_item_features, 0))

    def scaled(model):
        return Pipeline([("scaler", StandardScaler()), ("model", model)])

    specs = [
        ModelSpec(SAME_CTR_BASELINE, "baseline", lambda p: DummyClassifier(strategy="prior"), role="baseline"),
        ModelSpec(ITEM_BASELINE, "baseline", lambda p: Pipeline([
            ("items", ColumnTransformer([("item", StandardScaler(), item_cols)])),
            ("model", LogisticRegression(max_iter=5000))]), role="baseline"),
        ModelSpec(
            "Logistic Regression", "logistic_regression",
            lambda p: scaled(LogisticRegression(max_iter=10000, **p)),
            defaults={"C": 1.0},
            space={"C": ("float", 1e-3, 100.0, True)},
        ),
        ModelSpec(
            "Random Forest", "random_forest",
            lambda p: RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=1, **p),
            defaults={"max_depth": 5, "min_samples_leaf": 10, "max_features": 0.5},
            space={
                "max_depth": ("int", 2, 8, False),
                "min_samples_leaf": ("int", 5, 50, False),
                "max_features": ("float", 0.2, 1.0, False),
            },
        ),
        ModelSpec(
            "Gradient Boosting", "gradient_boosting",
            lambda p: GradientBoostingClassifier(random_state=seed, subsample=0.8, **p),
            defaults={"n_estimators": 60, "max_depth": 2, "learning_rate": 0.05, "min_samples_leaf": 20},
            space={
                "n_estimators": ("int", 30, 150, False),
                "max_depth": ("int", 1, 4, False),
                "learning_rate": ("float", 0.01, 0.2, True),
                "min_samples_leaf": ("int", 5, 60, False),
            },
        ),
        ModelSpec(
            "Hist Gradient Boosting", "hist_gradient_boosting",
            lambda p: HistGradientBoostingClassifier(random_state=seed, early_stopping=False, **p),
            defaults={"max_iter": 80, "learning_rate": 0.05, "max_leaf_nodes": 8, "min_samples_leaf": 30,
                      "l2_regularization": 1.0},
            space={
                "max_iter": ("int", 30, 200, False),
                "learning_rate": ("float", 0.01, 0.2, True),
                "max_leaf_nodes": ("int", 4, 31, False),
                "min_samples_leaf": ("int", 10, 100, False),
                "l2_regularization": ("float", 1e-3, 10.0, True),
            },
        ),
    ]

    try:
        import lightgbm as lgb

        specs.append(ModelSpec(
            "LightGBM", "lightgbm",
            lambda p: lgb.LGBMClassifier(random_state=seed, n_jobs=1, verbosity=-1, **p),
            defaults={"n_estimators": 80, "learning_rate": 0.05, "num_leaves": 7, "min_child_samples": 30},
            space={
                "n_estimators": ("int", 30, 200, False),
                "learning_rate": ("float", 0.01, 0.2, True),
                "num_leaves": ("int", 3, 31, False),
                "min_child_samples": ("int", 10, 100, False),
            },
        ))
    except ImportError:
        note("lightgbm not installed; skipping LightGBM")

    try:
        import xgboost as xgb

        specs.append(ModelSpec(
            "XGBoost", "xgboost",
            lambda p: xgb.XGBClassifier(random_state=seed, n_jobs=1, eval_metric="logloss", **p),
            defaults={"n_estimators": 80, "learning_rate": 0.05, "max_depth": 2, "min_child_weight": 1.0},
            space={
                "n_estimators": ("int", 30, 200, False),
                "learning_rate": ("float", 0.01, 0.2, True),
                "max_depth": ("int", 1, 5, False),
                "min_child_weight": ("float", 0.1, 10.0, True),
            },
        ))
    except ImportError:
        note("xgboost not installed; skipping XGBoost")

    try:
        import catboost as cb

        specs.append(ModelSpec(
            "CatBoost", "catboost",
            lambda p: cb.CatBoostClassifier(verbose=0, random_seed=seed, thread_count=1, **p),
            defaults={"iterations": 100, "depth": 3, "learning_rate": 0.05},
            space={
                "iterations": ("int", 50, 300, False),
                "depth": ("int", 2, 6, False),
                "learning_rate": ("float", 0.01, 0.2, True),
            },
        ))
    except ImportError:
        note("catboost not installed; skipping CatBoost")

    if families:
        wanted = set(families)
        specs = [s for s in specs if s.role == "baseline" or s.family in wanted]
    return specs


def build_estimator(spec: ModelSpec, params: dict[str, Any] | None = None):
    """Unfitted classifier for ``spec`` with ``params`` over its defaults."""
    return spec.build({**spec.defaults, **(params or {})})


# ----------------------------------------------------------- per-slot fitting
class _ConstantRate:
    """Stand-in for a slot or fold with no clicks (or only clicks) to learn from."""

    def __init__(self, rate: float):
        self.rate = rate

    def predict_proba(self, X):
        return np.c_[np.full(len(X), 1 - self.rate), np.full(len(X), self.rate)]


def fit_per_slot(estimator, X: np.ndarray, y: np.ndarray, slot: np.ndarray, rows: np.ndarray, n_slots: int) -> list:
    """Fit one clone of ``estimator`` for each slot on the given rows."""
    models = []
    for k in range(n_slots):
        r = rows[slot[rows] == k]
        if len(r) == 0 or y[r].min() == y[r].max():
            models.append(_ConstantRate(float(y[r].mean()) if len(r) else float(y[rows].mean())))
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            models.append(clone(estimator).fit(X[r], y[r]))
    return models


def predict_per_slot(models: list, X: np.ndarray, slot: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Click probability for each row from its own slot's model."""
    out = np.zeros(len(rows))
    for k, model in enumerate(models):
        m = slot[rows] == k
        if m.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[m] = model.predict_proba(X[rows[m]])[:, 1]
    return np.clip(out, EPS, 1 - EPS)


# -------------------------------------------------------------------- scoring
def score_predictions(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Log loss, Brier score and ROC AUC of click probabilities."""
    p = np.clip(p, EPS, 1 - EPS)
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
    }


def make_splits(y: np.ndarray, n_splits: int, seed: int) -> list:
    """Stratified folds, so every fold gets its share of the rare clicks."""
    n_splits = max(2, min(n_splits, int(y.sum()), int(len(y) - y.sum())))
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def out_of_fold(estimator_for_fold: Callable[[int, np.ndarray], Any], X: np.ndarray, y: np.ndarray,
                slot: np.ndarray, splits: list, n_slots: int) -> np.ndarray:
    """Predict every row from per-slot models that were not trained on it."""
    predictions = np.full(len(y), np.nan)
    for fold, (train_rows, test_rows) in enumerate(splits):
        models = fit_per_slot(estimator_for_fold(fold, train_rows), X, y, slot, train_rows, n_slots)
        predictions[test_rows] = predict_per_slot(models, X, slot, test_rows)
    return predictions


# --------------------------------------------------------------------- tuning
def _sample(space: dict[str, tuple], rng: np.random.Generator) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, (kind, low, high, log) in space.items():
        if kind == "int":
            params[name] = int(rng.integers(low, high + 1))
        elif log:
            params[name] = float(np.exp(rng.uniform(np.log(low), np.log(high))))
        else:
            params[name] = float(rng.uniform(low, high))
    return params


def tune_params(spec: ModelSpec, X: np.ndarray, y: np.ndarray, slot: np.ndarray, rows: np.ndarray,
                n_slots: int, n_trials: int, seed: int, inner_splits: int = 3,
                max_rows: int | None = None) -> dict[str, Any]:
    """Choose parameters using only the given rows.

    The rows are split again into inner folds and each trial is scored on its
    inner out-of-fold log loss. The defaults are always one of the trials.
    With ``max_rows``, the search runs on a stratified sample of that many of
    the rows (same click rate and slot mix); the chosen parameters are then
    used to train on all of them.
    """
    rows = np.asarray(rows, dtype=int)
    if max_rows and len(rows) > max_rows:
        from sklearn.model_selection import train_test_split
        rows, _ = train_test_split(rows, train_size=int(max_rows), random_state=seed,
                                   stratify=y[rows] * 10 + slot[rows])
        rows = np.sort(rows)
    if not spec.space or n_trials <= 0 or y[rows].sum() < 2 * inner_splits:
        return dict(spec.defaults)
    inner = [(rows[a], rows[b]) for a, b in make_splits(y[rows], inner_splits, seed)]

    def inner_loss(params: dict[str, Any]) -> float:
        predictions = np.full(len(y), np.nan)
        for train_rows, test_rows in inner:
            models = fit_per_slot(build_estimator(spec, params), X, y, slot, train_rows, n_slots)
            predictions[test_rows] = predict_per_slot(models, X, slot, test_rows)
        return score_predictions(y[rows], predictions[rows])["log_loss"]

    try:
        import optuna
    except ImportError:
        optuna = None

    if optuna is not None:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
        defaults = {name: spec.defaults[name] for name in spec.space if name in spec.defaults}
        if len(defaults) == len(spec.space):
            study.enqueue_trial(defaults)

        def objective(trial):
            params = {}
            for name, (kind, low, high, log) in spec.space.items():
                params[name] = (trial.suggest_int(name, low, high) if kind == "int"
                                else trial.suggest_float(name, low, high, log=log))
            return inner_loss(params)

        study.optimize(objective, n_trials=n_trials)
        return {**spec.defaults, **study.best_params}

    # Fallback when optuna is not installed: plain random search over the same ranges.
    rng = np.random.default_rng(seed)
    trials = [dict(spec.defaults)] + [{**spec.defaults, **_sample(spec.space, rng)} for _ in range(max(0, n_trials - 1))]
    losses = [inner_loss(params) for params in trials]
    return trials[int(np.argmin(losses))]


# ----------------------------------------------------------------------- main
def _run_candidate(spec: ModelSpec, tuned: bool, X, y, slot, splits, n_slots, n_trials, seed,
                   tune_max_rows: int | None = None) -> dict[str, Any]:
    """Score one family, either with its defaults or with nested tuning. Runs in a worker.

    Default and tuned versions are separate tasks so each one reports as it finishes.
    """
    started = time.monotonic()
    name = f"{spec.name}_Tuned" if tuned else spec.name
    try:
        if tuned:
            preds = out_of_fold(
                lambda fold, rows: build_estimator(spec, tune_params(spec, X, y, slot, rows, n_slots, n_trials,
                                                                     seed + fold, max_rows=tune_max_rows)),
                X, y, slot, splits, n_slots,
            )
            params = tune_params(spec, X, y, slot, np.arange(len(y)), n_slots, n_trials, seed, max_rows=tune_max_rows)
        else:
            preds = out_of_fold(lambda fold, rows: build_estimator(spec), X, y, slot, splits, n_slots)
            params = dict(spec.defaults)
        return {"name": name, "family": spec.family, "role": spec.role, "tuned": tuned,
                "params": params, "oof": preds, "seconds": time.monotonic() - started}
    except Exception as exc:  # one failing library should not stop the marathon
        return {"name": name, "family": spec.family, "role": spec.role, "error": str(exc),
                "seconds": time.monotonic() - started}


def train_and_evaluate(X: np.ndarray, y: np.ndarray, slot: np.ndarray, n_item_features: int,
                       params: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Train, tune and score every candidate click model.

    Args:
        X: Features, ``[user context, item context]`` per logged row.
        y: Clicks (0/1).
        slot: Slot of each row, 0 to ``n_slots - 1``.
        n_item_features: Item feature columns at the end of ``X``.
        params: ``random_state``, ``n_splits_cv``, ``model_trials``, ``tune_top``, ``winner_rule``,
            ``ensemble``, ``n_jobs``, ``model_families``,
            ``tune_max_rows`` (sample size for the tuning search) and ``cache`` (a ``TaskCache``
            that loads models already scored in an earlier session and saves new ones).

    Returns:
        ``(results table, winner, diagnostics)``. ``winner`` describes the best
        model as ``{"name", "family", "params"}`` (or ``"members"`` for the
        ensemble) so it can be rebuilt with ``estimator_from_winner``.
    """
    seed = int(params.get("random_state", 12345))
    n_splits = int(params.get("n_splits_cv", 3))
    n_trials = int(params.get("model_trials", 8))
    n_jobs = int(params.get("n_jobs", -1))
    n_slots = int(slot.max()) + 1
    specs = candidate_specs(seed, n_item_features, params.get("model_families"))
    splits = make_splits(y, n_splits, seed)
    LOG.info("Model marathon: %d candidate families, %d-fold stratified CV, %d tuning trials, %s clicks in %s rows",
             sum(s.role != "baseline" for s in specs), len(splits), n_trials, f"{int(y.sum()):,}", f"{len(y):,}")

    tune_top = params.get("tune_top", OPEConstants.TUNE_TOP)
    label = params.get("progress_label", "Model marathon")
    heartbeat = params.get("heartbeat_seconds", HEARTBEAT_SECONDS)
    cache = params.get("cache") or TaskCache(None)
    tune_rows = params.get("tune_max_rows")
    table, oof, entries = [], {}, {}

    def record(entry: dict[str, Any], model_type: str) -> None:
        scores = score_predictions(y, entry["oof"])
        fold_losses = [score_predictions(y[test], entry["oof"][test])["log_loss"] for _, test in splits]
        table.append({"Model": entry["name"], "Family": entry["family"], "Type": model_type, "Role": entry["role"],
                      "Log loss": scores["log_loss"], "Std log loss": float(np.std(fold_losses)),
                      "Brier": scores["brier"], "ROC AUC": scores["auc"],
                      "Seconds": float(entry.get("seconds", float("nan")))})
        oof[entry["name"]] = entry["oof"]
        entries[entry["name"]] = entry
        LOG.info("%s - out-of-fold log loss: %.5f, AUC: %.3f", entry["name"], scores["log_loss"], scores["auc"])

    def label_of(entry: dict[str, Any]) -> str:
        return "Baseline" if entry["role"] == "baseline" else ("Tuned" if entry["tuned"] else "Default")

    def run_round(jobs: list, round_label: str) -> None:
        """Score ``jobs`` in parallel; reuse checkpoints from an earlier session; save each new result."""
        progress = Progress(round_label, len(jobs), LOG, eta=False)
        pending = []
        for spec, tuned in jobs:
            name = f"{spec.name}_Tuned" if tuned else spec.name
            saved = cache.load(name, with_array=True)
            if saved is None:
                pending.append((spec, tuned))
                continue
            saved["oof"] = saved.pop("array")
            record(saved, label_of(saved))
            progress.done += 1
        if progress.done:
            LOG.info("%s: %d of %d models loaded from checkpoints; %d to run",
                     round_label, progress.done, progress.total, len(pending))
        tasks = [delayed(_run_candidate)(spec, tuned, X, y, slot, splits, n_slots, n_trials, seed, tune_rows)
                 for spec, tuned in pending]
        with Heartbeat(progress, heartbeat):
            for entry in iter_parallel(n_jobs, tasks):
                if "error" in entry:
                    LOG.warning("%s failed and was skipped after %s: %s", entry["name"],
                                format_duration(entry["seconds"]), entry["error"])
                    progress.update(f"{entry['name']} failed")
                    continue
                cache.save(entry["name"], {k: v for k, v in entry.items() if k != "oof"}, entry["oof"])
                record(entry, label_of(entry))
                progress.update(f"{entry['name']} took {format_duration(entry['seconds'])}")

    # Round 1: every family with its default settings, plus the baselines. Cheap.
    LOG.info("%s: round 1, %d models with default settings (%s)", label, len(specs),
             ", ".join(s.name for s in specs if s.role != "baseline"))
    run_round([(spec, False) for spec in specs], f"{label}, round 1 (defaults)")

    # Round 2: tune only the families that did best on their defaults.
    candidates = [n for n, e in entries.items() if e["role"] != "baseline"]
    if not candidates:
        raise RuntimeError("No candidate click model could be trained")
    tunable = [s for s in specs if s.role != "baseline" and s.space and s.name in entries]
    tunable.sort(key=lambda s: score_predictions(y, oof[s.name])["log_loss"])
    n_tune = 0 if n_trials <= 0 else (len(tunable) if tune_top is None or tune_top < 0 else min(tune_top, len(tunable)))
    to_tune = tunable[:n_tune]
    if to_tune:
        LOG.info("%s: round 2, tuning the top %d on their defaults (%s), %d trials each", label, len(to_tune),
                 ", ".join(s.name for s in to_tune), n_trials)
        run_round([(spec, True) for spec in to_tune], f"{label}, round 2 (tuning)")
    else:
        LOG.info("%s: round 2 skipped (no tuning requested)", label)
    candidates = [n for n, e in entries.items() if e["role"] != "baseline"]
    loss_of = {n: score_predictions(y, oof[n])["log_loss"] for n in candidates}

    # Ensemble: the best model from each of the three best families.
    members: list[str] = []
    if params.get("ensemble", True):
        best_per_family: dict[str, str] = {}
        for name in sorted(candidates, key=loss_of.get):
            best_per_family.setdefault(entries[name]["family"], name)
        members = list(best_per_family.values())[:3]
    if len(members) >= 2:
        ensemble = {"name": ENSEMBLE_NAME, "family": "ensemble", "role": "candidate", "tuned": False,
                    "params": {}, "oof": np.mean([oof[m] for m in members], axis=0),
                    "seconds": float(sum(entries[m].get("seconds", 0) for m in members)),
                    "members": [{"name": m, "family": entries[m]["family"], "params": entries[m]["params"]}
                                for m in members]}
        record(ensemble, "Ensemble")
        candidates.append(ENSEMBLE_NAME)
        loss_of[ENSEMBLE_NAME] = score_predictions(y, ensemble["oof"])["log_loss"]
    else:
        members = []

    results = pd.DataFrame(table).set_index("Model")
    baseline_loss = float(results.loc[SAME_CTR_BASELINE, "Log loss"])
    results["Skill vs baseline"] = 1.0 - results["Log loss"] / baseline_loss

    rule = params.get("winner_rule", OPEConstants.WINNER_RULE)
    best_name = min(candidates, key=loss_of.get)
    tolerance = float(params.get("winner_tolerance", OPEConstants.WINNER_TOLERANCE))
    within, margin = models_within_margin(y, {n: oof[n] for n in candidates}, best_name, splits, tolerance)
    results["Within margin of best"] = [name in within for name in results.index]
    results = results.sort_values("Log loss")
    chosen = simplest_model(within, entries) if rule == "simplest" else best_name
    chosen_entry = entries[chosen]
    winner = {"name": chosen, "family": chosen_entry["family"], "params": chosen_entry.get("params", {}),
              "rule": rule, "best_by_log_loss": best_name}
    if "members" in chosen_entry:
        winner["members"] = chosen_entry["members"]
    if rule == "simplest" and chosen != best_name:
        LOG.info("%s: lowest log loss is %s (%.5f), but %s (%.5f) is within the margin (%.5f) and simpler, "
                 "so it wins under the 'simplest' rule (%d models were within the margin)",
                 label, best_name, loss_of[best_name], chosen, loss_of[chosen], margin, len(within))

    notes = []
    if loss_of[chosen] >= baseline_loss:
        notes.append("No model predicts clicks better than the same CTR in every slot. "
                     "Model-based estimates will lean on a weak click model.")
    for note in notes:
        LOG.warning(note)
    diagnostics = {
        "rows": int(len(y)), "clicks": int(y.sum()), "slots": n_slots,
        "cross_validation": f"{len(splits)}-fold stratified cross-validation, one model per slot",
        "tuning_trials": n_trials, "tuned_families": [s.family for s in to_tune],
        "winner_rule": rule, "best_by_log_loss": best_name, "within_margin": sorted(within),
        "winner_tolerance": tolerance, "margin": margin,
        "baseline_log_loss": baseline_loss,
        "best_model": chosen, "best_log_loss": loss_of[chosen],
        "skill_vs_baseline": float(1.0 - loss_of[chosen] / baseline_loss),
        "ensemble_members": members,
        "error_unit": "log loss of click probabilities, on out-of-fold predictions", "notes": notes,
    }
    return results, winner, diagnostics


# ------------------------------------------------------------ winner rules
def best_model_standard_error(y: np.ndarray, p: np.ndarray, splits: list) -> float:
    """Standard error of a model's cross-validated log loss: std of its fold losses / sqrt(folds)."""
    fold_losses = np.array([score_predictions(y[test], p[test])["log_loss"] for _, test in splits])
    if len(fold_losses) < 2:
        return 0.0
    return float(fold_losses.std(ddof=1) / np.sqrt(len(fold_losses)))


def models_within_margin(y: np.ndarray, predictions: dict[str, np.ndarray], best: str, splits: list,
                         tolerance: float = 0.0) -> tuple[set[str], float]:
    """Models whose log loss is within a margin of the best model's, and that margin.

    The margin is the wider of two things:

    * a practical tolerance, ``tolerance`` times the best log loss (default
      0.25%): differences smaller than this are too small to matter for the
      off-policy estimates, but can make stage 4 several times slower;
    * one standard error of the best model's cross-validated log loss, from its
      spread across the folds (the classic one-standard-error rule: Breiman et
      al., 1984; The Elements of Statistical Learning, section 7.10).

    The tolerance is needed because the folds are stratified: every fold gets
    the same share of clicks, so the fold-to-fold spread, and with it the
    standard error, is tiny, and on its own the rule treats differences of a
    few hundred-thousandths of log loss as decisive. The best model is always
    included.
    """
    best_loss = score_predictions(y, predictions[best])["log_loss"]
    margin = max(tolerance * best_loss, best_model_standard_error(y, predictions[best], splits))
    within = {name for name, p in predictions.items()
              if name == best or score_predictions(y, p)["log_loss"] <= best_loss + margin}
    return within, float(margin)


def simplest_model(names: set[str], entries: dict[str, dict]) -> str:
    """The simplest of ``names``: earliest family in ``SIMPLICITY_ORDER``, defaults before tuned."""
    order = {family: i for i, family in enumerate(SIMPLICITY_ORDER)}

    def cost(name: str) -> tuple:
        entry = entries[name]
        return (order.get(entry["family"], len(order)), bool(entry.get("tuned")), name)

    return min(names, key=cost)


def estimator_from_winner(winner: dict[str, Any], seed: int, n_item_features: int):
    """Rebuild the winning classifier (single model or ensemble) from its description."""
    specs = {s.family: s for s in candidate_specs(seed, n_item_features, quiet=True) if s.role != "baseline"}
    if "members" in winner:
        return AveragedClassifier([build_estimator(specs[m["family"]], m["params"]) for m in winner["members"]])
    return build_estimator(specs[winner["family"]], winner["params"])
