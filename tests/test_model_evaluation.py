import numpy as np
from sklearn.base import clone

from src.model_evaluation import (
    ENSEMBLE_NAME,
    SAME_CTR_BASELINE,
    AveragedClassifier,
    candidate_specs,
    estimator_from_winner,
    make_splits,
    models_within_margin,
    score_predictions,
    train_and_evaluate,
)


def _synthetic(n=3000, seed=0):
    """Clicks driven by the first feature, so a real model should beat the baseline."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    p = 1 / (1 + np.exp(-(-3.0 + 1.5 * X[:, 0])))
    y = (rng.random(n) < p).astype(int)
    slot = rng.integers(0, 3, n)
    return X, y, slot


def test_marathon_picks_lowest_log_loss_and_beats_baseline():
    X, y, slot = _synthetic()
    results, winner, diag = train_and_evaluate(
        X, y, slot, n_item_features=2,
        params={"random_state": 0, "model_trials": 0, "n_jobs": 1,
                "model_families": ["logistic_regression", "hist_gradient_boosting"]},
    )
    candidates = results[results["Role"] == "candidate"]
    assert winner["name"] == candidates["Log loss"].idxmin()
    assert SAME_CTR_BASELINE in results.index
    assert diag["skill_vs_baseline"] > 0.05
    assert ENSEMBLE_NAME in results.index  # two families entered, so an ensemble is built


def test_winner_can_be_rebuilt_and_fitted():
    X, y, _ = _synthetic(800)
    single = estimator_from_winner({"name": "Logistic Regression", "family": "logistic_regression",
                                    "params": {"C": 0.5}}, seed=0, n_item_features=2)
    ensemble = estimator_from_winner({"name": ENSEMBLE_NAME, "family": "ensemble", "params": {},
                                      "members": [{"family": "logistic_regression", "params": {}},
                                                  {"family": "random_forest", "params": {"max_depth": 3}}]},
                                     seed=0, n_item_features=2)
    for est in (single, clone(ensemble)):
        p = est.fit(X, y).predict_proba(X)[:, 1]
        assert p.shape == (len(y),) and (p > 0).all() and (p < 1).all()


def test_averaged_classifier_is_the_mean_of_its_members():
    X, y, _ = _synthetic(500)
    specs = {s.family: s for s in candidate_specs(0, 2)}
    members = [specs["logistic_regression"].build({}), specs["random_forest"].build({"max_depth": 2})]
    avg = AveragedClassifier(members).fit(X, y)
    expected = np.mean([m.predict_proba(X) for m in avg.estimators_], axis=0)
    np.testing.assert_allclose(avg.predict_proba(X), expected)


def test_score_predictions_perfect_and_constant():
    y = np.array([0, 0, 1, 1])
    assert score_predictions(y, np.array([0.0, 0.0, 1.0, 1.0]))["auc"] == 1.0
    assert score_predictions(y, np.full(4, 0.5))["log_loss"] > score_predictions(y, np.array([.1, .1, .9, .9]))["log_loss"]


def _weak_signal(n=6000, seed=1):
    """Clicks barely related to the features, as in the Open Bandit Dataset."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    p = 1 / (1 + np.exp(-(-3.0 + 0.05 * X[:, 0])))
    return X, (rng.random(n) < p).astype(int), rng.integers(0, 3, n)


def test_round_two_tunes_only_the_top_families():
    X, y, slot = _synthetic()
    results, _, diag = train_and_evaluate(
        X, y, slot, n_item_features=2,
        params={"random_state": 0, "model_trials": 2, "tune_top": 1, "n_jobs": 1,
                "model_families": ["logistic_regression", "hist_gradient_boosting", "random_forest"]},
    )
    tuned = [m for m in results.index if m.endswith("_Tuned")]
    assert len(tuned) == 1 and len(diag["tuned_families"]) == 1
    defaults = results[(results["Type"] == "Default")]
    assert tuned[0].replace("_Tuned", "") == defaults["Log loss"].idxmin()  # the best on defaults got tuned


def test_simplest_rule_prefers_a_simple_model_that_is_not_measurably_worse():
    X, y, slot = _weak_signal()
    params = {"random_state": 0, "model_trials": 0, "n_jobs": 1,
              "model_families": ["logistic_regression", "random_forest", "hist_gradient_boosting"]}
    results, simple, diag = train_and_evaluate(X, y, slot, 2, {**params, "winner_rule": "simplest"})
    _, best, _ = train_and_evaluate(X, y, slot, 2, {**params, "winner_rule": "best"})
    assert best["name"] == diag["best_by_log_loss"]
    assert simple["name"] in diag["within_margin"]
    assert results.loc[simple["name"], "Within margin of best"]
    order = ["logistic_regression", "lightgbm", "hist_gradient_boosting", "xgboost", "catboost",
             "random_forest", "gradient_boosting", "ensemble"]
    within_families = [results.loc[m, "Family"] for m in diag["within_margin"]]
    assert simple["family"] == min(within_families, key=order.index)


def test_simplest_rule_still_picks_a_clearly_better_model():
    X, y, slot = _synthetic()  # strong signal: logistic regression is right, trees are not much worse
    _, winner, diag = train_and_evaluate(
        X, y, slot, 2, {"random_state": 0, "model_trials": 0, "n_jobs": 1, "winner_rule": "simplest",
                        "model_families": ["logistic_regression", "hist_gradient_boosting"]})
    assert SAME_CTR_BASELINE not in diag["within_margin"]  # baselines never compete
    assert winner["name"] in diag["within_margin"]


def test_no_ensemble_option():
    X, y, slot = _synthetic()
    results, _, diag = train_and_evaluate(
        X, y, slot, 2, {"random_state": 0, "model_trials": 0, "n_jobs": 1, "ensemble": False,
                        "model_families": ["logistic_regression", "hist_gradient_boosting"]})
    assert ENSEMBLE_NAME not in results.index and diag["ensemble_members"] == []


def test_one_se_rule_uses_fold_spread_not_a_paired_row_test():
    """A model that is consistently but negligibly worse counts as just as good; a clearly worse one does not."""
    rng = np.random.default_rng(3)
    n = 200_000
    p = np.full(n, 0.005)
    y = (rng.random(n) < p).astype(int)
    splits = make_splits(y, 3, 0)
    predictions = {"best": p, "tiny": p * 1.002, "bad": np.full(n, 0.02)}  # tiny: worse on every row, by a hair
    within, margin = models_within_margin(y, predictions, "best", splits, tolerance=0.0)
    assert within == {"best", "tiny"} and margin > 0


def test_tolerance_widens_the_margin_to_a_practical_size():
    """Stratified folds give a tiny standard error; the 0.25% tolerance lets a negligibly worse model through."""
    rng = np.random.default_rng(4)
    n = 200_000
    y = (rng.random(n) < 0.005).astype(int)
    splits = make_splits(y, 3, 0)
    best = np.full(n, y.mean())  # the best constant prediction
    predictions = {"best": best,
                   "slightly_worse": best * 1.08,  # about 0.05% worse in log loss
                   "clearly_worse": best * 1.6}    # well over 0.25% worse
    assert models_within_margin(y, predictions, "best", splits, tolerance=0.0)[0] == {"best"}
    within, margin = models_within_margin(y, predictions, "best", splits, tolerance=0.0025)
    assert within == {"best", "slightly_worse"}
    assert margin == 0.0025 * score_predictions(y, best)["log_loss"]
