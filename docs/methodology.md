# Methodology

This document describes how each stage works and why. For installation and usage, see the [README](../README.md).

## Contents

1. [A/B test](#1-ab-test)
2. [Reward model marathon](#2-reward-model-marathon)
3. [Off-policy evaluation](#3-off-policy-evaluation)
4. [Fair estimator selection](#4-fair-estimator-selection)
5. [Working at full scale](#5-working-at-full-scale)
6. [Checkpoints and resuming](#6-checkpoints-and-resuming)
7. [Logging](#7-logging)

## 1. A/B test

* **Metric**: click-through rate (CTR) per item impression.
* **Intervals**: Wilson intervals for each arm's CTR (reliable near zero), a normal interval for the difference, a delta-method interval on the log rate ratio for the relative lift, and a bootstrap interval on the lift as a cross-check.
* **Significance**: a pooled two-proportion z-test, plus Fisher's exact test because click counts can be small.
* **Sample ratio check**: each arm's share of impressions is always reported. Given the intended split (`--expected-control-share`, for example 0.5 for a 50/50 test), a chi-square test checks the arms received it; a mismatch usually means broken assignment or logging, and the result is labelled untrustworthy. The Open Bandit Dataset paper states that impressions were randomly assigned to a policy but does not give the split. The random arm received about 10% of impressions in every campaign, which suggests a deliberate 10/90 design, but because this is not documented the check is off by default. `--expected-control-share 0.1` turns it on with that assumption.
* **Power**: the power the test had for the observed lift, the minimum detectable effect, and the impressions per arm needed for a given lift. `ab_test_power.csv` is a planning table for lifts of 5% to 50%.

**Caveat.** Each page view shows three items, so rows from the same visit are not independent and the intervals are somewhat too narrow. The dataset has no user or session identifier to correct for this.

## 2. Reward model marathon

The model-based estimators (DM, DR and its variants) need a click model q̂(x, a, k): the probability that a user with context *x* clicks item *a* in slot *k*. Several model families compete and one is chosen per campaign.

1. **Scoring.** Every model, tuned or untuned, single or ensemble, is scored on the log loss of its pooled out-of-fold predictions (3-fold cross-validation, stratified so each fold receives its share of the rare clicks). Brier score and ROC AUC are reported alongside.
2. **Baselines.** "Same CTR in every slot" and "item features only". *Skill vs baseline* is the relative improvement in log loss over the first.
3. **Two rounds.** Round 1 scores every family with default settings. Round 2 tunes only the best families from round 1 (`--tune-top`, default 2).
4. **Nested tuning.** Hyperparameters are tuned inside each training fold (Optuna if installed, random search otherwise), so a tuned model's score is not inflated by tuning on its own test data. The default settings are always one of the trials. On large data each search runs on a 50,000-row stratified sample of the fold (`--tune-max-rows`), and the chosen settings are then trained on the whole fold.
5. **Ensemble.** The best model from each of the top three families is averaged (`--no-ensemble` leaves it out).
6. **Winner rule.** By default the winner is the simplest model whose log loss is within a margin of the lowest (`--winner-rule simplest`). The margin is the wider of:
   * one standard error of the best model's cross-validated log loss, from its spread across the folds (the one-standard-error rule; Breiman et al., 1984; Hastie, Tibshirani and Friedman, *The Elements of Statistical Learning*, section 7.10);
   * a practical-equivalence tolerance of 0.25% of the best log loss (`--winner-tolerance`).

   The tolerance is needed because the folds are stratified: every fold receives the same share of clicks, so the fold-to-fold spread, and with it the standard error, is very small, and differences of a few hundred-thousandths of log loss would otherwise count as decisive. The 0.25% value is a judgement call. "Simplest" means cheapest to train, in this order: logistic regression, LightGBM, histogram gradient boosting, XGBoost, CatBoost, random forest, gradient boosting, ensemble; untuned before tuned. Simplicity matters because the click model is refitted in every bootstrap run of the off-policy benchmark. `--winner-rule best` selects the lowest log loss regardless.
7. **No leakage.** The winner is chosen on click prediction alone. Thompson Sampling's true CTR is never used, so the off-policy benchmark remains a fair test.

The leaderboard (`model_performance.csv`) lists every model's log loss, AUC, Brier score, training time, and whether it was within the margin of the best. Clicks are hard to predict in this dataset: on the full data the best models improve on the baseline log loss by only about 0.5% to 1%.

## 3. Off-policy evaluation

**Question.** Using only the random recommender's logs, estimate the CTR that Thompson Sampling would have achieved.

* **Target policy.** Thompson Sampling's probability of showing each item in each slot is computed from 1,000,000 Monte Carlo draws of its production Beta prior.
* **Estimators.** Direct Method (DM), Inverse Propensity Weighting (IPW), self-normalised IPW (SNIPW), Doubly Robust (DR), self-normalised DR (SNDR), Switch-DR and DR with optimistic shrinkage (DR-os). Switch-DR and DR-os choose their hyperparameter with SLOPE++.
* **Click model.** The campaign's marathon winner, one model per slot, with 3-fold cross-fitting so that no row is scored by a model that was trained on it.
* **Ground truth.** Because both recommenders ran in parallel, Thompson Sampling's real CTR is known from its own logs. Each estimate is scored by its relative error against it, averaged over bootstrap resamples of the random logs (`--n-runs`, default 10).

The estimators are implemented in `src/obp_lite/`, a re-implementation of the core of ZOZO's [Open Bandit Pipeline](https://github.com/st-tech/zr-obp). The original package requires Python below 3.10, scikit-learn 1.0.2 and PyTorch, and fails on current pandas. `obp_lite` runs on current libraries; on identical inputs all seven estimators match the original code exactly, and the data loader produces identical features for all six arm and campaign splits.

## 4. Fair estimator selection

Choosing the estimator closest to the true CTR and reporting that same error overstates how well the method works: part of the winning margin is luck, and in practice there is no true CTR to choose with. The pipeline therefore asks: *if an estimator had been chosen by a rule, how wrong would it have been on data the rule never saw?*

* **Campaign split.** For each campaign, the estimator with the lowest error on the other two campaigns is chosen and scored on the held-out campaign. This reuses the benchmark's results at no extra cost. The "all" campaign shares many items with "men" and "women", so this is not a fully independent test.
* **Time split.** Each campaign's week is cut at the random arm's median timestamp. Every estimator is run on each half and scored against Thompson Sampling's CTR over the same period. The estimator with the lowest error on the first half is scored on the second half. This is the cleaner check (`--no-time-split` skips it).

Each result is shown next to the best estimator in hindsight (optimistic by construction) and the median estimator. Stage 5 then compares the A/B lift with the lift implied by the fairly chosen estimator: the conclusion that would have been reached without running the A/B test.

## 5. Working at full scale

The full dataset has about 27 million rows. The pipeline uses the following techniques to keep memory and run time bounded.

* **Prediction per distinct context.** The click model sees four one-hot user features, so there are at most 960 distinct contexts against up to 1.4M logged rows. Each distinct context is predicted for every item once and the result copied to its rows, which gives identical values at a fraction of the cost. Each row is predicted only at its own slot, the only value the estimators read.
* **Light loading.** Only the random arm's user features are loaded. The Thompson Sampling arm (about 24M rows) is read as five columns: timestamp, item, slot, click and propensity.
* **Marathon sampling.** The marathon uses a click-stratified sample of 200,000 rows per campaign (`--marathon-max-rows`), preserving the click rate and slot mix.
* **Click-model training cap in the benchmark.** In stage 4, each fold's click models are trained on at most 200,000 rows (`--ope-fit-max-rows`), a click-stratified sample of that fold's training rows. Every row is still predicted, and cross-fitting is unaffected. On the full dataset, a capped and an uncapped run gave the same results to within 0.1 percentage points (lowest errors of 6.7%, 9.3% and 2.2% for all, men and women, against 6.7%, 9.4% and 2.2% uncapped; identical campaign-split results).
* **Default model families.** scikit-learn's `GradientBoostingClassifier` is excluded by default because it is much slower than `HistGradientBoostingClassifier` at this size; `--model-families` adds it back.
* **Memory.** Large arrays are built in chunks and stored as float32, and the number of parallel jobs is capped by available memory, using a measured peak of about 2.6 GB for one bootstrap run on the "all" campaign.

## 6. Checkpoints and resuming

Every slow unit of work is saved as soon as it finishes, in the build's `checkpoints/` folder: each model scored in the marathon and each campaign's marathon result, each bootstrap run of the benchmark, and each time-split run. Files are written atomically (to a temporary file, then renamed).

`--continue-build auto` continues the newest unfinished build in `--output-dir`, or starts a new build if there is none. A continued build uses the settings recorded in its `run_manifest.json`, loads everything already finished, runs only what is missing, appends to the same log file and produces a single report. `--continue-build latest` does the same but stops with an error when there is nothing to continue, and a specific build folder can also be given. Only `--data-dir`, `--n-jobs`, `--log-level`, `--heartbeat-minutes`, `--show-plots` and `--expected-control-share` can change when continuing.

A lock file prevents two sessions from working on the same build at once. If the earlier session has died, the lock expires after two minutes.

Resuming was tested by force-killing runs part-way through the marathon and part-way through the benchmark and then continuing them: every output was identical to an uninterrupted run.

`--resume-build <build>` with `--from-stage` or `--only-stage` is different: it starts a new build that reuses an earlier build's outputs and copies them in, so the new build is self-contained. Only stages *before* the first re-run stage are reused; outputs of later stages would be out of date, so they are left out. A resumed build reads the same data folder as the earlier build unless `--data-dir` is given, and it stops with an error if the data does not match the earlier build's row and click counts, so results from different datasets are never combined.

## 7. Logging

Logging follows a file-first design. Every run writes a complete log to `builds/<run_id>/logs/`. All modules log through one named logger, never the root logger. The console handler tolerates Colab detaching the notebook's output stream during a long run: the pipeline continues and keeps writing to the log file.

Long steps report progress: the marathon logs each model as it finishes, with its duration; the benchmark logs each bootstrap run with an estimate of the time remaining; and if nothing finishes for three minutes a "still running" line is written (`--heartbeat-minutes`). Each line reads:

```
[2026-10-19 11:35:13] INFO     - A/B men: control CTR 0.460%, treatment CTR 0.690%, lift +50.0% (95% CI +3.4% to +117.6%), p=0.031 -> Treatment wins
```
