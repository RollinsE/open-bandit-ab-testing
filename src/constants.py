# -*- coding: utf-8 -*-
"""Project-wide constants: names, experiment settings and model hyperparameters.

Everything a user might want to tune for a run lives here or on the command
line, so the analysis modules contain no magic numbers.
"""

import numpy as np

# =============================================================================
# PROJECT
# =============================================================================

PROJECT_NAME = "Open Bandit A/B Testing"
LOGGER_NAME = "Open_Bandit_AB_Testing"
LOG_FILE_PREFIX = "open_bandit_pipeline"

# Environment variables read by src.config
ENV_RUNTIME = "OBD_ENV"            # "production" or "local" forces the runtime mode
ENV_DATA_DIR = "OBD_DATA_DIR"      # folder holding the full Open Bandit Dataset

STAGES = {
    1: "Data loading and validation",
    2: "A/B test",
    3: "Reward model marathon",
    4: "Off-policy evaluation",
    5: "A/B vs off-policy comparison",
    6: "Reporting",
}

# =============================================================================
# DATASET
# =============================================================================

BEHAVIOR_POLICIES = ("random", "bts")
CAMPAIGNS = ("all", "men", "women")

# Arm labels used in the A/B test. Control is the uniform random recommender,
# treatment is ZOZOTOWN's Bernoulli Thompson Sampling recommender.
CONTROL_POLICY = "random"
TREATMENT_POLICY = "bts"
POLICY_LABELS = {"random": "Random (control)", "bts": "Thompson Sampling (treatment)"}

# =============================================================================
# A/B TEST
# =============================================================================

class ABTestConstants:
    """Defaults for the A/B test stage."""

    ALPHA = 0.05                 # two-sided significance level
    POWER = 0.80                 # target power for sample-size and MDE figures
    N_BOOTSTRAP = 2000           # resamples for the bootstrap interval on the lift
    EXPECTED_SPLIT = 0.5         # intended control share when analysing your own test with analyse_ab_test
    PIPELINE_EXPECTED_SPLIT = None  # pipeline default: the dataset's intended split is not documented
    SRM_ALPHA = 0.001            # sample-ratio-mismatch alarm threshold


# =============================================================================
# OFF-POLICY EVALUATION
# =============================================================================

class OPEConstants:
    """Defaults for the off-policy evaluation stage."""

    N_RUNS = 10                  # bootstrap repetitions of the benchmark
    N_SIM_ACTION_DIST = 1_000_000  # Monte Carlo draws for the BTS action distribution
    N_FOLDS = 3                  # cross-validation and cross-fitting folds for the reward model
    MODEL_TRIALS = 8             # tuning trials per model family in the marathon (0 = no tuning)
    MARATHON_MAX_ROWS = 200_000  # stratified row cap for the marathon on the full dataset
    TUNE_MAX_ROWS = 50_000       # stratified sample used for each tuning search (0 = every training row)
    TUNE_TOP = 2                 # tune only the best N families on their defaults (-1 = all, 0 = none)
    WINNER_RULE = "simplest"     # "simplest": simplest model within the margin of the best; "best": lowest log loss
    OPE_FIT_MAX_ROWS = 200_000   # stage 4: cap on click-model training rows per fold (0 = every row)
    WINNER_TOLERANCE = 0.0025    # practical margin: models within 0.25% of the best log loss count as just as good
    LAMBDAS = [10, 50, 100, 500, 1000, 5000, 10000, np.inf]

# Model families that can enter the marathon. LightGBM, XGBoost and CatBoost
# join automatically when their library is installed.
MODEL_FAMILIES = ("logistic_regression", "random_forest", "gradient_boosting",
                  "hist_gradient_boosting", "lightgbm", "xgboost", "catboost")

# Entered by default. sklearn's plain GradientBoostingClassifier is left out: it was
# by far the slowest family on the full dataset, and HistGradientBoostingClassifier
# (which sklearn recommends from about 10,000 rows) does the same job much faster.
# Add it back with --model-families.
DEFAULT_MODEL_FAMILIES = tuple(f for f in MODEL_FAMILIES if f != "gradient_boosting")

# "Simplest" means cheapest to train and predict, in this order. Under the default
# winner rule, the click model is the first family here whose log loss is within one
# standard error of the best; default settings beat tuned ones of the same family.
# Every bootstrap run in stage 4 retrains the winner, so this choice sets most of
# the run time.
SIMPLICITY_ORDER = ("logistic_regression", "lightgbm", "hist_gradient_boosting", "xgboost",
                    "catboost", "random_forest", "gradient_boosting", "ensemble")

ESTIMATOR_LABELS = {
    "dm": "DM",
    "ipw": "IPW",
    "snipw": "SNIPW",
    "dr": "DR",
    "sndr": "SNDR",
    "switch-dr": "Switch-DR",
    "dr-os": "DR-os",
}

RANDOM_STATE = 12345

# Long steps log a "still running" line when nothing has finished for this long.
HEARTBEAT_SECONDS = 180
