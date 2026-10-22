# -*- coding: utf-8 -*-
"""Stage 2: the A/B test of Thompson Sampling against the random recommender.

How the test is analysed:

1. Metric. Click-through rate (CTR): clicks per item impression. Control is the
   uniform random recommender, treatment is Bernoulli Thompson Sampling.
2. Sample ratio check. Each arm's share of impressions is reported. When the
   intended split is known (``--expected-control-share``, e.g. 0.5 for a 50/50
   test), a chi-square test compares the observed share with it; a mismatch
   means the assignment or the logging is broken, and the result should not be
   trusted. The Open Bandit Dataset paper says impressions were randomly
   assigned to the two policies but does not state the split (the random arm
   received about 10% in every campaign), so by default the check is reported
   as "not tested" rather than judged against a guessed split.
3. Effect and intervals. For each campaign the pipeline reports both CTRs with
   Wilson intervals, the absolute difference with a normal interval, and the
   relative lift with a delta-method interval on the log ratio, plus a
   parametric bootstrap interval on the lift as a check.
4. Significance. A pooled two-proportion z-test gives the p-value; Fisher's
   exact test is reported alongside because click counts can be small.
5. Power. The pipeline reports the power the test had for the observed lift,
   the smallest lift it could reliably detect (MDE), and the impressions per arm
   needed to detect the observed lift.

Caveat: impressions are the unit of analysis, but one page view shows three
items, so rows from the same visit are not independent. Intervals are
therefore somewhat too narrow. The dataset has no user id to cluster on.
"""

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.constants import CONTROL_POLICY, TREATMENT_POLICY, ABTestConstants
from src.utils import get_logger


# =============================================================================
# STATISTICS
# =============================================================================

def wilson_interval(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Behaves well for rates near zero, where the normal interval can go negative.
    """
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (float(centre - half), float(centre + half))


def two_proportion_ztest(x_c: int, n_c: int, x_t: int, n_t: int) -> tuple[float, float]:
    """Pooled two-sided z-test for a difference in proportions.

    Returns:
        ``(z, p_value)``.
    """
    p_pool = (x_c + x_t) / (n_c + n_t)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    if se == 0:
        return (0.0, 1.0)
    z = (x_t / n_t - x_c / n_c) / se
    return (float(z), float(2 * stats.norm.sf(abs(z))))


def difference_interval(x_c: int, n_c: int, x_t: int, n_t: int, alpha: float = 0.05) -> tuple[float, float]:
    """Unpooled normal interval for ``p_t - p_c``."""
    p_c, p_t = x_c / n_c, x_t / n_t
    se = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    z = stats.norm.ppf(1 - alpha / 2)
    diff = p_t - p_c
    return (float(diff - z * se), float(diff + z * se))


def lift_interval(x_c: int, n_c: int, x_t: int, n_t: int, alpha: float = 0.05) -> tuple[float, float]:
    """Delta-method interval for the relative lift ``p_t / p_c - 1``.

    Built on the log of the rate ratio, which is closer to normal than the
    ratio itself, then transformed back.
    """
    if x_c == 0 or x_t == 0:
        return (np.nan, np.nan)
    p_c, p_t = x_c / n_c, x_t / n_t
    log_rr = np.log(p_t / p_c)
    se = np.sqrt((1 - p_t) / x_t + (1 - p_c) / x_c)
    z = stats.norm.ppf(1 - alpha / 2)
    return (float(np.exp(log_rr - z * se) - 1), float(np.exp(log_rr + z * se) - 1))


def bootstrap_lift_interval(x_c: int, n_c: int, x_t: int, n_t: int, n_bootstrap: int = 2000,
                            alpha: float = 0.05, random_state: Optional[int] = None) -> tuple[float, float]:
    """Parametric bootstrap percentile interval for the relative lift.

    Each resample redraws both arms' click counts from binomials at the
    observed rates, which is the same as resampling impressions with
    replacement.
    """
    rng = np.random.default_rng(random_state)
    c = rng.binomial(n_c, x_c / n_c, n_bootstrap) / n_c
    t = rng.binomial(n_t, x_t / n_t, n_bootstrap) / n_t
    ok = c > 0
    lifts = t[ok] / c[ok] - 1
    lo, hi = np.quantile(lifts, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def sample_ratio_mismatch(n_c: int, n_t: int, expected_control_share: float = 0.5) -> float:
    """Chi-square p-value that the observed arm sizes match the expected split."""
    total = n_c + n_t
    expected = [total * expected_control_share, total * (1 - expected_control_share)]
    return float(stats.chisquare([n_c, n_t], f_exp=expected).pvalue)


def achieved_power(p_c: float, p_t: float, n_c: int, n_t: int, alpha: float = 0.05) -> float:
    """Power of a two-sided two-proportion test to detect ``p_t`` vs ``p_c``."""
    se = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    if se == 0:
        return float("nan")
    z_a = stats.norm.ppf(1 - alpha / 2)
    shift = abs(p_t - p_c) / se
    return float(stats.norm.sf(z_a - shift) + stats.norm.cdf(-z_a - shift))


def minimum_detectable_effect(p_c: float, n_c: int, n_t: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest absolute CTR difference detectable at the given power.

    Uses the control rate's variance for both arms, the usual planning
    approximation.
    """
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * np.sqrt(p_c * (1 - p_c) * (1 / n_c + 1 / n_t)))


def required_sample_size(p_c: float, p_t: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Impressions needed per arm (equal split) to detect ``p_t`` vs ``p_c``."""
    if p_t == p_c:
        return float("inf")
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    p_bar = (p_c + p_t) / 2
    num = (z_a * np.sqrt(2 * p_bar * (1 - p_bar)) + z_b * np.sqrt(p_c * (1 - p_c) + p_t * (1 - p_t))) ** 2
    return float(np.ceil(num / (p_t - p_c) ** 2))


# =============================================================================
# ANALYSIS
# =============================================================================

@dataclass
class ABTestResult:
    """Everything reported for one campaign's A/B test."""

    campaign: str
    control_impressions: int
    control_clicks: int
    control_ctr: float
    control_ctr_lower: float
    control_ctr_upper: float
    treatment_impressions: int
    treatment_clicks: int
    treatment_ctr: float
    treatment_ctr_lower: float
    treatment_ctr_upper: float
    abs_diff: float
    abs_diff_lower: float
    abs_diff_upper: float
    rel_lift: float
    rel_lift_lower: float
    rel_lift_upper: float
    rel_lift_boot_lower: float
    rel_lift_boot_upper: float
    z_stat: float
    p_value: float
    fisher_p_value: float
    significant: bool
    control_share: float
    expected_control_share: float
    srm_p_value: float
    srm_flag: bool
    achieved_power: float
    mde_abs: float
    mde_rel: float
    n_per_arm_needed: float
    decision: str


def analyse_ab_test(campaign: str, x_c: int, n_c: int, x_t: int, n_t: int,
                    alpha: float = ABTestConstants.ALPHA, power: float = ABTestConstants.POWER,
                    n_bootstrap: int = ABTestConstants.N_BOOTSTRAP,
                    expected_split: Optional[float] = ABTestConstants.EXPECTED_SPLIT,
                    random_state: Optional[int] = None) -> ABTestResult:
    """Analyse one campaign's A/B test from click and impression counts.

    Args:
        campaign: Campaign name, carried into the result.
        x_c, n_c: Control clicks and impressions.
        x_t, n_t: Treatment clicks and impressions.
        alpha: Two-sided significance level.
        power: Target power for the MDE and sample-size figures.
        n_bootstrap: Resamples for the bootstrap lift interval.
        expected_split: Intended control share of impressions, for the sample
            ratio check. ``None`` skips the check (the observed share is still reported).
        random_state: Seed for the bootstrap.

    Returns:
        An ``ABTestResult``.
    """
    p_c, p_t = x_c / n_c, x_t / n_t
    z, p = two_proportion_ztest(x_c, n_c, x_t, n_t)
    fisher_p = float(stats.fisher_exact([[x_t, n_t - x_t], [x_c, n_c - x_c]]).pvalue)
    if expected_split is None:
        srm_p, srm_flag = float("nan"), False
    else:
        srm_p = sample_ratio_mismatch(n_c, n_t, expected_split)
        srm_flag = srm_p < ABTestConstants.SRM_ALPHA
    significant = p < alpha
    lift = p_t / p_c - 1 if p_c > 0 else np.nan
    mde = minimum_detectable_effect(p_c, n_c, n_t, alpha, power)
    ci_c = wilson_interval(x_c, n_c, alpha)
    ci_t = wilson_interval(x_t, n_t, alpha)
    ci_diff = difference_interval(x_c, n_c, x_t, n_t, alpha)
    ci_lift = lift_interval(x_c, n_c, x_t, n_t, alpha)
    ci_boot = bootstrap_lift_interval(x_c, n_c, x_t, n_t, n_bootstrap, alpha, random_state)

    if srm_flag:
        decision = "Do not trust: sample ratio mismatch"
    elif significant and lift > 0:
        decision = "Treatment wins"
    elif significant and lift < 0:
        decision = "Control wins"
    else:
        decision = "No significant difference"

    return ABTestResult(
        campaign=campaign,
        control_impressions=n_c, control_clicks=x_c, control_ctr=p_c,
        control_ctr_lower=ci_c[0], control_ctr_upper=ci_c[1],
        treatment_impressions=n_t, treatment_clicks=x_t, treatment_ctr=p_t,
        treatment_ctr_lower=ci_t[0], treatment_ctr_upper=ci_t[1],
        abs_diff=p_t - p_c, abs_diff_lower=ci_diff[0], abs_diff_upper=ci_diff[1],
        rel_lift=lift, rel_lift_lower=ci_lift[0], rel_lift_upper=ci_lift[1],
        rel_lift_boot_lower=ci_boot[0], rel_lift_boot_upper=ci_boot[1],
        z_stat=z, p_value=p, fisher_p_value=fisher_p, significant=bool(significant),
        control_share=n_c / (n_c + n_t),
        expected_control_share=float("nan") if expected_split is None else float(expected_split),
        srm_p_value=srm_p, srm_flag=bool(srm_flag),
        achieved_power=achieved_power(p_c, p_t, n_c, n_t, alpha),
        mde_abs=mde, mde_rel=mde / p_c if p_c > 0 else np.nan,
        n_per_arm_needed=required_sample_size(p_c, p_t, alpha, power),
        decision=decision,
    )


def execute_ab_test(datasets: Dict, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run stage 2 for every campaign and write the results.

    Args:
        datasets: Output of ``execute_data_pipeline``.
        config: Run configuration.

    Returns:
        ``(results, power_table)``: one row per campaign, and the sample sizes
        needed for a range of lifts. Both are written to the build's artifacts.
    """
    LOG = get_logger()
    rows = []
    for campaign in config["campaigns"]:
        control = datasets[(CONTROL_POLICY, campaign)]
        treatment = datasets[(TREATMENT_POLICY, campaign)]
        result = analyse_ab_test(
            campaign,
            x_c=int(control.reward.sum()), n_c=control.n_rounds,
            x_t=int(treatment.reward.sum()), n_t=treatment.n_rounds,
            alpha=config.get("alpha", ABTestConstants.ALPHA),
            expected_split=config.get("expected_control_share"),
            random_state=config.get("random_state"),
        )
        rows.append(asdict(result))
        LOG.info(
            "A/B %s: control CTR %.3f%%, treatment CTR %.3f%%, lift %+.1f%% "
            "(95%% CI %+.1f%% to %+.1f%%), p=%.3f -> %s",
            campaign, 100 * result.control_ctr, 100 * result.treatment_ctr,
            100 * result.rel_lift, 100 * result.rel_lift_lower, 100 * result.rel_lift_upper,
            result.p_value, result.decision,
        )
        if np.isnan(result.expected_control_share):
            LOG.info("A/B %s: control arm got %.1f%% of impressions (%s of %s); sample ratio check not run "
                     "because no intended split was given (--expected-control-share)",
                     campaign, 100 * result.control_share, f"{result.control_impressions:,}",
                     f"{result.control_impressions + result.treatment_impressions:,}")
        elif result.srm_flag:
            LOG.warning("A/B %s: sample ratio mismatch: control got %.2f%% of impressions against an intended "
                        "%.2f%% (p=%.2g). Check assignment and logging.", campaign, 100 * result.control_share,
                        100 * result.expected_control_share, result.srm_p_value)
        else:
            LOG.info("A/B %s: sample ratio check passed (control %.2f%% against an intended %.2f%%, p=%.2g)",
                     campaign, 100 * result.control_share, 100 * result.expected_control_share,
                     result.srm_p_value)
        if result.abs_diff == 0:
            LOG.warning("A/B %s: both arms have the same CTR, so there is no observed lift to size a test for.",
                        campaign)
        elif result.achieved_power < 0.5:
            LOG.warning(
                "A/B %s is underpowered (power %.0f%%). About %s impressions per arm are needed "
                "to detect the observed lift.",
                campaign, 100 * result.achieved_power, f"{result.n_per_arm_needed:,.0f}",
            )

    results = pd.DataFrame(rows)
    results.to_csv(config["output_files"]["ab_test_results"], index=False)
    LOG.info("Wrote A/B test results: %s", config["output_files"]["ab_test_results"])

    power_table = build_power_table(results, config)
    power_table.to_csv(config["output_files"]["ab_test_power"], index=False)
    LOG.info("Wrote A/B power table: %s", config["output_files"]["ab_test_power"])
    return results, power_table


def build_power_table(results: pd.DataFrame, config: dict,
                      lifts: tuple = (0.05, 0.10, 0.20, 0.30, 0.50)) -> pd.DataFrame:
    """Impressions per arm needed to detect a range of relative lifts.

    Uses each campaign's observed control CTR as the baseline, which is the
    planning table for a future, properly sized test.
    """
    alpha = config.get("alpha", ABTestConstants.ALPHA)
    rows = []
    for _, r in results.iterrows():
        for lift in lifts:
            rows.append({
                "campaign": r["campaign"],
                "baseline_ctr": r["control_ctr"],
                "target_lift": lift,
                "target_ctr": r["control_ctr"] * (1 + lift),
                "n_per_arm": required_sample_size(r["control_ctr"], r["control_ctr"] * (1 + lift),
                                                  alpha, ABTestConstants.POWER),
            })
    return pd.DataFrame(rows)
