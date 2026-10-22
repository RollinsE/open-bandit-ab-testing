# -*- coding: utf-8 -*-
"""Stage 5: compare the A/B test result with the off-policy estimates.

How the comparison works:

1. Same question, two answers. The A/B test measures Thompson Sampling's lift
   over random directly, from live traffic. OPE estimates the same lift from
   the random arm's logs alone: estimated treatment CTR / control CTR - 1.
2. Agreement. An estimator agrees with the A/B test when its estimated lift
   lies inside the A/B test's 95% interval for the lift.
3. Decision match. It also checks whether the OPE estimate would have led to
   the same call on direction (treatment better or worse than control).
4. Fair pick. The estimator chosen for each campaign by the leave-one-campaign-out
   rule (stage 4) is flagged, so the headline compares the A/B test with an
   estimator that was chosen without looking at that campaign.

This shows how far OPE can stand in for an A/B test on this data, which is the
practical reason to use it: to screen candidate policies before spending live
traffic on them.
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.utils import get_logger


def compare_ab_and_ope(ab_results: pd.DataFrame, ope_summary: pd.DataFrame, config: dict,
                       selection: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Join stage 2 and stage 4 into one table of lifts.

    Args:
        ab_results: Output of ``execute_ab_test``.
        ope_summary: Summary output of ``execute_ope_benchmark`` (stage 4).
        config: Run configuration.
        selection: Output of ``execute_estimator_selection``; its campaign-split
            picks are flagged in ``selected_by_campaign_split``.

    Returns:
        One row per campaign, reward model and estimator, written to
        ``output_files["ab_vs_ope"]``.
    """
    LOG = get_logger()
    ab = ab_results[["campaign", "control_ctr", "treatment_ctr", "rel_lift",
                     "rel_lift_lower", "rel_lift_upper", "decision"]].rename(columns={
        "control_ctr": "ab_control_ctr", "treatment_ctr": "ab_treatment_ctr",
        "rel_lift": "ab_lift", "rel_lift_lower": "ab_lift_lower",
        "rel_lift_upper": "ab_lift_upper", "decision": "ab_decision",
    })
    merged = ope_summary.merge(ab, on="campaign", how="inner")
    merged["ope_lift"] = merged["mean_estimate"] / merged["control_ctr"] - 1
    merged["lift_error"] = merged["ope_lift"] - merged["ab_lift"]
    merged["within_ab_ci"] = merged["ope_lift"].between(merged["ab_lift_lower"], merged["ab_lift_upper"])
    # Direction is undefined when the A/B test observed no difference at all.
    merged["same_direction"] = np.where(
        merged["ab_lift"] == 0, np.nan,
        (np.sign(merged["ope_lift"]) == np.sign(merged["ab_lift"])).astype(float),
    )

    picks = set()
    if selection is not None and not selection.empty:
        chosen = selection[selection["split"] == "campaign"]
        picks = set(zip(chosen["held_out"], chosen["selected_estimator"]))
    merged["selected_by_campaign_split"] = [(c, e) in picks for c, e in zip(merged["campaign"], merged["estimator"])]

    columns = ["campaign", "base_model", "estimator", "ab_lift", "ab_lift_lower", "ab_lift_upper",
               "ope_lift", "lift_error", "within_ab_ci", "same_direction", "selected_by_campaign_split",
               "ab_decision"]
    comparison = merged[columns].sort_values(["campaign", "base_model", "estimator"])
    comparison.to_csv(config["output_files"]["ab_vs_ope"], index=False)

    for campaign, grp in comparison.groupby("campaign"):
        direction = ("A/B lift is zero, so direction is undefined" if grp["same_direction"].isna().all()
                     else f"{int(grp['same_direction'].sum())} match its direction")
        LOG.info("Comparison %s: %d of %d estimates inside the A/B lift interval; %s",
                 campaign, int(grp["within_ab_ci"].sum()), len(grp), direction)
        pick = grp[grp["selected_by_campaign_split"]]
        if not pick.empty:
            r = pick.iloc[0]
            LOG.info("Comparison %s: fairly chosen estimator %s puts the lift at %+.1f%%; A/B measured %+.1f%% "
                     "(95%% CI %+.1f%% to %+.1f%%)", campaign, r["estimator"], 100 * r["ope_lift"],
                     100 * r["ab_lift"], 100 * r["ab_lift_lower"], 100 * r["ab_lift_upper"])
    LOG.info("Wrote A/B vs OPE comparison: %s", config["output_files"]["ab_vs_ope"])
    return comparison
