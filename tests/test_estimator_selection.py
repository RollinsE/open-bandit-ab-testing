import pandas as pd
import pytest

from src.estimator_selection import select_by_campaign, split_by_time
from src.obp_lite import OpenBanditDataset


def _summary(errors: dict) -> pd.DataFrame:
    rows = []
    for campaign, per_estimator in errors.items():
        for estimator, err in per_estimator.items():
            rows.append({"campaign": campaign, "estimator": estimator, "relative_error_mean": err,
                         "mean_estimate": 0.004 * (1 + err), "ground_truth": 0.004, "control_ctr": 0.0035})
    return pd.DataFrame(rows)


def test_campaign_split_chooses_on_other_campaigns_only():
    # "ipw" is best on men and women but worst on all; "dm" is best on all only.
    summary = _summary({
        "all": {"dm": 0.05, "ipw": 0.90},
        "men": {"dm": 0.40, "ipw": 0.10},
        "women": {"dm": 0.40, "ipw": 0.10},
    })
    sel = select_by_campaign(summary).set_index("held_out")
    assert sel.loc["all", "selected_estimator"] == "ipw"      # chosen without seeing "all"
    assert sel.loc["all", "test_error"] == pytest.approx(0.90)  # so it pays for that on "all"
    assert sel.loc["all", "oracle_estimator"] == "dm"
    assert sel.loc["all", "selected_rank"] == 2


def test_campaign_split_needs_two_campaigns():
    assert select_by_campaign(_summary({"all": {"dm": 0.1, "ipw": 0.2}})).empty


def test_time_split_halves_the_control_arm():
    control = OpenBanditDataset("random", "men")
    treatment = OpenBanditDataset("bts", "men", light=True)
    split = split_by_time(control, treatment)
    first, second = split["control_rows"]
    assert len(first) + len(second) == control.n_rounds
    assert abs(len(first) - len(second)) <= 1
    ts = control.data["timestamp"].to_numpy()
    assert ts[first].max() < ts[second].min()
    assert 0 < split["treatment_ctr"][0] < 1 and 0 < split["treatment_ctr"][1] < 1
