# -*- coding: utf-8 -*-
"""Open Bandit Dataset loader.

Mirrors ``obp.dataset.OpenBanditDataset``: same preprocessing (one-hot user
features, label-encoded item features), same position encoding, same bootstrap
and on-policy value used as ground truth in the OBD benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import LabelEncoder

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "obd"
BEHAVIOR_POLICIES = ("random", "bts")
CAMPAIGNS = ("all", "men", "women")


def parse_timestamps(values: pd.Series) -> pd.Series:
    """Parse the dataset's timestamps to UTC datetimes.

    The full dataset mixes two forms, with and without fractions of a second
    ("2019-11-25 21:27:37.123456+00:00" and "2019-11-25 21:27:37+00:00"), so
    the format must not be guessed from the first row.
    """
    return pd.to_datetime(values, format="ISO8601", utc=True)


@dataclass
class OpenBanditDataset:
    """Logged bandit feedback from ZOZOTOWN's fashion recommendation widget.

    behavior_policy: "random" (uniform) or "bts" (Bernoulli Thompson Sampling)
    campaign: "all", "men" or "women"
    data_path: folder holding ``<policy>/<campaign>/<campaign>.csv``. Defaults to
        the bundled 10k-row sample; point it at the full dataset (~27M rows,
        https://research.zozo.com/data.html) for real experiments.
    light: load only timestamp, item, slot, click and propensity, and skip the
        user features. Enough for CTRs and the A/B test, at a fraction of the
        memory: the full Thompson Sampling "all" file has about 12M rows.
    """

    behavior_policy: str
    campaign: str
    data_path: Path | None = None
    light: bool = False
    dataset_name: str = field(default="obd", init=False)

    def __post_init__(self) -> None:
        if self.behavior_policy not in BEHAVIOR_POLICIES:
            raise ValueError(f"behavior_policy must be one of {BEHAVIOR_POLICIES}")
        if self.campaign not in CAMPAIGNS:
            raise ValueError(f"campaign must be one of {CAMPAIGNS}")
        base = Path(self.data_path) if self.data_path else DEFAULT_DATA_PATH
        self.data_path = base / self.behavior_policy / self.campaign
        self.raw_data_file = f"{self.campaign}.csv"
        self.load_raw_data()
        self.pre_process()
        # Keep only the columns used after preprocessing. The raw table also holds
        # about 80 user-item affinity columns that the features never use; on the full
        # "all" campaign they are around 1 GB, copied to every parallel worker.
        self.data = self.data[[c for c in self.LIGHT_COLUMNS if c in self.data.columns]]

    # ------------------------------------------------------------------ loading
    LIGHT_COLUMNS = ("timestamp", "item_id", "position", "click", "propensity_score")

    def load_raw_data(self) -> None:
        path = self.data_path / self.raw_data_file
        if self.light:
            header = pd.read_csv(path, nrows=0).columns
            keep = [0] + [i for i, c in enumerate(header) if c in self.LIGHT_COLUMNS]
            self.data = pd.read_csv(path, index_col=0, usecols=keep)
        else:
            self.data = pd.read_csv(path, index_col=0)
        self.item_context = pd.read_csv(self.data_path / "item_context.csv", index_col=0)
        self.data["timestamp"] = parse_timestamps(self.data["timestamp"])
        self.data.sort_values("timestamp", inplace=True, kind="stable")
        self.action = self.data["item_id"].values.astype(int)
        self.position = (rankdata(self.data["position"].values, "dense") - 1).astype(int)
        self.reward = self.data["click"].values.astype(float)
        self.pscore = self.data["propensity_score"].values.astype(float)

    def pre_process(self) -> None:
        if self.light:
            self.context = np.zeros((self.n_rounds, 0))
        else:
            user_cols = self.data.columns.str.contains("user_feature")
            self.context = (
                pd.get_dummies(self.data.loc[:, user_cols].astype(str), drop_first=True)
                .values.astype(float)
            )
        item_feature_0 = self.item_context["item_feature_0"].to_frame()
        item_feature_cat = self.item_context.drop(
            columns=["item_id", "item_feature_0"]
        ).apply(LabelEncoder().fit_transform)
        self.action_context = pd.concat([item_feature_cat, item_feature_0], axis=1).values.astype(float)

    # ------------------------------------------------------------- properties
    @property
    def n_rounds(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_actions(self) -> int:
        return int(self.action.max() + 1)

    @property
    def len_list(self) -> int:
        return int(len(np.unique(self.position)))

    @property
    def dim_context(self) -> int:
        return int(self.context.shape[1])

    # --------------------------------------------------------------- feedback
    def obtain_batch_bandit_feedback(self, test_size: float = 0.3, is_timeseries_split: bool = False):
        full = dict(
            n_rounds=self.n_rounds,
            n_actions=self.n_actions,
            action=self.action,
            position=self.position,
            reward=self.reward,
            pscore=self.pscore,
            context=self.context,
            action_context=self.action_context,
        )
        if not is_timeseries_split:
            return full
        n_train = int(self.n_rounds * (1 - test_size))
        keys = ("action", "position", "reward", "pscore", "context")
        train = {**full, **{k: full[k][:n_train] for k in keys}, "n_rounds": n_train}
        test = {**full, **{k: full[k][n_train:] for k in keys}, "n_rounds": self.n_rounds - n_train}
        return train, test

    def sample_bootstrap_bandit_feedback(self, sample_size: int | None = None, random_state: int | None = None,
                                         rows: np.ndarray | None = None):
        """Resample rounds with replacement; ``rows`` restricts the draw to a subset (e.g. one time period)."""
        bf = self.obtain_batch_bandit_feedback()
        pool = np.arange(bf["n_rounds"]) if rows is None else np.asarray(rows)
        n = len(pool) if sample_size is None else sample_size
        idx = pool[np.random.default_rng(random_state).integers(0, len(pool), size=n)]
        for k in ("action", "position", "reward", "pscore", "context"):
            bf[k] = bf[k][idx]
        bf["n_rounds"] = n
        return bf

    @classmethod
    def calc_on_policy_policy_value_estimate(cls, behavior_policy: str, campaign: str, data_path: Path | None = None) -> float:
        """Average observed CTR of a policy on its own logs (the benchmark's ground truth)."""
        return float(cls(behavior_policy=behavior_policy, campaign=campaign, data_path=data_path).reward.mean())
