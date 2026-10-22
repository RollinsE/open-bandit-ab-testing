# -*- coding: utf-8 -*-
"""Stage 1: load and validate the Open Bandit Dataset.

How the data is prepared:

1. Two arms. The dataset was collected in an A/B test on ZOZOTOWN between a
   uniform random recommender and a Bernoulli Thompson Sampling recommender.
   Each arm's logs are loaded separately for every campaign (all, men, women).
2. One row per impression slot. Each row is one item shown in one of three
   slots, with a click indicator and the probability the logging policy had of
   showing that item there (the propensity score).
3. Checks. Clicks must be 0 or 1, propensities must lie in (0, 1], slots must
   be 0 to 2, and item ids must fall inside the item catalogue. A run stops on
   the first failed check, because every later stage depends on these.
4. Features. User features are one-hot encoded and item features label
   encoded, exactly as in the original Open Bandit Pipeline.
5. Memory. Only the control (random) arm's user features are loaded, because
   only its logs feed the click models and the off-policy estimators. The
   treatment arm needs clicks, slots and timestamps alone, which matters on
   the full dataset, where Thompson Sampling's "all" file has about 12M rows.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.constants import BEHAVIOR_POLICIES, CONTROL_POLICY, POLICY_LABELS
from src.obp_lite import OpenBanditDataset
from src.utils import get_logger

DatasetMap = Dict[Tuple[str, str], OpenBanditDataset]


def load_datasets(config: dict) -> DatasetMap:
    """Load every behavior policy x campaign split.

    Args:
        config: Run configuration with ``data_directory`` and ``campaigns``.

    Returns:
        Mapping of ``(policy, campaign)`` to a loaded ``OpenBanditDataset``.
    """
    LOG = get_logger()
    datasets: DatasetMap = {}
    for policy in BEHAVIOR_POLICIES:
        for campaign in config["campaigns"]:
            obd = OpenBanditDataset(
                behavior_policy=policy,
                campaign=campaign,
                data_path=Path(config["data_directory"]),
                light=(policy != CONTROL_POLICY),
            )
            datasets[(policy, campaign)] = obd
            LOG.info(
                "Loaded %s / %s: %s rows, %s items, %s clicks",
                policy, campaign, f"{obd.n_rounds:,}", obd.n_actions, int(obd.reward.sum()),
            )
    return datasets


def validate_dataset(obd: OpenBanditDataset) -> None:
    """Check one split's values are what the later stages assume.

    Raises:
        ValueError: Naming the split and the failed check.
    """
    name = f"{obd.behavior_policy}/{obd.campaign}"
    if not np.isin(obd.reward, [0.0, 1.0]).all():
        raise ValueError(f"{name}: click column has values other than 0 and 1")
    if not ((obd.pscore > 0) & (obd.pscore <= 1)).all():
        raise ValueError(f"{name}: propensity scores must be in (0, 1]")
    if obd.len_list != 3 or obd.position.min() != 0 or obd.position.max() != 2:
        raise ValueError(f"{name}: expected 3 recommendation slots coded 0-2")
    if obd.action.min() < 0 or obd.action.max() >= obd.action_context.shape[0]:
        raise ValueError(f"{name}: item ids fall outside the item catalogue")
    if np.isnan(obd.context).any():
        raise ValueError(f"{name}: user features contain missing values")


def summarise_datasets(datasets: DatasetMap) -> pd.DataFrame:
    """One row per split: size, clicks, CTR, catalogue size and time window."""
    rows = []
    for (policy, campaign), obd in datasets.items():
        ts = obd.data["timestamp"]  # already parsed to UTC datetimes by the loader
        rows.append({
            "policy": policy,
            "arm": POLICY_LABELS[policy],
            "campaign": campaign,
            "impressions": obd.n_rounds,
            "clicks": int(obd.reward.sum()),
            "ctr": float(obd.reward.mean()),
            "n_items": obd.n_actions,
            "n_slots": obd.len_list,
            "n_user_features": None if obd.light else obd.dim_context,
            "pscore_min": float(obd.pscore.min()),
            "pscore_max": float(obd.pscore.max()),
            "first_timestamp": ts.min().isoformat(),
            "last_timestamp": ts.max().isoformat(),
        })
    return pd.DataFrame(rows)


def execute_data_pipeline(config: dict) -> Tuple[DatasetMap, pd.DataFrame]:
    """Run stage 1: load, validate and summarise every split.

    Args:
        config: Run configuration from ``src.config``.

    Returns:
        ``(datasets, summary)``. The summary is also written to
        ``output_files["dataset_summary"]``.
    """
    LOG = get_logger()
    if config.get("using_bundled_sample"):
        LOG.warning(
            "Using the bundled 10,000-row sample per split. Results are illustrative; "
            "pass --data-dir with the full dataset for real conclusions."
        )
    datasets = load_datasets(config)
    for obd in datasets.values():
        validate_dataset(obd)
    LOG.info("All %d splits passed validation", len(datasets))

    summary = summarise_datasets(datasets)
    summary.to_csv(config["output_files"]["dataset_summary"], index=False)
    LOG.info("Wrote dataset summary: %s", config["output_files"]["dataset_summary"])
    return datasets, summary
