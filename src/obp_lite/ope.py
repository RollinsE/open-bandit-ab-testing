# -*- coding: utf-8 -*-
"""Run several estimators at once (mirrors ``obp.ope.OffPolicyEvaluation``)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .estimators import BaseEstimator


@dataclass
class OffPolicyEvaluation:
    bandit_feedback: dict
    ope_estimators: list[BaseEstimator]

    def _kw(self, action_dist, q_hat):
        bf = self.bandit_feedback
        return dict(
            reward=bf["reward"], action=bf["action"], position=bf["position"],
            pscore=bf["pscore"], action_dist=action_dist, q_hat=q_hat,
        )

    def estimate_policy_values(self, action_dist, estimated_rewards_by_reg_model=None) -> dict[str, float]:
        kw = self._kw(action_dist, estimated_rewards_by_reg_model)
        return {e.estimator_name: e.estimate_policy_value(**kw) for e in self.ope_estimators}

    def summarize_off_policy_estimates(self, action_dist, estimated_rewards_by_reg_model=None, n_bootstrap=1000, random_state=None) -> pd.DataFrame:
        kw = self._kw(action_dist, estimated_rewards_by_reg_model)
        rows = {e.estimator_name: e.estimate_interval(n_bootstrap=n_bootstrap, random_state=random_state, **kw) for e in self.ope_estimators}
        return pd.DataFrame(rows).T

    def evaluate_performance_of_estimators(self, ground_truth_policy_value, action_dist, estimated_rewards_by_reg_model=None, metric="relative-ee") -> dict[str, float]:
        est = self.estimate_policy_values(action_dist, estimated_rewards_by_reg_model)
        if metric == "relative-ee":
            return {k: abs(v - ground_truth_policy_value) / ground_truth_policy_value for k, v in est.items()}
        if metric == "se":
            return {k: (v - ground_truth_policy_value) ** 2 for k, v in est.items()}
        raise ValueError("metric must be 'relative-ee' or 'se'")
