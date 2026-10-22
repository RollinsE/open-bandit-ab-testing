# -*- coding: utf-8 -*-
"""Off-policy estimators for ranking bandit feedback (mirrors ``obp.ope``).

Shapes, as in obp:
    action, position, reward, pscore : (n,)
    action_dist, q_hat               : (n, n_actions, len_list)
Every estimator returns per-round contributions; the policy value is their mean.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


def _iw(action, position, pscore, action_dist):
    n = action.shape[0]
    return action_dist[np.arange(n), action, position] / pscore


def _dm_term(position, action_dist, q_hat, chunk: int = 100_000):
    """Sum over items of pi(a | slot) * q(x, a, slot), per row.

    Worked out in chunks of rows: the intermediate arrays are rows x items, which
    for the full "all" campaign would otherwise be about 1 GB each.
    """
    n = position.shape[0]
    out = np.empty(n)
    for start in range(0, n, chunk):
        r = np.arange(start, min(n, start + chunk))
        pi = action_dist[r, :, position[r]]
        q = q_hat[r, :, position[r]]
        out[r] = (pi * q).sum(axis=1)
    return out


def _q_factual(action, position, q_hat):
    return q_hat[np.arange(action.shape[0]), action, position]


@dataclass
class BaseEstimator:
    estimator_name: str = field(default="base", init=False)
    needs_model: bool = field(default=False, init=False)

    def round_rewards(self, reward, action, position, pscore, action_dist, q_hat=None) -> np.ndarray:
        raise NotImplementedError

    def estimate_policy_value(self, **kw) -> float:
        return float(self.round_rewards(**kw).mean())

    def estimate_interval(self, n_bootstrap: int = 1000, alpha: float = 0.05, random_state: int | None = None, **kw) -> dict:
        r = self.round_rewards(**kw)
        rng = np.random.default_rng(random_state)
        boots = np.array([r[rng.integers(0, len(r), len(r))].mean() for _ in range(n_bootstrap)])
        lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
        return {"mean": float(r.mean()), "lower": float(lo), "upper": float(hi)}


@dataclass
class DirectMethod(BaseEstimator):
    estimator_name: str = field(default="dm", init=False)
    needs_model: bool = field(default=True, init=False)

    def round_rewards(self, reward, action, position, pscore, action_dist, q_hat=None):
        return _dm_term(position, action_dist, q_hat)


@dataclass
class InverseProbabilityWeighting(BaseEstimator):
    lambda_: float = np.inf  # clipping threshold on the importance weight
    estimator_name: str = field(default="ipw", init=False)

    def round_rewards(self, reward, action, position, pscore, action_dist, q_hat=None):
        iw = _iw(action, position, pscore, action_dist)
        iw = np.where(iw <= self.lambda_, iw, 0.0)
        return iw * reward


@dataclass
class SelfNormalizedInverseProbabilityWeighting(BaseEstimator):
    estimator_name: str = field(default="snipw", init=False)

    def round_rewards(self, reward, action, position, pscore, action_dist, q_hat=None):
        iw = _iw(action, position, pscore, action_dist)
        return iw * reward / iw.mean()


@dataclass
class DoublyRobust(BaseEstimator):
    lambda_: float = np.inf
    estimator_name: str = field(default="dr", init=False)
    needs_model: bool = field(default=True, init=False)

    def _weight(self, iw):
        return np.where(iw <= self.lambda_, iw, 0.0)

    def round_rewards(self, reward, action, position, pscore, action_dist, q_hat=None):
        iw = _iw(action, position, pscore, action_dist)
        w = self._weight(iw)
        return _dm_term(position, action_dist, q_hat) + w * (reward - _q_factual(action, position, q_hat))


@dataclass
class SelfNormalizedDoublyRobust(DoublyRobust):
    estimator_name: str = field(default="sndr", init=False)

    def round_rewards(self, reward, action, position, pscore, action_dist, q_hat=None):
        iw = _iw(action, position, pscore, action_dist)
        return _dm_term(position, action_dist, q_hat) + iw / iw.mean() * (reward - _q_factual(action, position, q_hat))


@dataclass
class SwitchDoublyRobust(DoublyRobust):
    """DR where rounds with weight > lambda_ fall back to the model (Wang et al., 2017)."""

    estimator_name: str = field(default="switch-dr", init=False)


@dataclass
class DoublyRobustWithShrinkage(DoublyRobust):
    """DR with weights shrunk to λw/(w²+λ) (Su et al., 2020)."""

    estimator_name: str = field(default="dr-os", init=False)

    def _weight(self, iw):
        if np.isinf(self.lambda_):
            return iw
        if self.lambda_ == 0:
            return np.zeros_like(iw)
        return self.lambda_ * iw / (iw**2 + self.lambda_)


@dataclass
class _Tuned(BaseEstimator):
    """Pick lambda by SLOPE++ (Tucker & Lee, 2021), obp's default tuning method."""

    lambdas: list = field(default_factory=lambda: [10, 50, 100, 500, 1000, 5000, 10000, np.inf])
    delta: float = 0.1
    base: type = field(default=DoublyRobust, init=False)
    needs_model: bool = field(default=True, init=False)

    def _select(self, **kw) -> float:
        C = np.sqrt(6) - 1
        thetas, cnfs = [], []
        for lam in self.lambdas:
            r = self.base(lambda_=lam).round_rewards(**kw)
            n = len(r)
            thetas.append(r.mean())
            cnfs.append(stats.t(n - 1).ppf(1 - self.delta) * r.std(ddof=1) / np.sqrt(n))
        order = np.argsort(cnfs)[::-1]  # widest interval (least biased) first
        kept_t, kept_c = [], []
        for i, idx in enumerate(order):
            if kept_t and not (np.abs(np.array(kept_t) - thetas[idx]) <= cnfs[idx] + C * np.array(kept_c)).all():
                return self.lambdas[order[i - 1]]
            kept_t.append(thetas[idx]), kept_c.append(cnfs[idx])
        return self.lambdas[order[-1]]

    def round_rewards(self, **kw):
        self.best_lambda_ = self._select(**kw)
        return self.base(lambda_=self.best_lambda_).round_rewards(**kw)


@dataclass
class SwitchDoublyRobustTuning(_Tuned):
    estimator_name: str = field(default="switch-dr", init=False)
    base: type = field(default=SwitchDoublyRobust, init=False)


@dataclass
class DoublyRobustWithShrinkageTuning(_Tuned):
    estimator_name: str = field(default="dr-os", init=False)
    base: type = field(default=DoublyRobustWithShrinkage, init=False)
