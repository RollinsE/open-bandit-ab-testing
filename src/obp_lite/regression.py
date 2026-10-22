# -*- coding: utf-8 -*-
"""Reward regression q̂(x, a, k) for model-based OPE (mirrors ``obp.ope.RegressionModel``).

One classifier per slot, features = [user context, item context], with K-fold
cross-fitting so every row's q̂ comes from a model that never saw it.

Predictions are made in chunks of rows and stored as float32. q̂ has one value
per row, item and slot, so for the full "all" campaign (1.4M rows, 80 items,
3 slots) building it in one go would need tens of gigabytes.

When ``position`` is passed to ``fit_predict``, each row is only predicted for
its own slot: every estimator reads q̂ at the row's own slot and nowhere else,
so this gives identical estimates for a third of the prediction cost (which
dominates the run time with tree models). The other slots are left at zero.

Predictions are made once per distinct user context, not once per row. In the
Open Bandit Dataset the context is four one-hot user features, so there are at
most 960 distinct contexts against up to 1.4M rows; predicting each distinct
context for every item and copying the result to its rows gives exactly the
same q̂ at a tiny fraction of the cost. With data whose contexts are mostly
unique this simply does the same work as before.

``fit_max_rows`` caps how many rows each fold's models are trained on (a
random sample of that fold's training rows, stratified by click so the click
rate is kept). Fitting is the expensive part with slow models such as random
forests or ensembles, and it is repeated for every fold of every bootstrap
run; the cap bounds it without changing which rows are predicted, so
cross-fitting is unaffected: no row is ever scored by a model that saw it.
"""
from __future__ import annotations

from dataclasses import dataclass

import warnings

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold


@dataclass
class RegressionModel:
    n_actions: int
    len_list: int
    action_context: np.ndarray
    base_model: BaseEstimator
    chunk_rows: int = 20_000  # rows predicted at a time (each expands to chunk_rows x n_actions)
    fit_max_rows: int | None = None  # cap on training rows per fold (stratified by click); None = all

    def _fit_rows(self, rows: np.ndarray, reward: np.ndarray, seed: int | None) -> np.ndarray:
        """``rows``, or a click-stratified random sample of ``fit_max_rows`` of them."""
        cap = self.fit_max_rows
        if not cap or len(rows) <= cap:
            return rows
        rng = np.random.default_rng(seed)
        clicked = rows[reward[rows] > 0]
        unclicked = rows[reward[rows] <= 0]
        n_clicked = int(round(cap * len(clicked) / len(rows)))
        keep = np.r_[rng.choice(clicked, n_clicked, replace=False),
                     rng.choice(unclicked, cap - n_clicked, replace=False)]
        return np.sort(keep)

    def _features(self, context: np.ndarray, action: np.ndarray) -> np.ndarray:
        return np.c_[context, self.action_context[action]]

    def _fit(self, context, action, reward, position):
        models = []
        for k in range(self.len_list):
            m = position == k
            model = clone(self.base_model)
            y = reward[m].astype(int)  # some boosting libraries insist on integer class labels
            if y.min() == y.max():  # a fold/slot with no clicks: fall back to a constant
                model = _Constant(float(y.mean()))
            else:
                model.fit(self._features(context[m], action[m]), y)
            models.append(model)
        return models

    def _predict(self, models, context, position: np.ndarray | None = None) -> np.ndarray:
        """q̂ for every item; for every slot, or only each row's own slot when ``position`` is given."""
        n = context.shape[0]
        out = np.zeros((n, self.n_actions, self.len_list), dtype=np.float32)
        for k, model in enumerate(models):
            rows = np.arange(n) if position is None else np.flatnonzero(position == k)
            if len(rows) == 0:
                continue
            # Predict each distinct context once, then copy to the rows that share it.
            unique_ctx, inverse = np.unique(context[rows], axis=0, return_inverse=True)
            inverse = inverse.ravel()
            q_unique = np.empty((len(unique_ctx), self.n_actions), dtype=np.float32)
            for start in range(0, len(unique_ctx), self.chunk_rows):
                block = unique_ctx[start:start + self.chunk_rows]
                m = len(block)
                X = self._features(np.repeat(block, self.n_actions, axis=0), np.tile(np.arange(self.n_actions), m))
                with warnings.catch_warnings():
                    # LightGBM with recent scikit-learn warns about feature names even
                    # though it is fitted and used on plain arrays. Harmless; silenced.
                    warnings.filterwarnings("ignore", message="X does not have valid feature names")
                    q_unique[start:start + m] = model.predict_proba(X)[:, 1].reshape(m, self.n_actions)
            out[rows, :, k] = q_unique[inverse]
        return out

    def fit_predict(self, context, action, reward, position, n_folds: int = 3, random_state: int | None = None,
                    own_slot_only: bool = True, **_) -> np.ndarray:
        """Cross-fitted q̂. ``own_slot_only`` predicts each row at its own slot only (see module notes)."""
        if n_folds == 1:
            tr = self._fit_rows(np.arange(context.shape[0]), reward, random_state)
            models = self._fit(context[tr], action[tr], reward[tr], position[tr])
            return self._predict(models, context, position if own_slot_only else None)
        out = np.zeros((context.shape[0], self.n_actions, self.len_list), dtype=np.float32)
        splits = KFold(n_folds, shuffle=True, random_state=random_state).split(context)
        for fold, (tr, te) in enumerate(splits):
            tr = self._fit_rows(tr, reward, None if random_state is None else random_state + fold)
            models = self._fit(context[tr], action[tr], reward[tr], position[tr])
            out[te] = self._predict(models, context[te], position[te] if own_slot_only else None)
        return out


class _Constant:
    def __init__(self, p: float):
        self.p = p

    def predict_proba(self, X):
        return np.c_[np.full(len(X), 1 - self.p), np.full(len(X), self.p)]
