# -*- coding: utf-8 -*-
"""Context-free bandit policies used in ZOZOTOWN (mirrors ``obp.policy``)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PRIOR_FILE = Path(__file__).resolve().parents[2] / "data" / "obd" / "prior_bts.yaml"


def load_zozotown_prior(campaign: str, path: Path = PRIOR_FILE) -> dict[str, np.ndarray]:
    """Parse the Beta prior ZOZOTOWN's production BTS started from.

    Tiny parser for the fixed ``campaign: {alpha: [...], beta: [...]}`` layout so
    PyYAML isn't required.
    """
    prior: dict[str, dict[str, list[float]]] = {}
    camp = key = None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        token = line.strip()
        if indent == 0 and token.endswith(":"):
            camp = token[:-1]
            prior[camp] = {}
        elif token.endswith(":"):
            key = token[:-1]
            prior[camp][key] = []
        elif token.startswith("-"):
            prior[camp][key].append(float(token[1:]))
    return {k: np.asarray(v) for k, v in prior[campaign].items()}


@dataclass
class Random:
    """Uniform random ranking policy (the 'random' behavior policy in OBD)."""

    n_actions: int
    len_list: int = 1
    random_state: int | None = None
    policy_name: str = "random"

    def compute_batch_action_dist(self, n_rounds: int = 1, n_sim: int | None = None) -> np.ndarray:
        return np.full((n_rounds, self.n_actions, self.len_list), 1.0 / self.n_actions)


@dataclass
class EpsilonGreedy:
    """Context-free epsilon-greedy over fixed per-item CTR estimates."""

    n_actions: int
    len_list: int = 1
    epsilon: float = 0.1
    alpha: np.ndarray | None = None
    beta: np.ndarray | None = None
    random_state: int | None = None
    policy_name: str = field(default="egreedy", init=False)

    def compute_batch_action_dist(self, n_rounds: int = 1, n_sim: int = 100_000) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        a = np.ones(self.n_actions) if self.alpha is None else np.asarray(self.alpha)
        b = np.ones(self.n_actions) if self.beta is None else np.asarray(self.beta)
        greedy = np.argsort(-(a / (a + b)))[: self.len_list]
        counts = np.zeros((self.n_actions, self.len_list))
        for _ in range(n_sim):
            if rng.random() < self.epsilon:
                ranking = rng.choice(self.n_actions, self.len_list, replace=False)
            else:
                ranking = greedy
            counts[ranking, np.arange(self.len_list)] += 1
        return np.tile(counts / n_sim, (n_rounds, 1, 1))


@dataclass
class BernoulliTS:
    """Bernoulli Thompson Sampling ranking policy.

    With ``is_zozotown_prior=True`` it uses the Beta prior that ZOZOTOWN's
    production BTS ran with, so its action distribution reproduces the 'bts'
    behavior policy that generated the OBD logs.
    """

    n_actions: int
    len_list: int = 1
    alpha: np.ndarray | None = None
    beta: np.ndarray | None = None
    is_zozotown_prior: bool = False
    campaign: str | None = None
    random_state: int | None = None
    policy_name: str = "bts"

    def __post_init__(self) -> None:
        if self.is_zozotown_prior:
            if self.campaign is None:
                raise ValueError("campaign is required when is_zozotown_prior=True")
            prior = load_zozotown_prior(self.campaign)
            self.alpha, self.beta = prior["alpha"], prior["beta"]
        else:
            self.alpha = np.ones(self.n_actions) if self.alpha is None else np.asarray(self.alpha)
            self.beta = np.ones(self.n_actions) if self.beta is None else np.asarray(self.beta)
        self.random_ = np.random.default_rng(self.random_state)

    def compute_batch_action_dist(self, n_rounds: int = 1, n_sim: int = 100_000, chunk: int = 50_000) -> np.ndarray:
        """Monte Carlo estimate of P(item a is shown at slot k)."""
        counts = np.zeros((self.n_actions, self.len_list))
        done = 0
        while done < n_sim:
            m = min(chunk, n_sim - done)
            draws = self.random_.beta(self.alpha, self.beta, size=(m, self.n_actions))
            top = np.argsort(-draws, axis=1)[:, : self.len_list]
            for k in range(self.len_list):
                counts[:, k] += np.bincount(top[:, k], minlength=self.n_actions)
            done += m
        return np.tile(counts / n_sim, (n_rounds, 1, 1))
