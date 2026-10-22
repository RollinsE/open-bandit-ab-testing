# -*- coding: utf-8 -*-
"""obp_lite — a dependency-light re-implementation of the core of the Open Bandit Pipeline.

Same API shape as ``obp`` (https://github.com/st-tech/zr-obp, Apache-2.0) for the
OBD off-policy-evaluation workflow, but runs on current numpy / pandas /
scikit-learn without PyTorch.
"""
from .dataset import OpenBanditDataset
from .estimators import (
    DirectMethod,
    DoublyRobust,
    DoublyRobustWithShrinkage,
    DoublyRobustWithShrinkageTuning,
    InverseProbabilityWeighting,
    SelfNormalizedDoublyRobust,
    SelfNormalizedInverseProbabilityWeighting,
    SwitchDoublyRobust,
    SwitchDoublyRobustTuning,
)
from .ope import OffPolicyEvaluation
from .policy import BernoulliTS, EpsilonGreedy, Random
from .regression import RegressionModel

__all__ = [
    "BernoulliTS",
    "DirectMethod",
    "DoublyRobust",
    "DoublyRobustWithShrinkage",
    "DoublyRobustWithShrinkageTuning",
    "EpsilonGreedy",
    "InverseProbabilityWeighting",
    "OffPolicyEvaluation",
    "OpenBanditDataset",
    "Random",
    "RegressionModel",
    "SelfNormalizedDoublyRobust",
    "SelfNormalizedInverseProbabilityWeighting",
    "SwitchDoublyRobust",
    "SwitchDoublyRobustTuning",
]

__version__ = "0.1.0"
