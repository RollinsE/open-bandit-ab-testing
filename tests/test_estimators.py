import numpy as np
import pytest

from src.obp_lite import (
    BernoulliTS,
    DirectMethod,
    InverseProbabilityWeighting,
    OpenBanditDataset,
    Random,
    SelfNormalizedInverseProbabilityWeighting,
)


@pytest.fixture(scope="module")
def obd():
    return OpenBanditDataset(behavior_policy="random", campaign="all")


def _kw(obd, action_dist, q_hat=None):
    return dict(reward=obd.reward, action=obd.action, position=obd.position,
                pscore=obd.pscore, action_dist=action_dist, q_hat=q_hat)


def test_ipw_recovers_on_policy_value(obd):
    dist = Random(obd.n_actions, obd.len_list).compute_batch_action_dist(obd.n_rounds)
    assert InverseProbabilityWeighting().estimate_policy_value(**_kw(obd, dist)) == pytest.approx(obd.reward.mean())
    assert SelfNormalizedInverseProbabilityWeighting().estimate_policy_value(**_kw(obd, dist)) == pytest.approx(obd.reward.mean())


def test_dm_with_constant_model_returns_that_constant(obd):
    dist = Random(obd.n_actions, obd.len_list).compute_batch_action_dist(obd.n_rounds)
    q = np.full_like(dist, 0.01)
    assert DirectMethod().estimate_policy_value(**_kw(obd, dist, q)) == pytest.approx(0.01)


def test_bts_action_distribution_is_valid(obd):
    dist = BernoulliTS(obd.n_actions, obd.len_list, is_zozotown_prior=True, campaign="all",
                       random_state=0).compute_batch_action_dist(n_sim=20000)[0]
    assert dist.shape == (obd.n_actions, obd.len_list)
    np.testing.assert_allclose(dist.sum(axis=0), 1.0)
    assert (dist.sum(axis=1) <= 1.0 + 1e-9).all()  # an item appears at most once per list
