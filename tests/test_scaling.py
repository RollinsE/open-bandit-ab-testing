import numpy as np
from sklearn.linear_model import LogisticRegression

from src.model_training import cap_rows
from src.obp_lite import OpenBanditDataset, RegressionModel


def test_light_loading_keeps_what_the_ab_test_needs():
    full = OpenBanditDataset("bts", "women")
    light = OpenBanditDataset("bts", "women", light=True)
    np.testing.assert_array_equal(full.reward, light.reward)
    np.testing.assert_array_equal(full.position, light.position)
    np.testing.assert_array_equal(full.action, light.action)
    assert light.context.shape == (light.n_rounds, 0)
    assert list(full.data.columns) == list(light.data.columns)  # raw extras dropped after preprocessing
    assert full.context.shape[1] > 0


def test_chunked_prediction_matches_one_pass():
    obd = OpenBanditDataset("random", "men")
    args = dict(n_actions=obd.n_actions, len_list=obd.len_list, action_context=obd.action_context,
                base_model=LogisticRegression(max_iter=2000))
    big = RegressionModel(**args, chunk_rows=10**9)
    small = RegressionModel(**args, chunk_rows=777)
    models = big._fit(obd.context, obd.action, obd.reward, obd.position)
    q_big = big._predict(models, obd.context[:3000])
    q_small = small._predict(models, obd.context[:3000])
    assert q_small.dtype == np.float32
    np.testing.assert_allclose(q_small, q_big)


def test_marathon_row_cap_keeps_click_rate_and_slots():
    rng = np.random.default_rng(0)
    n = 50_000
    y = (rng.random(n) < 0.01).astype(int)
    slot = rng.integers(0, 3, n)
    X = rng.normal(size=(n, 4))
    Xc, yc, sc = cap_rows(X, y, slot, 10_000, seed=0)
    assert len(yc) == 10_000
    assert abs(yc.mean() - y.mean()) < 0.001
    assert np.allclose(np.bincount(sc) / len(sc), np.bincount(slot) / n, atol=0.01)


def test_own_slot_prediction_gives_identical_estimates():
    from src.obp_lite import BernoulliTS, DirectMethod, DoublyRobust, OffPolicyEvaluation
    obd = OpenBanditDataset("random", "men")
    bf = obd.obtain_batch_bandit_feedback()
    reg = RegressionModel(obd.n_actions, obd.len_list, obd.action_context, LogisticRegression(max_iter=2000))
    q_all = reg.fit_predict(bf["context"], bf["action"], bf["reward"], bf["position"], random_state=0, own_slot_only=False)
    q_own = reg.fit_predict(bf["context"], bf["action"], bf["reward"], bf["position"], random_state=0)
    dist = BernoulliTS(obd.n_actions, obd.len_list, is_zozotown_prior=True, campaign="men",
                       random_state=0).compute_batch_action_dist(n_rounds=bf["n_rounds"], n_sim=20000)
    ope = OffPolicyEvaluation(bf, [DirectMethod(), DoublyRobust()])
    a, b = ope.estimate_policy_values(dist, q_all), ope.estimate_policy_values(dist, q_own)
    for name in a:
        assert abs(a[name] - b[name]) < 1e-9


def test_timestamps_with_and_without_fractional_seconds(tmp_path):
    """The full dataset mixes both forms; loading must parse and order them correctly."""
    import pandas as pd
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "data" / "obd" / "random" / "men"
    data = pd.read_csv(src / "men.csv", index_col=0).head(50)
    stamps = pd.to_datetime(data["timestamp"], format="ISO8601", utc=True)
    whole = data.index[:25]
    data.loc[whole, "timestamp"] = stamps[whole].dt.floor("s").dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
    target = tmp_path / "random" / "men"
    target.mkdir(parents=True)
    data.to_csv(target / "men.csv")
    (target / "item_context.csv").write_text((src / "item_context.csv").read_text())

    obd = OpenBanditDataset("random", "men", data_path=tmp_path)
    ts = obd.data["timestamp"]
    assert str(ts.dtype).startswith("datetime64")
    assert ts.is_monotonic_increasing
    assert ts.notna().all()


def test_prediction_per_distinct_context_matches_row_by_row():
    """Predicting each distinct user once must give exactly the per-row predictions."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    obd = OpenBanditDataset("random", "men")
    reg = RegressionModel(obd.n_actions, obd.len_list, obd.action_context,
                          HistGradientBoostingClassifier(max_iter=20, random_state=0))
    models = reg._fit(obd.context, obd.action, obd.reward, obd.position)
    n = 2000
    ctx, pos = obd.context[:n], obd.position[:n]
    fast = reg._predict(models, ctx, pos)
    slow = np.zeros_like(fast)
    for i in range(n):
        X = reg._features(np.repeat(ctx[i:i + 1], obd.n_actions, axis=0), np.arange(obd.n_actions))
        slow[i, :, pos[i]] = models[pos[i]].predict_proba(X)[:, 1]
    np.testing.assert_allclose(fast, slow, rtol=1e-6)
    assert len(np.unique(ctx, axis=0)) < n  # the saving is real on this data


def test_fit_cap_keeps_click_rate_and_only_uses_training_rows():
    from src.obp_lite import RegressionModel
    rng = np.random.default_rng(0)
    reward = (rng.random(50_000) < 0.01).astype(int)
    rows = np.arange(10_000, 50_000)
    reg = RegressionModel(n_actions=2, len_list=1, action_context=np.eye(2), base_model=None, fit_max_rows=5_000)
    kept = reg._fit_rows(rows, reward, seed=1)
    assert len(kept) == 5_000 and len(np.unique(kept)) == 5_000
    assert np.isin(kept, rows).all()  # never trains on rows outside the fold
    assert abs(reward[kept].mean() - reward[rows].mean()) < 0.0005
    assert (reg._fit_rows(rows[:100], reward, seed=1) == rows[:100]).all()  # under the cap: unchanged
