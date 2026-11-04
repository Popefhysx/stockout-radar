import numpy as np
import pandas as pd
import pytest
from stockout import decide, evaluate as ev
from stockout.features import FEATURES, build
from stockout.sim import simulate


@pytest.fixture(scope="module")
def data():
    fac, prods, panel = simulate(seed=1, n_fac=20, months=30)
    return fac, prods, panel, build(panel)


def test_simulation_is_reproducible():
    a = simulate(seed=3, n_fac=5, months=6)[2]
    b = simulate(seed=3, n_fac=5, months=6)[2]
    pd.testing.assert_frame_equal(a, b)


def test_unreported_months_expose_nothing(data):
    p = data[2]
    obs = [c for c in p if c.startswith("obs_")]
    assert p.loc[~p.reported, obs].isna().all().all()


def test_features_never_read_future_or_truth(data):
    """Corrupting the future, or any true_* column, must leave features at t unchanged."""
    _, _, panel, base = data
    cut = 20
    bad = panel.copy()
    fut = bad.t > cut
    for c in ["obs_stock", "obs_dispensed", "obs_received", "obs_ordered", "obs_stockout_days", "obs_near_expiry"]:
        bad.loc[fut, c] = 999999.0
    bad.loc[fut, "reported"] = True
    for c in ["true_demand", "true_stock", "true_stockout_days", "true_expired"]:
        bad[c] = -1
    new = build(bad)
    keys = ["facility", "product", "t"]
    x = base[base.t <= cut].sort_values(keys)[FEATURES].reset_index(drop=True)
    y = new[new.t <= cut].sort_values(keys)[FEATURES].reset_index(drop=True)
    pd.testing.assert_frame_equal(x, y)


def test_training_only_uses_closed_label_windows(data):
    d = data[3]
    assert d[ev.train_mask(d, 25)].t.max() <= 22


def test_last_months_have_no_true_label(data):
    d = data[3]
    assert d[d.t > d.t.max() - 3].y_true.isna().all()


def _cur():
    return pd.DataFrame([
        dict(facility="A", district="D", tier="dispensary", product="X", p=0.9, amc=100.0, last_stock=20.0, months_since_report=0, obs_near_expiry_last=0.0),
        dict(facility="B", district="D", tier="hospital", product="X", p=0.1, amc=50.0, last_stock=500.0, months_since_report=0, obs_near_expiry_last=250.0),
        dict(facility="C", district="D", tier="hospital", product="X", p=0.1, amc=50.0, last_stock=500.0, months_since_report=5, obs_near_expiry_last=0.0),
    ])


FAC = pd.DataFrame(dict(facility=["A", "B", "C"], x=[0, 10, 1], y=[0, 0, 0]))
PR = pd.DataFrame(dict(product=["X"], base=[1], unit_cost=[1.0], shelf=[12], season=[0.0], priority=[1.0]))


def test_stale_reports_cannot_donate():
    assert "C" not in set(decide.surpluses(_cur()).facility)


def test_donor_keeps_own_cover_and_need_is_met():
    a = decide.plan(_cur(), FAC, PR)
    t = a[a.action == "Redistribute"]
    assert set(t.donor) == {"B"} and t.qty.sum() <= 500 - 200
    assert a.qty.sum() == 280


def test_budget_ceiling_is_never_exceeded():
    a = decide.plan(_cur(), FAC, PR)
    for b in [0, 1, 25, 100, 10000]:
        _, spent = decide.select(a, b)
        assert spent <= b


def test_ece_is_zero_when_perfect():
    p = np.array([0.0] * 50 + [0.99] * 50)
    y = np.array([0] * 50 + [1] * 50)
    assert ev.ece(y, p) < 0.02
