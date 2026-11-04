"""End-to-end run: simulate, backtest, final fit, plan. Writes output/results.json."""
import json
import numpy as np
import pandas as pd
from .sim import simulate
from .features import build, FEATURES, GROUPS
from . import evaluate as ev, decide

PRE = list(range(15, 21))    # label windows close before the drawdown
POST = list(range(24, 33))   # all origins after it
SEEDS = [42, 7, 2026]


def seed_metrics(seed):
    fac, prods, panel = simulate(seed=seed)
    d = build(panel)
    bt = ev.backtest(d, PRE + POST)
    return fac, prods, d, bt, {"pre": ev.metrics(bt[bt.t.isin(PRE)]), "post": ev.metrics(bt[bt.t.isin(POST)])}


def attributions(model, X, top=3):
    base = model.predict(X)
    med = X.median()
    out = np.zeros((len(X), len(GROUPS)))
    names = list(GROUPS)
    for j, g in enumerate(names):
        Z = X.copy()
        Z[GROUPS[g]] = med[GROUPS[g]].values
        out[:, j] = base - model.predict(Z)
    res = []
    for row in out:
        idx = [i for i in np.argsort(-row)[:top] if row[i] > 0.005]
        res.append([dict(signal=names[i], points=round(float(row[i] * 100), 1)) for i in idx])
    return res


def importance(model, df, y="y_true", seed=0):
    rng = np.random.default_rng(seed)
    from sklearn.metrics import average_precision_score as ap
    base = ap(df[y], model.predict(df[FEATURES]))
    rows = []
    for g, cols in GROUPS.items():
        X = df[FEATURES].copy()
        perm = rng.permutation(len(X))
        X[cols] = X[cols].values[perm]
        rows.append(dict(group=g, ap_drop=float(base - ap(df[y], model.predict(X)))))
    return sorted(rows, key=lambda r: -r["ap_drop"])


def run():
    all_seeds = {}
    keep = None
    for s in SEEDS:
        fac, prods, d, bt, m = seed_metrics(s)
        all_seeds[s] = m
        if s == SEEDS[0]:
            keep = fac, prods, d, bt
    fac, prods, d, bt = keep
    post = bt[bt.t.isin(POST)]
    T = int(d.t.max())
    model = ev.Calibrated().fit(d[ev.train_mask(d, T)])
    d["obs_near_expiry_last"] = d.groupby(["facility", "product"])["obs_near_expiry"].ffill()
    cur = d[(d.t == T) & d.eligible].copy()
    cur["p"] = model.predict(cur[FEATURES])
    actions = decide.plan(cur, fac, prods)
    attr = dict(zip(cur.index, attributions(model, cur[FEATURES])))
    risk = cur.sort_values("p", ascending=False).head(150)
    rows = [dict(facility=r.facility, district=r.district, tier=r.tier, product=r.product, risk=round(float(r.p), 3),
                 mos=None if pd.isna(r.mos) else round(float(r.mos), 1), months_since_report=None if pd.isna(r.months_since_report) else int(r.months_since_report),
                 signals=attr[r.Index]) for r in risk.itertuples()]
    res = dict(
        data_note="Synthetic data. No real facility, patient or country.",
        seeds={str(k): v for k, v in all_seeds.items()},
        mean_recall_gain_post=float(np.mean([v["post"]["gbm"]["recall"] - v["post"]["rule"]["recall"] for v in all_seeds.values()])),
        post=ev.metrics(post), pre=ev.metrics(bt[bt.t.isin(PRE)]),
        recall_gain_ci=ev.delta_ci(post), subgroups=ev.subgroups(post), calibration=ev.calibration_bins(post),
        importance=importance(model, post),
        at_risk_rows=int((cur.p >= decide.RISK_THRESHOLD).sum()), scored_rows=int(len(cur)),
        unscored_rows=int(((d.t == T) & ~d.eligible).sum()), risk=rows,
        actions=json.loads(actions.to_json(orient="records")), districts=sorted(fac.district.unique().tolist()),
    )
    return res


if __name__ == "__main__":
    r = run()
    json.dump(r, open("output/results.json", "w"), indent=1)
    print(json.dumps({k: r[k] for k in ["pre", "post", "recall_gain_ci", "mean_recall_gain_post", "subgroups", "importance", "at_risk_rows", "scored_rows", "unscored_rows"]}, indent=1))
    print({k: {"post_recall_gbm": round(v["post"]["gbm"]["recall"], 3), "post_recall_rule": round(v["post"]["rule"]["recall"], 3)} for k, v in r["seeds"].items()})
    print("actions", len(r["actions"]), "total cost", sum(a["cost"] for a in r["actions"]), "transfers", sum(a["action"] == "Redistribute" for a in r["actions"]))
