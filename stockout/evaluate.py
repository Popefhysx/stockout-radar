"""Rolling-origin backtest. Train only on rows whose label window closed by the origin."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .features import FEATURES

CAL_WINDOW = 6


def train_mask(d, origin, horizon=3):
    return d["eligible"] & d["y_obs"].notna() & (d["t"] <= origin - horizon)


class Calibrated:
    """Gradient boosting with Platt scaling fitted on the latest resolved months."""
    def fit(self, tr):
        cut = tr["t"].max() - CAL_WINDOW
        fit, cal = tr[tr["t"] <= cut], tr[tr["t"] > cut]
        self.m = HGB(max_iter=200, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=40,
                     l2_regularization=1.0, random_state=0).fit(fit[FEATURES], fit["y_obs"])
        self.c = LogisticRegression().fit(self._logit(cal[FEATURES]), cal["y_obs"])
        return self

    def _logit(self, X):
        p = np.clip(self.m.predict_proba(X)[:, 1], 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def predict(self, X):
        return self.c.predict_proba(self._logit(X))[:, 1]


def rule_score(d):
    """The standard practice: flag lowest months of stock. Unknown stock ranks last."""
    return (-d["mos"]).fillna(-1e6).values


def backtest(d, origins):
    out = []
    for o in origins:
        tr = d[train_mask(d, o)]
        te = d[(d["t"] == o) & d["eligible"] & d["y_true"].notna()].copy()
        te["p_gbm"] = Calibrated().fit(tr).predict(te[FEATURES])
        lin = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=500))
        te["p_lr"] = lin.fit(tr[FEATURES], tr["y_obs"]).predict_proba(te[FEATURES])[:, 1]
        te["s_rule"] = rule_score(te)
        out.append(te)
    return pd.concat(out)


def flag_top(df, col, share=0.10):
    """Flag the top share at each origin: a fixed visit capacity shared by every method."""
    r = df.groupby("t")[col].rank(ascending=False, method="first")
    return r <= np.ceil(df.groupby("t")[col].transform("size") * share)


def ece(y, p, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((b == i).mean() * abs(y[b == i].mean() - p[b == i].mean()) for i in range(bins) if (b == i).any()))


def metrics(df, y="y_true"):
    res = {"rows": int(len(df)), "stockout_rate": float(df[y].mean())}
    for name, col in [("rule", "s_rule"), ("logistic", "p_lr"), ("gbm", "p_gbm")]:
        f = flag_top(df, col)
        res[name] = dict(ap=float(average_precision_score(df[y], df[col])), auc=float(roc_auc_score(df[y], df[col])),
                         precision=float(df.loc[f, y].mean()), recall=float(df.loc[f, y].sum() / df[y].sum()))
    res["gbm"]["brier"] = float(brier_score_loss(df[y], df["p_gbm"]))
    res["gbm"]["ece"] = ece(df[y].values, df["p_gbm"].values)
    res["brier_base"] = float(brier_score_loss(df[y], np.full(len(df), df[y].mean())))
    return res


def delta_ci(df, y="y_true", n=1000, seed=0):
    """Recall gain over the rule, CI from resampling whole facilities."""
    fm, fr = flag_top(df, "p_gbm"), flag_top(df, "s_rule")
    a = pd.DataFrame({"f": df["facility"], "m": fm * df[y], "r": fr * df[y], "y": df[y]}).groupby("f").sum().values
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n):
        s = a[rng.integers(0, len(a), len(a))]
        v.append((s[:, 0].sum() - s[:, 1].sum()) / s[:, 2].sum())
    return [float(x) for x in np.percentile(v, [2.5, 97.5])]


def subgroups(df, y="y_true"):
    fm, fr = flag_top(df, "p_gbm"), flag_top(df, "s_rule")
    groups = {
        "Hospital": df["tier"] == "hospital", "Health centre": df["tier"] == "health_centre",
        "Dispensary": df["tier"] == "dispensary", "Less remote": df["remote"] < 0.4, "More remote": df["remote"] >= 0.4,
        "Reported this month": df["months_since_report"] == 0, "No report for 2+ months": df["months_since_report"] >= 2,
    }
    rows = []
    for name, m in groups.items():
        pos = df[m & (df[y] == 1)]
        rows.append(dict(group=name, rows=int(m.sum()), stockout_rate=float(df.loc[m, y].mean()),
                         recall_gbm=float(fm[pos.index].mean()), recall_rule=float(fr[pos.index].mean()),
                         flagged_gbm=float(fm[m].mean()), observed_vs_predicted=float(df.loc[m, y].mean() - df.loc[m, "p_gbm"].mean())))
    return rows


def calibration_bins(df, y="y_true"):
    q = pd.qcut(df["p_gbm"], 10, duplicates="drop")
    return [dict(predicted=float(g["p_gbm"].mean()), actual=float(g[y].mean()), n=int(len(g))) for _, g in df.groupby(q, observed=True)]
