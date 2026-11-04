"""Features read only obs_* columns at or before month t. Labels look at t+1..t+h."""
import numpy as np
import pandas as pd

FEATURES = ["last_stock", "amc", "amc_long", "mos", "est_mos", "months_since_report", "report_rate_6m",
            "received_3m", "ordered_3m", "unfilled_6m", "stockout_rate_6m", "near_expiry_share",
            "district_fill_3m", "national_fill_3m", "month_sin", "month_cos", "product_code", "tier_code", "remote"]
GROUPS = {
    "Stock on hand": ["last_stock", "mos", "est_mos", "near_expiry_share"],
    "Consumption": ["amc", "amc_long"],
    "Late or unfilled deliveries": ["received_3m", "ordered_3m", "unfilled_6m"],
    "Past stock-outs": ["stockout_rate_6m"],
    "Reporting gaps": ["months_since_report", "report_rate_6m"],
    "Wider supply shortage": ["district_fill_3m", "national_fill_3m"],
    "Season": ["month_sin", "month_cos"],
    "Product and facility type": ["product_code", "tier_code", "remote"],
}


def _ratio(a, b):
    return (a / b.where(b > 0)).astype(float)


def build(panel, horizon=3):
    d = panel.sort_values(["facility", "product", "t"]).copy()
    g = d.groupby(["facility", "product"], sort=False)
    roll = lambda c, w, f: g[c].transform(lambda s: getattr(s.rolling(w, min_periods=1), f)())
    d["last_stock"] = g["obs_stock"].ffill()
    d["amc"] = roll("obs_dispensed", 3, "mean")
    d["amc_long"] = roll("obs_dispensed", 12, "mean")
    d["amc"] = d["amc"].fillna(d["amc_long"])
    d["mos"] = _ratio(d["last_stock"], d["amc"])
    rep_t = d["t"].where(d["reported"])
    d["months_since_report"] = d["t"] - rep_t.groupby([d["facility"], d["product"]]).ffill()
    d["est_mos"] = _ratio((d["last_stock"] - d["amc"] * d["months_since_report"]).clip(lower=0), d["amc"])
    d["report_rate_6m"] = g["reported"].transform(lambda s: s.astype(float).rolling(6, min_periods=1).mean())
    d["received_3m"] = _ratio(roll("obs_received", 3, "sum"), d["amc"])
    d["ordered_3m"] = _ratio(roll("obs_ordered", 3, "sum"), d["amc"])
    d["unfilled_6m"] = _ratio(roll("obs_ordered", 6, "sum") - roll("obs_received", 6, "sum"), d["amc"])
    d["stockout_rate_6m"] = g["obs_stockout_days"].transform(
        lambda s: (s > 0).astype(float).where(s.notna()).rolling(6, min_periods=1).mean())
    d["near_expiry_share"] = _ratio(g["obs_near_expiry"].ffill(), d["last_stock"])

    def fill(keys):
        a = d.groupby(keys + ["t"], sort=False)[["obs_received", "obs_ordered"]].sum().reset_index()
        a = a.sort_values(keys + ["t"])
        roll_sum = a.groupby(keys)[["obs_received", "obs_ordered"]].transform(lambda s: s.rolling(3, min_periods=1).sum())
        a["v"] = _ratio(roll_sum["obs_received"], roll_sum["obs_ordered"]).clip(upper=1.5)
        return d[keys + ["t"]].merge(a[keys + ["t", "v"]], on=keys + ["t"], how="left")["v"].values
    d["district_fill_3m"] = fill(["district", "product"])
    d["national_fill_3m"] = fill(["product"])
    cal = d["t"] % 12 + 1
    d["month_sin"], d["month_cos"] = np.sin(2 * np.pi * cal / 12), np.cos(2 * np.pi * cal / 12)
    d["product_code"] = d["product"].astype("category").cat.codes
    d["tier_code"] = d["tier"].map({"hospital": 2, "health_centre": 1, "dispensary": 0})

    def future(col, fn):
        s = pd.concat([col.groupby([d["facility"], d["product"]]).shift(-k) for k in range(1, horizon + 1)], axis=1)
        return fn(s)
    d["y_true"] = future(d["true_stockout_days"], lambda s: (s > 0).any(axis=1).astype(float).where(s.notna().all(axis=1)))
    obs_out = (d["obs_stockout_days"] > 0).astype(float).where(d["obs_stockout_days"].notna())
    d["y_obs"] = future(obs_out, lambda s: (s.max(axis=1)))  # NaN when nothing was reported in the window
    d["eligible"] = d["last_stock"] > 0  # known to have had stock when last seen; never reads truth
    return d
