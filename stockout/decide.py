"""Turn risk into a costed action list. Prices are illustrative. A person approves every action."""
import numpy as np
import pandas as pd

RISK_THRESHOLD = 0.5
COVER_MONTHS = 3
DONOR_KEEP_MONTHS = 4
MAX_KM = 150


def needs(cur):
    r = cur[cur["p"] >= RISK_THRESHOLD].copy()
    r["need"] = (COVER_MONTHS * r["amc"] - r["last_stock"].fillna(0)).clip(lower=0).round()
    return r[r["need"] > 0].sort_values("p", ascending=False)


def surpluses(cur):
    """Only fresh reports can donate. Stock beyond the donor's own cover, or expiring before it can be used."""
    c = cur[(cur["months_since_report"] <= 1) & (cur["p"] < RISK_THRESHOLD) & cur["amc"].notna()].copy()
    over = c["last_stock"] - DONOR_KEEP_MONTHS * c["amc"]
    exp = c["obs_near_expiry_last"].fillna(0) - COVER_MONTHS * c["amc"]
    c["surplus"] = np.minimum(np.maximum(over, exp).clip(lower=0), (c["last_stock"] - c["amc"]).clip(lower=0)).round()
    c["expiring_surplus"] = np.minimum(exp.clip(lower=0), c["surplus"]).round()
    return c[c["surplus"] > 0]


def plan(cur, fac, prods):
    xy = fac.set_index("facility")[["x", "y"]]
    price = prods.set_index("product")
    don = surpluses(cur)
    left = {(r.facility, r.product): r.surplus for r in don.itertuples()}
    exp_left = {(r.facility, r.product): r.expiring_surplus for r in don.itertuples()}
    acts = []
    for r in needs(cur).itertuples():
        P = price.loc[r.product]
        need = r.need
        cands = []
        for (f, p), q in left.items():
            if p == r.product and q > 0 and f != r.facility:
                km = float(np.hypot(*(xy.loc[f].values - xy.loc[r.facility].values)))
                if km <= MAX_KM:
                    cands.append((-(exp_left[f, p] > 0), km, f))
        for _, km, f in sorted(cands):
            if need <= 0:
                break
            q = min(need, left[f, r.product])
            saved = min(q, exp_left[f, r.product])
            left[f, r.product] -= q
            exp_left[f, r.product] -= saved
            need -= q
            acts.append(_act("Redistribute", r, P, q, 15 + 0.5 * km, f, km, saved))
        if need > 0:
            acts.append(_act("Emergency order", r, P, need, 40 + 1.3 * P.unit_cost * need, None, None, 0))
    a = pd.DataFrame(acts)
    return a.sort_values("value_per_usd", ascending=False).reset_index(drop=True) if len(a) else a


def _act(kind, r, P, q, cost, donor, km, saved):
    days = min(30 * q / max(r.amc, 1), 90)
    value = r.p * days * P.priority
    return dict(action=kind, facility=r.facility, district=r.district, tier=r.tier, product=r.product, risk=float(r.p),
                qty=int(q), cost=float(round(cost, 2)), donor=donor, km=None if km is None else float(round(km)),
                saved_from_expiry=int(saved), days_covered=float(round(days, 1)), value=float(value),
                value_per_usd=float(value / cost))


def select(actions, budget):
    """Greedy best value per dollar under a hard ceiling."""
    chosen, spent = [], 0.0
    for a in actions.itertuples():
        if spent + a.cost <= budget:
            chosen.append(a.Index)
            spent += a.cost
    return chosen, spent
