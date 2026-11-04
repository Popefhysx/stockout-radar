"""Synthetic facility-level logistics data. NOT real data; no real facility or patient.

The simulator separates what is true (demand, stock) from what a logistics system
sees (late, missing and noisy reports). Models may only read the obs_* columns.
A donor drawdown at month `shift` makes resupply and reporting worse, so a model
trained before it faces a real distribution shift.
"""
import numpy as np
import pandas as pd

PRODUCTS = [  # name, monthly base demand, illustrative unit cost USD, shelf life months, malaria season amplitude, priority
    ("Artemether-lumefantrine 6x4", 180, 0.90, 18, 0.90, 1.0),
    ("Amoxicillin DT 250mg", 260, 0.03, 24, 0.25, 0.9),
    ("ORS + zinc co-pack", 120, 0.45, 24, 0.40, 1.0),
    ("Oxytocin 10IU", 25, 0.40, 12, 0.00, 1.0),
    ("Paracetamol 500mg", 600, 0.01, 36, 0.20, 0.4),
    ("Metformin 500mg", 200, 0.02, 30, 0.00, 0.6),
    ("DMPA injectable", 45, 0.85, 24, 0.00, 0.8),
    ("Malaria RDT", 240, 0.35, 15, 0.90, 0.9),
]
TIERS = {"hospital": 3.0, "health_centre": 1.0, "dispensary": 0.45}
PEAK_MONTH = 8  # season peak, calendar month


def simulate(seed=42, n_fac=120, months=36, shift=24, n_districts=8):
    rng = np.random.default_rng(seed)
    centres = rng.uniform(0, 300, size=(n_districts, 2))
    fac = pd.DataFrame({
        "facility": [f"F{i:03d}" for i in range(n_fac)],
        "district": rng.integers(0, n_districts, n_fac),
        "tier": rng.choice(list(TIERS), n_fac, p=[0.1, 0.4, 0.5]),
        "remote": rng.beta(1.6, 2.2, n_fac),
    })
    fac["district"] = "District " + (fac["district"] + 1).astype(str)
    idx = fac["district"].str.split().str[1].astype(int) - 1
    fac["x"] = centres[idx, 0] + rng.normal(0, 18, n_fac)
    fac["y"] = centres[idx, 1] + rng.normal(0, 18, n_fac)
    quality = rng.beta(5, 2, n_fac)    # hidden: facility management quality
    size = rng.lognormal(0, 0.35, n_fac)

    shortage = np.zeros(len(PRODUCTS), dtype=int)
    rows = []
    state = {}
    for f in range(n_fac):
        for p, prod in enumerate(PRODUCTS):
            mu0 = prod[1] * TIERS[fac.tier[f]] * size[f]
            state[f, p] = dict(batches=[[mu0 * rng.uniform(2, 5), int(rng.integers(6, prod[3] + 1))]],
                               pending=[], hist=[], mu0=mu0)
    season_shift = {}
    for t in range(months):
        cal = t % 12 + 1
        year = t // 12
        season_shift.setdefault(year, rng.integers(-1, 2))
        post = t >= shift
        for p in range(len(PRODUCTS)):  # national shortage state, hidden
            if shortage[p] > 0:
                shortage[p] -= 1
            elif rng.random() < (0.12 if post else 0.05):
                shortage[p] = rng.integers(1, 4)
        snapshot = shortage.copy()
        for f in range(n_fac):
            remote, q = fac.remote[f], quality[f]
            report_p = np.clip(0.97 - 0.25 * remote - (0.22 * (0.3 + remote) if post else 0) - 0.2 * (1 - q), 0.2, 0.99)
            reported = rng.random() < report_p
            for p, prod in enumerate(PRODUCTS):
                s = state[f, p]
                received = 0.0
                for o in [o for o in s["pending"] if o[1] <= t]:
                    received += o[0]
                    s["batches"].append([o[0], prod[3] - int(rng.integers(1, 7))])
                s["pending"] = [o for o in s["pending"] if o[1] > t]
                if rng.random() < 0.012:  # donor-style push delivery: unrequested and short-dated
                    push = s["mu0"] * rng.uniform(4, 9)
                    received += push
                    s["batches"].append([push, int(rng.integers(3, 8))])
                phase = 2 * np.pi * (cal - PEAK_MONTH - season_shift[year]) / 12
                mu = s["mu0"] * (1 + prod[4] * 0.5 * (1 + np.cos(phase)) * 1.0) * (1 + 0.004 * t)
                demand = rng.negative_binomial(4, 4 / (4 + mu)) if mu > 0 else 0
                # FEFO dispensing
                s["batches"].sort(key=lambda b: b[1])
                left = float(demand)
                for b in s["batches"]:
                    use = min(b[0], left)
                    b[0] -= use
                    left -= use
                dispensed = demand - left
                sod = 0 if demand == 0 or left <= 0 else int(np.ceil(30 * left / demand))
                expired = 0.0
                for b in s["batches"]:
                    b[1] -= 1
                    if b[1] <= 0:
                        expired += b[0]
                s["batches"] = [b for b in s["batches"] if b[1] > 0 and b[0] > 0.5]
                stock = sum(b[0] for b in s["batches"])
                near_exp = sum(b[0] for b in s["batches"] if b[1] <= 3)
                # adjusted consumption: facilities correct dispensing for days out of stock
                s["hist"].append(dispensed * 30 / max(30 - sod, 8))
                amc = np.mean(s["hist"][-3:]) * rng.uniform(0.85, 1.15)
                # report-and-request: no report, no order
                ordered = 0.0
                pend = sum(o[0] for o in s["pending"])
                if reported and stock + pend < 3 * amc and rng.random() < 0.65 + 0.35 * q:
                    ordered = max(0.0, 5 * amc - stock - pend)
                    fill = np.clip(rng.normal(0.95 - 0.10 * remote - (0.12 if post else 0), 0.1), 0.2, 1.0)
                    if snapshot[p] > 0:
                        fill *= rng.uniform(0.1, 0.5)
                    lead = rng.geometric(1 / (1.1 + 0.9 * remote + (0.5 if post else 0)))
                    if fill * ordered > 0:
                        s["pending"].append((fill * ordered, t + int(lead)))
                noise = rng.normal(1, 0.05)
                dirty_zero = reported and rng.random() < 0.012
                rows.append(dict(
                    t=t, facility=fac.facility[f], product=prod[0],
                    true_demand=demand, true_stock=stock, true_stockout_days=sod, true_expired=expired,
                    reported=reported,
                    obs_stock=(0.0 if dirty_zero else max(0.0, round(stock * noise))) if reported else np.nan,
                    obs_dispensed=(round(dispensed) if reported else np.nan),
                    obs_received=(round(received) if reported else np.nan),
                    obs_ordered=(round(ordered) if reported else np.nan),
                    obs_stockout_days=(sod if reported else np.nan),
                    obs_near_expiry=(round(near_exp) if reported else np.nan),
                ))
    panel = pd.DataFrame(rows).merge(fac[["facility", "district", "tier", "remote"]], on="facility")
    panel = panel.sort_values(["facility", "product", "t"]).reset_index(drop=True)
    prods = pd.DataFrame(PRODUCTS, columns=["product", "base", "unit_cost", "shelf", "season", "priority"])
    return fac, prods, panel
