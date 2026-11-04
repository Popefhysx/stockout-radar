# Stockout Radar

A prototype that forecasts which health facilities are likely to run out of an essential medicine within 3 months, explains the signals behind each forecast, and ranks redistribution and emergency orders under a budget ceiling. A person approves every action.

**All data is synthetic. No real facility, patient or country is involved. These results show the method, not real-world performance.**

## Why synthetic data
I looked for open facility-level stock data first. The closest match, the Côte d'Ivoire eLMIS extract from USAID's Intelligent Forecasting challenge, is hosted behind sign-in and accepted terms, and I could not confirm a licence that allows use here. So `stockout/sim.py` generates the data. To use real data, map an extract to the `obs_*` columns: features read nothing else, and a test enforces that.

## What the simulator does
It keeps truth (demand, stock) apart from what a logistics system sees (late, missing and noisy reports). Monthly data covers 120 facilities, 8 products and 36 months, with seasonal demand, national shortages, report-and-request ordering, short-dated push deliveries, and a donor drawdown at month 24 that worsens reporting, fill rates and lead times.

The first version gave implausible rates (66% of months out of stock after the drawdown, almost no expiry). I corrected the simulator once, against stock-out and expiry rates only, before any model was run. It has not changed since seeing model results.

## Results
Rolling-origin backtest. Each origin trains only on rows whose 3-month label window has closed. Every method may flag 10% of rows. Scored on the simulator's true stock-outs after the drawdown, seed 42, 6,901 test rows, 34.6% with a stock-out.

| Method | Flags correct | Stock-outs caught | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| Months-of-stock rule | 54.3% | 15.8% | 0.476 | 0.644 |
| Logistic regression | 67.7% | 19.7% | 0.575 | 0.732 |
| Gradient boosting | 70.7% | 20.5% | 0.596 | 0.739 |

- Gain in stock-outs caught over the rule: +4.8 points, 95% CI +0.9 to +8.6, resampling whole facilities. Across 3 seeds the mean gain is +4.9 points.
- At this capacity, no method could catch more than about 29% of stock-outs.

## Known problems
- Written by one author: the simulator may favour the model. Real data is the real test.
- Calibration error rose from 1.8% before the drawdown to 12.8% after, with risks running too low. Two likely causes, not yet separated: the drawdown, and learning from reported stock-outs when silent facilities are often the ones running out.
- In the seed 42 run the model catches less than the rule for hospitals (8.4% vs 15.8%), less remote facilities (11.8% vs 16.1%) and facilities that reported this month (6.9% vs 19.1%). Its gain comes mainly from facilities silent for 2+ months (59.2% vs 10.1%).
- Simple logistic regression gets most of the gain.
- Rows last reported at zero stock, or never reported, are not scored. They need direct follow-up.
- Prices, distances and transport costs are illustrative.

## Run it
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m stockout.pipeline
.venv/bin/python -m stockout.build_report
```
Then open `output/index.html`.
