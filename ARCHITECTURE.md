# Architecture

Batch Python pipeline that writes one static HTML page.

- `stockout/sim.py`: synthetic data. `true_*` columns are truth, `obs_*` are what a system sees.
- `stockout/features.py`: features from `obs_*` at or before month t; labels from t+1 to t+3.
- `stockout/evaluate.py`: rolling-origin backtest, baselines, calibration, subgroup checks.
- `stockout/decide.py`: redistribution and emergency orders, ranked under a budget ceiling.
- `stockout/pipeline.py`: runs everything and writes `output/results.json`.
- `stockout/build_report.py` + `template.html`: renders `output/index.html`. Design tokens are the CSS variables at the top of the template; the page is its only shell.
