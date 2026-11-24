# Moving-average and momentum strategies on the S&P 500

Long-short backtests of two strategy families on current S&P 500 stocks: an MA crossover (baseline 50/200 days, daily) and cross-sectional momentum (baseline 12-1 months, monthly, top/bottom 20%). Both are 100/100 long/short, equal weight, at least 20 names per side, with trading and borrow costs.

Samples: development 2006-2014, validation 2015-2019, test 2020-2025.

## Setup

```bash
pip install pandas numpy matplotlib requests lxml yfinance
```

Run commands from this folder. Internet is only needed if `data/raw/` is missing.

## Run

Build the dataset (only when the raw data or cleaning rules change; each build gets a new version in `data/`, and scripts use the newest):

```bash
python3 build_dataset.py
```

Analyses, in any order; each writes to its own folder in `results/`:

```bash
python3 eda.py                              # data checks
python3 strategies/ma_strategy.py           # MA and 12-1 baselines
python3 strategies/percentile_momentum.py   # momentum diagnostics
python3 compare_families.py                 # MA vs momentum
python3 strategies/sector_analysis.py       # sector analysis
python3 strategies/trade_log.py             # trade logs
python3 strategies/walk_forward.py          # validation, 2015-2019
```

After a rebuild, run `ma_strategy.py` and `percentile_momentum.py` first; later scripts check against their results.

## Final test

```bash
python3 strategies/final_test.py
```

Runs once on 2020-2025 using only the frozen settings in `config/final.json`
