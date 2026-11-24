import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # build_dataset.py
from pipeline import (to_weights, backtest, summarize, COST_SCENARIOS, DEV_START, DEV_END,
                      MOM_LOOKBACK, MOM_SKIP, TOP, BOTTOM, ROOT)

MA_SHORT, MA_LONG = 50, 200
BASELINE_DIR = ROOT / "results" / "baseline"      # MA strategy results vs the S&P 500


def compare_to_sp500(strategy_returns, spy_returns):
    import matplotlib
    matplotlib.use("Agg")                 # save to a file; no pop-up window needed
    import matplotlib.pyplot as plt

    all_r = pd.DataFrame({**strategy_returns, "S&P 500 (SPY)": spy_returns})
    all_r = all_r.loc[DEV_START:DEV_END].dropna()
    spy = all_r["S&P 500 (SPY)"]

    rows = {}
    for name in all_r.columns:
        r = all_r[name]
        equity = (1 + r).cumprod()
        rows[name] = {
            "growth_of_$1": equity.iloc[-1],
            "ann_return": equity.iloc[-1] ** (252 / len(r)) - 1,
            "ann_vol": r.std() * np.sqrt(252),
            "sharpe": r.mean() / r.std() * np.sqrt(252),
            "max_drawdown": (equity / equity.cummax() - 1).min(),
            "beta_to_sp500": r.cov(spy) / spy.var(),
            "correlation_to_sp500": r.corr(spy),
        }
    table = pd.DataFrame(rows).T
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(BASELINE_DIR / "comparison_vs_sp500.csv")
    print("\nStrategies (base costs) vs the S&P 500, development years "
          f"{DEV_START[:4]}-{DEV_END[:4]}:")
    print(table.round(3).to_string())

    fig, ax = plt.subplots(figsize=(10, 5))
    (1 + all_r).cumprod().plot(ax=ax, logy=True)
    ax.set_title(f"Growth of $1, {DEV_START[:4]}-{DEV_END[:4]} (strategies net of base costs)")
    ax.set_ylabel("Value of $1 (log scale)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(BASELINE_DIR / "comparison_vs_sp500.png", dpi=150)
    print("Saved results/baseline/comparison_vs_sp500.csv and .png")


def ma_weights(prices, eligible, short=MA_SHORT, long=MA_LONG):
    side = np.sign(prices.rolling(short).mean() - prices.rolling(long).mean())
    return to_weights(side.where(eligible).fillna(0))


def main():
    # Load the clean, versioned dataset built by build_dataset.py.
    # Prices, returns, eligibility and the benchmark all come from there.
    from build_dataset import load_clean_panel
    panel = load_clean_panel()
    print(f"Using clean dataset version {panel['version']}")
    # Signals use signal_prices (E010: invalid returns counted as 0%); the backtest uses returns
    prices, returns = panel["signal_prices"], panel["returns"]
    eligible, spy = panel["eligible"], panel["spy"]

    month_ends = prices.groupby(prices.index.to_period("M")).tail(1).index

    ma_target = ma_weights(prices, eligible)

    mom = prices.shift(MOM_SKIP) / prices.shift(MOM_LOOKBACK) - 1
    pct = mom.where(eligible).loc[month_ends].rank(axis=1, pct=True, method="first")
    xs_side = (pct >= TOP).astype(int) - (pct <= BOTTOM).astype(int)
    xs_target = to_weights(xs_side).reindex(prices.index).ffill().fillna(0)

    rows = []
    for name, target in [("MA 50/200", ma_target), ("12-1 momentum", xs_target)]:
        for scenario, (cost, borrow) in COST_SCENARIOS.items():
            rows.append({"strategy": name, "scenario": scenario,
                         "cost_bps": cost, "borrow_bps": borrow,
                         **summarize(backtest(target, returns, cost, borrow))})

    results = pd.DataFrame(rows)
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(BASELINE_DIR / "baseline_results.csv", index=False)
    print(results.round(3).to_string(index=False))

    # Compare against the S&P 500 (base cost scenario)
    cost, borrow = COST_SCENARIOS["base"]
    spy_returns = spy.pct_change(fill_method=None)
    compare_to_sp500({
        "MA 50/200": backtest(ma_target, returns, cost, borrow)["net"],
        "12-1 momentum": backtest(xs_target, returns, cost, borrow)["net"],
    }, spy_returns)


if __name__ == "__main__":
    main()
