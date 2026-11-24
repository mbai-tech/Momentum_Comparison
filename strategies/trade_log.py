import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # build_dataset.py
from pipeline import backtest, summarize, COST_SCENARIOS, DEV_START, DEV_END, ROOT
from build_dataset import load_clean_panel, DATA_DIR
from ma_strategy import ma_weights, MA_SHORT, MA_LONG
from percentile_momentum import percentile_momentum

OUT = os.path.join(ROOT, "results", "trade_log")
TOLERANCE = 1e-12            # weight changes smaller than this are floating-point noise


def trade_rows(target, raw_close, adj_close, signals):
    w = target.shift(1).fillna(0)                      # weights after trading (as in backtest)
    old = w.shift(1).fillna(0)
    period = (w.index >= DEV_START) & (w.index <= DEV_END)
    dw = (w - old)[period]
    i, j = np.nonzero(dw.abs().to_numpy() > TOLERANCE)
    dates, tickers = dw.index[i], dw.columns[j]
    o, n = old[period].to_numpy()[i, j], w[period].to_numpy()[i, j]
    rows = pd.DataFrame({"date": dates, "ticker": tickers, "old_weight": o, "new_weight": n,
                         "trade_size": n - o})
    rows["trade_type"] = np.select(
        [o == 0, n == 0, np.sign(o) != np.sign(n)], ["entry", "exit", "flip"], "resize")
    rows["price"] = raw_close.reindex_like(w)[period].to_numpy()[i, j]
    rows["adj_close"] = adj_close.reindex_like(w)[period].to_numpy()[i, j]
    decision = pd.Series(w.index, index=w.index).shift(1)[period]
    rows["signal_date"] = decision.to_numpy()[i]
    for name, frame in signals.items():
        rows[name] = frame.reindex_like(w).shift(1)[period].to_numpy()[i, j]
    cost_bps = COST_SCENARIOS["base"][0]
    rows["estimated_cost"] = rows["trade_size"].abs() * cost_bps / 10_000
    return rows


def summary(name, log, bt):
    years = log["date"].dt.year
    n_years = (pd.Timestamp(DEV_END) - pd.Timestamp(DEV_START)).days / 365.25
    dev = bt.loc[DEV_START:DEV_END]
    borrow = dev["gross"] - dev["net"] - (dev["turnover"] * 2 * COST_SCENARIOS["base"][0] / 10_000)
    by_year = pd.DataFrame({"trades": log.groupby(years).size(),
                            "trading_cost": log.groupby(years)["estimated_cost"].sum(),
                            "borrow_cost": borrow.groupby(borrow.index.year).sum()})
    by_year.index.name = "year"
    types = log["trade_type"].value_counts(normalize=True).reindex(["entry", "exit", "flip", "resize"])
    stats = {"strategy": name, "trades": len(log), "trades_per_year": len(log) / n_years,
             "trade_days": log["date"].nunique(),
             "avg_abs_trade_size": log["trade_size"].abs().mean(),
             "median_abs_trade_size": log["trade_size"].abs().median(),
             **{f"share_{t}": v for t, v in types.items()},
             "trading_cost_per_year": by_year["trading_cost"].sum() / n_years,
             "borrow_cost_per_year": by_year["borrow_cost"].sum() / n_years}
    largest = log.loc[log["trade_size"].abs().nlargest(5).index]
    return stats, by_year, largest


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 220)
    panel = load_clean_panel()
    sp, returns, eligible = panel["signal_prices"], panel["returns"], panel["eligible"]
    long = pd.read_pickle(f"{DATA_DIR}/clean_panel_v{panel['version']}.pkl")
    long["ticker"] = long["ticker"].astype(str)
    raw_close = long.pivot(index="date", columns="ticker", values="raw_close").reindex_like(sp)
    cost, borrow = COST_SCENARIOS["base"]
    print(f"Using clean dataset version {panel['version']}; development {DEV_START[:4]}-{DEV_END[:4]}; "
          f"base costs ({cost} bps trading, {borrow} bps borrow)\n")

    # The baseline weights, unchanged
    ma_target = ma_weights(sp, eligible)
    mom, pct, side, xs_target = percentile_momentum(sp, eligible)
    ma_spread = sp.rolling(MA_SHORT).mean() / sp.rolling(MA_LONG).mean() - 1
    mom_daily, pct_daily = mom.reindex(sp.index), pct.reindex(sp.index)   # month-ends only
    strategies = {
        "MA 50/200": (ma_target, {"signal_ma_spread": ma_spread}),
        "12-1 momentum": (xs_target, {"signal_mom_12_1": mom_daily, "signal_percentile": pct_daily}),
    }
    baseline = pd.read_csv(os.path.join(ROOT, "results", "baseline", "baseline_results.csv"))
    baseline = baseline[baseline.scenario == "base"].set_index("strategy")

    all_stats = []
    for name, (target, signals) in strategies.items():
        bt = backtest(target, returns, cost, borrow)
        ok = np.isclose(summarize(bt)["sharpe_net"], baseline.loc[name, "sharpe_net"])
        log = trade_rows(target, raw_close, panel["prices"], signals)
        dev = bt.loc[DEV_START:DEV_END]
        bt_cost = (dev["turnover"] * 2 * cost / 10_000).sum()
        cost_ok = np.isclose(log["estimated_cost"].sum(), bt_cost)
        slug = name.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        log.to_csv(f"{OUT}/trades_{slug}.csv.gz", index=False, float_format="%.10g")

        stats, by_year, largest = summary(name, log, bt)
        by_year.to_csv(f"{OUT}/by_year_{slug}.csv")
        largest.to_csv(f"{OUT}/largest_trades_{slug}.csv", index=False)
        all_stats.append(stats)

        print(f"== {name}: same weights as the baseline (Sharpe matches: {ok}); "
              f"log costs = backtest trading cost: {cost_ok}")
        print(f"{stats['trades']:,} trades on {stats['trade_days']:,} days, "
              f"{stats['trades_per_year']:,.0f} per year; average |trade size| "
              f"{stats['avg_abs_trade_size']:.5f} (median {stats['median_abs_trade_size']:.5f})")
        print("Trade types: " + ", ".join(f"{t} {stats[f'share_{t}']:.1%}"
                                          for t in ["entry", "exit", "flip", "resize"]))
        print(f"Cost per year: trading {stats['trading_cost_per_year']:.2%}, "
              f"borrow {stats['borrow_cost_per_year']:.2%}")
        print(by_year.to_string(formatters={"trades": "{:,}".format,
                                            "trading_cost": "{:.2%}".format,
                                            "borrow_cost": "{:.2%}".format}))
        sig_cols = list(signals)
        print("5 largest trades:")
        print(largest[["date", "ticker", "trade_type", "old_weight", "new_weight", "trade_size",
                       "price", "signal_date", *sig_cols, "estimated_cost"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print()

    pd.DataFrame(all_stats).to_csv(f"{OUT}/summary.csv", index=False)
    print(f"Saved in {OUT}/")


if __name__ == "__main__":
    main()
