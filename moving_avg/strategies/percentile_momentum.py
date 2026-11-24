import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # build_dataset.py
from pipeline import (to_weights, backtest, summarize, COST_SCENARIOS, DEV_START, DEV_END,
                      MIN_NAMES, MOM_LOOKBACK, MOM_SKIP, TOP, BOTTOM, ROOT)
from build_dataset import load_clean_panel

OUT = os.path.join(ROOT, "results", "percentile")
N_QUANTILES = 5          # quintiles for the bucket diagnostics (spec 4.6)


# The strategy (spec 4.3)
def percentile_momentum(prices, eligible):
    # Step 1: fixed rebalance dates = last AVAILABLE trading day of each month
    month_ends = prices.groupby(prices.index.to_period("M")).tail(1).index

    # Step 2: 12-1 momentum, MOM(L=252, S=21) = P[t-21] / P[t-252] - 1   (spec 4.1)
    mom = (prices.shift(MOM_SKIP) / prices.shift(MOM_LOOKBACK) - 1).where(eligible)
    mom = mom.loc[month_ends]

    # Step 3 (optional winsorising) is not needed: ranks are unaffected by outliers.

    # Step 4: rank across ELIGIBLE stocks only; columns are alphabetical and
    # method="first" breaks ties in that order -> deterministic (spec 4.3 step 2)
    pct = mom.rank(axis=1, method="first", pct=True)          # rank / N_t, in (0, 1]

    # Step 5: tails. Long if percentile >= 0.80, short if <= 0.20 (spec 4.3)
    side = (pct >= TOP).astype(int) - (pct <= BOTTOM).astype(int)

    # Step 6: equal weight within each leg, 100% gross long and 100% gross short;
    #         a leg with fewer than MIN_NAMES stocks goes to cash (minimum basket size)
    w = to_weights(side, MIN_NAMES)

    # Step 7: hold the month-end weights until the next rebalance. The ONE-DAY LAG to the
    #         first trading day of month m+1 is applied inside backtest(), exactly as for MA.
    target = w.reindex(prices.index).ffill().fillna(0)
    return mom, pct, side, target


# Signal diagnostics (spec 4.6)
def holding_period_returns(prices, rebal_dates):
    pos = prices.index.get_indexer(rebal_dates)
    exec_dates = prices.index[np.minimum(pos + 1, len(prices.index) - 1)]
    p = prices.loc[exec_dates]
    fwd = p.shift(-1) / p - 1
    fwd.index = rebal_dates
    return fwd


def signal_diagnostics(mom, pct, side, prices, eligible, sectors):
    # Only rebalance dates whose whole holding period ends inside the development sample
    dates = mom.index[(mom.index >= DEV_START) & (mom.index <= DEV_END)][:-1]
    fwd = holding_period_returns(prices, mom.index).loc[dates]
    m, p, s = mom.loc[dates], pct.loc[dates], side.loc[dates]
    out = {}

    # Coverage
    n_elig = eligible.loc[dates].sum(axis=1)
    n_valid = m.notna().sum(axis=1)
    coverage = pd.DataFrame({"eligible": n_elig, "valid_signal": n_valid,
                             "n_long": (s > 0).sum(axis=1), "n_short": (s < 0).sum(axis=1)})
    coverage.to_csv(f"{OUT}/coverage_by_date.csv")
    out["avg_eligible_stocks"] = n_elig.mean()
    out["min_eligible_stocks"] = n_elig.min()
    out["avg_pct_eligible_with_signal"] = (n_valid / n_elig).mean()
    out["avg_names_long"] = coverage["n_long"].mean()
    out["avg_names_short"] = coverage["n_short"].mean()
    out["months_with_a_leg_in_cash"] = int(((coverage["n_long"] < MIN_NAMES) |
                                            (coverage["n_short"] < MIN_NAMES)).sum())

    # Cross-sectional distribution of the raw signal
    dist = pd.DataFrame({"mean": m.mean(axis=1), "std": m.std(axis=1), "skew": m.skew(axis=1),
                         "p01": m.quantile(0.01, axis=1), "p50": m.median(axis=1),
                         "p99": m.quantile(0.99, axis=1), "min": m.min(axis=1),
                         "max": m.max(axis=1)})
    dist.to_csv(f"{OUT}/signal_distribution_by_date.csv")
    for c in ["mean", "std", "skew", "p01", "p50", "p99"]:
        out[f"signal_{c}_avg"] = dist[c].mean()
    out["signal_min_ever"], out["signal_max_ever"] = dist["min"].min(), dist["max"].max()

    # Persistence: rank autocorrelation and names staying in each tail
    rank_autocorr = p.corrwith(p.shift(1), axis=1, method="spearman").dropna()
    stay_long = ((s > 0) & (s.shift(1) > 0)).sum(axis=1) / (s.shift(1) > 0).sum(axis=1)
    stay_short = ((s < 0) & (s.shift(1) < 0)).sum(axis=1) / (s.shift(1) < 0).sum(axis=1)
    out["rank_autocorrelation_avg"] = rank_autocorr.mean()
    out["pct_long_names_kept_next_month"] = stay_long.mean()
    out["pct_short_names_kept_next_month"] = stay_short.mean()

    # Information coefficient: Spearman(signal, next holding-period return) each month
    mask = m.notna() & fwd.notna()
    ic = m.where(mask).corrwith(fwd.where(mask), axis=1, method="spearman").dropna()
    ic.rename("rank_ic").to_csv(f"{OUT}/rank_ic_by_date.csv")
    out["ic_mean"] = ic.mean()
    out["ic_tstat"] = ic.mean() / (ic.std() / np.sqrt(len(ic)))
    out["ic_pct_positive_months"] = (ic > 0).mean()

    # Quantile (bucket) returns and monotonicity -- every bucket, not only the tails
    bucket = np.ceil(p * N_QUANTILES).clip(1, N_QUANTILES)
    q = pd.DataFrame({f"Q{k}": fwd.where(bucket == k).mean(axis=1)
                      for k in range(1, N_QUANTILES + 1)})
    q[f"Q{N_QUANTILES}-Q1"] = q[f"Q{N_QUANTILES}"] - q["Q1"]
    q.to_csv(f"{OUT}/quantile_returns_by_date.csv")
    q_mean = q.mean()
    for k, v in q_mean.items():
        out[f"avg_monthly_return_{k}"] = v
    steps = np.diff(q_mean[[f"Q{k}" for k in range(1, N_QUANTILES + 1)]].values)
    out["quantiles_monotonic_increasing"] = bool((steps > 0).all())

    # Sector composition of the tails (current GICS sectors, not point-in-time)
    sec = sectors.reindex(s.columns).fillna("Unknown")
    long_share = (s > 0).T.groupby(sec).sum().T.div((s > 0).sum(axis=1), axis=0)
    short_share = (s < 0).T.groupby(sec).sum().T.div((s < 0).sum(axis=1), axis=0)
    univ_share = eligible.loc[dates].T.groupby(sec).sum().T.div(n_elig, axis=0)
    sector_table = pd.DataFrame({"share_of_universe": univ_share.mean(),
                                 "share_of_long_leg": long_share.mean(),
                                 "share_of_short_leg": short_share.mean()}).fillna(0)
    sector_table["long_minus_short"] = sector_table["share_of_long_leg"] - \
                                       sector_table["share_of_short_leg"]
    sector_table.sort_values("long_minus_short").to_csv(f"{OUT}/sector_composition.csv")
    out["largest_sector_share_long_avg"] = long_share.max(axis=1).mean()
    out["largest_sector_share_short_avg"] = short_share.max(axis=1).mean()

    # Turnover forecast from rank migration, before costs are applied
    w = to_weights(s, MIN_NAMES)
    out["forecast_monthly_one_way_turnover"] = (0.5 * (w - w.shift(1)).abs().sum(axis=1)).iloc[1:].mean()

    return pd.Series(out, name="value"), q_mean, ic, sector_table


# Long/short attribution (spec 9.4)
def leg_attribution(target, returns):
    held = target.shift(2).fillna(0)          # same timing as backtest(): signal -> trade -> earn
    r = returns.fillna(0)
    legs = pd.DataFrame({"long_leg": (held.clip(lower=0) * r).sum(axis=1),
                         "short_leg": (held.clip(upper=0) * r).sum(axis=1)}).loc[DEV_START:DEV_END]
    yearly = legs.groupby(legs.index.year).sum()          # sum of daily returns
    yearly["total"] = yearly.sum(axis=1)
    yearly.index.name = "year"
    annual = (legs.mean() * 252).rename("annualised_contribution")
    return yearly, annual


def save_figures(q_mean, ic, yearly):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    buckets = [f"Q{k}" for k in range(1, N_QUANTILES + 1)]
    ax.bar(buckets, q_mean[buckets] * 100)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"Average next-month return by 12-1 momentum quintile, {DEV_START[:4]}-{DEV_END[:4]}")
    ax.set_xlabel("Quintile (Q1 = past losers, Q5 = past winners)")
    ax.set_ylabel("Average monthly return (%)")
    fig.tight_layout(); fig.savefig(f"{OUT}/xs_quintile_returns.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(ic.index, ic.values, width=20)
    ax.plot(ic.index, ic.rolling(12).mean(), color="black", label="12-month average")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("Monthly rank information coefficient, 12-1 momentum")
    ax.set_ylabel("Spearman IC"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/xs_rank_ic.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    yearly[["long_leg", "short_leg"]].mul(100).plot.bar(ax=ax)
    ax.plot(range(len(yearly)), yearly["total"] * 100, "ko-", label="total")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("12-1 momentum: gross return by year and leg (sum of daily returns)")
    ax.set_ylabel("%"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/xs_yearly_legs.png", dpi=150); plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    # Clean, versioned dataset from build_dataset.py -- same data as the MA strategy
    panel = load_clean_panel()
    print(f"Using clean dataset version {panel['version']}")
    # Signals and diagnostic forward returns use signal_prices (E010); the backtest uses returns
    prices, returns = panel["signal_prices"], panel["returns"]
    eligible, sectors = panel["eligible"], panel["sectors"]

    mom, pct, side, target = percentile_momentum(prices, eligible)

    # Sanity checks: exposures must match the baseline design
    active = target.abs().sum(axis=1) > 0
    gross = target.abs().sum(axis=1)[active]
    net = target.sum(axis=1)[active]
    print(f"Exposure check: gross {gross.mean():.3f} (target 2.0), "
          f"largest |net| {net.abs().max():.2e} (target 0)")

    # 1. Signal diagnostics (judge the signal before the portfolio)
    diag, q_mean, ic, sector_table = signal_diagnostics(mom, pct, side, prices, eligible, sectors)
    diag.to_csv(f"{OUT}/signal_diagnostics.csv")
    print("\nSignal diagnostics, development sample:")
    print(diag.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nSector composition of the tails (average share):")
    print(sector_table.round(3).to_string())

    # 2. Backtest under all four cost scenarios (same backtest() as MA)
    rows = []
    for scenario, (cost, borrow) in COST_SCENARIOS.items():
        rows.append({"strategy": "12-1 momentum", "scenario": scenario,
                     "cost_bps": cost, "borrow_bps": borrow,
                     **summarize(backtest(target, returns, cost, borrow))})
    results = pd.DataFrame(rows)
    results.to_csv(f"{OUT}/percentile_baseline_results.csv", index=False)
    print("\nBacktest, development sample:")
    print(results.round(3).to_string(index=False))

    # 3. Long/short attribution
    yearly, annual = leg_attribution(target, returns)
    yearly.to_csv(f"{OUT}/yearly_leg_returns.csv")
    print("\nAnnualised contribution by leg (gross):")
    print(annual.round(4).to_string())
    print("\nGross return by year and leg:")
    print(yearly.round(3).to_string())

    save_figures(q_mean, ic, yearly)
    print(f"\nAll tables and figures saved in {OUT}/")


if __name__ == "__main__":
    main()
