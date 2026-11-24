import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # build_dataset.py
from pipeline import (to_weights, backtest, summarize, COST_SCENARIOS, MIN_NAMES, TOP, BOTTOM, ROOT)
from build_dataset import load_clean_panel
from ma_strategy import ma_weights

OUT = os.path.join(ROOT, "results", "walk_forward")
WF_YEARS = list(range(2015, 2020))
WINDOW_YEARS = 5
CANDIDATES = {
    "MA": {"MA (20,100)": ("ma", 20, 100), "MA (50,200)": ("ma", 50, 200),
           "MA (100,250)": ("ma", 100, 250)},
    "Momentum": {"6-1": ("mom", 126, 21), "9-1": ("mom", 189, 21), "12-1": ("mom", 252, 21)},
}
FIXED_BASELINES = {"MA": "MA (50,200)", "Momentum": "12-1"}


def momentum_weights(sp, eligible, lookback, skip):
    dates = sp.index.to_series().groupby(sp.index.to_period("M")).tail(1).index
    mom = (sp.shift(skip) / sp.shift(lookback) - 1).where(eligible).loc[dates]
    pct = mom.rank(axis=1, method="first", pct=True)
    side = (pct >= TOP).astype(int) - (pct <= BOTTOM).astype(int)
    return to_weights(side, MIN_NAMES).reindex(sp.index).ffill().fillna(0)


def candidate_target(spec, sp, eligible):
    kind, a, b = spec
    return ma_weights(sp, eligible, a, b) if kind == "ma" else momentum_weights(sp, eligible, a, b)


def sharpe(r):
    r = r.dropna()
    return r.mean() / r.std() * np.sqrt(252)


def metrics(r, turnover=None):
    r = r.dropna()
    eq = (1 + r).cumprod()
    out = {"ann_return": eq.iloc[-1] ** (252 / len(r)) - 1, "ann_vol": r.std() * np.sqrt(252),
           "sharpe_net": sharpe(r), "max_drawdown": (eq / eq.cummax() - 1).min(),
           "growth_of_$1": eq.iloc[-1]}
    if turnover is not None:
        out["annual_one_way_turnover"] = turnover.mean() * 252
    return out


def last_trading_day(index, year):
    return index[index.year == year][-1]


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 220)
    panel = load_clean_panel()
    sp, returns, eligible = panel["signal_prices"], panel["returns"], panel["eligible"]
    spy_r = panel["spy"].pct_change(fill_method=None)
    cost, borrow = COST_SCENARIOS["base"]
    print(f"Using clean dataset version {panel['version']}; walk-forward {WF_YEARS[0]}-{WF_YEARS[-1]}, "
          f"{WINDOW_YEARS}-year selection window, base costs ({cost} bps trading, {borrow} bps borrow)\n")

    # 1. Every candidate on its own
    targets, bts = {}, {}
    for fam, cands in CANDIDATES.items():
        for name, spec in cands.items():
            targets[name] = candidate_target(spec, sp, eligible)
            bts[name] = backtest(targets[name], returns, cost, borrow)

    # Check: the fixed baselines reproduce the development-sample results
    base = pd.read_csv(os.path.join(ROOT, "results", "baseline", "baseline_results.csv"))
    base = base[base.scenario == "base"].set_index("strategy")["sharpe_net"]
    for cand, stored in [("MA (50,200)", base["MA 50/200"]), ("12-1", base["12-1 momentum"])]:
        got = summarize(bts[cand])["sharpe_net"]
        print(f"Check {cand}: development Sharpe {got:.4f} vs baseline {stored:.4f} -> "
              f"{'OK' if np.isclose(got, stored) else 'MISMATCH'}")

    # 2. Selection at the start of each year from the previous 5 years
    selection, chosen = [], {fam: {} for fam in CANDIDATES}
    for fam, cands in CANDIDATES.items():
        for y in WF_YEARS:
            a, b = f"{y - WINDOW_YEARS}-01-01", f"{y - 1}-12-31"
            window = {c: sharpe(bts[c]["net"].loc[a:b]) for c in cands}
            used = {c: sharpe(bts[c]["net"].loc[str(y)]) for c in cands}
            pick = max(window, key=window.get)
            chosen[fam][y] = pick
            rank = sorted(used, key=used.get, reverse=True).index(pick) + 1
            selection.append({"family": fam, "year": y, "window": f"{y - WINDOW_YEARS}-{y - 1}",
                              **{f"window_sharpe {c}": v for c, v in window.items()},
                              "chosen": pick,
                              **{f"same_year_sharpe {c}": v for c, v in used.items()},
                              "chosen_rank_in_year": rank})
    selection = pd.DataFrame(selection)
    selection.to_csv(f"{OUT}/selection.csv", index=False)

    # 3. Stitched walk-forward portfolios (switching is traded and charged)
    idx = sp.index
    bounds = [last_trading_day(idx, y - 1) for y in WF_YEARS] + [idx[-1] + pd.Timedelta(days=1)]
    wf_bt, wf_net = {}, {}
    for fam in CANDIDATES:
        stitched = targets[chosen[fam][WF_YEARS[0]]].copy()      # already holding the first pick
        for y, start, end in zip(WF_YEARS, bounds[:-1], bounds[1:]):
            rows = (idx >= start) & (idx < end)
            stitched.loc[rows] = targets[chosen[fam][y]].loc[rows]
        wf_bt[fam] = backtest(stitched, returns, cost, borrow)
        wf_net[f"Walk-forward {fam}"] = wf_bt[fam]["net"]

    period = slice(f"{WF_YEARS[0]}-01-01", f"{WF_YEARS[-1]}-12-31")
    series = {**{k: (v.loc[period], wf_bt[k.split()[-1]]["turnover"].loc[period])
                 for k, v in wf_net.items()},
              **{f"Fixed {c}": (bts[c]["net"].loc[period], bts[c]["turnover"].loc[period])
                 for c in FIXED_BASELINES.values()},
              "S&P 500 (SPY)": (spy_r.loc[period], None)}
    summary = pd.DataFrame({k: metrics(r, t) for k, (r, t) in series.items()}).T
    summary.index.name = "strategy"
    summary.to_csv(f"{OUT}/walk_forward_summary.csv")

    yearly = []
    for k, (r, _) in series.items():
        for y, x in r.groupby(r.index.year):
            fam = k.split()[-1] if k.startswith("Walk-forward") else None
            yearly.append({"strategy": k, "year": y, "net_return": (1 + x).prod() - 1,
                           "sharpe": sharpe(x), "candidate": chosen[fam][y] if fam else ""})
    yearly = pd.DataFrame(yearly)
    yearly.to_csv(f"{OUT}/walk_forward_yearly.csv", index=False)

    switch = {fam: [(y, chosen[fam][y - 1], chosen[fam][y],
                     wf_bt[fam]["turnover"].loc[str(y)].iloc[:2].sum())
                    for y in WF_YEARS[1:] if chosen[fam][y] != chosen[fam][y - 1]]
              for fam in CANDIDATES}

    every = pd.DataFrame({c: metrics(bts[c]["net"].loc[period], bts[c]["turnover"].loc[period])
                          for cands in CANDIDATES.values() for c in cands}).T
    every.index.name = "candidate"
    every.to_csv(f"{OUT}/candidates_2015_2019.csv")

    # 4. Figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for k, (r, _) in series.items():
        (1 + r.fillna(0)).cumprod().plot(ax=ax, label=k, lw=1.6 if k.startswith("Walk") else 1,
                                         ls="-" if k.startswith("Walk") else "--")
    for y in WF_YEARS[1:]:
        ax.axvline(pd.Timestamp(f"{y}-01-01"), color="grey", lw=0.6)
    ax.set_title(f"Walk-forward {WF_YEARS[0]}-{WF_YEARS[-1]}: growth of $1, net of base costs")
    ax.set_ylabel("Value of $1"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/walk_forward_equity.png", dpi=150); plt.close(fig)

    # 5. Print
    for fam, cands in CANDIDATES.items():
        s = selection[selection.family == fam].set_index("year")
        print(f"\n{fam}: net Sharpe over the previous {WINDOW_YEARS} years -> chosen candidate")
        print(s[[f"window_sharpe {c}" for c in cands]].rename(columns=lambda c: c.split(" ", 1)[1])
              .assign(chosen=s["chosen"], rank_in_year=s["chosen_rank_in_year"])
              .to_string(float_format=lambda x: f"{x:+.3f}"))
        for y, old, new, t in switch[fam]:
            print(f"  switch at start of {y}: {old} -> {new} (one-way turnover on switch days {t:.2f})")
    print(f"\nNet return by year ({WF_YEARS[0]}-{WF_YEARS[-1]}, base costs):")
    print(yearly.pivot(index="year", columns="strategy", values="net_return")[list(series)]
          .to_string(float_format=lambda x: f"{x:+.1%}"))
    print(f"\nWhole period {WF_YEARS[0]}-{WF_YEARS[-1]}:")
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nEvery candidate over the same years (hindsight reference, not selectable):")
    print(every[["ann_return", "sharpe_net", "max_drawdown"]].to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nAll tables and figures saved in {OUT}/")


if __name__ == "__main__":
    main()
