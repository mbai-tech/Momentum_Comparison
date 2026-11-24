import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # build_dataset.py
from pipeline import (to_weights, backtest, summarize, COST_SCENARIOS, DEV_START, DEV_END,
                      MIN_NAMES, MOM_LOOKBACK, MOM_SKIP, TOP, BOTTOM, ROOT)
from build_dataset import load_clean_panel
from ma_strategy import ma_weights

OUT = os.path.join(ROOT, "results", "sector")
VARIANTS = ["raw", "within_sector", "sector_neutral"]
DEFENSIVE = ["Consumer Staples", "Health Care", "Utilities"]      # all other sectors: cyclical
SUBPERIODS = {"2006-2010": ("2006-01-01", "2010-12-31"), "2011-2014": ("2011-01-01", "2014-12-31")}
CRISIS_YEARS = [2008, 2009]


# Building blocks
def rebalance_dates(index):
    return index.to_series().groupby(index.to_period("M")).tail(1).index


def dev_only(dates):
    return dates[(dates >= DEV_START) & (dates <= DEV_END)]


def by_sector(frame, groups):
    return frame.T.groupby(groups).sum().T


def tails(pct):
    return (pct >= TOP).astype(int) - (pct <= BOTTOM).astype(int)


def momentum_scores(sp, eligible, dates):
    return (sp.shift(MOM_SKIP) / sp.shift(MOM_LOOKBACK) - 1).where(eligible).loc[dates]


def within_sector_pct(mom, groups):
    pct = pd.DataFrame(np.nan, index=mom.index, columns=mom.columns)
    for _, cols in mom.columns.groupby(groups.reindex(mom.columns).values).items():
        pct[cols] = mom[cols].rank(axis=1, method="first", pct=True)
    return pct


def neutralise(w, groups):
    longs, shorts = w.clip(lower=0), -w.clip(upper=0)
    L, S = by_sector(longs, groups), by_sector(shorts, groups)
    both = (L > 0) & (S > 0)                                  # sectors with only one side dropped
    long_scale = ((L + S) / 2 / L).where(both, 0)
    short_scale = ((L + S) / 2 / S).where(both, 0)
    labels = groups.reindex(w.columns).values
    new_long = longs * long_scale[labels].set_axis(w.columns, axis=1).values
    new_short = shorts * short_scale[labels].set_axis(w.columns, axis=1).values
    ok = ((new_long > 0).sum(axis=1) >= MIN_NAMES) & ((new_short > 0).sum(axis=1) >= MIN_NAMES)
    out = new_long.div(new_long.sum(axis=1), axis=0) - new_short.div(new_short.sum(axis=1), axis=0)
    return out.where(ok, 0).fillna(0)                         # back to 100/100, or cash


def build_variants(sp, eligible, groups, dates):
    mom = momentum_scores(sp, eligible, dates)
    pct_raw = mom.rank(axis=1, method="first", pct=True)
    pct_within = within_sector_pct(mom, groups)
    w_raw = to_weights(tails(pct_raw), MIN_NAMES)
    weights = {"raw": w_raw,
               "within_sector": to_weights(tails(pct_within), MIN_NAMES),
               "sector_neutral": neutralise(w_raw, groups)}
    return weights, pct_raw, pct_within


def daily_target(w, index):
    return w.reindex(index).ffill().fillna(0)


# Diagnostics
def portfolio_profile(w, groups, dates):
    wd = w.loc[dates]
    active = wd.abs().sum(axis=1) > 0
    wd = wd[active]
    lw, sw = by_sector(wd.clip(lower=0), groups), by_sector(-wd.clip(upper=0), groups)
    net = lw - sw
    gross_share = (lw + sw).div((lw + sw).sum(axis=1), axis=0)
    return {
        "avg_names_long": (wd > 0).sum(axis=1).mean(),
        "avg_names_short": (wd < 0).sum(axis=1).mean(),
        "largest_sector_share_long_avg": lw.max(axis=1).mean(),
        "largest_sector_share_short_avg": sw.max(axis=1).mean(),
        "sector_hhi_gross_avg": (gross_share ** 2).sum(axis=1).mean(),
        "sum_abs_net_sector_exposure_avg": net.abs().sum(axis=1).mean(),
        "max_abs_net_sector_exposure_avg": net.abs().max(axis=1).mean(),
        "months_in_cash": int((~active).sum()),
    }


def sector_weights_table(w, eligible, groups, dates, variant):
    wd = w.loc[dates]
    parts = {"n_eligible": by_sector(eligible.loc[dates].astype(int), groups),
             "n_long": by_sector((wd > 0).astype(int), groups),
             "n_short": by_sector((wd < 0).astype(int), groups),
             "long_weight": by_sector(wd.clip(lower=0), groups),
             "short_weight": by_sector(-wd.clip(upper=0), groups)}
    t = pd.concat({k: v.stack() for k, v in parts.items()}, axis=1)
    t.index.names = ["date", "sector"]
    t["net_weight"] = t["long_weight"] - t["short_weight"]
    t.insert(0, "variant", variant)
    return t.reset_index()


def sector_contributions(target, returns, groups, sector_ret):
    held = target.shift(2).fillna(0)
    r = returns.fillna(0)
    contrib = by_sector(held * r, groups).loc[DEV_START:DEV_END]
    long_c = by_sector(held.clip(lower=0) * r, groups).loc[DEV_START:DEV_END]
    short_c = by_sector(held.clip(upper=0) * r, groups).loc[DEV_START:DEV_END]
    net_w = by_sector(held, groups).loc[DEV_START:DEV_END]
    allocation = (net_w * sector_ret.loc[DEV_START:DEV_END].fillna(0)).sum(axis=1)
    total = contrib.sum(axis=1)
    split = {"total_gross_per_yr": total.mean() * 252,
             "sector_allocation_per_yr": allocation.mean() * 252,
             "stock_selection_per_yr": (total - allocation).mean() * 252}
    annual = pd.DataFrame({"long_leg_per_yr": long_c.mean() * 252,
                           "short_leg_per_yr": short_c.mean() * 252,
                           "total_per_yr": contrib.mean() * 252})
    return contrib, annual, split


def sector_turnover(target, groups):
    w = target.shift(1).fillna(0)
    traded = by_sector(w.diff().abs(), groups).loc[DEV_START:DEV_END]
    held = by_sector(w.abs(), groups).loc[DEV_START:DEV_END]
    return pd.DataFrame({"annual_one_way_turnover": traded.mean() * 252 / 2,
                         "share_of_turnover": traded.sum() / traded.sum().sum(),
                         "share_of_gross_weight": held.sum() / held.sum().sum()})


def leg_contributions(target, returns, labels):
    held = target.shift(2).fillna(0)
    r = returns.fillna(0)
    long_c = by_sector(held.clip(lower=0) * r, labels).loc[DEV_START:DEV_END]
    short_c = by_sector(held.clip(upper=0) * r, labels).loc[DEV_START:DEV_END]
    return pd.DataFrame({"long_side_per_yr": long_c.mean() * 252,
                         "short_side_per_yr": short_c.mean() * 252,
                         "total_per_yr": (long_c + short_c).mean() * 252})


def cash_days(target):
    held = target.shift(2).fillna(0).loc[DEV_START:DEV_END]
    no_long, no_short = held.clip(lower=0).sum(axis=1) == 0, held.clip(upper=0).sum(axis=1) == 0
    return {"days_in_cash": int((no_long & no_short).sum()),
            "days_with_a_side_in_cash": int((no_long | no_short).sum()),
            "days_total": len(held),
            "avg_names_long": (held > 0).sum(axis=1).mean(),
            "avg_names_short": (held < 0).sum(axis=1).mean()}


def performance_by_period(nets):
    def sharpe(x):
        return x.mean() / x.std() * np.sqrt(252)
    yearly, sub = [], []
    for name, r in nets.items():
        r = r.loc[DEV_START:DEV_END]
        for y, x in r.groupby(r.index.year):
            yearly.append({"strategy": name, "year": y, "net_return": (1 + x).prod() - 1,
                           "sharpe": sharpe(x)})
        for label, (a, b) in SUBPERIODS.items():
            x = r.loc[a:b]
            sub.append({"strategy": name, "period": label,
                        "ann_return": (1 + x).prod() ** (252 / len(x)) - 1, "sharpe": sharpe(x)})
    return pd.DataFrame(yearly), pd.DataFrame(sub)


def run_backtests(weights, sp_index, returns, scenarios=COST_SCENARIOS):
    rows = []
    for variant, w in weights.items():
        target = daily_target(w, sp_index)
        for scenario, (cost, borrow) in scenarios.items():
            rows.append({"variant": variant, "scenario": scenario, "cost_bps": cost,
                         "borrow_bps": borrow, **summarize(backtest(target, returns, cost, borrow))})
    return pd.DataFrame(rows)


def save_figures(contrib_cum, net_by_date, loso):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, v in zip(axes, VARIANTS):
        (contrib_cum[v] * 100).plot(ax=ax, legend=False, lw=1, colormap="tab20")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(v.replace("_", " ")); ax.set_ylabel("cumulative gross contribution (%)")
        ax.grid(alpha=0.3)
    axes[-1].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"Cumulative sector contribution, {DEV_START[:4]}-{DEV_END[:4]} (sum of daily returns)")
    fig.tight_layout(); fig.savefig(f"{OUT}/sector_contribution_cumulative.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    (net_by_date * 100).plot(ax=ax, lw=1, colormap="tab20")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Raw 12-1 momentum: net sector weight on each rebalance date")
    ax.set_ylabel("net weight (% of one leg)"); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/raw_net_sector_weights.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    loso.pivot(index="sector_removed", columns="variant", values="sharpe_net")[VARIANTS].plot.bar(ax=ax)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Leave-one-sector-out: net Sharpe (base costs); 'none' = full universe")
    ax.set_ylabel("Sharpe"); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(f"{OUT}/leave_one_sector_out.png", dpi=150); plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 220)
    panel = load_clean_panel()
    print(f"Using clean dataset version {panel['version']}; development sample "
          f"{DEV_START[:4]}-{DEV_END[:4]}; sectors are current GICS labels\n")
    sp, returns, eligible = panel["signal_prices"], panel["returns"], panel["eligible"]
    groups = panel["sectors"].reindex(sp.columns).fillna("Unknown")
    sectors = sorted(groups.unique())
    dates = rebalance_dates(sp.index)
    dev_dates = dev_only(dates)
    cost, borrow = COST_SCENARIOS["base"]

    # 1. The three variants
    weights, pct_raw, pct_within = build_variants(sp, eligible, groups, dates)
    targets = {v: daily_target(w, sp.index) for v, w in weights.items()}
    for v, t in targets.items():
        h = t.loc[DEV_START:DEV_END]
        a = h.abs().sum(axis=1) > 0
        print(f"Exposure check {v}: gross {h.abs().sum(axis=1)[a].mean():.3f} (target 2.0), "
              f"largest |net| {h.sum(axis=1)[a].abs().max():.1e}")

    results = run_backtests(weights, sp.index, returns)
    results.to_csv(f"{OUT}/variant_results_all_costs.csv", index=False)
    check = pd.read_csv(os.path.join(ROOT, "results", "percentile", "percentile_baseline_results.csv"))
    same = np.allclose(results[results.variant == "raw"]["sharpe_net"].values, check["sharpe_net"].values)
    print(f"Raw variant matches percentile_momentum.py results: {same}\n")

    base = results[results.scenario == "base"].set_index("variant")
    table = pd.DataFrame({v: {**base.loc[v, ["ann_return", "ann_vol", "sharpe_net", "sharpe_gross",
                                             "max_drawdown"]].to_dict(),
                              "annual_one_way_turnover": base.loc[v, "avg_daily_turnover"] * 252,
                              **portfolio_profile(weights[v], groups, dev_dates)}
                          for v in VARIANTS}).T
    for c in ["ann_return", "sharpe_net", "annual_one_way_turnover"]:
        table[f"{c}_vs_raw"] = table[c] - table.loc["raw", c]
    table.index.name = "variant"
    table.to_csv(f"{OUT}/three_variant_table.csv")
    print("Three-variant table (spec 10.4), base costs:")
    print(table.T.to_string(float_format=lambda x: f"{x:.4f}"))

    # 2. Sector counts and weights on every rebalance date
    sw = pd.concat([sector_weights_table(weights[v], eligible, groups, dev_dates, v) for v in VARIANTS])
    sw.to_csv(f"{OUT}/sector_weights_by_date.csv", index=False)
    avg = sw.groupby(["variant", "sector"])[["n_eligible", "n_long", "n_short", "long_weight",
                                             "short_weight", "net_weight"]].mean()
    univ = avg.loc["raw", "n_eligible"] / avg.loc["raw", "n_eligible"].sum()
    sides = pd.DataFrame({"share_of_universe": univ,
                          "raw_long_weight": avg.loc["raw", "long_weight"],
                          "raw_short_weight": avg.loc["raw", "short_weight"],
                          "raw_net_weight": avg.loc["raw", "net_weight"]}).sort_values("raw_net_weight")
    print("\nRaw variant: average sector weight in each leg vs share of the eligible universe:")
    print(sides.to_string(float_format=lambda x: f"{x:.3f}"))

    # 3. Sector returns and contribution over time; allocation vs selection
    prev_elig = eligible.shift(1, fill_value=False)
    sector_ret = returns.where(prev_elig).T.groupby(groups).mean().T        # equal weight
    dev_ret = sector_ret.loc[DEV_START:DEV_END]
    monthly = (1 + dev_ret.fillna(0)).groupby(dev_ret.index.to_period("M")).prod() - 1
    monthly.to_csv(f"{OUT}/sector_returns_monthly.csv")
    contrib_cum, contrib_daily, annual_rows, split_rows = {}, {}, [], {}
    for v in VARIANTS:
        contrib, annual, split = sector_contributions(targets[v], returns, groups, sector_ret)
        contrib_daily[v] = contrib
        contrib_cum[v] = contrib.cumsum()
        annual_rows.append(annual.assign(variant=v))
        split_rows[v] = split
    cum = pd.concat({v: c.groupby(c.index.to_period("M")).tail(1) for v, c in contrib_cum.items()},
                    names=["variant", "date"])
    cum.to_csv(f"{OUT}/sector_contribution_cumulative.csv")
    annual = pd.concat(annual_rows).reset_index().rename(columns={"index": "sector"})
    annual.to_csv(f"{OUT}/sector_contribution_annual.csv", index=False)
    split = pd.DataFrame(split_rows).T
    split.index.name = "variant"
    split.to_csv(f"{OUT}/allocation_vs_selection.csv")
    print("\nGross return split (per year): sector allocation (rotation) vs stock selection:")
    print(split.to_string(float_format=lambda x: f"{x:+.4f}"))
    print("\nRaw variant: gross contribution by sector (per year):")
    print(annual[annual.variant == "raw"].set_index("sector").drop(columns="variant")
          .sort_values("total_per_yr").to_string(float_format=lambda x: f"{x:+.4f}"))

    # 4. Suitability by sector: breadth and turnover (raw), and single-sector performance
    turn = sector_turnover(targets["raw"], groups)
    single = []
    for s in sectors:
        e = eligible & (groups == s).values
        mom = momentum_scores(sp, e, dates)
        side = tails(mom.rank(axis=1, method="first", pct=True))
        n_l, n_s = (side.loc[dev_dates] > 0).sum(axis=1), (side.loc[dev_dates] < 0).sum(axis=1)
        tradeable = ((n_l >= MIN_NAMES) & (n_s >= MIN_NAMES)).mean()
        row = {"sector": s, "avg_eligible": e.loc[dev_dates].sum(axis=1).mean(),
               "max_names_per_side": int(min(n_l.max(), n_s.max())),
               "share_of_months_tradeable": tradeable}
        if tradeable == 1:
            row.update(summarize(backtest(daily_target(to_weights(side, MIN_NAMES), sp.index),
                                          returns, cost, borrow)))
            row["status"] = "ok"
        else:
            row["status"] = f"too small (< {MIN_NAMES} names per side)"
        single.append(row)
    single = pd.DataFrame(single).set_index("sector")
    single.to_csv(f"{OUT}/single_sector_performance.csv")
    suit = single[["avg_eligible", "max_names_per_side", "status"]].join(turn) \
        .join(avg.loc["raw", ["n_long", "n_short"]].rename(columns=lambda c: f"raw_avg_{c}"))
    suit["turnover_per_unit_weight"] = suit["share_of_turnover"] / suit["share_of_gross_weight"]
    suit.to_csv(f"{OUT}/sector_suitability.csv")
    print("\nSector suitability (raw variant turnover; single-sector test at base costs):")
    print(suit.sort_values("avg_eligible").to_string(float_format=lambda x: f"{x:.3f}"))

    # 5. Raw vs within-sector ranks
    pr, pw = pct_raw.loc[dev_dates], pct_within.loc[dev_dates]
    lr, lw_, sr, sw_ = pr >= TOP, pw >= TOP, pr <= BOTTOM, pw <= BOTTOM
    ranks = pd.DataFrame({"rank_correlation": pr.corrwith(pw, axis=1, method="spearman"),
                          "long_overlap": (lr & lw_).sum(axis=1) / lr.sum(axis=1),
                          "short_overlap": (sr & sw_).sum(axis=1) / sr.sum(axis=1)})
    ranks.to_csv(f"{OUT}/rank_comparison_by_date.csv")
    print("\nRaw vs within-sector ranks (average over rebalance dates; overlap = share of raw "
          "names also chosen within sector):")
    print(ranks.mean().to_string(float_format=lambda x: f"{x:.3f}"))

    # 6. Leave-one-sector-out (spec 10.2): all three variants, base costs
    loso = []
    for s in ["none"] + sectors:
        e = eligible if s == "none" else eligible & (groups != s).values
        w_s, _, _ = build_variants(sp, e, groups, dates)
        for v in VARIANTS:
            m = summarize(backtest(daily_target(w_s[v], sp.index), returns, cost, borrow))
            loso.append({"sector_removed": s, "variant": v, **m})
    loso = pd.DataFrame(loso)
    full = loso[loso.sector_removed == "none"].set_index("variant")
    loso["sharpe_net_change"] = loso["sharpe_net"] - loso["variant"].map(full["sharpe_net"])
    loso["ann_return_change"] = loso["ann_return"] - loso["variant"].map(full["ann_return"])
    loso.to_csv(f"{OUT}/leave_one_sector_out.csv", index=False)
    print("\nLeave-one-sector-out, net Sharpe at base costs (change vs full universe):")
    show = loso.pivot(index="sector_removed", columns="variant", values="sharpe_net")[VARIANTS]
    chg = loso.pivot(index="sector_removed", columns="variant", values="sharpe_net_change")[VARIANTS]
    print(show.round(3).astype(str).add(" (").add(chg.map(lambda x: f"{x:+.3f}")).add(")")
          .to_string())

    # 7. MA 50/200: cyclical vs defensive sectors (spec 10.5)
    labels = pd.Series(np.where(groups.isin(DEFENSIVE), "defensive", "cyclical"), index=groups.index)
    ma_full = ma_weights(sp, eligible)
    ma_contrib = leg_contributions(ma_full, returns, labels)
    ma_contrib.index.name = "group"
    ma_contrib.to_csv(f"{OUT}/ma_group_contribution.csv")
    ma_rows = []
    for g in ["all sectors", "cyclical", "defensive"]:
        e = eligible if g == "all sectors" else eligible & (labels == g).values
        t = ma_full if g == "all sectors" else ma_weights(sp, e)
        m = summarize(backtest(t, returns, cost, borrow))
        ma_rows.append({"group": g, "ann_return": m["ann_return"], "ann_vol": m["ann_vol"],
                        "sharpe_net": m["sharpe_net"], "max_drawdown": m["max_drawdown"],
                        "avg_eligible": e.loc[DEV_START:DEV_END].sum(axis=1).mean(), **cash_days(t)})
    ma_groups = pd.DataFrame(ma_rows).set_index("group")
    ma_groups.to_csv(f"{OUT}/ma_by_group.csv")
    print(f"\nMA 50/200, cyclical vs defensive (defensive = {', '.join(DEFENSIVE)}), base costs")
    print("Gross contribution per year inside the full MA strategy:")
    print(ma_contrib.to_string(float_format=lambda x: f"{x:+.4f}"))
    print("MA run on each group alone:")
    print(ma_groups.to_string(float_format=lambda x: f"{x:.3f}"))

    # 8. Stability over time (spec 10.5): MA and the three 12-1 variants, base costs
    nets = {"MA 50/200": backtest(ma_full, returns, cost, borrow)["net"]}
    nets.update({f"12-1 {v}": backtest(targets[v], returns, cost, borrow)["net"] for v in VARIANTS})
    yearly, sub = performance_by_period(nets)
    yearly.to_csv(f"{OUT}/yearly_performance.csv", index=False)
    sub.to_csv(f"{OUT}/subperiod_performance.csv", index=False)
    print("\nNet return by year, base costs:")
    print(yearly.pivot(index="year", columns="strategy", values="net_return")[list(nets)]
          .to_string(float_format=lambda x: f"{x:+.1%}"))
    print("Sharpe by year:")
    print(yearly.pivot(index="year", columns="strategy", values="sharpe")[list(nets)]
          .to_string(float_format=lambda x: f"{x:+.2f}"))
    print("Subperiods (annualised return, Sharpe):")
    print(sub.pivot(index="strategy", columns="period", values=["ann_return", "sharpe"])
          .loc[list(nets)].to_string(float_format=lambda x: f"{x:+.3f}"))

    raw_c = contrib_daily["raw"]
    by_year = raw_c.groupby(raw_c.index.year).sum()                # sum of daily contributions
    by_year["Total"] = by_year.sum(axis=1)
    by_year.index.name = "year"
    by_year.to_csv(f"{OUT}/raw_sector_contribution_by_year.csv")
    crisis = by_year.loc[CRISIS_YEARS].sum()
    other = by_year.drop(index=CRISIS_YEARS).sum()
    crisis_tbl = pd.DataFrame({"contribution_2008_2009": crisis, "contribution_other_years": other,
                               "contribution_2006_2014": crisis + other})
    crisis_tbl["share_from_2008_2009"] = crisis_tbl["contribution_2008_2009"] / \
        crisis_tbl["contribution_2006_2014"]
    crisis_tbl.index.name = "sector"
    crisis_tbl.to_csv(f"{OUT}/raw_sector_contribution_2008_2009.csv")
    print("\n12-1 raw: gross sector contribution by year (sum of daily returns):")
    print(by_year.T.to_string(float_format=lambda x: f"{x:+.3f}"))
    print("Contribution from 2008-2009 vs other years (share is only meaningful when both have "
          "the same sign):")
    print(crisis_tbl.to_string(float_format=lambda x: f"{x:+.3f}"))

    raw_net = sw[sw.variant == "raw"].pivot(index="date", columns="sector", values="net_weight")
    save_figures(contrib_cum, raw_net, loso)
    print(f"\nAll tables and figures saved in {OUT}/")


if __name__ == "__main__":
    main()
