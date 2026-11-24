import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "strategies"))
from build_dataset import DATA_DIR, latest_version
from pipeline import MIN_HISTORY, MIN_PRICE, MIN_DOLLAR_VOLUME, EXTREME_RETURN, ROOT

OUT = str(ROOT / "results" / "eda")
SEED = 42                 # fixed, so the "random" samples are reproducible
N_SAMPLES = 5             # cases per event type in step 6
WINDOW = 15               # trading days shown either side of each event


def load_long_panel(version=None):
    if version is None:
        version = latest_version()
    long = pd.read_pickle(f"{DATA_DIR}/clean_panel_v{version}.pkl")
    long["ticker"] = long["ticker"].astype(str)
    long["sector"] = long["sector"].astype(str)
    return long, version


def wide(long, col):
    return long.pivot(index="date", columns="ticker", values=col).sort_index().sort_index(axis=1)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}", dpi=150)
    plt.close(fig)


def step1_eligible(px, eligible):
    counts = pd.DataFrame({"with_price": px.notna().sum(axis=1),
                           "eligible": eligible.sum(axis=1)})
    counts.to_csv(f"{OUT}/eligible_count_by_date.csv")
    fig, ax = plt.subplots(figsize=(9, 4))
    counts["with_price"].plot(ax=ax, label="Stocks with a price")
    counts["eligible"].plot(ax=ax, label="Eligible (history, price, liquidity filters)")
    ax.set_title("Number of securities through time")
    ax.set_ylabel("Stocks"); ax.legend(); ax.grid(alpha=0.3)
    save(fig, "eligible_count.png")
    yearly = counts.groupby(counts.index.year).mean().round(0)
    return {"eligible_min": int(counts["eligible"].min()),
            "eligible_min_date": str(counts["eligible"].idxmin().date()),
            "eligible_max": int(counts["eligible"].max()),
            "eligible_mean": round(float(counts["eligible"].mean()), 1)}, yearly


def step2_missing(long, px):
    # "Inside history" = between a stock's first and last price. Missing there is a real gap;
    # before the first price (not yet listed) is expected and not counted as missing.
    first = px.notna().idxmax()
    last = px[::-1].notna().idxmax()
    inside = pd.DataFrame({t: (px.index >= first[t]) & (px.index <= last[t]) for t in px.columns},
                          index=px.index)
    gap = inside & px.isna()

    fields = ["adj_close", "raw_close", "volume", "return", "dollar_volume", "adv20_dollar"]
    by_field = pd.DataFrame({
        "missing_rows": [int(long[f].isna().sum()) for f in fields],
        "share_of_rows": [long[f].isna().mean() for f in fields]}, index=fields)
    by_field.loc["adj_close (gaps inside history)", "missing_rows"] = int(gap.sum().sum())
    by_field.to_csv(f"{OUT}/missing_by_field.csv")

    by_stock = pd.DataFrame({"first_date": first.dt.date, "last_date": last.dt.date,
                             "days_inside_history": inside.sum(),
                             "missing_days_inside": gap.sum()})
    by_stock["share_missing"] = by_stock["missing_days_inside"] / by_stock["days_inside_history"]
    by_stock.sort_values("missing_days_inside", ascending=False).to_csv(f"{OUT}/missing_by_stock.csv")

    by_date = gap.sum(axis=1).rename("stocks_with_gap")
    by_date.to_csv(f"{OUT}/missing_by_date.csv")
    fig, ax = plt.subplots(figsize=(9, 3.5))
    by_date.plot(ax=ax)
    ax.set_title("Stocks missing a price inside their own history, by date")
    ax.set_ylabel("Stocks"); ax.grid(alpha=0.3)
    save(fig, "missing_by_date.png")

    # zero volume while a price exists is suspicious (stale or bad data)
    zero_vol = int(((long["volume"] == 0) & long["adj_close"].notna()).sum())
    return {"gap_days_total": int(gap.sum().sum()),
            "stocks_with_any_gap": int((by_stock["missing_days_inside"] > 0).sum()),
            "dates_with_any_gap": int((by_date > 0).sum()),
            "zero_volume_rows_with_price": zero_vol,
            "stocks_starting_after_2005": int((first > px.index[0]).sum())}, by_stock


def step3_returns(long):
    r = long["return"].dropna()
    q = r.quantile([0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999])
    stats = {"n_daily_returns": int(len(r)), "mean": r.mean(), "std": r.std(),
             "skew": r.skew(), "excess_kurtosis": r.kurt(),
             **{f"p{k*100:g}": v for k, v in q.items()},
             f"n_beyond_{EXTREME_RETURN:.0%}": int((r.abs() > EXTREME_RETURN).sum()),
             "n_beyond_20%": int((r.abs() > 0.20).sum())}

    extremes = long.loc[r.abs().sort_values(ascending=False).index[:30],
                        ["date", "ticker", "sector", "return", "adj_close", "raw_close", "volume"]]
    extremes.to_csv(f"{OUT}/extreme_returns_top30.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(r.clip(-0.25, 0.25), bins=200)
    axes[0].set_title("Daily returns, all stocks (clipped at ±25% for display)")
    axes[0].set_xlabel("Daily return")
    axes[1].hist(r, bins=300, log=True)
    axes[1].set_title("Daily returns, full range (log count)")
    axes[1].set_xlabel("Daily return")
    save(fig, "return_distribution.png")

    yearly = long.dropna(subset=["return"]).groupby(long["date"].dt.year)["return"].agg(
        ["std", lambda x: (x.abs() > 0.2).sum()])
    yearly.columns = ["daily_std", "moves_beyond_20pct"]
    fig, ax = plt.subplots(figsize=(9, 3.5))
    (yearly["daily_std"] * 100).plot.bar(ax=ax)
    ax.set_title("Cross-sectional daily return volatility by year")
    ax.set_ylabel("Std of daily returns (%)")
    save(fig, "return_volatility_by_year.png")
    return stats, extremes

def step4_sectors(long):
    el = long[long["eligible"]]
    month = el["date"].dt.to_period("M")
    counts = el.groupby([month, "sector"])["ticker"].nunique().unstack(fill_value=0)
    dv = el.groupby([month, "sector"])["dollar_volume"].sum().unstack(fill_value=0)
    dv_share = dv.div(dv.sum(axis=1), axis=0)
    counts.to_csv(f"{OUT}/sector_counts_by_month.csv")
    dv_share.to_csv(f"{OUT}/sector_dollar_volume_share_by_month.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    counts.index = counts.index.to_timestamp()
    dv_share.index = dv_share.index.to_timestamp()
    counts.plot.area(ax=axes[0], legend=False, linewidth=0)
    axes[0].set_title("Eligible stocks by sector")
    axes[0].set_ylabel("Stocks")
    dv_share.plot.area(ax=axes[1], linewidth=0)
    axes[1].set_title("Sector share of dollar volume\n(proxy for market value; shares outstanding not available)")
    axes[1].set_ylabel("Share"); axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    save(fig, "sectors_through_time.png")

    summary = pd.DataFrame({"avg_eligible_stocks": counts.mean(),
                            "first_month": counts.iloc[0], "last_month": counts.iloc[-1],
                            "avg_dollar_volume_share": dv_share.mean()}).round(3)
    summary.to_csv(f"{OUT}/sector_summary.csv")
    return summary


def step5_liquidity(long):
    has_px = long[long["adj_close"].notna()]
    fail_hist = has_px["history_days"] < MIN_HISTORY
    fail_price = has_px["raw_close"] < MIN_PRICE
    fail_liq = ~(has_px["adv20_dollar"] >= MIN_DOLLAR_VOLUME)          # missing ADV counts as fail
    removed = pd.DataFrame({"history": fail_hist, "price": fail_price,
                            "liquidity": fail_liq, "any_filter": fail_hist | fail_price | fail_liq})
    by_year = removed.groupby(has_px["date"].dt.year).mean()
    by_year.to_csv(f"{OUT}/share_removed_by_filter_by_year.csv")

    fig, ax = plt.subplots(figsize=(9, 3.5))
    (by_year * 100).plot(ax=ax, marker="o")
    ax.set_title("Share of stock-days with a price that each filter removes")
    ax.set_ylabel("% of stock-days"); ax.grid(alpha=0.3)
    save(fig, "filters_removed_share.png")

    adv = has_px["adv20_dollar"].dropna()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.hist(np.log10(adv[adv > 0]), bins=120)
    ax.axvline(np.log10(MIN_DOLLAR_VOLUME), color="red", ls="--",
               label=f"Filter: ${MIN_DOLLAR_VOLUME/1e6:.0f}M")
    ax.set_title("20-day average daily dollar volume, all stock-days")
    ax.set_xlabel("log10(dollars)"); ax.legend()
    save(fig, "liquidity_distribution.png")

    overall = removed.mean()
    return {f"share_removed_{k}": round(float(v), 4) for k, v in overall.items()} | \
           {"adv20_median_$M": round(float(adv.median()) / 1e6, 1),
            "adv20_p05_$M": round(float(adv.quantile(0.05)) / 1e6, 1)}


def step6_samples(long, px, raw, by_stock, extremes):
    rng = np.random.default_rng(SEED)
    events = []

    # Splits: days where raw_close / adj_close jumps by more than 1.5x (a split changes the ratio)
    ratio = raw / px
    jump = ratio / ratio.shift(1)
    splits = jump[(jump > 1.5) | (jump < 1 / 1.5)].stack().dropna().reset_index()
    splits.columns = ["date", "ticker", "factor"]
    for _, row in splits.sample(min(N_SAMPLES, len(splits)), random_state=SEED).iterrows():
        events.append(("split", row["ticker"], row["date"], f"ratio change x{row['factor']:.2f}"))

    # Large returns: random draws from all moves beyond EXTREME_RETURN
    big = long[long["return"].abs() > EXTREME_RETURN]
    for _, row in big.sample(min(N_SAMPLES, len(big)), random_state=SEED).iterrows():
        events.append(("large_return", row["ticker"], row["date"], f"return {row['return']:+.1%}"))

    # Mergers / ticker changes / new listings show up as stocks whose data START after 2005
    # (this dataset cannot see stocks that were acquired and disappeared -- survivorship bias)
    late = by_stock[pd.to_datetime(by_stock["first_date"]) > px.index[0]]
    for t in rng.choice(late.index, min(N_SAMPLES, len(late)), replace=False):
        events.append(("late_start", t, pd.Timestamp(late.loc[t, "first_date"]),
                       "first price in dataset (IPO, spin-off, merger or ticker change)"))

    # Stocks whose data stop before the end (delisted, acquired or renamed)
    early = by_stock[pd.to_datetime(by_stock["last_date"]) < px.index[-1]]
    for t in rng.choice(early.index, min(N_SAMPLES, len(early)), replace=False):
        events.append(("early_end", t, pd.Timestamp(early.loc[t, "last_date"]), "last price in dataset"))

    rows = []
    for kind, t, d, note in events:
        i = px.index.get_loc(d)
        lo, hi = max(i - WINDOW, 0), min(i + WINDOW, len(px.index) - 1)
        seg_adj, seg_raw = px[t].iloc[lo:hi + 1], raw[t].iloc[lo:hi + 1]
        fig, ax = plt.subplots(figsize=(6, 3))
        seg_adj.plot(ax=ax, label="adj_close")
        seg_raw.plot(ax=ax, label="raw_close (traded)", ls="--")
        ax.axvline(d, color="red", lw=0.8)
        ax.set_title(f"{kind}: {t} on {d.date()} ({note})", fontsize=8)
        ax.legend(fontsize=7)
        fname = f"sample_{kind}_{t}_{d.date()}.png"
        save(fig, fname)
        r = px[t].pct_change(fill_method=None)
        rows.append({"type": kind, "ticker": t, "date": d.date(), "note": note,
                     "adj_before": seg_adj.iloc[0], "adj_after": seg_adj.iloc[-1],
                     "max_abs_daily_return_in_window": r.iloc[lo:hi + 1].abs().max(),
                     "figure": fname,
                     "your_verdict": ""})      # fill in: real event / data error / unclear
    samples = merge_verdicts(pd.DataFrame(rows), f"{OUT}/event_samples.csv")
    return samples, len(splits), len(big), len(late), len(early)


def merge_verdicts(samples, path):
    key = ["type", "ticker", "date"]
    samples["date"] = samples["date"].astype(str)
    if os.path.exists(path):
        old = pd.read_csv(path, dtype=str, keep_default_na=False)
        reviewed = old[old["your_verdict"].str.strip() != ""]
        verdict = dict(zip(map(tuple, reviewed[key].values), reviewed["your_verdict"]))
        samples["your_verdict"] = [verdict.get(k, "") for k in map(tuple, samples[key].values)]
        sampled = set(map(tuple, samples[key].values))
        kept = reviewed[[k not in sampled for k in map(tuple, reviewed[key].values)]]
        samples = pd.concat([samples.astype(str).replace("nan", ""), kept], ignore_index=True)
    samples.to_csv(path, index=False)
    return samples


def main():
    os.makedirs(OUT, exist_ok=True)
    long, version = load_long_panel()
    print(f"Exploratory data analysis of clean dataset version {version}\n")
    px, raw = wide(long, "adj_close"), wide(long, "raw_close")
    eligible = wide(long, "eligible").reindex_like(px).fillna(False).astype(bool)

    s1, yearly_counts = step1_eligible(px, eligible)
    print("1. Eligible securities:", s1)
    print("   Average per year (stocks with a price / eligible):")
    print(yearly_counts.T.to_string())

    s2, by_stock = step2_missing(long, px)
    print("\n2. Missing values:", s2)
    print("   Stocks with the most missing days inside their history:")
    print(by_stock.sort_values("missing_days_inside", ascending=False).head(5).to_string())

    s3, extremes = step3_returns(long)
    print("\n3. Daily return distribution:")
    print(pd.Series(s3).to_string(float_format=lambda x: f"{x:.4f}"))
    print("   Ten largest moves (also in extreme_returns_top30.csv):")
    print(extremes.head(10).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    s4 = step4_sectors(long)
    print("\n4. Sectors (eligible stocks; dollar-volume share as market-value proxy):")
    print(s4.to_string())

    s5 = step5_liquidity(long)
    print("\n5. Liquidity and filters (share of stock-days with a price removed):", s5)

    samples, n_split, n_big, n_late, n_early = step6_samples(long, px, raw, by_stock, extremes)
    print(f"\n6. Events found: {n_split} splits, {n_big} moves beyond {EXTREME_RETURN:.0%}, "
          f"{n_late} late starts, {n_early} early ends. Random sample (seed {SEED}):")
    print(samples[["type", "ticker", "date", "note"]].to_string(index=False))

    summary = pd.Series({**s1, **s2, **{f"returns_{k}": v for k, v in s3.items()}, **s5,
                         "splits_detected": n_split, "late_starts": n_late,
                         "early_ends": n_early}, name="value")
    summary.to_csv(f"{OUT}/eda_summary.csv")
    print(f"\nAll tables and figures saved in {OUT}/. Open each sample_*.png, decide whether it is a "
          "real event or a data error, and write it in the 'your_verdict' column of event_samples.csv.")


if __name__ == "__main__":
    main()
