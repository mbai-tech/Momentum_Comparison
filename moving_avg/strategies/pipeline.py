import io
import os

import numpy as np
import pandas as pd
import requests

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent   # moving_avg/ (this file is in strategies/)
RAW_DIR = ROOT / "data" / "raw"                    # cached downloads (prices, volume, SPY, sectors)
QUALITY_DIR = ROOT / "results" / "data_quality"   # data-cleaning checklist outputs

DATA_START, DATA_END = "2005-01-01", "2025-12-31"
DEV_START, DEV_END = "2006-01-01", "2014-12-31"

MIN_HISTORY = 252
MIN_PRICE = 5.0
MIN_DOLLAR_VOLUME = 10e6
MIN_NAMES = 20

MOM_LOOKBACK, MOM_SKIP = 252, 21
TOP, BOTTOM = 0.80, 0.20

COST_SCENARIOS = {"frictionless": (0, 0), "low": (5, 25), "base": (10, 50), "stress": (20, 200)}

BENCHMARK = "SPY"   # S&P 500 ETF; adjusted close includes dividends

# ---- Data cleaning rules (spec 3.3-3.4). Fixed in advance; do not tune to results.
# Duplicate share classes of the same company: keep the Class A (voting) share.
DUPLICATE_SHARE_CLASSES = {"GOOG": "GOOGL", "FOX": "FOXA", "NWS": "NWSA"}   # drop -> keep
EXTREME_RETURN = 0.50     # flag daily moves larger than +/-50% for manual inspection
STALE_DAYS = 5            # flag runs of 5+ identical closing prices in a row
WINSORIZE = False         # baseline: no winsorising; extremes are flagged and counted instead
# Known stock splits: adjusted prices should show NO large move on these dates
SPLIT_CHECKS = [("AAPL", "2014-06-09", "7-for-1"), ("AAPL", "2020-08-31", "4-for-1"),
                ("TSLA", "2020-08-31", "5-for-1"), ("AMZN", "2022-06-06", "20-for-1"),
                ("GOOGL", "2022-07-18", "20-for-1"), ("NVDA", "2024-06-10", "10-for-1")]


def get_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    table = pd.read_html(io.StringIO(html))[0]
    return sorted(table["Symbol"].str.replace(".", "-").tolist())   # BRK.B -> BRK-B


def load_prices():
    files = [RAW_DIR / f for f in ["prices.pkl", "raw_close.pkl", "volume.pkl"]]
    if all(os.path.exists(f) for f in files):
        return pd.read_pickle(RAW_DIR / "prices.pkl"), pd.read_pickle(RAW_DIR / "volume.pkl")
    import yfinance as yf
    raw = yf.download(get_tickers(), start=DATA_START, end=DATA_END,
                      auto_adjust=False, actions=True, progress=False)
    prices, volume = raw["Adj Close"], raw["Volume"]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    prices.to_pickle(RAW_DIR / "prices.pkl")
    volume.to_pickle(RAW_DIR / "volume.pkl")
    unadjusted_close(raw["Close"], raw["Stock Splits"]).to_pickle(RAW_DIR / "raw_close.pkl")
    return prices, volume


def unadjusted_close(split_adjusted_close, stock_splits):
    ratio = stock_splits.replace(0, 1).fillna(1)                 # e.g. 4.0 on a 4-for-1 day
    later_splits = ratio[::-1].cumprod()[::-1].shift(-1).fillna(1)
    return split_adjusted_close * later_splits


def load_raw_close():
    if not os.path.exists(RAW_DIR / "raw_close.pkl"):
        raise FileNotFoundError("raw_close.pkl missing: delete prices.pkl and volume.pkl, "
                                "then run build_dataset.py to download again.")
    return pd.read_pickle(RAW_DIR / "raw_close.pkl")


def load_benchmark():
    if os.path.exists(RAW_DIR / "spy.pkl"):
        return pd.read_pickle(RAW_DIR / "spy.pkl")
    import yfinance as yf
    spy = yf.download(BENCHMARK, start=DATA_START, end=DATA_END,
                      auto_adjust=True, progress=False)["Close"].squeeze()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    spy.to_pickle(RAW_DIR / "spy.pkl")
    return spy


def clean_data(prices, volume, spy):
    q = {}   # check name -> result

    # Check 1: date uniqueness (wide format: one row per date for every ticker)
    q["duplicate_dates_removed"] = int(prices.index.duplicated().sum())
    prices = prices[~prices.index.duplicated(keep="first")]
    volume = volume[~volume.index.duplicated(keep="first")]

    # Check 2: sort by date and ticker before any rolling calculation
    prices = prices.sort_index()[sorted(prices.columns)]
    volume = volume.reindex_like(prices)

    # Universe rule 3.3(2): one share class per company
    dupes = [t for t in DUPLICATE_SHARE_CLASSES if t in prices.columns]
    prices, volume = prices.drop(columns=dupes), volume.drop(columns=dupes)
    q["duplicate_share_classes_removed"] = ", ".join(dupes) if dupes else "none"

    # Tickers with no data at all
    empty = prices.columns[prices.notna().sum() == 0].tolist()
    q["tickers_with_no_data"] = ", ".join(empty) if empty else "none"
    q["tickers_with_data"] = int(prices.shape[1] - len(empty))
    q["first_date"], q["last_date"] = str(prices.index[0].date()), str(prices.index[-1].date())

    # Check 3a: zero or negative prices -> invalid, set to missing
    bad = prices <= 0
    q["zero_or_negative_prices_removed"] = int(bad.sum().sum())
    prices = prices.mask(bad)
    q["negative_volume_removed"] = int((volume < 0).sum().sum())
    volume = volume.mask(volume < 0)

    # Check 3b: implausibly large moves -> flag for inspection (not changed)
    r = prices.pct_change(fill_method=None)
    big = r.abs() > EXTREME_RETURN
    flagged = r[big].stack().dropna().rename("return").reset_index()
    flagged.columns = ["date", "ticker", "return"]
    flagged["price_before"] = [prices[t].shift(1).loc[d] for d, t in zip(flagged["date"], flagged["ticker"])]
    flagged["price_after"] = [prices.loc[d, t] for d, t in zip(flagged["date"], flagged["ticker"])]
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    flagged.to_csv(QUALITY_DIR / "flagged_returns.csv", index=False)
    q[f"returns_beyond_{EXTREME_RETURN:.0%}_flagged"] = int(big.sum().sum())
    # a spike that fully reverses the next day is the classic signature of a bad price
    spike = (r > EXTREME_RETURN) & (r.shift(-1) < -EXTREME_RETURN / (1 + EXTREME_RETURN))
    q["spike_then_reversal_flagged"] = int(spike.sum().sum())

    # Check 3c: stale prices (same close repeated) -> flag
    same = prices.eq(prices.shift(1)) & prices.notna()
    in_run = same.astype(int).rolling(STALE_DAYS - 1).sum() == STALE_DAYS - 1   # last N closes equal
    stale = in_run & ~in_run.shift(1, fill_value=False)                         # first day of each run
    stale_list = stale.stack()
    stale_list = stale_list[stale_list].reset_index().iloc[:, :2]
    stale_list.columns = ["date_run_reached_limit", "ticker"]
    stale_list.to_csv(QUALITY_DIR / "stale_prices.csv", index=False)
    q[f"stale_runs_of_{STALE_DAYS}plus_days_flagged"] = int(stale.sum().sum())

    # Check 4: prices adjusted for splits -> known split dates should show no big move
    rows = []
    for t, d, ratio in SPLIT_CHECKS:
        if t in prices.columns and pd.Timestamp(d) in r.index:
            move = r.loc[pd.Timestamp(d), t]
            rows.append({"ticker": t, "date": d, "split": ratio, "return": move,
                         "adjusted_ok": bool(abs(move) < 0.25) if pd.notna(move) else None})
    splits = pd.DataFrame(rows)
    splits.to_csv(QUALITY_DIR / "split_checks.csv", index=False)
    q["known_splits_checked"] = len(splits)
    q["known_splits_adjusted_ok"] = int(splits["adjusted_ok"].sum()) if len(splits) else 0

    # Checks 5-6: no forward-filling; gaps inside a stock's history are counted, not filled
    inside = prices.notna().cummax() & prices[::-1].notna().cummax()[::-1]   # between first and last price
    q["missing_prices_inside_history"] = int((prices.isna() & inside).sum().sum())
    q["gap_rule"] = "no filling; stock ineligible until its rolling windows refill"

    # Check 7: full lookback required -> enforced later (rolling windows, 252-day history)
    q["full_lookback_required"] = "yes (rolling windows need complete data; 252-day history)"

    # Check 8: extreme returns only with a pre-specified symmetric rule
    q["winsorizing"] = "off (extremes flagged and counted above)" if not WINSORIZE else "on"

    # Check 9: benchmark on the same calendar
    spy = spy[~spy.index.duplicated(keep="first")].sort_index()
    q["benchmark_dates_missing_vs_stocks"] = int(len(prices.index.difference(spy.index)))
    spy = spy.reindex(prices.index)

    # Check 10: save the data-quality summary BEFORE any backtest
    summary = pd.Series(q, name="result")
    summary.index.name = "check"
    summary.to_csv(QUALITY_DIR / "data_quality_summary.csv")
    print("Data-quality summary (saved to results/data_quality/data_quality_summary.csv):")
    print(summary.to_string(), "\n")
    return prices, volume, spy


def to_weights(side, min_names=MIN_NAMES):
    longs = (side > 0).astype(float)
    shorts = (side < 0).astype(float)
    n_long, n_short = longs.sum(axis=1), shorts.sum(axis=1)
    w_long = longs.div(n_long.where(n_long >= min_names), axis=0)
    w_short = shorts.div(n_short.where(n_short >= min_names), axis=0)
    return w_long.fillna(0) - w_short.fillna(0)


def backtest(target, returns, cost_bps, borrow_bps):
    w = target.shift(1).fillna(0)
    held = w.shift(1).fillna(0)
    r = returns.fillna(0)
    gross = (held * r).sum(axis=1)
    traded = w.diff().abs().sum(axis=1)
    trade_cost = traded * cost_bps / 10_000
    borrow_cost = held.clip(upper=0).abs().sum(axis=1) * borrow_bps / 10_000 / 252
    return pd.DataFrame({"gross": gross,
                         "net": gross - trade_cost - borrow_cost,
                         "turnover": traded / 2,
                         "gross_exposure": held.abs().sum(axis=1),
                         "net_exposure": held.sum(axis=1)})


def summarize(bt):
    bt = bt.loc[DEV_START:DEV_END]
    net = bt["net"]
    equity = (1 + net).cumprod()
    return {
        "ann_return": equity.iloc[-1] ** (252 / len(net)) - 1,
        "ann_vol": net.std() * np.sqrt(252),
        "sharpe_net": net.mean() / net.std() * np.sqrt(252),
        "sharpe_gross": bt["gross"].mean() / bt["gross"].std() * np.sqrt(252),
        "max_drawdown": (equity / equity.cummax() - 1).min(),
        "avg_daily_turnover": bt["turnover"].mean(),
        "avg_gross_exposure": bt["gross_exposure"].mean(),
        "avg_net_exposure": bt["net_exposure"].mean(),
    }
