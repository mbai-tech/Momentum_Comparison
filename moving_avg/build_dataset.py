import datetime
import glob
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "strategies"))
from pipeline import (load_prices, load_raw_close, load_benchmark, clean_data, DUPLICATE_SHARE_CLASSES,
                      MIN_HISTORY, MIN_PRICE, MIN_DOLLAR_VOLUME, DATA_START, DATA_END,
                      DEV_START, DEV_END, ROOT, RAW_DIR)

DATA_DIR = str(ROOT / "data")
ADV_WINDOW = 20                                       # days in the dollar-volume average
# Correction E009 (spec 3.4, pre-specified): an adjusted return is invalid when it is larger than
# this in absolute value AND has the opposite sign to the raw-close return on the same day
INVALID_RETURN_THRESHOLD = 0.20
E009_RULE = ("E009: return set to NaN where |adj return| > 20% and the adj return and raw-close "
             "return have opposite signs (mis-adjusted corporate action)")
# Correction E010: exemption from E009 for real moves on split days. A flagged return is restored
# only if Yahoo records a split for that ticker on that date AND the raw_close / adj_close ratio
# changes that day by the split factor (1 / Yahoo split value) within this tolerance
SPLIT_TOLERANCE = 0.02
SPLITS_CACHE = os.path.join(DATA_DIR, "splits_checked.csv")
E010_RULE = ("E010: an E009-flagged return is restored if Yahoo records a split on that ticker and "
             "date AND raw_close/adj_close changes that day by the split factor within 2%")
VERIFY_TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ"]
VERIFY_DATES = ["2008-03-03", "2010-06-01", "2013-09-03"]   # inside the development sample

DATA_DICTIONARY = [
    ("date", "date", "Trading date"),
    ("ticker", "string", "Yahoo Finance ticker; dots replaced by dashes (BRK.B -> BRK-B)"),
    ("adj_close", "float", "Split- and dividend-adjusted closing price (Yahoo 'Adj Close'); "
                           "used for returns"),
    ("raw_close", "float", "Price that actually traded that day (Yahoo 'Close' with later "
                           "splits undone); used only for the price filter"),
    ("volume", "float", "Shares traded that day"),
    ("return", "float", "Simple daily return adj_close[t] / adj_close[t-1] - 1; "
                        "missing if the previous price is missing (no filling), or if marked "
                        "invalid by correction E009 and not exempted by E010 "
                        "(see data/invalid_returns.csv)"),
    ("signal_close", "float", "Price used for signals: the stock's first adj_close chained with "
                              "its daily returns, counting invalid returns as 0%; equals "
                              "adj_close until the stock's first invalid return"),
    ("dollar_volume", "float", "adj_close x volume"),
    ("adv20_dollar", "float", "Average dollar_volume over the last 20 trading days (past data only)"),
    ("history_days", "int", "Number of days with a valid price up to and including this date"),
    ("sector", "string", "Current GICS sector from Wikipedia (not point-in-time)"),
    ("in_universe", "bool", "Member of the investment universe on this date "
                            "(current S&P 500 list: True for every listed stock)"),
    ("eligible", "bool", f"history_days >= {MIN_HISTORY} AND raw_close >= {MIN_PRICE} AND "
                         f"adv20_dollar >= {MIN_DOLLAR_VOLUME:,.0f} AND in_universe"),
]

def load_sectors():
    if os.path.exists(RAW_DIR / "sectors.csv"):
        return pd.read_csv(RAW_DIR / "sectors.csv", index_col=0)["sector"]
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    table = pd.read_html(io.StringIO(html))[0]
    sectors = pd.Series(table["GICS Sector"].values,
                        index=table["Symbol"].str.replace(".", "-"), name="sector")
    sectors.index.name = "ticker"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    sectors.to_csv(RAW_DIR / "sectors.csv")
    return sectors


def yahoo_splits_on(pairs):
    cols = ["ticker", "date", "yahoo_split", "all_splits_on_record", "checked_on"]
    cache = pd.read_csv(SPLITS_CACHE, dtype={"ticker": str, "date": str}) \
        if os.path.exists(SPLITS_CACHE) else pd.DataFrame(columns=cols)
    known = set(zip(cache["ticker"], cache["date"]))
    new = []
    for t, d in pairs:
        if (t, d) in known:
            continue
        import yfinance as yf
        splits = yf.Ticker(t).splits
        on_day = [v for x, v in splits.items() if x.date().isoformat() == d]
        new.append({"ticker": t, "date": d, "yahoo_split": float(on_day[0]) if on_day else 0.0,
                    "all_splits_on_record": "; ".join(f"{x.date()}:{v:g}" for x, v in splits.items()),
                    "checked_on": datetime.date.today().isoformat()})
    if new:
        cache = pd.concat([cache, pd.DataFrame(new)], ignore_index=True)
        cache.to_csv(SPLITS_CACHE, index=False)
    return {(r.ticker, r.date): float(r.yahoo_split) for r in cache.itertuples()}


def chain_prices(prices, returns):
    first = prices.bfill().iloc[0]
    growth = (1 + returns.where(prices.notna()).fillna(0)).cumprod()
    return (growth * first).where(prices.notna())


def file_date(path):
    return datetime.date.fromtimestamp(os.path.getmtime(path)).isoformat() \
        if os.path.exists(path) else "unknown"


def version_key(version):
    date, _, n = version.partition("_")
    return date, int(n) if n else 1


def list_versions():
    files = glob.glob(f"{DATA_DIR}/clean_panel_v*.pkl")
    return sorted((os.path.basename(f)[len("clean_panel_v"):-len(".pkl")] for f in files),
                  key=version_key)


def latest_version():
    versions = list_versions()
    if not versions:
        raise FileNotFoundError("No clean dataset found. Run: python3 build_dataset.py")
    return versions[-1]


def next_version():
    base = datetime.date.today().isoformat()
    used = [version_key(v)[1] for v in list_versions() if version_key(v)[0] == base]
    return f"{base}_{max(used) + 1}" if used else base

def manual_verification(prices, raw_close, volume, wide):
    rows = []
    for t in [t for t in VERIFY_TICKERS if t in prices.columns]:
        for d in VERIFY_DATES:
            d = prices.index[prices.index.searchsorted(pd.Timestamp(d))]   # next trading day
            i = prices.index.get_loc(d)
            close, prev = prices[t].iloc[i], prices[t].iloc[i - 1]
            dv_last20 = (prices[t] * volume[t]).iloc[i - ADV_WINDOW + 1:i + 1]
            by_hand = {"return": close / prev - 1,
                       "dollar_volume": close * volume[t].iloc[i],
                       "adv20_dollar": dv_last20.mean()}
            row = {"ticker": t, "date": d.date(), "prev_adj_close": prev, "adj_close": close,
                   "raw_close": raw_close[t].iloc[i], "volume": volume[t].iloc[i]}
            for k, v in by_hand.items():
                row[f"{k}_stored"] = wide[k].loc[d, t]
                row[f"{k}_by_hand"] = v
                row[f"{k}_match"] = bool(np.isclose(wide[k].loc[d, t], v, equal_nan=True))
            row["eligible_stored"] = bool(wide["eligible"].loc[d, t])
            row["eligible_by_hand"] = bool(wide["history_days"].loc[d, t] >= MIN_HISTORY and
                                           raw_close[t].iloc[i] >= MIN_PRICE and
                                           by_hand["adv20_dollar"] >= MIN_DOLLAR_VOLUME)
            rows.append(row)
    return pd.DataFrame(rows)

def build():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Step 1: raw data (cached .pkl files; downloaded only if missing)
    prices, volume = load_prices()
    spy_raw = load_benchmark()
    sectors = load_sectors()

    # Step 3: identifiers already standardised at download (dots -> dashes);
    # clean_data() removes duplicate share classes and runs the spec 3.4 checklist
    prices, volume, spy = clean_data(prices, volume, spy_raw)
    raw_close = load_raw_close()
    raw_close = raw_close[~raw_close.index.duplicated(keep="first")].reindex_like(prices)
    raw_close = raw_close.where(prices.notna())          # same missing days as adj_close

    # Step 4: returns and dollar volume
    returns = prices.pct_change(fill_method=None)

    # Correction E009: a large adjusted return with the opposite sign to the traded-price return
    # means a corporate action (e.g. a spin-off) was adjusted incorrectly, so the adjusted move is
    # not real. Mark it invalid (NaN) and save the rows. Prices themselves are not changed.
    raw_returns = raw_close.pct_change(fill_method=None)
    invalid = (returns.abs() > INVALID_RETURN_THRESHOLD) & \
              (np.sign(returns) * np.sign(raw_returns) < 0)
    ratio = raw_close / prices
    ratio_change = ratio / ratio.shift(1)
    invalid_rows = pd.DataFrame({"adj_return": returns.where(invalid).stack(future_stack=True),
                                 "raw_return": raw_returns.where(invalid).stack(future_stack=True),
                                 "ratio_change": ratio_change.where(invalid).stack(future_stack=True)}
                                ).dropna(subset=["adj_return"]).reset_index()
    invalid_rows.columns = ["date", "ticker", "adj_return", "raw_return", "ratio_change"]

    # Correction E010: exempt flagged returns on real split days (see SPLIT_TOLERANCE)
    keys = [(t, d.date().isoformat()) for t, d in zip(invalid_rows["ticker"], invalid_rows["date"])]
    split_on = yahoo_splits_on(keys)
    invalid_rows["yahoo_split"] = [split_on[k] for k in keys]
    invalid_rows["split_factor"] = [1 / v if v > 0 else np.nan for v in invalid_rows["yahoo_split"]]
    invalid_rows["exempt_E010"] = (invalid_rows["yahoo_split"] > 0) & \
        ((invalid_rows["ratio_change"] / invalid_rows["split_factor"] - 1).abs() <= SPLIT_TOLERANCE)
    invalid_rows["status"] = np.where(invalid_rows["exempt_E010"], "restored (E010)", "invalid")
    invalid_rows.to_csv(f"{DATA_DIR}/invalid_returns.csv", index=False)
    for _, row in invalid_rows[invalid_rows["exempt_E010"]].iterrows():
        invalid.loc[row["date"], row["ticker"]] = False
    returns = returns.mask(invalid)
    n_invalid = int(invalid.sum().sum())
    print(f"E009: {len(invalid_rows)} adjusted returns flagged; E010: "
          f"{int(invalid_rows['exempt_E010'].sum())} restored on split days, {n_invalid} stay "
          f"invalid (saved to {DATA_DIR}/invalid_returns.csv)")
    print(invalid_rows.to_string(index=False), "\n")

    # Signal prices: adj_close with invalid returns counted as 0% (no jump on a bad day)
    signal_close = chain_prices(prices, returns)
    first_bad = invalid.any(axis=1)
    first_bad = first_bad.idxmax() if first_bad.any() else prices.index[-1] + pd.Timedelta(days=1)
    before = prices.index < first_bad
    assert np.allclose(signal_close[before], prices[before], rtol=1e-9, equal_nan=True), \
        "signal_close differs from adj_close before the first invalid return"

    dollar_volume = prices * volume
    adv20 = dollar_volume.rolling(ADV_WINDOW).mean()
    history = prices.notna().cumsum()

    # Step 5: point-in-time eligibility. Universe membership = current S&P 500 list, so every
    # stock is a member on every date. Replace this panel with historical membership to remove
    # survivorship bias (planned experiment E006).
    in_universe = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    # Price filter on the price that actually traded (raw_close), NOT the adjusted price:
    # adjusted prices are lowered by splits that happen later, which is look-ahead bias.
    eligible = (history >= MIN_HISTORY) & (raw_close >= MIN_PRICE) & \
               (adv20 >= MIN_DOLLAR_VOLUME) & in_universe

    wide = {"adj_close": prices, "raw_close": raw_close, "volume": volume, "return": returns,
            "signal_close": signal_close,
            "dollar_volume": dollar_volume, "adv20_dollar": adv20, "history_days": history,
            "in_universe": in_universe, "eligible": eligible}

    # Step 2: long format, one row per date-security pair that has a price
    long = pd.concat({k: v.stack(future_stack=True) for k, v in wide.items()}, axis=1)
    long.index.names = ["date", "ticker"]
    long = long[long["adj_close"].notna()].reset_index()
    long["sector"] = long["ticker"].map(sectors).fillna("Unknown")
    long["history_days"] = long["history_days"].astype(int)
    long["ticker"] = long["ticker"].astype("category")     # smaller file
    long["sector"] = long["sector"].astype("category")
    long = long[[c for c, _, _ in DATA_DICTIONARY]]
    assert not long.duplicated(["date", "ticker"]).any(), "duplicate date-ticker rows"

    # Step 4 (continued): verification sample
    verify = manual_verification(prices, raw_close, volume, wide)

    # Step 6: save a clean, versioned dataset + metadata
    version = next_version()
    long.to_pickle(f"{DATA_DIR}/clean_panel_v{version}.pkl")
    spy.rename("SPY").to_pickle(f"{DATA_DIR}/benchmark_v{version}.pkl")
    verify.to_csv(f"{DATA_DIR}/manual_verification_v{version}.csv", index=False)
    pd.DataFrame(DATA_DICTIONARY, columns=["field", "type", "description"]).to_csv(
        f"{DATA_DIR}/data_dictionary.csv", index=False)
    meta = {
        "version": version,
        "built_on": datetime.datetime.now().isoformat(timespec="seconds"),
        "raw_files_saved_on": {f: file_date(RAW_DIR / f) for f in ["prices.pkl", "volume.pkl",
                                                          "spy.pkl", "sectors.csv"]},
        "source": "Yahoo Finance via yfinance (auto_adjust=False, actions=True: Adj Close, Close, Stock Splits, Volume); universe and sectors from "
                  "Wikipedia 'List of S&P 500 companies' (current list)",
        "requested_sample": [DATA_START, DATA_END],
        "first_date": str(prices.index[0].date()), "last_date": str(prices.index[-1].date()),
        "n_trading_days": int(len(prices.index)),
        "n_tickers_with_data": int(long["ticker"].nunique()),
        "n_rows": int(len(long)),
        "rules": {"min_history_days": MIN_HISTORY, "min_price": MIN_PRICE,
                  "min_adv20_dollar": MIN_DOLLAR_VOLUME, "adv_window": ADV_WINDOW,
                  "price_filter_uses": "raw_close (actually traded price), not adj_close",
                  "duplicate_share_classes_dropped": DUPLICATE_SHARE_CLASSES,
                  "universe": "current S&P 500 list (not point-in-time)",
                  "missing_prices": "not filled", "risk_free": "not used",
                  "invalid_returns": E009_RULE,
                  "invalid_returns_exemption": E010_RULE,
                  "signal_prices": "signal_close = first adj_close chained with returns, "
                                   "invalid returns counted as 0%"},
        "n_flagged_returns_E009": int(len(invalid_rows)),
        "n_restored_returns_E010": int(invalid_rows["exempt_E010"].sum()),
        "n_invalid_returns": n_invalid,
    }
    with open(f"{DATA_DIR}/clean_panel_v{version}.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved data/clean_panel_v{version}.pkl: {meta['n_rows']:,} rows "
          f"(date x ticker), {meta['n_tickers_with_data']} tickers, "
          f"{meta['first_date']} to {meta['last_date']}")
    print(f"\nManual verification sample (data/manual_verification_v{version}.csv):")
    show = ["ticker", "date", "adj_close", "raw_close", "return_stored",
            "return_match", "dollar_volume_match", "adv20_dollar_match", "eligible_stored"]
    print(verify[show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    all_ok = verify[[c for c in verify if c.endswith("_match")]].all().all() and \
             (verify["eligible_stored"] == verify["eligible_by_hand"]).all()
    print("\nAll stored values match the by-hand calculations." if all_ok
          else "\nWARNING: some stored values do not match -- inspect the CSV.")
    print("Now check a few adj_close values against Yahoo Finance's historical data page "
          "(adjusted close) and record the check in your research log.")


# Used by the strategy scripts
def load_clean_panel(version=None):
    if version is None:
        version = latest_version()
    long = pd.read_pickle(f"{DATA_DIR}/clean_panel_v{version}.pkl")
    long["ticker"] = long["ticker"].astype(str)

    def wide(col):
        return long.pivot(index="date", columns="ticker", values=col).sort_index() \
                   .sort_index(axis=1)

    if "signal_close" not in long:
        raise ValueError(f"Dataset version {version} has no signal_close (built before E010). "
                         "Run: python3 build_dataset.py")
    prices = wide("adj_close")
    eligible = wide("eligible").reindex_like(prices).fillna(False).astype(bool)
    return {
        "version": version,
        "prices": prices,
        "signal_prices": wide("signal_close").reindex_like(prices),   # use for signals only
        "volume": wide("volume").reindex_like(prices),
        "returns": wide("return").reindex_like(prices),
        "eligible": eligible,
        "sectors": long.drop_duplicates("ticker").set_index("ticker")["sector"].astype(str),
        "spy": pd.read_pickle(f"{DATA_DIR}/benchmark_v{version}.pkl").reindex(prices.index),
    }


if __name__ == "__main__":
    build()
