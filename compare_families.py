import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "strategies"))
from build_dataset import load_clean_panel

import pipeline as P
from pipeline import to_weights, backtest, COST_SCENARIOS, DEV_START, DEV_END

MIN_NAMES = getattr(P, "MIN_NAMES", 20)
MA_SHORT = getattr(P, "MA_SHORT", 50)
MA_LONG = getattr(P, "MA_LONG", 200)
MOM_LOOKBACK = getattr(P, "MOM_LOOKBACK", 252)
MOM_SKIP = getattr(P, "MOM_SKIP", 21)
TOP = getattr(P, "TOP", 0.80)
BOTTOM = getattr(P, "BOTTOM", 0.20)

OUT = str(P.ROOT / "results" / "compare")
N_QUANTILES = 5
CROSSOVER_HORIZON = 63      # trading days (~3 months) used to judge a crossover; fixed in advance
WHIPSAW_DAYS = 20           # a crossover reversed within this many trading days counts as a whipsaw
ROLLING_WINDOW = 252        # one year for rolling metrics
# Stress periods chosen from market events, independently of the strategy results (spec 9.3)
CRISIS_PERIODS = {
    "Lehman to market low (2008-09-15 to 2009-03-09)": ("2008-09-15", "2009-03-09"),
    "2009 rebound (2009-03-10 to 2009-09-30)": ("2009-03-10", "2009-09-30"),
    "2010 flash crash / euro crisis (2010-04-23 to 2010-07-02)": ("2010-04-23", "2010-07-02"),
    "2011 US downgrade (2011-07-22 to 2011-10-03)": ("2011-07-22", "2011-10-03"),
}
# Neighbouring settings from the pre-registered grid (spec 8.4) -- a stability view, not selection
MA_NEIGHBOURS = [(20, 100), (50, 200), (100, 250)]
MOM_NEIGHBOURS = [(126, 21), (189, 21), (252, 21)]

def month_ends(index):
    return pd.Series(index, index=index).groupby(index.to_period("M")).last().values


def ma_signal(prices, short, long):
    return np.sign(prices.rolling(short).mean() - prices.rolling(long).mean())


def ma_target(prices, eligible, short=MA_SHORT, long=MA_LONG, long_only=False):
    side = ma_signal(prices, short, long).where(eligible).fillna(0)
    if long_only:
        side = side.clip(lower=0)
    return to_weights(side)


def momentum(prices, lookback=MOM_LOOKBACK, skip=MOM_SKIP):
    return prices.shift(skip) / prices.shift(lookback) - 1


def xs_target(prices, eligible, lookback=MOM_LOOKBACK, skip=MOM_SKIP):
    dates = pd.DatetimeIndex(month_ends(prices.index))
    pct = momentum(prices, lookback, skip).where(eligible).loc[dates] \
        .rank(axis=1, method="first", pct=True)
    side = (pct >= TOP).astype(int) - (pct <= BOTTOM).astype(int)
    return to_weights(side, MIN_NAMES).reindex(prices.index).ffill().fillna(0)


def dev(x):
    return x.loc[DEV_START:DEV_END]

def drawdown_details(r):
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    trough = dd.idxmin()
    peak = eq.loc[:trough].idxmax()
    after = eq.loc[trough:]
    recovered = after[after >= eq.loc[peak]]
    return dd.min(), peak, trough, (recovered.index[0] if len(recovered) else None)


def performance(r):
    r = r.dropna()
    eq = (1 + r).cumprod()
    ann_ret = eq.iloc[-1] ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    downside = np.sqrt((np.minimum(r, 0) ** 2).mean()) * np.sqrt(252)
    mdd, peak, trough, rec = drawdown_details(r)
    monthly = (1 + r).groupby(r.index.to_period("M")).prod() - 1
    return {
        "annualised return": ann_ret,
        "annualised volatility": vol,
        "Sharpe": r.mean() / r.std() * np.sqrt(252),
        "Sortino": r.mean() * 252 / downside if downside > 0 else np.nan,
        "max drawdown": mdd,
        "drawdown peak": peak.date(),
        "drawdown trough": trough.date(),
        "drawdown recovery": rec.date() if rec is not None else "not recovered by 2014",
        "Calmar": ann_ret / abs(mdd) if mdd < 0 else np.nan,
        "hit rate, daily": (r > 0).mean(),
        "hit rate, monthly": (monthly > 0).mean(),
        "growth of $1": eq.iloc[-1],
    }

def trading_intensity(target, bt):
    w = dev(target.shift(1).fillna(0))                  # weights after trading (execution lag)
    s = np.sign(w)
    held_days = (s != 0).sum().sum()
    starts = ((s != 0) & (s != s.shift(1))).sum().sum()             # entries and flips
    exits = ((s == 0) & (s.shift(1).fillna(0) != 0)).sum().sum()
    years = len(w) / 252
    b = dev(bt)
    return {
        "avg daily one-way turnover": b["turnover"].mean(),
        "annual one-way turnover": b["turnover"].mean() * 252,
        "cost drag, total (return/yr)": (b["gross"] - b["net"]).mean() * 252,
        "cost drag, trading (return/yr)": (b["gross"] - b["net"]).mean() * 252
                                           - borrow_drag(target),
        "cost drag, borrow (return/yr)": borrow_drag(target),
        "avg holding period (trading days)": held_days / starts if starts else np.nan,
        "position changes per year (entries, exits, flips)": (starts + exits) / years,
        "weight adjustments per year (any trade)": ((w - w.shift(1)).abs() > 1e-12).sum().sum() / years,
    }


def borrow_drag(target):
    tc, br = COST_SCENARIOS["base"]
    held = dev(target.shift(2).fillna(0))
    return (held.clip(upper=0).abs().sum(axis=1) * br / 1e4 / 252).mean() * 252

def diversification(target, sectors):
    held = dev(target.shift(2).fillna(0))
    active = held.abs().sum(axis=1) > 0
    held = held[active]
    longs, shorts = held.clip(lower=0), held.clip(upper=0).abs()

    def eff_n(x):
        g = x.sum(axis=1)
        return (g ** 2 / (x ** 2).sum(axis=1)).where(g > 0)

    sec = sectors.reindex(held.columns).fillna("Unknown")
    net_by_sector = held.T.groupby(sec).sum().T
    gross_by_sector = held.abs().T.groupby(sec).sum().T
    avg_abs_net = net_by_sector.abs().mean()
    return {
        "avg positions, long": (longs > 0).sum(axis=1).mean(),
        "avg positions, short": (shorts > 0).sum(axis=1).mean(),
        "effective positions, long": eff_n(longs).mean(),
        "effective positions, short": eff_n(shorts).mean(),
        "top single weight, average": held.abs().max(axis=1).mean(),
        "top single weight, maximum": held.abs().max(axis=1).max(),
        "largest avg |net sector exposure|": avg_abs_net.max(),
        "sector with largest avg |net exposure|": avg_abs_net.idxmax(),
        "largest avg sector gross exposure": gross_by_sector.mean().max(),
    }, net_by_sector.mean()

def return_source(target, returns, bt, spy_r, sectors):
    held = dev(target.shift(2).fillna(0))
    r = dev(returns.reindex_like(target).fillna(0))
    net = dev(bt["net"])
    both = pd.concat([net, dev(spy_r)], axis=1, keys=["s", "m"]).dropna()
    beta = both["s"].cov(both["m"]) / both["m"].var()
    alpha = (both["s"].mean() - beta * both["m"].mean()) * 252
    contrib = held * r
    sec = sectors.reindex(held.columns).fillna("Unknown")
    by_sector = contrib.T.groupby(sec).sum().T.mean() * 252
    crisis = {}
    for name, (a, b) in CRISIS_PERIODS.items():
        crisis[name] = (1 + net.loc[a:b]).prod() - 1
    return {
        "long leg (gross return/yr)": (held.clip(lower=0) * r).sum(axis=1).mean() * 252,
        "short leg (gross return/yr)": (held.clip(upper=0) * r).sum(axis=1).mean() * 252,
        "market beta (vs SPY)": beta,
        "alpha vs SPY (net, per year)": alpha,
        "correlation with SPY": both["s"].corr(both["m"]),
        "best sector contribution": f"{by_sector.idxmax()} ({by_sector.max():+.2%}/yr)",
        "worst sector contribution": f"{by_sector.idxmin()} ({by_sector.min():+.2%}/yr)",
    }, by_sector, crisis

def holding_returns(prices, dates):
    pos = prices.index.get_indexer(dates)
    exec_dates = prices.index[np.minimum(pos + 1, len(prices.index) - 1)]
    p = prices.loc[exec_dates]
    fwd = p.shift(-1) / p - 1
    fwd.index = dates
    return fwd

def ic_and_quintiles(signal, prices, eligible):
    all_dates = pd.DatetimeIndex(month_ends(prices.index))
    fwd_all = holding_returns(prices, all_dates)
    dates = all_dates[(all_dates >= DEV_START) & (all_dates <= DEV_END)][:-1]
    sig = signal.where(eligible).loc[dates]
    fwd = fwd_all.loc[dates]
    mask = sig.notna() & fwd.notna()
    ic = sig.where(mask).corrwith(fwd.where(mask), axis=1, method="spearman").dropna()
    pct = sig.rank(axis=1, method="first", pct=True)
    bucket = np.ceil(pct * N_QUANTILES).clip(1, N_QUANTILES)
    q = pd.Series({f"Q{k}": fwd.where(bucket == k).mean(axis=1).mean()
                   for k in range(1, N_QUANTILES + 1)})
    rho = pd.Series(range(1, N_QUANTILES + 1)).corr(pd.Series(q.values), method="spearman")
    return ic, q, rho, pct

def crossover_quality(prices, eligible):
    sig = ma_signal(prices, MA_SHORT, MA_LONG).where(eligible)
    fwd = prices.shift(-(CROSSOVER_HORIZON + 1)) / prices.shift(-1) - 1   # from execution day
    last_ok = prices.index[prices.index.get_indexer([pd.Timestamp(DEV_END)], method="pad")[0]
                           - CROSSOVER_HORIZON - 1]
    hits, total, whipsaw, gaps = 0, 0, 0, []
    for t in sig.columns:
        s = sig[t].loc[DEV_START:last_ok]
        prev = sig[t].shift(1).loc[DEV_START:last_ok]
        cross = s[(s != prev) & s.notna() & prev.notna() & (s != 0) & (prev != 0)]
        if cross.empty:
            continue
        f = fwd[t].reindex(cross.index)
        ok = f.notna()
        hits += int((np.sign(f[ok]) == cross[ok]).sum())
        total += int(ok.sum())
        pos = prices.index.get_indexer(cross.index)
        d = np.diff(pos)
        whipsaw += int((d <= WHIPSAW_DAYS).sum())
        gaps.extend(d.tolist())
    return {
        "crossovers": total,
        f"crossover hit rate ({CROSSOVER_HORIZON}-day)": hits / total if total else np.nan,
        f"whipsaw share (reversed within {WHIPSAW_DAYS} days)": whipsaw / total if total else np.nan,
        "median days between crossovers": float(np.median(gaps)) if gaps else np.nan,
    }


def persistence(pct, tail=True):
    autocorr = pct.corrwith(pct.shift(1), axis=1, method="spearman").mean()
    out = {"rank autocorrelation (month to month)": autocorr}
    if tail:
        side = (pct >= TOP).astype(int) - (pct <= BOTTOM).astype(int)
        out["share of long names kept next month"] = (
            ((side > 0) & (side.shift(1) > 0)).sum(axis=1) / (side.shift(1) > 0).sum(axis=1)).mean()
        out["share of short names kept next month"] = (
            ((side < 0) & (side.shift(1) < 0)).sum(axis=1) / (side.shift(1) < 0).sum(axis=1)).mean()
    return out


def yearly(r):
    r = dev(r).dropna()
    g = r.groupby(r.index.year)
    return pd.DataFrame({"return": g.apply(lambda x: (1 + x).prod() - 1),
                         "Sharpe": g.apply(lambda x: x.mean() / x.std() * np.sqrt(252))})


def subperiods(r):
    out = {}
    for name, (a, b) in {"2006-2010": ("2006-01-01", "2010-12-31"),
                         "2011-2014": ("2011-01-01", "2014-12-31")}.items():
        x = r.loc[a:b].dropna()
        out[f"Sharpe {name}"] = x.mean() / x.std() * np.sqrt(252)
    return out

def main():
    os.makedirs(OUT, exist_ok=True)
    panel = load_clean_panel()
    print(f"Comparison on clean dataset version {panel['version']}, "
          f"development sample {DEV_START[:4]}-{DEV_END[:4]}, base costs\n")
    # Signals, MA strength, crossover checks and diagnostic forward returns use signal_prices
    # (E010: invalid returns counted as 0%); backtests use returns
    prices, returns, eligible = panel["signal_prices"], panel["returns"], panel["eligible"]
    sectors, spy_r = panel["sectors"], panel["spy"].pct_change(fill_method=None)
    tc, br = COST_SCENARIOS["base"]

    targets = {"MA 50/200": ma_target(prices, eligible),
               "MA 50/200 long/cash": ma_target(prices, eligible, long_only=True),
               "12-1 momentum": xs_target(prices, eligible)}
    bts = {k: backtest(t, returns, tc, br) for k, t in targets.items()}
    nets = {k: dev(b["net"]) for k, b in bts.items()}
    nets["S&P 500 (SPY)"] = dev(spy_r)

    # Exposure check: the portfolios must be what the baseline says
    for k, t in targets.items():
        h = dev(t.shift(2).fillna(0))
        a = h.abs().sum(axis=1) > 0
        print(f"  {k}: average gross {h.abs().sum(axis=1)[a].mean():.2f}, "
              f"net {h.sum(axis=1)[a].mean():+.2f}")

    rows, sector_net, sector_contrib, crisis = {}, {}, {}, {}
    for k in nets:
        rows[k] = {("1 Performance", m): v for m, v in performance(nets[k]).items()}
    for k, t in targets.items():
        for m, v in trading_intensity(t, bts[k]).items():
            rows[k][("2 Trading intensity", m)] = v
        div, sector_net[k] = diversification(t, sectors)
        for m, v in div.items():
            rows[k][("3 Diversification", m)] = v
        src, sector_contrib[k], crisis[k] = return_source(t, returns, bts[k], spy_r, sectors)
        for m, v in src.items():
            rows[k][("4 Return source", m)] = v
    crisis["S&P 500 (SPY)"] = {n: (1 + nets["S&P 500 (SPY)"].loc[a:b]).prod() - 1
                               for n, (a, b) in CRISIS_PERIODS.items()}

    # Signal quality: the same IC / quintile test for both signals, plus family-specific measures
    ma_strength = prices.rolling(MA_SHORT).mean() / prices.rolling(MA_LONG).mean() - 1
    signals = {"MA 50/200": ma_strength, "12-1 momentum": momentum(prices)}
    quintiles, ics = {}, {}
    for k, sig in signals.items():
        ic, q, rho, pct = ic_and_quintiles(sig, prices, eligible)
        quintiles[k], ics[k] = q, ic
        sq = {"IC mean": ic.mean(), "IC t-statistic": ic.mean() / (ic.std() / np.sqrt(len(ic))),
              "IC positive months": (ic > 0).mean(),
              "quintile monotonicity (Spearman, +1 = rising)": rho,
              "Q5 - Q1 monthly return": q.iloc[-1] - q.iloc[0]}
        sq.update(persistence(pct, tail=(k == "12-1 momentum")))
        if k == "MA 50/200":
            sq.update(crossover_quality(prices, eligible))
        for m, v in sq.items():
            rows[k][("5 Signal quality", m)] = v

    # Stability: subperiods, rolling, neighbouring parameters
    for k in nets:
        for m, v in subperiods(nets[k]).items():
            rows[k][("6 Stability", m)] = v
        roll = nets[k].rolling(ROLLING_WINDOW)
        rs = (roll.mean() / roll.std() * np.sqrt(252)).dropna()
        rows[k][("6 Stability", "rolling 1-yr Sharpe, min")] = rs.min()
        rows[k][("6 Stability", "rolling 1-yr Sharpe, max")] = rs.max()
        rows[k][("6 Stability", "share of time rolling Sharpe > 0")] = (rs > 0).mean()

    neighbours = []
    for s, l in MA_NEIGHBOURS:
        r = dev(backtest(ma_target(prices, eligible, s, l), returns, tc, br)["net"])
        p = performance(r)
        neighbours.append({"family": "MA", "setting": f"({s},{l})",
                           "Sharpe": p["Sharpe"], "annualised return": p["annualised return"]})
    for lb, sk in MOM_NEIGHBOURS:
        r = dev(backtest(xs_target(prices, eligible, lb, sk), returns, tc, br)["net"])
        p = performance(r)
        neighbours.append({"family": "12-1", "setting": f"{lb // 21}-{sk // 21}",
                           "Sharpe": p["Sharpe"], "annualised return": p["annualised return"]})
    neighbours = pd.DataFrame(neighbours)

    # ---- save tables
    summary = pd.DataFrame(rows)
    order = sorted(range(len(summary.index)), key=lambda i: (summary.index[i][0], i))
    summary = summary.iloc[order]
    summary.index = pd.MultiIndex.from_tuples(summary.index, names=["view", "metric"])
    summary.to_csv(f"{OUT}/comparison.csv")
    pd.DataFrame({k: yearly(v)["return"] for k, v in nets.items()}).to_csv(f"{OUT}/yearly_returns.csv")
    pd.DataFrame({k: yearly(v)["Sharpe"] for k, v in nets.items()}).to_csv(f"{OUT}/yearly_sharpe.csv")
    pd.DataFrame(crisis).to_csv(f"{OUT}/crisis_periods.csv")
    pd.DataFrame(sector_contrib).to_csv(f"{OUT}/sector_contribution.csv")
    pd.DataFrame(sector_net).to_csv(f"{OUT}/sector_net_exposure.csv")
    pd.DataFrame(quintiles).to_csv(f"{OUT}/quintile_returns.csv")
    neighbours.to_csv(f"{OUT}/neighbouring_settings.csv", index=False)

    # ---- figures
    fig, ax = plt.subplots(figsize=(10, 4.5))
    (1 + pd.DataFrame(nets).fillna(0)).cumprod().plot(ax=ax, logy=True)
    for (a, b) in CRISIS_PERIODS.values():
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="grey", alpha=0.15)
    ax.set_title("Growth of $1, 2006-2014 (net of base costs); shaded = stress periods")
    ax.set_ylabel("Value of $1 (log)"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/equity_curves.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    for k, r in nets.items():
        eq = (1 + r.fillna(0)).cumprod()
        (eq / eq.cummax() - 1).plot(ax=ax, label=k)
    ax.set_title("Drawdown (underwater chart)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/drawdowns.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    for k, r in nets.items():
        roll = r.rolling(ROLLING_WINDOW)
        (roll.mean() / roll.std() * np.sqrt(252)).plot(ax=ax, label=k)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Rolling 1-year Sharpe ratio (252 trading days)"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/rolling_sharpe.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    (pd.DataFrame({k: yearly(v)["return"] for k, v in nets.items()}) * 100).plot.bar(ax=ax)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Net return by year (%)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/yearly_returns.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    (pd.DataFrame(quintiles) * 100).plot.bar(ax=ax)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Average next-month return by signal quintile (Q1 = weakest)")
    ax.set_ylabel("% per month")
    fig.tight_layout(); fig.savefig(f"{OUT}/quintile_returns.png", dpi=150); plt.close(fig)

    # ---- print
    pd.set_option("display.width", 200)
    fmt = summary.copy().astype(object)
    for idx in fmt.index:
        for c in fmt.columns:
            v = fmt.loc[idx, c]
            if isinstance(v, (float, np.floating)):
                fmt.loc[idx, c] = "" if pd.isna(v) else f"{v:.3f}"
    print(fmt.fillna("").to_string())
    print("\nCrisis periods (net cumulative return):")
    print(pd.DataFrame(crisis).map(lambda x: f"{x:+.1%}").to_string())
    print("\nNeighbouring parameter settings (base costs, development):")
    print(neighbours.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nAll tables and figures saved in {OUT}/")


if __name__ == "__main__":
    main()