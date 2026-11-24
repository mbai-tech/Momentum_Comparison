import argparse
import datetime
import hashlib
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT]
from pipeline import to_weights, backtest          # frozen code (checksummed)
from build_dataset import load_clean_panel          # frozen code (checksummed)

CONFIG = os.path.join(ROOT, "config", "final.json")
OUT = os.path.join(ROOT, "results", "final_test")


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def verify_frozen_files(frozen):
    files = {**frozen["sha256_py"], **frozen["data"]["sha256"]}
    changed = [rel for rel, h in files.items()
               if not os.path.exists(os.path.join(ROOT, rel)) or sha256(os.path.join(ROOT, rel)) != h]
    if changed:
        raise SystemExit("STOP: these frozen files are missing or changed since the freeze: "
                         + ", ".join(changed) + ". Do not run the final test.")
    return len(files)


# Strategies (frozen rules)
def ma_target(sp, eligible, p, min_names):
    side = np.sign(sp.rolling(p["short"]).mean() - sp.rolling(p["long"]).mean())
    return to_weights(side.where(eligible).fillna(0), min_names)


def momentum_target(sp, eligible, p, tails, min_names):
    dates = sp.index.to_series().groupby(sp.index.to_period("M")).tail(1).index
    mom = (sp.shift(p["skip_days"]) / sp.shift(p["lookback_days"]) - 1).where(eligible).loc[dates]
    pct = mom.rank(axis=1, method="first", pct=True)
    side = (pct >= tails["top"]).astype(int) - (pct <= tails["bottom"]).astype(int)
    return to_weights(side, min_names).reindex(sp.index).ffill().fillna(0)


# Measurement
def sharpe(r):
    return r.mean() / r.std() * np.sqrt(252)


def metrics(r, spy, turnover=None):
    eq = (1 + r).cumprod()
    out = {"ann_return": eq.iloc[-1] ** (252 / len(r)) - 1, "ann_vol": r.std() * np.sqrt(252),
           "sharpe_net": sharpe(r), "max_drawdown": (eq / eq.cummax() - 1).min(),
           "growth_of_$1": eq.iloc[-1], "beta_to_benchmark": r.cov(spy) / spy.var(),
           "correlation_to_benchmark": r.corr(spy)}
    if turnover is not None:
        out["annual_one_way_turnover"] = turnover.mean() * 252
    return out


def block_bootstrap_sharpe(r, block, n_resamples, seed):
    x = r.to_numpy()
    n = len(x)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(n_resamples, math.ceil(n / block)))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_resamples, -1)[:, :n]
    s = x[idx]
    return s.mean(axis=1) / s.std(axis=1, ddof=1) * np.sqrt(252)


def run(cfg, start, end, out_dir):
    fz = cfg["frozen"]
    panel = load_clean_panel(fz["data"]["dataset_version"])
    sp, returns, eligible = panel["signal_prices"], panel["returns"], panel["eligible"]
    spy = panel["spy"].pct_change(fill_method=None).loc[start:end]
    min_names = fz["min_names_per_side"]["value"]
    ma, mom = fz["final_settings"]["MA"], fz["final_settings"]["Momentum"]
    names = {"MA": f"MA ({ma['short']},{ma['long']})",
             "Momentum": f"Momentum {mom['lookback_days'] // 21}-{mom['skip_days'] // 21}"}
    targets = {names["MA"]: ma_target(sp, eligible, ma, min_names),
               names["Momentum"]: momentum_target(sp, eligible, mom, fz["momentum_tails"], min_names)}
    bench = f"Benchmark ({fz['benchmark']['ticker']})"

    rows, nets = [], {}
    for name, target in targets.items():
        for scenario, c in fz["cost_scenarios"].items():
            if not isinstance(c, dict):
                continue                                   # the "model" description
            bt = backtest(target, returns, c["trading_bps"], c["borrow_bps_per_year"]).loc[start:end]
            nets[(name, scenario)] = bt["net"]
            rows.append({"strategy": name, "scenario": scenario, **c,
                         **metrics(bt["net"], spy, bt["turnover"])})
    rows.append({"strategy": bench, "scenario": "-", **metrics(spy, spy)})
    results = pd.DataFrame(rows)
    assert not spy.isna().any() and all(len(v) == len(spy) for v in nets.values())

    b = fz["bootstrap"]
    series = {f"{n} | {s}": r for (n, s), r in nets.items()}
    series[bench] = spy
    ci = []
    for label, r in series.items():
        boot = block_bootstrap_sharpe(r, b["block_length_days"], b["resamples"], b["seed"])
        lo, hi = np.percentile(boot, [100 * q for q in b["interval_percentiles"]])
        ci.append({"series": label, "sharpe": sharpe(r), "ci_low": lo, "ci_high": hi,
                   "share_of_resamples_above_0": (boot > 0).mean()})
    ci = pd.DataFrame(ci)

    yearly = []
    for name in targets:
        r = nets[(name, "base")]
        for y, x in r.groupby(r.index.year):
            yearly.append({"strategy": name, "year": y, "net_return": (1 + x).prod() - 1,
                           "sharpe": sharpe(x)})
    for y, x in spy.groupby(spy.index.year):
        yearly.append({"strategy": bench, "year": y, "net_return": (1 + x).prod() - 1,
                       "sharpe": sharpe(x)})
    yearly = pd.DataFrame(yearly)

    os.makedirs(out_dir, exist_ok=True)
    results.to_csv(f"{out_dir}/results_all_costs.csv", index=False)
    ci.to_csv(f"{out_dir}/bootstrap_sharpe_ci.csv", index=False)
    yearly.to_csv(f"{out_dir}/yearly_base_costs.csv", index=False)

    ps = fz["plot_style"]
    import matplotlib
    matplotlib.use(ps["backend"])
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name in targets:
        (1 + nets[(name, "base")]).cumprod().plot(ax=ax, label=f"{name}, base costs")
    (1 + spy).cumprod().plot(ax=ax, label=bench, ls="--")
    ax.set_title(f"Final test {start[:4]}-{end[:4]}: growth of $1, net of base costs")
    ax.set_ylabel("Value of $1"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{out_dir}/final_test_equity.png", dpi=ps["dpi"]); plt.close(fig)
    return results, ci, yearly


def main():
    parser = argparse.ArgumentParser(description="One-time final test on the frozen test period.")
    parser.add_argument("--rerun", action="store_true",
                        help="overwrite existing results (deliberate reruns only)")
    args = parser.parse_args()
    if os.path.exists(OUT) and not args.rerun:
        raise SystemExit(f"STOP: {OUT} already exists; the final test has been run. "
                         "Use --rerun only for a deliberate, logged rerun.")

    cfg = json.load(open(CONFIG))
    n_checked = verify_frozen_files(cfg["frozen"])
    test = cfg["frozen"]["periods"]["test"]
    print(f"Frozen files verified ({n_checked} checksums). Dataset "
          f"{cfg['frozen']['data']['dataset_version']}; test period {test['start']} to {test['end']}\n")
    results, ci, yearly = run(cfg, test["start"], test["end"], OUT)

    with open(f"{OUT}/run_manifest.json", "w") as f:
        json.dump({"run_at": datetime.datetime.now().isoformat(timespec="seconds"),
                   "config_sha256": sha256(CONFIG), "frozen_files_verified": n_checked,
                   "test_period": [test["start"], test["end"]], "rerun": args.rerun}, f, indent=2)

    pd.set_option("display.width", 220)
    print("Results, all cost scenarios:")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    p = cfg["frozen"]["bootstrap"]["interval_percentiles"]
    print(f"\nMoving-block bootstrap of the annualised Sharpe ratio "
          f"({p[0]:.1%}-{p[1]:.1%} interval):")
    print(ci.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nNet return by year, base costs:")
    print(yearly.pivot(index="year", columns="strategy", values="net_return")
          .to_string(float_format=lambda x: f"{x:+.1%}"))
    print(f"\nSaved in {OUT}/")


if __name__ == "__main__":
    main()
