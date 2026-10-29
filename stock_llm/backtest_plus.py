"""Portfolio backtest of Time-LLM+ test predictions (pooled over the rolling test years).

Each selected event is a trade entered at the open of the event day and held H trading days.
  neutral : side * (stock - equal-weight peer basket), the quantity the model predicts
  raw     : side * stock (long/short the stock itself)
The portfolio holds every open trade with equal weight; days without open trades earn 0.
Rules: q (val-quantile thresholds on the score), conf (confidence head, direction = sign of score),
ltt (only trades certified by Learn-then-Test). Benchmarks: all events long, and random sides
(permutation test on the Sharpe ratio, same trades and holding periods).
Usage: python -m stock_llm.backtest_plus --pattern "reports/plus/pred_qwen1.5b_f*_h30_p-news_k1_bank_text_aligned_full_s0.npz"
"""
import argparse
import glob

import numpy as np
import pandas as pd

from stock_llm.data.prices import load_prices
from stock_llm.data.universe import TICKERS


def daily_returns():
    """Per-ticker daily log returns, with the event-day return measured from the open."""
    close, intraday = {}, {}
    for t in TICKERS:
        px = load_prices(t).set_index("Date")
        close[t] = np.log(px["Close"]).diff()
        intraday[t] = np.log(px["Close"] / px["Open"])
    c, i = pd.DataFrame(close), pd.DataFrame(intraday)
    days = c.dropna(how="all").index
    return c.loc[days], i.loc[days]


def trade_paths(trades, H, c, i, mode):
    """Daily P&L of each trade over its holding window (day 0 = open-to-close of the event day)."""
    peer_c, peer_i = c.mean(1), i.mean(1)
    days = c.index
    rows = []
    for k, tr in enumerate(trades.itertuples()):
        p = days.get_indexer([pd.Timestamp(tr.date)])[0]
        if p < 0 or p + H > len(days):
            continue
        win = days[p:p + H]
        stock = np.r_[i[tr.ticker].iloc[p], c[tr.ticker].iloc[p + 1:p + H].to_numpy()]
        if mode == "neutral":
            stock = stock - np.r_[peer_i.iloc[p], peer_c.iloc[p + 1:p + H].to_numpy()]
        rows.append(pd.DataFrame({"trade": k, "day": win, "pnl": tr.side * np.nan_to_num(stock)}))
    return pd.concat(rows) if rows else pd.DataFrame(columns=["trade", "day", "pnl"])


def stats(paths, all_days, n_trades):
    if paths.empty:
        return {"trades": 0}
    port = paths.groupby("day")["pnl"].mean().reindex(all_days, fill_value=0.0)
    per_trade = paths.groupby("trade")["pnl"].sum()
    eq = port.cumsum()
    return {"trades": n_trades, "win_%": float((per_trade > 0).mean() * 100),
            "avg_trade_%": float(per_trade.mean() * 100),
            "t_stat": float(per_trade.mean() / (per_trade.std() / np.sqrt(len(per_trade)))),
            "ann_ret_%": float(port.mean() * 252 * 100), "ann_vol_%": float(port.std() * np.sqrt(252) * 100),
            "sharpe": float(port.mean() / port.std() * np.sqrt(252)) if port.std() > 0 else 0.0,
            "max_dd_%": float((eq - eq.cummax()).min() * 100),
            "exposure_%": float((paths.groupby("day").size().reindex(all_days, fill_value=0) > 0).mean() * 100)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--n_random", type=int, default=200)
    a = ap.parse_args()
    files = sorted(glob.glob(a.pattern))
    d = pd.concat([pd.DataFrame({k: np.load(f, allow_pickle=True)[k] for k in
                                 ["p", "conf", "rel", "ret", "side_q", "sel_conf", "sel_ltt", "ticker", "date"]})
                   for f in files], ignore_index=True)
    print(f"{len(files)} prediction files, {len(d)} test events, {d.date.min()} -> {d.date.max()}")
    c, i = daily_returns()
    all_days = c.index[(c.index >= pd.Timestamp(d.date.min())) & (c.index <= pd.Timestamp(d.date.max()))]
    rules = {"q (val quantiles)": d.assign(side=d.side_q)[d.side_q != 0],
             "conf (confidence head)": d.assign(side=np.sign(d.p))[d.sel_conf.astype(bool)],
             "ltt (certified)": d.assign(side=np.sign(d.p))[d.sel_ltt.astype(bool)],
             "benchmark: all events long": d.assign(side=1)}
    rng = np.random.default_rng(0)
    rows = []
    for mode in ["neutral", "raw"]:
        for name, tr in rules.items():
            r = {"pnl": mode, "rule": name, **stats(trade_paths(tr, a.horizon, c, i, mode), all_days, len(tr))}
            if not name.startswith("benchmark") and len(tr) and a.n_random:
                sh = [stats(trade_paths(tr.assign(side=rng.choice([-1, 1], len(tr))), a.horizon, c, i, mode),
                            all_days, len(tr))["sharpe"] for _ in range(a.n_random)]
                r["p_vs_random"] = float(np.mean(np.array(sh) >= r["sharpe"]))
            rows.append(r)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
