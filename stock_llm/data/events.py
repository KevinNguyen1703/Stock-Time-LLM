"""Event-driven samples with long holding horizons.

A sample is (ticker, day t) where at least one non-routine item tagged to the ticker becomes usable
before the open of t. Trade: enter at the open of t, exit at the close of t+H-1 (H trading days).
Labels: raw log return (ret), excess return vs VN-Index (exc) and return relative to the equal-weight
average of the universe tickers over the same window (rel = the company-specific part).

Context given to models (all known before the open of t):
  - time series: daily features for the SEQ_LEN rows up to t-1 (price, foreign flow, VN-Index),
    scaled with statistics from the training period only
  - text: up to MAX_ITEMS non-routine items from the last LOOKBACK trading days, newest first,
    with event type, LLM sentiment score and age
Splits are purged: a training/validation label window never overlaps the following split.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stock_llm.data.news import bucket_to_trading_day
from stock_llm.data.prices import add_features, load_prices
from stock_llm.data.universe import TICKERS

SEQ_LEN, LOOKBACK, MAX_ITEMS, VAL_DAYS = 60, 5, 8, 120
FOLDS = [("2023-01-01", "2024-01-01"), ("2024-01-01", "2025-01-01"), ("2025-01-01", "2026-11-01")]


def load_index():
    d = pd.read_csv("data/prices/VNINDEX.csv", parse_dates=["Date"]).sort_values("Date").set_index("Date")
    return d


def ts_frame(ticker, idx):
    """Daily features; row t uses information up to the close of t."""
    px = load_prices(ticker)
    f = add_features(px).set_index("Date")
    fl = pd.read_csv(f"data/flows/{ticker}_foreign.csv", parse_dates=["Date"]).set_index("Date")
    traded = (px.set_index("Date")["Volume"] * px.set_index("Date")["Close"]).rolling(20).mean()
    fl = fl.reindex(f.index)
    f["fnet_1"] = (fl["net_val"].fillna(0) / traded.reindex(f.index)).clip(-2, 2)
    f["fnet_20"] = f["fnet_1"].rolling(20).mean()
    f["fown_chg20"] = fl["own_pct"].ffill().diff(20)
    ir = np.log(idx["Close"]).reindex(f.index)
    f["mkt_ret_1"], f["mkt_ret_20"] = ir.diff(), ir.diff(20)
    f["mkt_vol_20"] = f["mkt_ret_1"].rolling(20).std()
    f["rel_ret_20"] = f["ret_20"] - f["mkt_ret_20"]
    f["ret_60"] = np.log(f["Adj Close"]).diff(60)
    return f.drop(columns=["Adj Close"]).fillna(0.0), px.set_index("Date")


@dataclass
class EventSet:
    ticker: list = field(default_factory=list)
    date: list = field(default_factory=list)
    ts: list = field(default_factory=list)          # [SEQ_LEN, F] unscaled
    items: list = field(default_factory=list)       # list of dicts (title, lead, event, score, age)
    y: dict = field(default_factory=dict)           # name -> list


def build_events(horizons=(5, 30, 45)):
    from stock_llm.text.corpus import ALIASES  # noqa: F401  (keeps tagging definitions in one place)
    c = pd.read_parquet("data/text/corpus.parquet")
    ev = pd.read_parquet("data/text/event_qwen25_7b.parquet")
    sc = pd.read_parquet("data/text/llm_qwen25_7b.parquet")
    items = ev.merge(sc[["url", "ticker", "score"]], on=["url", "ticker"]).merge(
        c[["url", "published", "title", "lead", "source"]], on="url")
    items = items[items["event"] != "routine"].copy()
    date_only = items["published"].dt.time == pd.Timestamp("00:00").time()
    items.loc[date_only, "published"] += pd.Timedelta(hours=23, minutes=59)
    idx = load_index()
    # equal-weight universe: average over tickers of log(close[t+h-1] / open[t]) for every entry day
    pxs = {t: load_prices(t).set_index("Date") for t in TICKERS}
    all_days = pxs[TICKERS[0]].index
    for p_ in pxs.values():
        all_days = all_days.intersection(p_.index)
    U_open = pd.DataFrame({t: pxs[t]["Open"].reindex(all_days) for t in TICKERS})
    U_close = pd.DataFrame({t: pxs[t]["Close"].reindex(all_days) for t in TICKERS})
    peer = {h: np.log(U_close.shift(-(h - 1)) / U_open).mean(axis=1) for h in horizons}
    rows = []
    for t in TICKERS:
        f, px = ts_frame(t, idx)
        days = f.index
        b = bucket_to_trading_day(items[items["ticker"] == t], days)
        pos = pd.Series(np.arange(len(days)), index=days)
        b["tpos"] = pos.loc[b["target_day"]].to_numpy()
        lo_, op, cl = np.log(px["Close"]).reindex(days), px["Open"].reindex(days), px["Close"].reindex(days)
        iop, icl = idx["Open"].reindex(days), idx["Close"].reindex(days)
        for p in sorted(b["tpos"].unique()):
            if p < SEQ_LEN:
                continue
            ctx = b[(b["tpos"] <= p) & (b["tpos"] > p - LOOKBACK)].sort_values("published", ascending=False)
            r = {"ticker": t, "date": days[p], "ts": f.iloc[p - SEQ_LEN:p].to_numpy(np.float32),
                 "n_today": int((ctx["tpos"] == p).sum()),
                 "items": [{"title": x.title, "lead": x.lead, "event": x.event, "score": int(x.score),
                            "age": int(p - x.tpos), "source": x.source} for x in ctx.head(MAX_ITEMS).itertuples()]}
            for h in horizons:
                e = p + h - 1
                if e < len(days):
                    r[f"ret_{h}"] = float(np.log(cl.iloc[e] / op.iloc[p]))
                    r[f"exc_{h}"] = r[f"ret_{h}"] - float(np.log(icl.iloc[e] / iop.iloc[p]))
                    r[f"rel_{h}"] = r[f"ret_{h}"] - float(peer[h].get(days[p], np.nan))
                    r[f"end_{h}"] = days[e]
                else:
                    r[f"ret_{h}"] = r[f"exc_{h}"] = r[f"rel_{h}"] = np.nan
                    r[f"end_{h}"] = pd.NaT
            rows.append(r)
    return pd.DataFrame(rows), list(f.columns)


def split_fold(df, fold, h, val_days=VAL_DAYS, test_range=None):
    """Purged rolling-origin split for horizon h. Returns train, val, test frames and feature scaler.
    test_range=(start, end) overrides the fold's test period (used for walk-forward retraining)."""
    lo, hi = map(pd.Timestamp, test_range or FOLDS[fold])
    all_days = np.sort(df["date"].unique())
    d = df.dropna(subset=[f"ret_{h}"])
    test = d[(d["date"] >= lo) & (d["date"] < hi)]
    val_start = pd.Timestamp(all_days[max(0, np.searchsorted(all_days, lo.to_datetime64()) - val_days)])
    val = d[(d["date"] >= val_start) & (d[f"end_{h}"] < lo)]
    train = d[d[f"end_{h}"] < val_start]
    ts = np.stack(train["ts"].to_list())
    mu, sd = ts.reshape(-1, ts.shape[-1]).mean(0), ts.reshape(-1, ts.shape[-1]).std(0) + 1e-8
    return train, val, test, (mu.astype(np.float32), sd.astype(np.float32))
