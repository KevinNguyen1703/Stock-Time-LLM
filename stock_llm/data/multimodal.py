"""Multimodal samples: price window + the news usable before the open of the prediction day.

For ticker i and prediction day t:
  company tokens: up to K_CO items tagged to i with target_day in the last LOOKBACK trading days (newest first)
  market tokens : up to K_MK untagged market items with target_day == t (i.e. published since the last open)
Each token carries a corpus row index (into the embedding matrix), its age in trading days, its source id
and the LLM score for (item, ticker) (0 for market tokens). Padding index is -1.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from stock_llm.data.news import bucket_to_trading_day
from stock_llm.data.prices import add_features, load_prices
from stock_llm.data.universe import TICKERS
from stock_llm.data.windows import make_fold

LOOKBACK, K_CO, K_MK = 5, 16, 16
SOURCES = ["filing", "press", "market_stock", "market_bank", "market_macro"]


@dataclass
class MMSplit:
    X: np.ndarray        # [n, T, F]
    y: np.ndarray        # [n]
    ticker: np.ndarray   # [n]
    dates: np.ndarray    # [n]
    co_idx: np.ndarray   # [n, K_CO] corpus rows, -1 = pad
    co_age: np.ndarray
    co_src: np.ndarray
    co_score: np.ndarray
    mk_idx: np.ndarray   # [n, K_MK]
    mk_age: np.ndarray
    mk_src: np.ndarray


class TextStore:
    def __init__(self, emb_name="bge_m3", score_name="qwen25_7b"):
        c = pd.read_parquet("data/text/corpus.parquet")
        z = np.load(f"data/text/emb_{emb_name}.npz", allow_pickle=True)
        row = pd.Series(np.arange(len(z["url"])), index=z["url"])
        c = c[c["url"].isin(row.index)].copy()
        c["row"] = row.loc[c["url"]].to_numpy()
        date_only = c["published"].dt.time == pd.Timestamp("00:00").time()
        c.loc[date_only, "published"] += pd.Timedelta(hours=23, minutes=59)
        c["src"] = c["source"].map({s: i for i, s in enumerate(SOURCES)})
        self.corpus, self.emb = c, z["emb"]
        s = pd.read_parquet(f"data/text/llm_{score_name}.parquet")
        self.score = s.set_index(["url", "ticker"])["score"].to_dict()


def _gather(b, days, pred_days, k, lookback, score_of):
    """For each prediction day, newest-first items with target_day in the last `lookback` trading days."""
    n = len(pred_days)
    idx, age, src, sc = (np.full((n, k), -1, np.int64), np.zeros((n, k), np.int64),
                         np.zeros((n, k), np.int64), np.zeros((n, k), np.float32))
    if len(b) == 0:
        return idx, age, src, sc
    day_pos = pd.Series(np.arange(len(days)), index=days)
    b = b.assign(tpos=day_pos.loc[b["target_day"]].to_numpy()).sort_values("published", ascending=False)
    groups = {p: g for p, g in b.groupby("tpos")}
    for j, d in enumerate(pred_days):
        p = day_pos[d]
        parts = [groups[q] for q in range(p, p - lookback, -1) if q in groups]
        if not parts:
            continue
        sel = pd.concat(parts).head(k)
        m = len(sel)
        idx[j, :m] = sel["row"].to_numpy()
        age[j, :m] = p - sel["tpos"].to_numpy()
        src[j, :m] = sel["src"].to_numpy()
        sc[j, :m] = [score_of(u) for u in sel["url"]]
    return idx, age, src, sc


def build_fold(store, fold, seq_len=60, tickers=TICKERS):
    from stock_llm.train import FOLDS
    c = store.corpus
    general = c[c["tickers"] == ""]
    out = [[] for _ in range(3)]
    cols = None
    for ti, t in enumerate(tickers):
        feat = add_features(load_prices(t))
        splits, cols = make_fold(feat, seq_len, *FOLDS[fold])
        days = pd.DatetimeIndex(feat["Date"])
        co = bucket_to_trading_day(c[c["tickers"].str.contains(t)], days)
        mk = bucket_to_trading_day(general, days)
        for si, s in enumerate(splits):
            pd_days = pd.DatetimeIndex(s.dates)
            ci = _gather(co, days, pd_days, K_CO, LOOKBACK, lambda u: store.score.get((u, t), 0))
            mi = _gather(mk, days, pd_days, K_MK, 1, lambda u: 0)
            out[si].append(MMSplit(s.X, s.y, np.full(len(s.y), ti), s.dates, *ci, *mi[:3]))
    merged = []
    for parts in out:
        merged.append(MMSplit(*[np.concatenate([getattr(p, f) for p in parts])
                                for f in MMSplit.__dataclass_fields__]))
    return merged, cols
