"""Shared inputs for the SOTA comparison, built from the same event dataset and splits as Time-LLM+.

Per event (one row of data/text/events.pkl):
  x        [60, 18]   the event's feature window (rows up to the close of t-1), scaled with training stats,
                      channels reordered so the target channel ret_1 is LAST (TSLib "MS" convention)
  fut      [30]       realised daily return path of the holding window (training target for forecasters),
                      scaled by the training std of ret_1
  text_win [60, 1024] daily BGE-M3 news embedding aligned to the window (TaTS): step i carries the news that
                      became usable before the open of the day after row i, so the last step is the event
                      day's pre-open news; zero vector on days without news
  items    [8, 1024]  embeddings of the event's (up to 8) news items, mask (MM-TSFlib, REST-TS)
  ret, rel            30-day raw and peer-relative returns (evaluation)
"""
import json

import numpy as np
import pandas as pd

from stock_llm.data.events import SEQ_LEN, split_fold
from stock_llm.data.multimodal import TextStore
from stock_llm.data.news import bucket_to_trading_day
from stock_llm.data.prices import load_prices
from stock_llm.data.universe import TICKERS
from stock_llm.train_plus import future_paths

TARGET = "ret_1"
MAX_ITEMS = 8


def channel_order(cols):
    others = [c for c in cols if c != TARGET]
    return [cols.index(c) for c in others + [TARGET]], others + [TARGET]


class TextTables:
    """Daily mean news embedding per ticker (by first usable trading day) and a title -> row map."""
    def __init__(self):
        st = TextStore()
        self.emb = st.emb
        c = st.corpus
        self.days, self.daily = {}, {}
        for t in TICKERS:
            days = pd.DatetimeIndex(load_prices(t)["Date"])
            b = bucket_to_trading_day(c[c["tickers"].str.contains(t)], days)
            pos = pd.Series(np.arange(len(days)), index=days)
            tab = np.zeros((len(days), self.emb.shape[1]), np.float32)
            cnt = np.zeros(len(days))
            for p, r in zip(pos.loc[b["target_day"]].to_numpy(), b["row"].to_numpy()):
                tab[p] += self.emb[r]; cnt[p] += 1
            tab[cnt > 0] /= cnt[cnt > 0, None]
            self.days[t], self.daily[t] = pos, tab.astype(np.float16)
        self.title_row = dict(zip(c["title"], c["row"]))


def build_split(df, scaler, order, sd_target, h, tt):
    ts = np.stack(df["ts"].to_list())
    x = ((ts - scaler[0]) / scaler[1])[:, :, order].astype(np.float32)
    fut = (future_paths(df, h) / sd_target).astype(np.float32)
    text = np.zeros((len(df), SEQ_LEN, tt.emb.shape[1]), np.float16)
    items = np.zeros((len(df), MAX_ITEMS, tt.emb.shape[1]), np.float16)
    mask = np.zeros((len(df), MAX_ITEMS), bool)
    for j, (t, d, its) in enumerate(zip(df["ticker"], pd.to_datetime(df["date"]), df["items"])):
        p = tt.days[t].get(d)
        if p is not None and p >= SEQ_LEN - 1:
            text[j] = tt.daily[t][p - SEQ_LEN + 1:p + 1]
        for k, it in enumerate(its[:MAX_ITEMS]):
            r = tt.title_row.get(it["title"])
            if r is not None:
                items[j, k] = tt.emb[r]; mask[j, k] = True
    return {"x": x, "fut": fut, "text": text, "items": items, "mask": mask,
            "ret": df[f"ret_{h}"].to_numpy(), "rel": df[f"rel_{h}"].to_numpy(),
            "ticker": df["ticker"].to_numpy(), "date": df["date"].to_numpy(), "idx": df.index.to_numpy()}


def build_fold(fold, h=30, val_days=250, test_range=None, tt=None):
    df = pd.read_pickle("data/text/events.pkl")
    cols = json.load(open("data/text/events_cols.json"))
    order, names = channel_order(cols)
    tr, va, te, scaler = split_fold(df, fold, h, val_days, test_range)
    tr, va, te = [d.dropna(subset=[f"rel_{h}", f"ret_{h}"]) for d in (tr, va, te)]
    sd_target = float(scaler[1][cols.index(TARGET)])
    tt = tt or TextTables()
    out = {k: build_split(d, scaler, order, sd_target, h, tt) for k, d in (("train", tr), ("val", va), ("test", te))}
    return out, {"channels": names, "sd_target": sd_target}
