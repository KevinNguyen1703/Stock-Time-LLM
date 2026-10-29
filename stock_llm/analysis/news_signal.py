"""Does the scraped ticker news carry information about next-day moves?

Uses only items usable before the open of day t (stock_llm.data.news alignment). Reports:
  - size: |return| on news vs no-news days (volatility information)
  - direction: a crude Vietnamese finance lexicon score vs the sign of the day-t return
"""
import re

import numpy as np
import pandas as pd

from stock_llm.data.news import bucket_to_trading_day
from stock_llm.data.prices import load_prices
from stock_llm.data.universe import TICKERS

POS = ["lãi", "tăng", "lợi nhuận", "kỷ lục", "vượt", "cổ tức", "mua vào", "nâng", "tích cực", "khả quan",
       "bứt phá", "đột biến", "trúng thầu", "hợp đồng", "mở rộng", "mua ròng"]
NEG = ["lỗ", "giảm", "bán ra", "bán ròng", "phạt", "khởi tố", "vi phạm", "nợ xấu", "thua", "sụt", "đình chỉ",
       "cảnh báo", "thoái vốn", "lao dốc", "thủng", "tiêu cực", "bắt"]


def score(title):
    t = title.lower()
    return sum(len(re.findall(r"\b" + re.escape(w) + r"\b", t)) for w in POS) - \
        sum(len(re.findall(r"\b" + re.escape(w) + r"\b", t)) for w in NEG)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    rows, ev = [], []
    for s in TICKERS:
        px = load_prices(s)
        px["ret"] = np.log(px["Adj Close"]).diff()
        n = pd.read_csv(f"data/news/{s}.csv", parse_dates=["published"])
        n = n[n["published"] >= "2020-01-01"].copy()
        n["kind"] = np.where(n["url"].str.contains("/du-lieu/"), "filing", "press")
        date_only = n["published"].dt.time == pd.Timestamp("00:00").time()
        n.loc[date_only, "published"] += pd.Timedelta(hours=23, minutes=59)
        b = bucket_to_trading_day(n, px["Date"])
        b["s"] = b["title"].map(score)
        g = b.groupby("target_day").agg(n_press=("kind", lambda k: (k == "press").sum()),
                                        n_filing=("kind", lambda k: (k == "filing").sum()), sent=("s", "sum"))
        d = px.set_index("Date").join(g).fillna({"n_press": 0, "n_filing": 0, "sent": 0}).dropna(subset=["ret"])
        d["abs_ret"] = d["ret"].abs()
        d["ticker"] = s
        ev.append(d)
    d = pd.concat(ev)
    print("== |next-day return| by news presence (pooled, %)")
    for name, mask in [("no news", (d.n_press == 0) & (d.n_filing == 0)), ("filing only", (d.n_filing > 0) & (d.n_press == 0)),
                       ("press >=1", d.n_press > 0), ("press >=3", d.n_press >= 3)]:
        print(f"  {name:12s} days={mask.sum():5d}  mean|ret|={d.loc[mask,'abs_ret'].mean()*100:.2f}  share |ret|>2%={(d.loc[mask,'abs_ret']>.02).mean()*100:.1f}")
    print("\n== direction vs lexicon sentiment (days with nonzero score, non-flat)")
    m = (d.sent != 0) & (d.ret != 0)
    hit = (np.sign(d.loc[m, "sent"]) == np.sign(d.loc[m, "ret"]))
    print(f"  days={m.sum()}  hit={hit.mean()*100:.1f}%   per-year:",
          hit.groupby(d.loc[m].index.year).agg(["mean", "size"]).assign(mean=lambda x: (x["mean"]*100).round(1)).to_dict("index"))
    for thr in [2, 3]:
        mm = (d.sent.abs() >= thr) & (d.ret != 0)
        print(f"  |score|>={thr}: days={mm.sum()} hit={(np.sign(d.loc[mm,'sent'])==np.sign(d.loc[mm,'ret'])).mean()*100:.1f}%")
