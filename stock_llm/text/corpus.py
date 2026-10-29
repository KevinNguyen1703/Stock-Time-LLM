"""One table of every headline since 2020: ticker news + market categories, with company tags.

Ticker-page items are tagged with their ticker. Market items are tagged with every universe ticker
whose code or common name appears in the title/lead; untagged market items stay as general context.
Output: data/text/corpus.parquet (one row per unique URL, stable integer `nid`).
"""
import glob
import os
import re

import numpy as np
import pandas as pd

from stock_llm.data.universe import TICKERS

ALIASES = {
    "ACB": ["ACB", "Á Châu"], "CTG": ["CTG", "VietinBank", "Vietinbank"], "FPT": ["FPT"],
    "HPG": ["HPG", "Hòa Phát", "Hoà Phát"], "MBB": ["MBB", "MB Bank", "MBBank", "Ngân hàng Quân đội"],
    "SSI": ["SSI"], "VCB": ["VCB", "Vietcombank"], "VNM": ["VNM", "Vinamilk"],
}
PATTERNS = {t: re.compile(r"(?<![\wÀ-ỹ])(" + "|".join(map(re.escape, a)) + r")(?![\wÀ-ỹ])") for t, a in ALIASES.items()}
START = "2020-01-01"


def tag(text):
    return "|".join(t for t, p in PATTERNS.items() if p.search(text))


def build(news_dir="data/news", out="data/text/corpus.parquet"):
    frames = []
    for t in TICKERS:
        d = pd.read_csv(f"{news_dir}/{t}.csv")
        d["published"] = pd.to_datetime(d["published"], format="mixed")
        d["lead"] = ""
        d["source"] = np.where(d["url"].str.contains("/du-lieu/"), "filing", "press")
        frames.append(d[["published", "title", "lead", "url", "source"]].assign(page_ticker=t))
    for f in sorted(glob.glob(f"{news_dir}/market_*.csv")):
        d = pd.read_csv(f)
        d["published"] = pd.to_datetime(d["published"], format="mixed")
        d["lead"] = d["lead"].fillna("")
        d["source"] = "market_" + d["category"]
        frames.append(d[["published", "title", "lead", "url", "source"]].assign(page_ticker=""))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["published"] >= START]
    # same URL may come from several pages: merge their ticker tags, keep the most specific source
    df["mention"] = (df["title"] + " " + df["lead"]).map(tag)
    df["tickers"] = df.apply(lambda r: "|".join(sorted(set(filter(None, [r.page_ticker, *r.mention.split("|")])))), axis=1)
    agg = df.groupby("url").agg(published=("published", "min"), title=("title", "first"), lead=("lead", "max"),
                                source=("source", lambda s: sorted(s, key=lambda x: x.startswith("market"))[0]),
                                tickers=("tickers", lambda s: "|".join(sorted(set("|".join(s).split("|")) - {""}))))
    agg = agg.reset_index().sort_values("published").reset_index(drop=True)
    agg["nid"] = np.arange(len(agg))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    agg.to_parquet(out, index=False)
    return agg


if __name__ == "__main__":
    c = build()
    print(f"{len(c)} unique items, {c.published.min()} -> {c.published.max()}")
    print(c["source"].value_counts().to_string())
    tagged = c[c.tickers != ""]
    print(f"\nitems tagged to >=1 universe ticker: {len(tagged)}")
    print(pd.Series("|".join(tagged.tickers).split("|")).value_counts().to_string())
