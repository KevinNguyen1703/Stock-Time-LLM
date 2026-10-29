"""Direct test of the zero-shot LLM score: does the score of news known before the open of day t
predict the sign of day t's return (and of the next 5 days)? Reported per year, since years
before the LLM's cutoff (~2024) may contain hindsight leakage."""
import numpy as np
import pandas as pd

from stock_llm.data.news import bucket_to_trading_day
from stock_llm.data.prices import load_prices
from stock_llm.data.universe import TICKERS


def daily_scores(name="qwen25_7b"):
    s = pd.read_parquet(f"data/text/llm_{name}.parquet")
    c = pd.read_parquet("data/text/corpus.parquet")[["url", "published", "source"]]
    s = s.merge(c, on="url")
    date_only = s["published"].dt.time == pd.Timestamp("00:00").time()
    s.loc[date_only, "published"] += pd.Timedelta(hours=23, minutes=59)
    out = []
    for t in TICKERS:
        px = load_prices(t)
        lp = np.log(px["Adj Close"])
        px["r1"] = lp.diff()                    # close t-1 -> close t
        px["r5"] = lp.shift(-4) - lp.shift(1)   # close t-1 -> close t+4
        b = bucket_to_trading_day(s[s.ticker == t], px["Date"])
        g = b.groupby("target_day").agg(score=("score", "sum"), n=("score", "size"),
                                        n_nonzero=("score", lambda x: (x != 0).sum()))
        d = px.set_index("Date")[["r1", "r5"]].join(g, how="inner")
        d["ticker"] = t
        out.append(d)
    return pd.concat(out)


if __name__ == "__main__":
    d = daily_scores()
    d = d[d.score != 0]
    d["year"] = d.index.year
    for h in ["r1", "r5"]:
        x = d[(d[h] != 0) & d[h].notna()]
        hit = np.sign(x.score) == np.sign(x[h])
        print(f"\n== {h}: days with nonzero net score = {len(x)}, hit {hit.mean()*100:.1f}%  (always-up {np.mean(x[h]>0)*100:.1f}%)")
        print("  by year :", {y: f"{v['mean']*100:.1f}% (n={v['size']}, up {u*100:.0f}%)" for (y, v), u in
                              zip(hit.groupby(x.year).agg(["mean", "size"]).iterrows(), (x[h] > 0).groupby(x.year).mean())})
        for name, m in [("positive score", x.score > 0), ("negative score", x.score < 0), ("|score|>=2", x.score.abs() >= 2)]:
            print(f"  {name:15s} n={m.sum():5d} hit={hit[m].mean()*100:.1f}%  mean ret={x.loc[m,h].mean()*100:+.3f}%")
