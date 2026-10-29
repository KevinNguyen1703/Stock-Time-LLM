"""Attach news to trading days without look-ahead.

Rule: an item is usable for predicting day t only if it was published before the open of t
(09:00). Each item is therefore bucketed to the first trading day whose open is after its
timestamp. Date-only stamps (00:00) are treated as end-of-day, the conservative choice.
"""
import glob

import numpy as np
import pandas as pd

MARKET_OPEN = pd.Timedelta(hours=9)


def load_news(symbol, news_dir="data/news"):
    """Ticker-specific items plus market-wide categories, with a `kind` column."""
    frames = []
    t = pd.read_csv(f"{news_dir}/{symbol}.csv", parse_dates=["published"])
    t["kind"] = np.where(t["url"].str.contains("/du-lieu/"), "filing", "press")
    t["lead"] = ""
    frames.append(t[["published", "title", "lead", "url", "kind"]])
    for f in sorted(glob.glob(f"{news_dir}/market_*.csv")):
        m = pd.read_csv(f, parse_dates=["published"])
        m["kind"] = "market_" + m["category"]
        frames.append(m[["published", "title", "lead", "url", "kind"]])
    df = pd.concat(frames, ignore_index=True).drop_duplicates("url")
    date_only = df["published"].dt.time == pd.Timestamp("00:00").time()
    df.loc[date_only, "published"] += pd.Timedelta(hours=23, minutes=59)
    return df.sort_values("published").reset_index(drop=True)


def bucket_to_trading_day(news, trading_days):
    """Add `target_day`: first trading day whose 09:00 open is strictly after `published`."""
    days = pd.DatetimeIndex(sorted(trading_days))
    opens = days + MARKET_OPEN
    pos = np.searchsorted(opens.values, news["published"].values, side="right")
    out = news.copy()
    out["target_day"] = pd.NaT
    ok = pos < len(days)
    out.loc[ok, "target_day"] = days[pos[ok]]
    return out.dropna(subset=["target_day"])


def texts_for_day(bucketed, day, lookback_days=5, trading_days=None, max_items=8):
    """Most recent items usable to predict `day`, newest first."""
    days = pd.DatetimeIndex(sorted(trading_days))
    i = days.get_loc(pd.Timestamp(day))
    lo = days[max(0, i - lookback_days + 1)]
    sel = bucketed[(bucketed["target_day"] >= lo) & (bucketed["target_day"] <= pd.Timestamp(day))]
    return sel.sort_values("published", ascending=False).head(max_items)
