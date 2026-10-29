"""Download headlines (title, timestamp, url) of company news/announcements from CafeF.

Only metadata is stored, not article bodies. Output: data/news/<SYMBOL>.csv
"""
import argparse
import os
import re

import pandas as pd
from bs4 import BeautifulSoup

from common import BASE, get


def fetch_symbol(symbol, page_size=20, max_pages=2000, stop_before=None):
    rows, seen = [], set()
    for page in range(1, max_pages + 1):
        r = get(f"{BASE}/Ajax/Events_RelatedNews_New.aspx", params={
            "symbol": symbol, "floorID": 0, "configID": 0,
            "PageIndex": page, "PageSize": page_size, "Type": 2})
        items = BeautifulSoup(r.text, "lxml").select("li")
        new = 0
        for li in items:
            t, a = li.select_one(".timeTitle"), li.select_one("a")
            if not (t and a):
                continue
            url = a["href"].split("?")[0]
            if url in seen:
                continue
            seen.add(url)
            new += 1
            rows.append({"symbol": symbol, "published": t.get_text(strip=True),
                         "title": re.sub(r"\s+", " ", a.get_text(strip=True)),
                         "url": "https://cafef.vn" + url if url.startswith("/") else url})
        if page % 10 == 0:
            print(f"{symbol}: page {page}, {len(rows)} items, oldest {rows[-1]['published'] if rows else '-'}")
        if new == 0:
            break
    df = pd.DataFrame(rows)
    df["published"] = pd.to_datetime(df["published"], format="%d/%m/%Y %H:%M")
    return df.sort_values("published").reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["ACB", "CTG", "FPT", "HPG", "MBB", "SSI", "VCB", "VNM"])
    ap.add_argument("--out", default="data/news")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for s in a.symbols:
        df = fetch_symbol(s)
        df.to_csv(f"{a.out}/{s}.csv", index=False)
        print(f"{s}: {len(df)} items {df.published.min()} -> {df.published.max()}")
