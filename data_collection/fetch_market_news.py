"""Download market-wide news (stock market / banking / macro) from CafeF's category timelines.

Stores headline, lead sentence (the site's own `sapo` snippet), timestamp and URL only.
Pages back until the oldest item predates --start. Resumable: re-running restarts a few pages before
the end of the existing file (pages hold 15 items).
Output: data/news/market_<category>.csv
"""
import argparse
import csv
import os
import re

from bs4 import BeautifulSoup

from common import get

CATEGORIES = {"stock": 18831, "macro": 18833, "bank": 18834}
FIELDS = ["category", "published", "title", "lead", "url"]


def parse_page(html, category):
    out = []
    for it in BeautifulSoup(html, "lxml").select("div.tlitem"):
        a, t, p = it.select_one("h3 a"), it.select_one("span.time"), it.select_one("p.sapo")
        if not (a and t):
            continue
        out.append({"category": category, "published": t["title"][:19],
                    "title": re.sub(r"\s+", " ", a.get_text(strip=True)),
                    "lead": re.sub(r"\s+", " ", p.get_text(strip=True)) if p else "",
                    "url": "https://cafef.vn" + a["href"] if a["href"].startswith("/") else a["href"]})
    return out


def run(category, zone, start, out_dir, max_pages=6000):
    path = f"{out_dir}/market_{category}.csv"
    seen = set()
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            seen = {r["url"] for r in csv.DictReader(f)}
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        if new_file:
            w.writeheader()
        for page in range(max(1, len(seen) // 15 - 5), max_pages + 1):
            items = parse_page(get(f"https://cafef.vn/timelinelist/{zone}/{page}.chn").text, category)
            if not items:
                break
            fresh = [i for i in items if i["url"] not in seen]
            for i in fresh:
                seen.add(i["url"])
                w.writerow(i)
            f.flush()
            if page % 25 == 0:
                print(f"{category}: page {page}, {len(seen)} items, oldest {items[-1]['published']}", flush=True)
            if items[-1]["published"][:10] < start:
                break
    print(f"{category}: done, {len(seen)} items", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=list(CATEGORIES), choices=list(CATEGORIES))
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--out", default="data/news")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for c in a.categories:
        run(c, CATEGORIES[c], a.start, a.out)
