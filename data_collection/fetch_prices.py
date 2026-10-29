"""Download daily OHLCV for VN tickers from CafeF into data/prices/<SYMBOL>.csv.

CafeF quotes prices in thousand VND; we scale by 1000 (VND).
Raw Close is NOT adjusted for splits/stock dividends; Change is the day's move vs the exchange's
adjusted reference price, which stock_llm.data.prices uses to build a clean adjusted series.
"""
import argparse
import datetime as dt
import os

import pandas as pd

from common import BASE, get

COLS = {"Ngay": "Date", "GiaMoCua": "Open", "GiaCaoNhat": "High", "GiaThapNhat": "Low",
        "GiaDongCua": "Close", "GiaDieuChinh": "Adj Close", "KhoiLuongKhopLenh": "Volume",
        "ThayDoi": "Change"}


def parse_change(s):
    """'-0,70 (-1,83%)' -> -0.70 (thousand VND, change vs the exchange's adjusted reference price)."""
    return float(s.split(" ")[0].replace(",", "."))


def fetch_window(symbol, start, end, page_size=30):
    """The endpoint serves at most ~1 year per request and caps page size at 30."""
    rows, page = [], 1
    while True:
        r = get(f"{BASE}/Ajax/PageNew/DataHistory/PriceHistory.ashx", params={
            "Symbol": symbol, "StartDate": start.strftime("%m/%d/%Y"),
            "EndDate": end.strftime("%m/%d/%Y"), "PageIndex": page, "PageSize": page_size})
        data = r.json()["Data"]["Data"]
        if not data:
            break
        rows += data
        page += 1
    return rows


def fetch_symbol(symbol, start, end):
    rows = []
    for year in range(start.year, end.year + 1):
        rows += fetch_window(symbol, max(start, dt.date(year, 1, 1)), min(end, dt.date(year, 12, 31)))
        print(f"{symbol}: {year} done, {len(rows)} rows")
    df = pd.DataFrame(rows)[list(COLS)].rename(columns=COLS)
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    df["Change"] = df["Change"].map(parse_change)
    for c in ["Open", "High", "Low", "Close", "Adj Close", "Change"]:
        df[c] = df[c] * 1000
    return df.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["ACB", "CTG", "FPT", "HPG", "MBB", "SSI", "VCB", "VNM"])
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--out", default="data/prices")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for s in a.symbols:
        df = fetch_symbol(s, dt.date.fromisoformat(a.start), dt.date.today())
        df.to_csv(f"{a.out}/{s}.csv", index=False)
        print(f"{s}: {len(df)} rows {df.Date.min().date()} -> {df.Date.max().date()}")
