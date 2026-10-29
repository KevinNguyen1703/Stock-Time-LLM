"""Download daily foreign and proprietary (tự doanh) trading flows per ticker, plus VN-Index prices.

Outputs data/flows/<SYMBOL>_foreign.csv, data/flows/<SYMBOL>_prop.csv, data/prices/VNINDEX.csv.
Values are in VND. Proprietary data on CafeF only exists from a later start date than foreign data.
"""
import argparse
import datetime as dt
import os

import pandas as pd

from common import BASE, get
from fetch_prices import fetch_symbol as fetch_price

ENDPOINTS = {
    "foreign": ("GDKhoiNgoai", lambda d: d["Data"],
                {"Ngay": "Date", "KLGDRong": "net_vol", "GTDGRong": "net_val", "KLMua": "buy_vol",
                 "GtMua": "buy_val", "KLBan": "sell_vol", "GtBan": "sell_val", "RoomConLai": "room_left",
                 "DangSoHuu": "own_pct"}),
    "prop": ("GDTuDoanh", lambda d: d["Data"]["ListDataTudoanh"],
             {"Date": "Date", "KLcpMua": "buy_vol", "KlcpBan": "sell_vol", "GtMua": "buy_val", "GtBan": "sell_val"}),
}


def fetch(symbol, kind, start, end, page_size=30):
    ep, extract, cols = ENDPOINTS[kind]
    rows = []
    for year in range(start.year, end.year + 1):
        page = 1
        while True:
            r = get(f"{BASE}/Ajax/PageNew/DataHistory/{ep}.ashx", params={
                "Symbol": symbol, "StartDate": max(start, dt.date(year, 1, 1)).strftime("%m/%d/%Y"),
                "EndDate": min(end, dt.date(year, 12, 31)).strftime("%m/%d/%Y"),
                "PageIndex": page, "PageSize": page_size})
            data = extract(r.json()["Data"])
            if not data:
                break
            rows += data
            page += 1
    if not rows:
        return pd.DataFrame(columns=list(cols.values()))
    df = pd.DataFrame(rows)[list(cols)].rename(columns=cols)
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    if kind == "prop":
        df["net_vol"] = df["buy_vol"] - df["sell_vol"]
        df["net_val"] = df["buy_val"] - df["sell_val"]
    return df.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["ACB", "CTG", "FPT", "HPG", "MBB", "SSI", "VCB", "VNM"])
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--out", default="data/flows")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    start, end = dt.date.fromisoformat(a.start), dt.date.today()
    idx = fetch_price("VNINDEX", start, end)
    for c in ["Open", "High", "Low", "Close", "Adj Close", "Change"]:
        idx[c] = idx[c] / 1000          # fetch_prices scales to VND; the index is in points
    idx.to_csv(os.path.join(os.path.dirname(a.out), "prices", "VNINDEX.csv"), index=False)
    print(f"VNINDEX: {len(idx)} rows", flush=True)
    for s in a.symbols:
        for kind in ENDPOINTS:
            df = fetch(s, kind, start, end)
            df.to_csv(f"{a.out}/{s}_{kind}.csv", index=False)
            print(f"{s} {kind}: {len(df)} rows {df.Date.min()} -> {df.Date.max()}", flush=True)
