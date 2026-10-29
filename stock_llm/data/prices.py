"""Load a ticker's raw CafeF prices and turn them into a model-ready feature frame.

Raw Close is not adjusted for splits / stock dividends (it jumps, e.g. CTG -32% on 2025-12-17).
CafeF's `Change` column is each day's move versus the exchange's adjusted reference price, so
`Close / (Close - Change) - 1` is the true daily return on every day, corporate actions included.
We compound those returns into a series anchored at the latest raw close, and rescale O/H/L by
the same per-day factor. Volume is left raw (only its log-change is used).
"""
import numpy as np
import pandas as pd

PRICE_DIR = "data/prices"
MAX_LEGAL_RET = 0.105  # HNX band is 10%; larger "returns" are unadjusted corporate-action jumps
TARGET = "Adj Close"


def load_prices(symbol, price_dir=PRICE_DIR):
    df = pd.read_csv(f"{price_dir}/{symbol}.csv", parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    ret = df["Close"] / (df["Close"] - df["Change"]) - 1
    ret.iloc[0] = 0.0
    artifact = ret.abs() > MAX_LEGAL_RET
    if artifact.any():
        print(f"{symbol}: neutralised {int(artifact.sum())} corporate-action day(s): "
              + ", ".join(f"{d:%Y-%m-%d} ({r:+.1%})" for d, r in zip(df.loc[artifact, 'Date'], ret[artifact])))
    ret[artifact] = 0.0
    logp = np.log1p(ret).cumsum()
    adj = df["Close"].iloc[-1] * np.exp(logp - logp.iloc[-1])
    factor = adj / df["Close"]
    for c in ["Open", "High", "Low"]:
        df[c] = df[c] * factor
    df["Close"] = df["Adj Close"] = adj
    return df.drop(columns=["Change"])


def add_features(df):
    """Technical features computed only from past/current rows (no look-ahead)."""
    c = df["Adj Close"]
    out = pd.DataFrame({"Date": df["Date"], "Adj Close": c})
    out["ret_1"] = np.log(c).diff()
    out["hl_range"] = (df["High"] - df["Low"]) / c
    out["oc_gap"] = (df["Close"] - df["Open"]) / df["Open"]
    out["vol_chg"] = np.log(df["Volume"].replace(0, np.nan)).diff()
    for w in (5, 20):
        out[f"ret_{w}"] = np.log(c).diff(w)
    out["vol_20"] = out["ret_1"].rolling(20).std()
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    out["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    out["bb_pos"] = (c - ma20) / (2 * sd20)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    out["macd"] = (ema12 - ema26) / c
    return out.dropna().reset_index(drop=True)
