"""Where is there any predictable signal? Tabular features at close of day t-1, forward h-day return.

Compares horizons (1/5/10/20 days), absolute vs market-relative targets (minus the equal-weight
average of the universe), and linear vs gradient-boosted models, on the rolling-origin folds.
Labels for h>1 overlap, so each fold's train set ends h days before validation starts (purge).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from stock_llm.data.prices import add_features, load_prices
from stock_llm.data.universe import TICKERS

FOLDS = [("2023-01-01", "2024-01-01"), ("2024-01-01", "2025-01-01"), ("2025-01-01", "2026-11-01")]
HORIZONS = [1, 5, 10, 20]


def panel():
    frames = []
    for t in TICKERS:
        f = add_features(load_prices(t))
        f["ticker"] = t
        lp = np.log(f["Adj Close"])
        for h in HORIZONS:  # features on row t are known at close t; target = return from close t to close t+h
            f[f"fwd_{h}"] = lp.shift(-h) - lp
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    for h in HORIZONS:
        df[f"rel_{h}"] = df[f"fwd_{h}"] - df.groupby("Date")[f"fwd_{h}"].transform("mean")
    df["mkt_ret_1"] = df.groupby("Date")["ret_1"].transform("mean")
    df["mkt_ret_5"] = df.groupby("Date")["ret_5"].transform("mean")
    return df


FEATS = ["ret_1", "hl_range", "oc_gap", "vol_chg", "ret_5", "ret_20", "vol_20", "rsi_14", "bb_pos", "macd",
         "mkt_ret_1", "mkt_ret_5"]


def evaluate(df, target, h, model):
    out = []
    for lo, hi in FOLDS:
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        days = np.sort(df["Date"].unique())
        cut = days[np.searchsorted(days, lo.to_datetime64()) - h]          # purge: train labels must end before test starts
        tr = df[(df["Date"] < cut)].dropna(subset=[target])
        te = df[(df["Date"] >= lo) & (df["Date"] < hi)].dropna(subset=[target])
        m = Ridge(alpha=10.0) if model == "ridge" else HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.03, max_iter=300, min_samples_leaf=100, l2_regularization=1.0)
        mu, sd = tr[FEATS].mean(), tr[FEATS].std()
        m.fit((tr[FEATS] - mu) / sd, tr[target])
        p = m.predict((te[FEATS] - mu) / sd)
        y = te[target].to_numpy()
        nz = y != 0
        conf = np.abs(p) >= np.quantile(np.abs(p), 0.8)
        ic_daily = pd.DataFrame({"d": te["Date"].values, "p": p, "y": y}).groupby("d").apply(
            lambda g: g["p"].corr(g["y"], method="spearman") if g["p"].std() > 0 else np.nan, include_groups=False)
        out.append({"dir": np.mean((p[nz] > 0) == (y[nz] > 0)) * 100,
                    "win_top20": np.mean((p[conf & nz] > 0) == (y[conf & nz] > 0)) * 100,
                    "base_up": np.mean(y[nz] > 0) * 100,
                    "IC": np.corrcoef(p, y)[0, 1], "rankIC": ic_daily.mean()})
    return pd.DataFrame(out)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    df = panel()
    rows = []
    for kind in ["fwd", "rel"]:
        for h in HORIZONS:
            for model in ["ridge", "gbm"]:
                r = evaluate(df, f"{kind}_{h}", h, model)
                rows.append({"target": f"{kind}_{h}d", "model": model,
                             **{f"{c}_f{i}": r[c].iloc[i] for c in ["dir", "win_top20"] for i in range(3)},
                             "base_up_mean": r["base_up"].mean(), "IC_mean": r["IC"].mean(), "rankIC_mean": r["rankIC"].mean()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
