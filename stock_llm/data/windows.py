"""Chronological train/val/test windows over a feature frame.

Each sample: X = [seq_len, n_features] (all features jointly), y = next-day log return.
Split borders are in time order; windows never straddle a border (a window belongs to the
split of its prediction day), and the scaler is fit on training rows only.
"""
from dataclasses import dataclass

import numpy as np

FEATURES_EXCLUDE = ("Date", "Adj Close")


@dataclass
class Split:
    X: np.ndarray       # [n, seq_len, F] scaled
    y: np.ndarray       # [n] next-day log return (unscaled)
    dates: np.ndarray   # [n] prediction date
    last_ret: np.ndarray  # [n] return on the last input day (momentum baseline)
    price_prev: np.ndarray  # [n] Adj Close on the last input day


def make_splits(feat, seq_len=60, ratios=(0.7, 0.1, 0.2)):
    cols = [c for c in feat.columns if c not in FEATURES_EXCLUDE]
    vals = feat[cols].to_numpy(dtype=np.float64)
    price = feat["Adj Close"].to_numpy()
    dates = feat["Date"].to_numpy()
    logp = np.log(price)
    n = len(feat)
    n_tr, n_te = int(n * ratios[0]), int(n * ratios[2])
    borders = [(0, n_tr), (n_tr, n - n_te), (n - n_te, n)]  # prediction-day ranges

    mu = vals[:n_tr].mean(0)
    sd = vals[:n_tr].std(0) + 1e-8
    scaled = (vals - mu) / sd

    splits = []
    for lo, hi in borders:
        idx = np.arange(max(lo, seq_len), hi)  # predict day t from rows [t-seq_len, t-1]
        X = np.stack([scaled[t - seq_len:t] for t in idx])
        y = logp[idx] - logp[idx - 1]
        splits.append(Split(X.astype(np.float32), y, dates[idx], logp[idx - 1] - logp[idx - 2], price[idx - 1]))
    return splits, cols


def make_fold(feat, seq_len, test_start, test_end, val_days=120):
    """One rolling-origin fold: test = [test_start, test_end), val = the val_days before it, train = the rest before.
    The scaler is fit on train rows only; windows are assigned to the split of their prediction day."""
    cols = [c for c in feat.columns if c not in FEATURES_EXCLUDE]
    vals = feat[cols].to_numpy(dtype=np.float64)
    price = feat["Adj Close"].to_numpy()
    dates = feat["Date"].to_numpy()
    logp = np.log(price)
    t0 = int(np.searchsorted(dates, np.datetime64(test_start)))
    t1 = int(np.searchsorted(dates, np.datetime64(test_end)))
    borders = [(0, t0 - val_days), (t0 - val_days, t0), (t0, t1)]
    mu, sd = vals[:borders[0][1]].mean(0), vals[:borders[0][1]].std(0) + 1e-8
    scaled = (vals - mu) / sd
    out = []
    for lo, hi in borders:
        idx = np.arange(max(lo, seq_len), hi)
        X = np.stack([scaled[t - seq_len:t] for t in idx]) if len(idx) else np.zeros((0, seq_len, len(cols)))
        out.append(Split(X.astype(np.float32), logp[idx] - logp[idx - 1], dates[idx],
                         logp[idx - 1] - logp[idx - 2], price[idx - 1]))
    return out, cols
