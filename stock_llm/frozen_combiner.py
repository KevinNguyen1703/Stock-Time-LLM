"""Frozen LLM as news reader + small regularised combiner with time-series features.

The LLM is not trained: its zero-shot judgements (sentiment score and event type per headline) are
aggregated into per-event features and combined with the time-series features of the same event by a
low-capacity model (L2 logistic regression or shallow gradient boosting). Variants:
  ts_only | llm_only | combined | combined_shuffled (text features permuted across events, train and test)
Evaluation per purged rolling-origin fold: test AUC with a bootstrap 95% CI, paired bootstrap of
AUC(combined) - AUC(ts_only), and selective trading with thresholds set on validation.
Usage: python -m stock_llm.frozen_combiner --target rel --horizon 30
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from stock_llm.data.events import split_fold
from stock_llm.models.ts_llm import EVENT_VI

EVENTS = list(EVENT_VI)
TS_COLS = ["ret_1", "ret_5", "ret_20", "ret_60", "vol_20", "rsi_14", "bb_pos", "macd", "fnet_20", "fown_chg20",
           "mkt_ret_20", "mkt_vol_20", "rel_ret_20"]


def text_features(items):
    f = {"n_items": len(items), "n_today": sum(i["age"] == 0 for i in items),
         "n_press": sum(i["source"] != "filing" for i in items),
         "s_all": sum(i["score"] for i in items), "s_today": sum(i["score"] for i in items if i["age"] == 0),
         "s_max": max((i["score"] for i in items), default=0), "s_min": min((i["score"] for i in items), default=0),
         "n_neg": sum(i["score"] < 0 for i in items)}
    for e in EVENTS:
        sel = [i for i in items if i["event"] == e]
        f[f"n_{e}"] = len(sel)
        f[f"s_{e}"] = sum(i["score"] for i in sel)
    return f


def features(df, cols):
    ci = {c: i for i, c in enumerate(cols)}
    ts = pd.DataFrame([{c: t[-1][ci[c]] for c in TS_COLS} for t in df["ts"]], index=df.index)
    tx = pd.DataFrame([text_features(it) for it in df["items"]], index=df.index)
    return ts, tx


def model(kind, seed):
    if kind == "logit":
        return LogisticRegression(C=0.05, max_iter=2000)
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=200, min_samples_leaf=80,
                                          l2_regularization=1.0, random_state=seed)


def fit_predict(kind, Xtr, ytr, Xs, seed=0):
    mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
    m = model(kind, seed).fit((Xtr - mu) / sd, ytr)
    return [m.predict_proba((X - mu) / sd)[:, 1] for X in Xs]


def boot_auc(y, p, q=None, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    a, d = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].mean() < 1:
            a.append(roc_auc_score(y[i], p[i]))
            if q is not None:
                d.append(a[-1] - roc_auc_score(y[i], q[i]))
    out = {"ci_lo": float(np.percentile(a, 2.5)), "ci_hi": float(np.percentile(a, 97.5))}
    if q is not None:
        out["p_not_better_than_ts"] = float(np.mean(np.array(d) <= 0))
    return out


def trades(pv, pt, ret, q=0.2):
    lo, hi = np.quantile(pv, q), np.quantile(pv, 1 - q)
    long, short = pt >= hi, pt <= lo
    t = long | short
    side = np.where(long, 1, -1)
    y = ret > 0
    return {"n_trades": int(t.sum()), "trade_win": float(np.mean((side[t] > 0) == y[t]) * 100) if t.any() else np.nan,
            "avg_ret_pct": float(np.mean(side[t] * ret[t]) * 100) if t.any() else np.nan,
            "floor": float(max(y.mean(), 1 - y.mean()) * 100)}


def run(target, h, kind):
    df = pd.read_pickle("data/text/events.pkl")
    cols = json.load(open("data/text/events_cols.json"))
    rows = []
    for fold in range(3):
        tr, va, te, _ = split_fold(df, fold, h)
        tr, va, te = [d.dropna(subset=[f"{target}_{h}"]) for d in (tr, va, te)]
        (ts_tr, tx_tr), (ts_va, tx_va), (ts_te, tx_te) = [features(d, cols) for d in (tr, va, te)]
        ytr, yte = (tr[f"{target}_{h}"] > 0).to_numpy(), (te[f"{target}_{h}"] > 0).to_numpy()
        ret_te = te[f"{target}_{h}"].to_numpy()
        rng = np.random.default_rng(fold)
        shuf = lambda x: x.iloc[rng.permutation(len(x))].set_axis(x.index)  # noqa: E731
        variants = {
            "ts_only": [ts_tr, ts_va, ts_te],
            "llm_only": [tx_tr, tx_va, tx_te],
            "combined": [pd.concat([a, b], axis=1) for a, b in [(ts_tr, tx_tr), (ts_va, tx_va), (ts_te, tx_te)]],
            "combined_shuffled": [pd.concat([a, shuf(b)], axis=1) for a, b in [(ts_tr, tx_tr), (ts_va, tx_va), (ts_te, tx_te)]],
        }
        preds = {k: fit_predict(kind, v[0], ytr, v[1:], seed=fold) for k, v in variants.items()}
        for k, (pv, pt) in preds.items():
            r = {"fold": fold, "variant": k, "test_auc": roc_auc_score(yte, pt),
                 **boot_auc(yte, pt, preds["ts_only"][1] if k != "ts_only" else None), **trades(pv, pt, ret_te)}
            rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="rel", choices=["ret", "exc", "rel"])
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--model", default="logit", choices=["logit", "gbm"])
    a = ap.parse_args()
    res = run(a.target, a.horizon, a.model)
    pd.set_option("display.width", 250)
    print(f"== target={a.target} H={a.horizon} model={a.model}")
    print(res.round(3).to_string(index=False))
    res.to_csv(f"reports/frozen_{a.target}_h{a.horizon}_{a.model}.csv", index=False)
