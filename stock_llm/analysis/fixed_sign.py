"""Overfitting check for the text features: time-series model + only the LLM net sentiment with a FIXED
positive sign. score = logit(p_ts) + w * z(s_all), with w >= 0 picked on validation from a small grid.
If text helps only through year-specific fitted signs, this version will not help."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from stock_llm.data.events import split_fold
from stock_llm.frozen_combiner import features, fit_predict

W = [0.0, 0.05, 0.1, 0.2, 0.4]

df = pd.read_pickle("data/text/events.pkl")
cols = json.load(open("data/text/events_cols.json"))
rows = []
for target in ["ret", "rel"]:
    for kind in ["logit", "gbm"]:
        pooled = {"ts": [], "fixed": [], "y": []}
        for fold in range(3):
            tr, va, te, _ = split_fold(df, fold, 30)
            tr, va, te = [d.dropna(subset=[f"{target}_30"]) for d in (tr, va, te)]
            (ts_tr, tx_tr), (ts_va, tx_va), (ts_te, tx_te) = [features(d, cols) for d in (tr, va, te)]
            ytr, yva, yte = [(d[f"{target}_30"] > 0).to_numpy() for d in (tr, va, te)]
            pv, pt = fit_predict(kind, ts_tr, ytr, [ts_va, ts_te], seed=fold)
            mu, sd = tx_tr["s_all"].mean(), tx_tr["s_all"].std()
            lv = np.log(pv / (1 - pv)); lt = np.log(pt / (1 - pt))
            zv, zt = (tx_va["s_all"] - mu) / sd, (tx_te["s_all"] - mu) / sd
            w = max(W, key=lambda w: roc_auc_score(yva, lv + w * zv))
            ft = lt + w * zt
            rows.append({"target": target, "model": kind, "fold": fold, "w_chosen": w,
                         "ts_only": roc_auc_score(yte, pt), "ts+fixed_sentiment": roc_auc_score(yte, ft),
                         "sentiment_alone": roc_auc_score(yte, zt)})
            pooled["ts"].append(pt); pooled["fixed"].append(1 / (1 + np.exp(-ft))); pooled["y"].append(yte)
        y = np.concatenate(pooled["y"])
        rows.append({"target": target, "model": kind, "fold": "pooled", "w_chosen": np.nan,
                     "ts_only": roc_auc_score(y, np.concatenate(pooled["ts"])),
                     "ts+fixed_sentiment": roc_auc_score(y, np.concatenate(pooled["fixed"])), "sentiment_alone": np.nan})
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).round(3).to_string(index=False))
