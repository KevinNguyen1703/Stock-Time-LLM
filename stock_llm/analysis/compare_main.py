"""Compare Time-LLM+ variants (full model) per test year and pooled, against the anchor and the floors.
Usage: python -m stock_llm.analysis.compare_main "label=glob_for_json" ["label2=glob2" ...]"""
import glob
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

rows = []
for spec in sys.argv[1:]:
    label, pattern = spec.split("=", 1)
    files = sorted(glob.glob(pattern))
    preds = []
    for f in files:
        r = json.load(open(f))
        t, a = r["test"], r["anchor_test"]
        rows.append({"variant": label, "year": ["2023", "2024", "2025-26"][r["args"]["fold"]],
                     "auc_total": t["auc_total"], "anchor_auc": a["auc_total"], "auc_company": t["auc_company"],
                     "trades": t["n_trades"], "win": t["trade_win"], "anchor_win": a["trade_win"],
                     "floor": t["floor_primary"], "avg_%": t["trade_avg_pct"]})
        preds.append(np.load(f.replace("/qwen", "/pred_qwen").replace(".json", ".npz"), allow_pickle=True))
    if len(preds) == 3:
        y = np.concatenate([z["ret"] > 0 for z in preds])
        rows.append({"variant": label, "year": "pooled",
                     "auc_total": roc_auc_score(y, np.concatenate([z["p_total"] for z in preds])),
                     "anchor_auc": roc_auc_score(y, np.concatenate([z["anchor"][:, 0] for z in preds])),
                     "auc_company": roc_auc_score(np.concatenate([z["rel"] > 0 for z in preds]),
                                                  np.concatenate([z["p_company"] for z in preds]))})
pd.set_option("display.width", 220)
print(pd.DataFrame(rows).round(3).to_string(index=False))
