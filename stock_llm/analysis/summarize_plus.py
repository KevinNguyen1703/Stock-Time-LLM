"""Ablation table for reports/plus/*.json: per config, test metrics by fold (mean over seeds) and pooled
over folds (predictions concatenated, each made by a model trained on earlier data only)."""
import glob
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

rows, pooled = [], {}
for f in sorted(glob.glob("reports/plus/*.json")):
    if "limit" in f:
        continue
    r = json.load(open(f))
    a, t = r["args"], r["test"]
    cfg = (a["backbone"], a["horizon"], a["prompt"], a["knowledge"], a["patching"], a["loss"])
    rows.append({"cfg": cfg, "fold": a["fold"], "seed": a["seed"], "auc_company": t["auc_company"],
                 "trade_win": t["trade_win"], "n_trades": t["n_trades"], "floor": t["floor_company"],
                 "sel_win": t.get("sel_win", np.nan), "sel_n": t.get("sel_n", np.nan)})
    z = np.load(f.replace(f"/{f.split('/')[-1]}", "/pred_" + f.split("/")[-1].replace(".json", ".npz")))
    pooled.setdefault((cfg, a["seed"]), []).append(pd.DataFrame({"p": z["p"], "y": z["rel"] > 0}))
df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("no results yet")
by_fold = df.groupby(["cfg", "fold"])["auc_company"].mean().unstack().add_prefix("auc_f")
pool = pd.Series({k: roc_auc_score(pd.concat(v).y, pd.concat(v).p) for k, v in pooled.items() if len(v) == 3})
pool = pool.groupby(level=0).agg(["mean", "std"]).add_prefix("pooled_auc_") if len(pool) else pd.DataFrame()
summ = df.groupby("cfg").agg(seeds=("seed", "nunique"), trade_win=("trade_win", "mean"), floor=("floor", "mean"),
                             sel_win=("sel_win", "mean"))
out = summ.join(by_fold).join(pool)
out.index = [" ".join(map(str, c[2:])) for c in out.index]
pd.set_option("display.width", 250)
print(out.round(3).to_string())
