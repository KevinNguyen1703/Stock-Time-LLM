"""Summarise reports/mm/<text>_f<fold>_mm<x>.json over folds and seeds."""
import glob
import json
import sys

import numpy as np
import pandas as pd

tag = sys.argv[1] if len(sys.argv) > 1 else "mm0.0"
rows = []
for f in sorted(glob.glob(f"reports/mm/*_{tag}.json")):
    d = json.load(open(f))
    for r in d["results"]:
        t = r["test"]
        rows.append({"text": d["args"]["text"], "fold": d["args"]["fold"], "seed": r["seed"],
                     "dir": t["reg"]["dir_acc"], "IC": t["reg"]["IC"],
                     "top20_win": t["cls"]["trade_top20"]["win"], "top20_ret_bp": t["cls"]["trade_top20"]["avg_ret"] * 1e4,
                     "gate0": r["gates"].get("layer0", [np.nan])[0] if r["gates"] else np.nan})
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
by_fold = df.pivot_table(index="text", columns="fold", values="top20_win", aggfunc="mean").round(1)
summ = df.groupby("text").agg(dir=("dir", "mean"), IC=("IC", "mean"), IC_sd=("IC", "std"),
                              top20_win=("top20_win", "mean"), top20_ret_bp=("top20_ret_bp", "mean"),
                              gate0=("gate0", "mean")).round(3)
print(summ.join(by_fold.add_prefix("win_f")).to_string())
