"""Table of reports/llm/*.json (excluding overfit sanity runs): val/test AUC, trade winrate vs floors."""
import glob
import json

import pandas as pd

rows = []
for f in sorted(glob.glob("reports/llm/*.json")):
    if "overfit" in f:
        continue
    r = json.load(open(f))
    a, v, t = r["args"], r["val"], r["test"]
    rows.append({"target": a["target"], "H": a["horizon"], "fold": a["fold"], "mode": a["mode"], "ep": a["epochs"],
                 "val_auc": v["auc"], "test_auc": t["auc"], "test_n": t["n"], "trades": t["n_trades"],
                 "trade_win": t.get("trade_win"), "floor_up": t["always_up"],
                 "floor_best": max(t["always_up"], 100 - t["always_up"]),
                 "long_win": t.get("long_win"), "short_win": t.get("short_win"),
                 "avg_ret_pct": t.get("trade_avg_ret_pct")})
df = pd.DataFrame(rows).sort_values(["target", "H", "fold", "mode"])
pd.set_option("display.width", 250)
print(df.round(3).to_string(index=False))
