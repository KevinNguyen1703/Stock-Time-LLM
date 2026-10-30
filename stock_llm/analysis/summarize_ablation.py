"""Ablation table for reports/ablation: per configuration, test AUC (total and company-specific) per year and
pooled over years, trade winrate per year vs that year's floor, and a paired bootstrap of pooled AUC
against the final configuration."""
import glob
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from stock_llm.ablations import FULL, ROWS

YEARS = ["2023", "2024", "2025-26"]


def load():
    runs = {}
    for f in glob.glob("reports/ablation/*.json"):
        r = json.load(open(f))
        a = r["args"]
        key = (a["prompt"], a["knowledge"], a["patching"], a["loss"])
        z = np.load(f.replace("ablation/qwen", "ablation/pred_qwen").replace(".json", ".npz"), allow_pickle=True)
        runs.setdefault(key, {})[a["fold"]] = (r["test"], z)
    return runs


def pooled(fold_map, field, ycol):
    p = np.concatenate([fold_map[f][1][field] for f in range(3)])
    y = np.concatenate([fold_map[f][1][ycol] > 0 for f in range(3)])
    return p, y


def boot_diff(p1, p0, y, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        d.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p0[i]))
    return float(np.mean(np.array(d) <= 0))   # p-value that the row is not better than the reference


def main():
    runs = load()
    full_key = (FULL["prompt"], FULL["knowledge"], FULL["patching"], FULL["loss"])
    ref_p, ref_y = pooled(runs[full_key], "p_total", "ret")
    rows = []
    for block, items in ROWS.items():
        for it in items:
            c = {**FULL, **it}
            key = (c["prompt"], c["knowledge"], c["patching"], c["loss"])
            if key not in runs or len(runs[key]) < 3:
                continue
            fm = runs[key]
            p, y = pooled(fm, "p_total", "ret")
            pc, yc = pooled(fm, "p_company", "rel")
            row = {"block": block, "row": " ".join(f"{k}={v}" for k, v in it.items()),
                   "AUC_total_pooled": roc_auc_score(y, p), "AUC_company_pooled": roc_auc_score(yc, pc)}
            for f in range(3):
                t = fm[f][0]
                row[f"AUC_{YEARS[f]}"] = t["auc_total"]
                row[f"win_{YEARS[f]}"] = f"{t['trade_win']:.1f}/{t['floor_primary']:.1f}"
            row["P(row beats final)"] = np.nan if key == full_key else 1 - boot_diff(p, ref_p, y)
            rows.append(row)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))
    df.to_csv("reports/ablation_table.csv", index=False)


if __name__ == "__main__":
    main()
