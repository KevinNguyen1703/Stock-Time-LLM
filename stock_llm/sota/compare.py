"""SOTA comparison on the Time-LLM+ event dataset: every method on the three test years with identical scoring.

Methods: numeric TSLib models, text+series methods (TaTS, MM-TSFlib, REST-TS on two backbones), zero-shot
foundation models (Chronos-2, TimesFM 3, TimesFM 2.5), CALF, the original Time-LLM (re-implemented in our
framework: statistics prompt, fixed patches, MSE) and Time-LLM+ (C1 news prompt + C2 news-conditioned filter
bank, MSE; results from reports/c3/mse). One seed, each method's own default hyperparameters.
Usage: python -m stock_llm.sota.compare [--run 0]   (--run 0 only rebuilds the table)
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from stock_llm.c3_search import COMMON
from stock_llm.runmany import run_jobs

TRAINED = ["patchtst", "itransformer", "timemixer", "timexer",
           "tats_itransformer", "tats_patchtst", "mmtsf_itransformer", "mmtsf_patchtst",
           "restts_itransformer", "restts_patchtst"]
ZERO = ["chronos2", "timesfm3", "timesfm25"]
FOLDS = (0, 1, 2)
YEARS = ["2023", "2024", "2025-26"]
GROUP = {"patchtst": "numeric", "itransformer": "numeric", "timemixer": "numeric", "timexer": "numeric",
         "chronos2": "foundation (zero-shot)", "timesfm3": "foundation (zero-shot)", "timesfm25": "foundation (zero-shot)",
         "calf": "LLM (no text)", "timellm": "LLM (no text)",
         "tats_itransformer": "text + series", "tats_patchtst": "text + series",
         "mmtsf_itransformer": "text + series", "mmtsf_patchtst": "text + series",
         "restts_itransformer": "text + series", "restts_patchtst": "text + series",
         "timellm_plus": "ours (text + series)"}
SOURCES = {"timellm_plus": "reports/c3/mse", "timellm": "reports/sota/timellm"}


def jobs(include_calf):
    out = []
    for m in TRAINED:
        for f in FOLDS:
            out.append((f"sota_{m}_f{f}", ["--method", m, "--fold", str(f)], {"module": "stock_llm.sota.run"}))
    for m in ZERO:
        for f in FOLDS:
            out.append((f"sota_{m}_f{f}", ["--method", m, "--fold", str(f)],
                        {"module": "stock_llm.sota.run", "local_only": True}))
    for f in FOLDS:   # original Time-LLM in our framework: statistics prompt, fixed patching, MSE
        args = ["--fold", str(f), "--loss", "mse", "--coverage", "0.2", "--out", "reports/sota/timellm", *COMMON]
        args[args.index("--prompt") + 1] = "static"
        args[args.index("--knowledge") + 1] = "0"
        args[args.index("--patching") + 1] = "fixed"
        out.append((f"sota_timellm_f{f}", args))
    if include_calf:
        for f in FOLDS:
            out.append((f"sota_calf_f{f}", ["--fold", str(f)],
                        {"module": "stock_llm.sota.calf_run", "python": ".venv-calf/bin/python", "local_only": True}))
    return out


def load(method, fold):
    d = SOURCES.get(method, f"reports/sota/{method}")
    f = sorted(glob.glob(f"{d}/pred_*_f{fold}_*.npz"))
    f = [x for x in f if "_wf" not in x]
    return np.load(f[0], allow_pickle=True) if f else None


def table():
    rows = []
    for m in [*GROUP]:
        zs = [load(m, f) for f in FOLDS]
        if any(z is None for z in zs):
            continue
        r = {"method": m, "family": GROUP[m]}
        for y, z in zip(YEARS, zs):
            p, ret, side = z["p_total"], z["ret"], z["side_q"]
            t = side != 0
            r[f"AUC {y}"] = roc_auc_score(ret > 0, p)
            r[f"win {y}"] = np.mean((side[t] > 0) == (ret[t] > 0)) * 100 if t.any() else np.nan
        p = np.concatenate([(z["p_total"] - np.median(z["p_total"])) / (z["p_total"].std() + 1e-12) for z in zs])
        y = np.concatenate([z["ret"] > 0 for z in zs])
        r["AUC pooled"] = roc_auc_score(y, p)
        side = np.concatenate([z["side_q"] for z in zs]); ret = np.concatenate([z["ret"] for z in zs])
        t = side != 0
        r["trades"] = int(t.sum())
        r["win pooled"] = np.mean((side[t] > 0) == (ret[t] > 0)) * 100
        r["avg/trade %"] = np.mean(side[t] * ret[t]) * 100
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("AUC pooled", ascending=False)
    floors = [max(np.mean(load("timellm_plus", f)["ret"] > 0), np.mean(load("timellm_plus", f)["ret"] <= 0)) * 100
              for f in FOLDS] if load("timellm_plus", 0) is not None else []
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))
    if floors:
        print("floors (majority direction, % of events):", dict(zip(YEARS, np.round(floors, 1))))
    df.to_csv("reports/sota_comparison.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, default=1)
    a = ap.parse_args()
    if a.run:
        calf_ok = os.path.exists("logs/calf_env.log") and "CALF ENV OK" in open("logs/calf_env.log").read()
        print("CALF environment ready:", calf_ok, flush=True)
        run_jobs(jobs(calf_ok))
    table()


if __name__ == "__main__":
    main()
