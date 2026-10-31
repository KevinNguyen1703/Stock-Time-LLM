"""Time-LLM+FM: foundation-model forecasts (TimesFM 3 + Chronos-2, zero-shot) in the anchor, news + Time-LLM+
as a shrunk learned correction. Compared with each foundation model alone, the anchor alone (FM ensemble +
sentiment, no learning) and an FM + Time-LLM variant without news.

Pre-declared gate: on 2023 and 2024, Time-LLM+FM must beat TimesFM 3 alone on AUC in both years; only then is
2025-26 reported as the check."""
import glob
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from stock_llm.c3_search import COMMON
from stock_llm.runmany import run_jobs

YEARS = {0: "2023", 1: "2024", 2: "2025-26"}


def args(fold, news):
    a = ["--fold", str(fold), "--loss", "dir", "--coverage", "0.2", "--anchor_fm", "1",
         "--out", f"reports/fm/{'news' if news else 'nonews'}", *COMMON]
    if not news:
        a[a.index("--prompt") + 1] = "static"
        a[a.index("--knowledge") + 1] = "0"
    return a


def auc_trade(p, ret, side):
    t = side != 0
    return (roc_auc_score(ret > 0, p), np.mean((side[t] > 0) == (ret[t] > 0)) * 100 if t.any() else np.nan,
            np.mean(side[t] * ret[t]) * 100 if t.any() else np.nan, int(t.sum()))


def report():
    rows = {}
    for f, y in YEARS.items():
        out = {}
        for name, d in (("TimesFM 3 alone", "reports/sota/timesfm3"), ("Chronos-2 alone", "reports/sota/chronos2")):
            z = np.load(glob.glob(f"{d}/pred_*_f{f}_s0.npz")[0], allow_pickle=True)
            out[name] = auc_trade(z["p_total"], z["ret"], z["side_q"])
        for name, d in (("Time-LLM+FM (ours)", "reports/fm/news"), ("FM + Time-LLM, no news", "reports/fm/nonews")):
            j = json.load(open(glob.glob(f"{d}/*_f{f}_*_fm.json")[0]))
            z = np.load(glob.glob(f"{d}/pred_*_f{f}_*_fm.npz")[0], allow_pickle=True)
            out[name] = auc_trade(z["p_total"], z["ret"], z["side_q"])
            if name.startswith("Time-LLM+FM"):
                at = j["anchor_test"]
                out["Anchor alone (FM + sentiment)"] = (at["auc_total"], at["trade_win"], at["trade_avg_pct"], at["n_trades"])
        rows[y] = out
    names = list(rows["2023"])
    print(f"{'method':32s} " + " | ".join(f"{y:^26s}" for y in rows))
    for n in names:
        print(f"{n:32s} " + " | ".join(f"AUC {rows[y][n][0]:.3f} win {rows[y][n][1]:5.1f} n {rows[y][n][3]:4d}" for y in rows))
    gate = all(rows[y]["Time-LLM+FM (ours)"][0] > rows[y]["TimesFM 3 alone"][0] for y in ("2023", "2024"))
    print("\nDEV GATE (beat TimesFM 3 alone on AUC in 2023 and 2024):", "PASS" if gate else "FAIL")
    return rows, gate


if __name__ == "__main__":
    run_jobs([(f"fm_{'news' if n else 'nonews'}_f{f}", args(f, n)) for n in (True, False) for f in YEARS])
    report()
