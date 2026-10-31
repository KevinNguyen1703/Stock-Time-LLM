"""Walk-forward (quarterly retraining) for the C3 design and its references on 2025-26.

Models (same base: news prompt + knowledge tokens, patching=bank_text):
  c3   : loss sel_evidence (evidence-aware, profit-weighted selective head); trades = confidence above the
         validation-quantile threshold (coverage 0.2), direction = sign of the anchored direction score
  dir  : same anchored direction head without the selective head; trades = top/bottom 10% of scores by
         validation-quantile thresholds (about the same 20% coverage)
  mse  : plain regression; trades chosen the same way as dir
All thresholds come from each quarter's own validation window (causal: no test-quarter information).
Usage: python -m stock_llm.walk_forward_c3
"""
import glob

import numpy as np
import pandas as pd

from stock_llm.c3_search import COMMON
from stock_llm.runmany import run_jobs
from stock_llm.walk_forward import QUARTERS

OUT = "reports/walkforward_c3"
MODELS = {"c3": "sel_evidence", "dir": "dir", "mse": "mse", "c3rel": "sel_evidence_rel", "ra": "regime_adapt"}


def jobs():
    out = []
    for m, loss in MODELS.items():
        for lo, hi in QUARTERS:
            out.append((f"wfc3_{m}_{lo[:7]}", ["--fold", "2", "--loss", loss, "--coverage", "0.2",
                                                "--test_start", lo, "--test_end", hi, "--out", f"{OUT}/{m}",
                                                "--tag_suffix", f"_wf{lo[:7]}", *COMMON]))
    return out


def trades(m):
    rows = []
    for f in sorted(glob.glob(f"{OUT}/{m}/pred_*_wf*.npz")):
        z = np.load(f, allow_pickle=True)
        if m.startswith("c3"):
            centre = float(z["p_center"]) if "p_center" in z.files else 0.0
            side = np.where(z["sel_conf"].astype(bool), np.sign(z["p_total"] - centre), 0)
        else:
            side = z["side_q"]
        t = side != 0
        rows.append(pd.DataFrame({"q": f.split("_wf")[1][:7], "side": side[t], "ret": z["ret"][t],
                                  "date": z["date"][t]}))
    return pd.concat(rows) if rows else pd.DataFrame()


def summary():
    allr = np.concatenate([np.load(f, allow_pickle=True)["ret"] for f in glob.glob(f"{OUT}/c3/pred_*_wf*.npz")])
    print(f"2025-26 events: {len(allr)}, up-rate {np.mean(allr > 0)*100:.1f}% (floor {max(np.mean(allr > 0), np.mean(allr <= 0))*100:.1f}%)")
    for m in MODELS:
        d = trades(m)
        if d.empty:
            continue
        win = (d.side > 0) == (d.ret > 0)
        L, S = d.side > 0, d.side < 0
        print(f"{m:4s}: trades {len(d):4d} win {win.mean()*100:5.1f}% avg {np.mean(d.side*d.ret)*100:+.2f}% | "
              f"longs {L.sum()} (win {np.mean(d.ret[L] > 0)*100:.1f}%) shorts {S.sum()} "
              f"(win {np.mean(d.ret[S] <= 0)*100 if S.any() else float('nan'):.1f}%)")
        print("      per quarter win%: " + " ".join(f"{q}:{g.mean()*100:.0f}({len(g)})" for q, g in win.groupby(d.q.values)))


if __name__ == "__main__":
    import sys
    only = sys.argv[1:]          # optional: model names to run (e.g. c3rel); finished runs are skipped anyway
    run_jobs([j for j in jobs() if not only or j[0].split("_")[1] in only])
    summary()
