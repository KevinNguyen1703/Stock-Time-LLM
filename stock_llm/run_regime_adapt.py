"""C3 final (regime-adaptive re-centring): development test vs plain MSE, then the 2025-26 walk-forward.

Pre-declared gate: on 2023 and 2024 (single fit each), regime_adapt must beat MSE on trade winrate AND
average return (validation-quantile long/short rule, top/bottom 10%) with AUC no more than 0.01 lower.
Only if the gate passes is the 2025-26 walk-forward (vs MSE walk-forward) reported as the check."""
import glob

import numpy as np
from sklearn.metrics import roc_auc_score

from stock_llm.c3_search import COMMON
from stock_llm.runmany import run_jobs
from stock_llm.walk_forward_c3 import jobs as wf_jobs, summary as wf_summary

DEV = [(f"c3_regime_adapt_f{f}", ["--fold", str(f), "--loss", "regime_adapt", "--coverage", "0.2",
                                  "--out", "reports/c3/regime_adapt", *COMMON]) for f in (0, 1, 2)]


def rule(z):
    side, r = z["side_q"], z["ret"]
    t = side != 0
    return {"n": int(t.sum()), "win": float(np.mean((side[t] > 0) == (r[t] > 0)) * 100),
            "avg": float(np.mean(side[t] * r[t]) * 100), "auc": float(roc_auc_score(r > 0, z["p_total"])),
            "shorts": int((side == -1).sum())}


def dev_gate():
    ok = True
    for f, y in ((0, "2023"), (1, "2024"), (2, "2025-26 (check)")):
        ra = rule(np.load(glob.glob(f"reports/c3/regime_adapt/pred_*_f{f}_*.npz")[0], allow_pickle=True))
        ms = rule(np.load(glob.glob(f"reports/c3/mse/pred_*_f{f}_*.npz")[0], allow_pickle=True))
        good = ra["win"] > ms["win"] and ra["avg"] > ms["avg"] and ra["auc"] >= ms["auc"] - 0.01
        if f < 2:
            ok &= good
        print(f"{y:16s} regime_adapt win {ra['win']:.1f} avg {ra['avg']:+.2f}% auc {ra['auc']:.3f} "
              f"n {ra['n']} shorts {ra['shorts']} | mse win {ms['win']:.1f} avg {ms['avg']:+.2f}% auc {ms['auc']:.3f} "
              f"n {ms['n']} shorts {ms['shorts']} " + ("BETTER" if good else ""), flush=True)
    print("DEV GATE:", "PASS" if ok else "FAIL", flush=True)
    return ok


if __name__ == "__main__":
    run_jobs(DEV + [j for j in wf_jobs() if j[0].startswith("wfc3_ra_")])
    passed = dev_gate()
    print("\nwalk-forward 2025-26 " + ("(check, gate passed)" if passed else "(gate FAILED - shown for transparency only)"))
    wf_summary()
