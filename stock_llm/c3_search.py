"""C3 redesign loop: train selection-focused objectives and test them against the plain MSE model.

Base (fixed): C1 news prompt + knowledge tokens, C2 filter bank + news hypernetwork (patching=bank_text).
Success (per dev year 2023 and 2024): at equal coverage (top --coverage of test events, chosen from scores
only, no labels), the variant's confidence-selected trades beat, on BOTH winrate and average return, each
reference's most-extreme-score trades: "mse" (plain regression), "dir" (the same anchored direction head
without the selective head; isolates C3) and "anchor" (news sentiment alone); and the variant's AUC is
within 0.01 of the better reference model.
Usage: python -m stock_llm.c3_search --variants sel hybrid_sel sel_profit sel_regime [--folds 0 1]
"""
import argparse
import glob
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from stock_llm.runmany import run_jobs

COMMON = ["--prompt", "news", "--knowledge", "1", "--patching", "bank_text", "--shrink", "0.1",
          "--trade_q", "0.1", "--horizon", "30", "--lr", "0.001", "--epochs", "3", "--hyper_lr_mult", "5"]
OUT = "reports/c3"
YEARS = {0: "2023", 1: "2024", 2: "2025-26"}


def jobs(variants, folds, coverage, extra):
    out = []
    for v in ["mse", "dir", *variants]:
        for f in folds:
            name = f"c3_{v}{extra.replace(' ', '').replace('--', '_')}_f{f}"
            out.append((name, ["--fold", str(f), "--loss", v, "--coverage", str(coverage),
                               "--out", f"{OUT}/{v}{extra.replace(' ', '').replace('--', '_')}", *COMMON,
                               *extra.split()]))
    return out


def selection(z, coverage, by):
    p = z["anchor"][:, 0] if by == "anchor" else z["p_total"]
    score = z["conf"] if by == "conf" else np.abs(p - np.median(p))
    sel = score >= np.quantile(score, 1 - coverage)
    side, r = np.sign(p[sel] - (0 if by == "conf" else np.median(p))), z["ret"][sel]
    return {"n": int(sel.sum()), "win": float(np.mean((side > 0) == (r > 0)) * 100),
            "avg": float(np.mean(side * r) * 100), "auc": float(roc_auc_score(z["ret"] > 0, p))}


def evaluate(variants, folds, coverage, extra):
    tag = extra.replace(" ", "").replace("--", "_")
    load = lambda v, f: np.load(glob.glob(f"{OUT}/{v}{tag}/pred_*_f{f}_*.npz")[0], allow_pickle=True)  # noqa: E731
    report = {}
    for v in variants:
        ok, rows = True, []
        for f in folds:
            s = selection(load(v, f), coverage, "conf")
            refs = {"mse": selection(load("mse", f), coverage, "abs"),
                    "dir": selection(load("dir", f), coverage, "abs"),        # same model, no selective head
                    "anchor": selection(load("dir", f), coverage, "anchor")}  # news sentiment only, no learning
            good = all(s["win"] > b["win"] and s["avg"] > b["avg"] for b in refs.values()) and \
                s["auc"] >= max(refs["mse"]["auc"], refs["dir"]["auc"]) - 0.01
            ok &= good
            rows.append({"year": YEARS[f], "variant": s, **refs, "better": good})
            print(f"{v:12s} {YEARS[f]}: SEL win {s['win']:.1f} avg {s['avg']:+.2f}% auc {s['auc']:.3f} | "
                  + " | ".join(f"{k} win {b['win']:.1f} avg {b['avg']:+.2f}% auc {b['auc']:.3f}" for k, b in refs.items())
                  + f" | n {s['n']} " + ("BETTER" if good else ""), flush=True)
        report[v] = {"pass": bool(ok), "rows": rows}
        print(f"{v}: {'PASS' if ok else 'fail'}", flush=True)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--coverage", type=float, default=0.2)
    ap.add_argument("--extra", default="", help="extra train_plus args shared by all runs of this round")
    a = ap.parse_args()
    run_jobs(jobs(a.variants, a.folds, a.coverage, a.extra))
    rep = evaluate(a.variants, a.folds, a.coverage, a.extra)
    json.dump(rep, open(f"{OUT}/round{a.extra.replace(' ', '').replace('--', '_')}_{'_'.join(a.variants)}.json", "w"),
              indent=1)


if __name__ == "__main__":
    main()
