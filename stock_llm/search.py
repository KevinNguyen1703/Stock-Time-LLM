"""Configuration search for Time-LLM+ with a protected final holdout.

Development = test folds 0 and 1 (2023, 2024). Every candidate configuration is trained and scored on
those two folds only. Fold 2 (2025-26) is the final holdout: it is run once, for the confirmed winner.

Success criterion for a trading rule (q = val-quantile thresholds, conf = confidence head), per dev year:
  winrate >= --target, trades >= --min_trades, and winrate >= floor + --margin, where floor = the winrate of
  the trivial rule "take every event in the majority direction" in that year.
A candidate that passes is re-run with two more seeds; it is confirmed if the seed-averaged winrate still
passes in both years. Workers: this machine and (once idle) the ssh host "gpu2".
Usage: python -m stock_llm.search [--max_configs 30]
"""
import argparse
import itertools
import json
import os
import random
import subprocess
import threading
import time

import numpy as np

OUT = "reports/search"
SPACE = {"shrink": [0.03, 0.1, 0.3], "coverage": [0.1, 0.15, 0.2, 0.3], "horizon": [30, 45],
         "lr": [5e-4, 1e-3, 2e-3], "epochs": [2, 3], "hyper_lr_mult": [1.0, 5.0], "trade_q": [0.1, 0.2]}
BASE = {"shrink": 0.1, "coverage": 0.3, "horizon": 30, "lr": 1e-3, "epochs": 3, "hyper_lr_mult": 5.0, "trade_q": 0.2}
# hand-ordered first candidates (one change at a time from the current best), then random samples
FIRST = [{"coverage": 0.2}, {"coverage": 0.1}, {"coverage": 0.2, "trade_q": 0.1}, {"horizon": 45, "coverage": 0.2},
         {"shrink": 0.03, "coverage": 0.2}, {"shrink": 0.3, "coverage": 0.2}, {"lr": 5e-4, "coverage": 0.2},
         {"epochs": 2, "coverage": 0.2}, {"horizon": 45, "coverage": 0.1}]


def cfg_id(c, seed):
    return "c_" + "_".join(f"{k}{c[k]:g}" for k in sorted(c)) + f"_s{seed}"


def args_for(c, fold, seed):
    a = ["--fold", fold, "--seed", seed, "--out", f"{OUT}/{cfg_id(c, seed)}", "--tag_suffix", "_" + cfg_id(c, seed)]
    for k, v in c.items():
        a += [f"--{k}", v]
    return [str(x) for x in a]


# ---------------------------------------------------------------- workers
class Worker(threading.Thread):
    def __init__(self, name, jobs, lock, remote=False):
        super().__init__(daemon=True)
        self.name_, self.jobs, self.lock, self.remote = name, jobs, lock, remote

    def remote_busy(self):
        # gpu2 is free once the 7B baseline (started separately) has written its last fold; running beside it
        # (41 of 48 GB in use) would run out of memory
        r = subprocess.run(["ssh", "gpu2", "ls ~/Gumiho/Stock-Time-LLM/reports/plus/qwen7b_f2_*full_s0.json"],
                           capture_output=True)
        return r.returncode != 0

    def run(self):
        while True:
            if self.remote and self.remote_busy():
                time.sleep(120)
                continue
            with self.lock:
                if not self.jobs:
                    return
                c, fold, seed = self.jobs.pop(0)
            args = args_for(c, fold, seed)
            log = f"logs/search/{cfg_id(c, seed)}_f{fold}.log"
            t0 = time.time()
            if self.remote:
                subprocess.run(["bash", "stock_llm/remote.sh", "sync"], check=True)
                cmd = (f"cd Gumiho/Stock-Time-LLM && mkdir -p logs/search && .venv/bin/python -m stock_llm.train_plus "
                       f"{' '.join(args)} > {log} 2>&1")
                subprocess.run(["ssh", "gpu2", cmd])
                subprocess.run(["rsync", "-az", f"gpu2:Gumiho/Stock-Time-LLM/{OUT}/{cfg_id(c, seed)}/",
                                f"{OUT}/{cfg_id(c, seed)}/"])
                subprocess.run(["rsync", "-az", f"gpu2:Gumiho/Stock-Time-LLM/{log}", log])
            else:
                with open(log, "w") as f:
                    subprocess.run([".venv/bin/python", "-m", "stock_llm.train_plus", *args], stdout=f, stderr=f)
            print(f"[{self.name_}] {cfg_id(c, seed)} fold {fold} done in {time.time()-t0:.0f}s", flush=True)


def result_files(c, seed):
    d = f"{OUT}/{cfg_id(c, seed)}"
    return {f: [x for x in (os.listdir(d) if os.path.isdir(d) else [])
                if x.startswith(f"pred_") and f"_f{f}_" in x] for f in (0, 1, 2)}


def all_done(c, seed, folds=(0, 1)):
    r = result_files(c, seed)
    return all(r[f] for f in folds)


# ---------------------------------------------------------------- scoring
def rule_stats(z, rule):
    r = z["ret"]
    if rule == "q":
        side = z["side_q"]
    else:
        side = np.where(z["sel_conf"].astype(bool), np.sign(z["p"]), 0)
    t = side != 0
    win = float(np.mean((side[t] > 0) == (r[t] > 0)) * 100) if t.any() else float("nan")
    # floor for the same trades' directions: majority direction over all events of the year
    floor = float(max(np.mean(r > 0), np.mean(r <= 0)) * 100)
    return {"n": int(t.sum()), "win": win, "floor": floor, "avg": float(np.mean(side[t] * r[t]) * 100) if t.any() else float("nan")}


def score(c, seeds, folds=(0, 1)):
    out = {}
    for rule in ("q", "conf"):
        per = []
        for f in folds:
            st = [rule_stats(np.load(f"{OUT}/{cfg_id(c, s)}/{result_files(c, s)[f][0]}", allow_pickle=True), rule)
                  for s in seeds]
            per.append({k: float(np.nanmean([x[k] for x in st])) for k in ("n", "win", "floor", "avg")})
        out[rule] = per
    return out


def passes(per, a):
    return all(p["win"] >= a.target and p["n"] >= a.min_trades and p["win"] >= p["floor"] + a.margin for p in per)


def run_jobs(configs, folds, seeds, a, lock):
    jobs = [(c, f, s) for c in configs for s in seeds for f in folds]
    workers = [Worker("local", jobs, lock)] + ([Worker("gpu2", jobs, lock, remote=True)] if a.use_gpu2 else [])
    for w in workers:
        w.start()
    for w in workers:
        w.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=60.0)
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--min_trades", type=int, default=80)
    ap.add_argument("--max_configs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=2, help="configs evaluated per round (one per worker)")
    ap.add_argument("--use_gpu2", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); os.makedirs("logs/search", exist_ok=True)
    lock = threading.Lock()
    rng = random.Random(0)
    keys = sorted(SPACE)
    queue = [{**BASE, **d} for d in FIRST]
    seen = {cfg_id(c, 0) for c in queue}
    while len(queue) < a.max_configs:
        c = {k: rng.choice(SPACE[k]) for k in keys}
        if cfg_id(c, 0) not in seen:
            seen.add(cfg_id(c, 0)); queue.append(c)
    board = []
    for i in range(0, len(queue), a.batch):
        batch = [c for c in queue[i:i + a.batch] if not all_done(c, 0)]
        run_jobs(batch, (0, 1), (0,), a, lock)
        for c in queue[i:i + a.batch]:
            if not all_done(c, 0):
                continue
            sc = score(c, (0,))
            board.append({"cfg": c, "score": sc})
            for rule in ("q", "conf"):
                per = sc[rule]
                print(f"{cfg_id(c, 0)} [{rule}] " + " | ".join(
                    f"{y}: win {p['win']:.1f} floor {p['floor']:.1f} n {p['n']:.0f} avg {p['avg']:+.2f}%"
                    for y, p in zip(("2023", "2024"), per)) + ("  PASS" if passes(per, a) else ""), flush=True)
                if passes(per, a):
                    print(f"candidate: {cfg_id(c, 0)} rule {rule}; confirming with seeds 1, 2", flush=True)
                    run_jobs([c], (0, 1), (1, 2), a, lock)
                    conf = score(c, (0, 1, 2))[rule]
                    print("seed-averaged:", [{k: round(v, 2) for k, v in p.items()} for p in conf], flush=True)
                    if passes(conf, a):
                        print(f"CONFIRMED {cfg_id(c, 0)} rule {rule}; running final holdout (2025-26) once", flush=True)
                        run_jobs([c], (2,), (0, 1, 2), a, lock)
                        hold = score(c, (0, 1, 2), folds=(2,))[rule]
                        json.dump({"cfg": c, "rule": rule, "dev": conf, "holdout_2025_26": hold},
                                  open(f"{OUT}/WINNER.json", "w"), indent=1)
                        print("HOLDOUT 2025-26:", [{k: round(v, 2) for k, v in p.items()} for p in hold], flush=True)
                        return
        json.dump(board, open(f"{OUT}/leaderboard.json", "w"), indent=1, default=str)
    print("search exhausted without a confirmed configuration", flush=True)


if __name__ == "__main__":
    main()
