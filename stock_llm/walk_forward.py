"""Walk-forward evaluation of the confirmed Time-LLM+ configuration on the 2025-26 holdout.

For each calendar quarter, the model is retrained on all data before it (purged 12-month validation
window immediately before the quarter sets the trade thresholds) and tested on that quarter only. The
configuration is fixed in advance (reports/search/WINNER.json); nothing is tuned on these quarters.
Runs are split between this machine and the ssh host "gpu2".
Usage: python -m stock_llm.walk_forward [--seeds 0 1 2]
"""
import argparse
import glob
import json
import os
import subprocess
import threading
import time

import numpy as np

QUARTERS = [("2025-01-01", "2025-04-01"), ("2025-04-01", "2025-07-01"), ("2025-07-01", "2025-10-01"),
            ("2025-10-01", "2026-01-01"), ("2026-01-01", "2026-04-01"), ("2026-04-01", "2026-07-01"),
            ("2026-07-01", "2026-10-01")]
OUT = "reports/walkforward"


def job_args(cfg, q, seed):
    lo, hi = QUARTERS[q]
    a = ["--fold", "2", "--seed", seed, "--test_start", lo, "--test_end", hi,
         "--out", f"{OUT}/s{seed}", "--tag_suffix", f"_wf{lo[:7]}"]
    for k, v in cfg.items():
        a += [f"--{k}", v]
    return [str(x) for x in a]


def reachable():
    return subprocess.run(["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", "gpu2", "true"],
                          capture_output=True).returncode == 0


def worker(name, jobs, lock, cfg, remote):
    synced = False
    while True:
        if remote:   # gpu2 may be offline: only take a job while it is reachable
            if not reachable():
                with lock:
                    if not jobs:
                        return
                time.sleep(120)
                continue
            if not synced:
                synced = subprocess.run(["bash", "stock_llm/remote.sh", "sync"]).returncode == 0
        with lock:
            if not jobs:
                return
            q, seed = jobs.pop(0)
        args, log, t0 = job_args(cfg, q, seed), f"logs/walkforward/s{seed}_q{q}.log", time.time()
        if remote:
            r = subprocess.run(["ssh", "gpu2", "cd Gumiho/Stock-Time-LLM && mkdir -p logs/walkforward && "
                            f".venv/bin/python -m stock_llm.train_plus {' '.join(args)} > {log} 2>&1"])
            if r.returncode != 0:          # connection lost mid-run: give the job back
                with lock:
                    jobs.append((q, seed))
                continue
            subprocess.run(["rsync", "-az", f"gpu2:Gumiho/Stock-Time-LLM/{OUT}/", f"{OUT}/"])
            subprocess.run(["rsync", "-az", f"gpu2:Gumiho/Stock-Time-LLM/{log}", log])
        else:
            with open(log, "w") as f:
                subprocess.run([".venv/bin/python", "-m", "stock_llm.train_plus", *args], stdout=f, stderr=f)
        print(f"[{name}] quarter {QUARTERS[q][0]} seed {seed} done in {time.time()-t0:.0f}s", flush=True)


def summarize(seeds):
    rows = []
    for s in seeds:
        files = sorted(glob.glob(f"{OUT}/s{s}/pred_*_wf*.npz"))
        z = [np.load(f, allow_pickle=True) for f in files]
        side = np.concatenate([x["side_q"] for x in z]); r = np.concatenate([x["ret"] for x in z])
        t, L, S = side != 0, side == 1, side == -1
        rows.append({"seed": s, "quarters": len(files), "trades": int(t.sum()),
                     "win": float(np.mean((side[t] > 0) == (r[t] > 0)) * 100),
                     "floor": float(max(np.mean(r > 0), np.mean(r <= 0)) * 100),
                     "avg_%": float(np.mean(side[t] * r[t]) * 100),
                     "long_win": float(np.mean(r[L] > 0) * 100), "short_win": float(np.mean(r[S] <= 0) * 100),
                     "up_rate": float(np.mean(r > 0) * 100)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--use_gpu2", type=int, default=1)
    a = ap.parse_args()
    cfg = json.load(open("reports/search/WINNER.json"))["cfg"]
    os.makedirs("logs/walkforward", exist_ok=True)
    for s in a.seeds:
        os.makedirs(f"{OUT}/s{s}", exist_ok=True)
    jobs = [(q, s) for s in a.seeds for q in range(len(QUARTERS))
            if not glob.glob(f"{OUT}/s{s}/pred_*_wf{QUARTERS[q][0][:7]}.npz")]   # resumable
    lock = threading.Lock()
    ws = [threading.Thread(target=worker, args=("local", jobs, lock, cfg, False))]
    if a.use_gpu2:
        ws.append(threading.Thread(target=worker, args=("gpu2", jobs, lock, cfg, True)))
    for w in ws:
        w.start()
    for w in ws:
        w.join()
    rows = summarize(a.seeds)
    json.dump({"cfg": cfg, "per_seed": rows}, open(f"{OUT}/summary.json", "w"), indent=1)
    for r in rows:
        print({k: round(v, 2) if isinstance(v, float) else v for k, v in r.items()}, flush=True)


if __name__ == "__main__":
    main()
