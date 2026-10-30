"""Ablation grid for Time-LLM+ around the confirmed configuration, split across this machine and gpu2.

Each block varies one contribution with the other two fixed at the final setting:
  C1 prompting  : none | static (original Time-LLM) | data (earlier dynamic prompt) | news | news + knowledge |
                  shuffled news + knowledge
  C2 patching   : fixed | fft_attn (earlier FFT+attention) | bank | bank_text | bank_text_aligned (KASD) |
                  KASD with shuffled news
  C3 objective  : mse | hybrid (earlier hybrid loss) | decomp | dro | dro_rank | full | full with shuffled news
Common settings come from reports/search/WINNER.json (shrink, coverage, trade_q, horizon, lr, epochs).
Results: reports/ablation/<tag>.json (+ pred_*.npz). Resumable.
Usage: python -m stock_llm.ablations [--folds 0 1 2] [--seeds 0]
"""
import argparse
import glob
import json
import os
import subprocess
import threading
import time

OUT = "reports/ablation"
FULL = {"prompt": "news", "knowledge": 1, "patching": "bank_text_aligned", "loss": "full"}
ROWS = {
    "C1": [{"prompt": "none", "knowledge": 0}, {"prompt": "static", "knowledge": 0}, {"prompt": "data", "knowledge": 0},
           {"prompt": "news", "knowledge": 0}, {"prompt": "news", "knowledge": 1},
           {"prompt": "news_shuffled", "knowledge": 1}],
    "C2": [{"patching": "fixed"}, {"patching": "fft_attn"}, {"patching": "bank"}, {"patching": "bank_text"},
           {"patching": "bank_text_aligned"}, {"patching": "bank_text_aligned", "prompt": "news_shuffled"}],
    "C3": [{"loss": "mse"}, {"loss": "hybrid"}, {"loss": "decomp"}, {"loss": "dro"}, {"loss": "dro_rank"},
           {"loss": "full"}, {"loss": "full", "prompt": "news_shuffled"}],
}


def configs():
    seen, out = set(), []
    for block, rows in ROWS.items():
        for r in rows:
            c = {**FULL, **r}
            key = json.dumps(c, sort_keys=True)
            if key not in seen:
                seen.add(key); out.append(c)
    return out


def tag(c, fold, seed, common):
    return (f"qwen1.5b_f{fold}_h{common['horizon']}_p-{c['prompt']}_k{c['knowledge']}_{c['patching']}_{c['loss']}"
            f"_s{seed}_shrink{common['shrink']:g}")


def job_args(c, fold, seed, common):
    a = ["--fold", fold, "--seed", seed, "--out", OUT]
    for k, v in {**common, **c}.items():
        a += [f"--{k}", v]
    return [str(x) for x in a]


def reachable():
    return subprocess.run(["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", "gpu2", "true"],
                          capture_output=True).returncode == 0


def worker(name, jobs, lock, common, remote):
    synced = False
    while True:
        if remote:
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
            c, fold, seed = jobs.pop(0)
        args, t0 = job_args(c, fold, seed, common), time.time()
        log = f"logs/ablation/{tag(c, fold, seed, common)}.log"
        if remote:
            r = subprocess.run(["ssh", "gpu2", "cd Gumiho/Stock-Time-LLM && mkdir -p logs/ablation && "
                                f".venv/bin/python -m stock_llm.train_plus {' '.join(args)} > {log} 2>&1"])
            if r.returncode != 0:
                with lock:
                    jobs.append((c, fold, seed))
                continue
            subprocess.run(["rsync", "-az", f"gpu2:Gumiho/Stock-Time-LLM/{OUT}/", f"{OUT}/"])
            subprocess.run(["rsync", "-az", f"gpu2:Gumiho/Stock-Time-LLM/{log}", log])
        else:
            with open(log, "w") as f:
                subprocess.run([".venv/bin/python", "-m", "stock_llm.train_plus", *args], stdout=f, stderr=f)
        print(f"[{name}] {tag(c, fold, seed, common)} done in {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--use_gpu2", type=int, default=1)
    a = ap.parse_args()
    w = json.load(open("reports/search/WINNER.json"))["cfg"]
    common = {k: w[k] for k in ("shrink", "coverage", "trade_q", "horizon", "lr", "epochs", "hyper_lr_mult")}
    os.makedirs(OUT, exist_ok=True); os.makedirs("logs/ablation", exist_ok=True)
    jobs = [(c, f, s) for c in configs() for s in a.seeds for f in a.folds
            if not os.path.exists(f"{OUT}/{tag(c, f, s, common)}.json")]
    print(f"{len(configs())} configurations, {len(jobs)} runs to do", flush=True)
    lock = threading.Lock()
    ws = [threading.Thread(target=worker, args=("local", jobs, lock, common, False))]
    if a.use_gpu2:
        ws.append(threading.Thread(target=worker, args=("gpu2", jobs, lock, common, True)))
    for t in ws:
        t.start()
    for t in ws:
        t.join()
    print("ablation grid complete", flush=True)


if __name__ == "__main__":
    main()
