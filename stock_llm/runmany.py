"""Run a list of train_plus jobs split between this machine and the ssh host "gpu2" (if reachable).
Each job is a name plus train_plus arguments; finished jobs (log contains the final JSON) are skipped.
Usage (from code): run_jobs([("name", ["--fold", "0", ...]), ...])"""
import os
import subprocess
import threading
import time


def reachable():
    return subprocess.run(["ssh", "-o", "ConnectTimeout=10", "-o", "BatchMode=yes", "gpu2", "true"],
                          capture_output=True).returncode == 0


def done(name):
    log = f"logs/runs/{name}.log"
    return os.path.exists(log) and '"anchor_test"' in open(log, errors="ignore").read()


def _worker(label, jobs, lock, remote, out_dirs):
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
            name, args = jobs.pop(0)
        log, t0 = f"logs/runs/{name}.log", time.time()
        if remote:
            r = subprocess.run(["ssh", "gpu2", "cd Gumiho/Stock-Time-LLM && mkdir -p logs/runs && "
                                f".venv/bin/python -m stock_llm.train_plus {' '.join(args)} > {log} 2>&1"])
            if r.returncode != 0:
                with lock:
                    jobs.append((name, args))
                continue
            for d in out_dirs:
                subprocess.run(["rsync", "-az", f"gpu2:Gumiho/Stock-Time-LLM/{d}/", f"{d}/"])
            subprocess.run(["rsync", "-az", f"gpu2:Gumiho/Stock-Time-LLM/{log}", log])
        else:
            with open(log, "w") as f:
                subprocess.run([".venv/bin/python", "-m", "stock_llm.train_plus", *args], stdout=f, stderr=f)
        print(f"[{label}] {name} done in {time.time()-t0:.0f}s", flush=True)


def run_jobs(jobs, use_gpu2=True):
    os.makedirs("logs/runs", exist_ok=True)
    out_dirs = sorted({a[a.index("--out") + 1] for _, a in jobs if "--out" in a})
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
    jobs = [j for j in jobs if not done(j[0])]
    lock = threading.Lock()
    ws = [threading.Thread(target=_worker, args=("local", jobs, lock, False, out_dirs))]
    if use_gpu2:
        ws.append(threading.Thread(target=_worker, args=("gpu2", jobs, lock, True, out_dirs)))
    for t in ws:
        t.start()
    for t in ws:
        t.join()
