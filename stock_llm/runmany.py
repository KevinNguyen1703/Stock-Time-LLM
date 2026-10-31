"""Run a list of jobs split between this machine and the ssh host "gpu2" (if reachable).
A job is (name, args) for stock_llm.train_plus, or (name, args, opts) with opts keys:
  module (default "stock_llm.train_plus"), python (default ".venv/bin/python"), local_only (default False).
Finished jobs (log contains the final JSON with "auc_total") are skipped.
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
    return os.path.exists(log) and '"auc_total"' in open(log, errors="ignore").read()


def _opts(job):
    o = job[2] if len(job) > 2 else {}
    return o.get("module", "stock_llm.train_plus"), o.get("python", ".venv/bin/python"), o.get("local_only", False)


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
            pick = next((j for j in jobs if not (remote and _opts(j)[2])), None)
            if pick is None:
                return
            jobs.remove(pick)
        name, args = pick[0], pick[1]
        module, python, _ = _opts(pick)
        log, t0 = f"logs/runs/{name}.log", time.time()
        if remote:
            r = subprocess.run(["ssh", "gpu2", "cd Gumiho/Stock-Time-LLM && mkdir -p logs/runs && "
                                f"{python} -m {module} {' '.join(args)} > {log} 2>&1"])
            if r.returncode != 0:
                with lock:
                    jobs.append(pick)
                continue
            for d in out_dirs:
                subprocess.run(["rsync", "-az", f"gpu2:Gumiho/Stock-Time-LLM/{d}/", f"{d}/"])
            subprocess.run(["rsync", "-az", f"gpu2:Gumiho/Stock-Time-LLM/{log}", log])
        else:
            with open(log, "w") as f:
                subprocess.run([python, "-m", module, *args], stdout=f, stderr=f)
        print(f"[{label}] {name} done in {time.time()-t0:.0f}s", flush=True)


def run_jobs(jobs, use_gpu2=True):
    os.makedirs("logs/runs", exist_ok=True)
    out_dirs = sorted({j[1][j[1].index("--out") + 1] for j in jobs if "--out" in j[1]})
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
