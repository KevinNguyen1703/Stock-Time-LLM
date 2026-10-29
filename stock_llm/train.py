"""Train one model on the pooled tickers and report validation + (once) test metrics.

Test is touched exactly once, with the checkpoint chosen by validation loss.
Usage: python -m stock_llm.train --model lstm --seeds 0 1 2
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from stock_llm.baselines import direction_acc
from stock_llm.data.prices import add_features, load_prices
from stock_llm.data.universe import TICKERS
from stock_llm.data.windows import make_fold, make_splits
from stock_llm.models.baselines import REGISTRY


FOLDS = [("2023-01-01", "2024-01-01"), ("2024-01-01", "2025-01-01"), ("2025-01-01", "2026-11-01")]


def build_data(tickers, seq_len, fold=None):
    per = {}
    for t in tickers:
        feat = add_features(load_prices(t))
        (tr, va, te), cols = (make_fold(feat, seq_len, *FOLDS[fold]) if fold is not None
                              else make_splits(feat, seq_len))
        per[t] = (tr, va, te)
    return per, cols


def pool(per, which):
    xs = [per[t][which] for t in per]
    return (np.concatenate([s.X for s in xs]), np.concatenate([s.y for s in xs]),
            np.concatenate([np.full(len(s.y), i) for i, s in enumerate(xs)]))


def predict(model, X, device, bs=512):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(model(torch.from_numpy(X[i:i + bs]).to(device)).cpu().numpy())
    return np.concatenate(out)


def selective(pred, y, thr):
    """Trade only when |pred| > thr (long if pred>0, short if <0). Flat days are not counted as wins or losses."""
    t = (np.abs(pred) > thr) & (y != 0)
    n = int(t.sum())
    if n == 0:
        return {"n": 0, "win": float("nan"), "avg_ret": float("nan")}
    side = np.sign(pred[t])
    return {"n": n, "win": float(np.mean(side == np.sign(y[t])) * 100), "avg_ret": float(np.mean(side * y[t]))}


def metrics(pred, y):
    m = {"ret_MAE": float(np.mean(np.abs(pred - y))), "dir_acc": direction_acc(pred, y),
         "IC": float(np.corrcoef(pred, y)[0, 1]) if pred.std() > 0 else 0.0,
         "trade_1pct": selective(pred, y, 0.01)}
    q = np.quantile(np.abs(pred), 0.8)          # most confident 20% of days, scale-free
    m["trade_top20"] = selective(pred, y, q)
    big = np.abs(y) > 0.01                       # DIAGNOSTIC ONLY: uses the outcome to pick days, not tradable
    m["diag_big_move_dir_acc"] = direction_acc(pred[big], y[big]) if big.sum() else float("nan")
    return m


def run(args, seed, per, n_feat, device):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr, _ = pool(per, 0)
    Xva, yva, _ = pool(per, 1)
    if args.min_move > 0:                        # drop minor-fluctuation days from TRAINING only
        keep = np.abs(ytr) >= args.min_move
        Xtr, ytr = Xtr[keep], ytr[keep]
    y_sd = ytr.std()
    model = REGISTRY[args.model](n_feat, args.seq_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    best, best_state, bad = 1e9, None, 0
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy((ytr / y_sd).astype(np.float32))
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), args.batch_size):
            idx = perm[i:i + args.batch_size]
            loss = nn.functional.huber_loss(model(Xt[idx].to(device)), yt[idx].to(device), delta=2.0)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        v = float(np.mean(np.abs(predict(model, Xva, device) * y_sd - yva)))
        if v < best - 1e-7:
            best, bad = v, 0
            best_state = {k: x.detach().clone() for k, x in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    res = {"seed": seed, "epochs_run": epoch + 1, "val": metrics(predict(model, Xva, device) * y_sd, yva), "test": {}}
    for t, (_, _, te) in per.items():
        res["test"][t] = metrics(predict(model, te.X, device) * y_sd, te.y)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lstm", choices=list(REGISTRY))
    ap.add_argument("--tickers", nargs="+", default=TICKERS)
    ap.add_argument("--seq_len", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min_move", type=float, default=0.0, help="drop train samples with |return| below this")
    ap.add_argument("--fold", type=int, default=None, help="rolling-origin fold index (see FOLDS); default: single 70/10/20 split")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="reports/results")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    per, cols = build_data(args.tickers, args.seq_len, args.fold)
    results = []
    for s in args.seeds:
        t0 = time.time()
        r = run(args, s, per, len(cols), device)
        results.append(r)
        print(f"{args.model} seed {s}: {r['epochs_run']} ep, {time.time()-t0:.0f}s | val {r['val']}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    json.dump({"args": vars(args), "results": results}, open(f"{args.out}/{args.model}_f{args.fold}_mm{args.min_move}.json", "w"), indent=1)
