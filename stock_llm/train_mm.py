"""Train the price-news fusion model on one rolling-origin fold.

--text none       : price only (same architecture, cross-attention skipped)
--text real       : headline embeddings + LLM score
--text embed_only : headline embeddings, no LLM score
--text score_only : LLM score tokens only (no embeddings)
--text shuffled   : real tokens, but each sample's news sets are permuted across samples within each split
Selection by validation loss; test evaluated once.
Usage: python -m stock_llm.train_mm --fold 2 --text real --seeds 0 1 2
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from stock_llm.data.multimodal import TextStore, build_fold
from stock_llm.data.universe import TICKERS
from stock_llm.models.fusion import FusionModel
from stock_llm.train import metrics

NEWS_FIELDS = {"co": ["co_idx", "co_age", "co_src", "co_score"], "mk": ["mk_idx", "mk_age", "mk_src", "co_score"]}


def to_tensors(s, device, shuffle_rng=None):
    perm = shuffle_rng.permutation(len(s.y)) if shuffle_rng is not None else np.arange(len(s.y))
    t = {"X": torch.from_numpy(s.X), "ticker": torch.from_numpy(s.ticker), "y": torch.from_numpy(s.y.astype(np.float32))}
    for k in ["co_idx", "co_age", "co_src", "co_score", "mk_idx", "mk_age", "mk_src"]:
        t[k] = torch.from_numpy(getattr(s, k)[perm])
    t["mk_score"] = torch.zeros_like(t["mk_idx"], dtype=torch.float32)
    return {k: v.to(device) for k, v in t.items()}


def batch_news(t, i):
    return {"co": (t["co_idx"][i], t["co_age"][i], t["co_src"][i], t["co_score"][i]),
            "mk": (t["mk_idx"][i], t["mk_age"][i], t["mk_src"][i], t["mk_score"][i])}


@torch.no_grad()
def predict(model, t, E, bs=1024):
    model.eval()
    ret, logit = [], []
    for i in range(0, len(t["y"]), bs):
        sl = slice(i, i + bs)
        r, lg, _ = model(t["X"][sl], t["ticker"][sl], E, batch_news(t, sl))
        ret.append(r); logit.append(lg)
    return torch.cat(ret).cpu().numpy(), torch.sigmoid(torch.cat(logit)).cpu().numpy()


def run(args, seed, splits, n_feat, E, device):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed) if args.text == "shuffled" else None
    tr, va, te = [to_tensors(s, device, rng) for s in splits]
    y_sd = float(tr["y"].std())
    model = FusionModel(n_feat, args.seq_len, len(TICKERS), E.shape[1],
                        text="real" if args.text == "shuffled" else args.text).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    keep = (tr["y"].abs() >= args.min_move).nonzero().squeeze(1)
    best, best_state, bad = 1e9, None, 0
    for epoch in range(args.epochs):
        model.train()
        for i in keep[torch.randperm(len(keep), device=device)].split(args.batch_size):
            r, lg, _ = model(tr["X"][i], tr["ticker"][i], E, batch_news(tr, i))
            yt = tr["y"][i]
            loss = F.huber_loss(r, yt / y_sd, delta=2.0) + args.cls_weight * \
                F.binary_cross_entropy_with_logits(lg[yt != 0], (yt[yt != 0] > 0).float())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        pr, pp = predict(model, va, E)
        yv = va["y"].cpu().numpy()
        v = float(np.mean(np.abs(pr * y_sd - yv)))
        if v < best - 1e-7:
            best, bad, best_state = v, 0, {k: x.detach().clone() for k, x in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    res = {"seed": seed, "epochs_run": epoch + 1, "gates": model.gates() if args.text != "none" else {}}
    for name, t in [("val", va), ("test", te)]:
        pr, pp = predict(model, t, E)
        y = t["y"].cpu().numpy()
        res[name] = {"reg": metrics(pr * y_sd, y), "cls": metrics(pp - 0.5, y)}
        if name == "test":
            np.savez(f"{args.out}/pred_{args.text}_f{args.fold}_s{seed}.npz", ret=pr * y_sd, p_up=pp, y=y,
                     ticker=t["ticker"].cpu().numpy(), dates=splits[2].dates)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="real", choices=["none", "real", "embed_only", "score_only", "shuffled"])
    ap.add_argument("--fold", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=5e-2)
    ap.add_argument("--cls_weight", type=float, default=0.5)
    ap.add_argument("--min_move", type=float, default=0.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="reports/mm")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    store = TextStore()
    E = torch.from_numpy(store.emb).to(device)
    splits, cols = build_fold(store, args.fold, args.seq_len)
    results = []
    for s in args.seeds:
        t0 = time.time()
        r = run(args, s, splits, len(cols), E, device)
        results.append(r)
        t = r["test"]
        print(f"[{args.text} f{args.fold} s{s}] {r['epochs_run']}ep {time.time()-t0:.0f}s | test dir {t['reg']['dir_acc']:.1f} "
              f"IC {t['reg']['IC']:+.3f} top20(reg) {t['reg']['trade_top20']['win']:.1f}/{t['reg']['trade_top20']['n']} "
              f"top20(cls) {t['cls']['trade_top20']['win']:.1f}/{t['cls']['trade_top20']['n']} gates {r['gates']}", flush=True)
    json.dump({"args": vars(args), "results": results},
              open(f"{args.out}/{args.text}_f{args.fold}_mm{args.min_move}.json", "w"), indent=1)
