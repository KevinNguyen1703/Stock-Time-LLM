"""Train/evaluate the event-driven text + time-series LLM on one purged rolling-origin fold.

Modes: full | text_only | ts_only | ts_as_text | shuffle_ts | shuffle_text
Label: up/down of the H-day return (or excess return with --target exc) entered at the open of the event day.
Selective trading: long if p >= the val 1-q quantile, short if p <= the val q quantile (q = --trade_q).
--overfit N trains and evaluates on the same N training samples (pipeline sanity check, not a result).
Usage: python -m stock_llm.train_llm --fold 2 --horizon 30 --mode full
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from stock_llm.data.events import SEQ_LEN, split_fold
from stock_llm.models.ts_llm import EVENT_VI, TS_TOKEN, TSLLM
from stock_llm.text.corpus import ALIASES

BASE = "Qwen/Qwen2.5-7B-Instruct"
TS_TEXT_COLS = ["ret_5", "ret_20", "ret_60", "vol_20", "rsi_14", "fnet_20", "mkt_ret_20", "rel_ret_20"]


def age_vi(a):
    return "hôm nay" if a == 0 else f"{a} phiên trước"


def make_prompt(tok, row, mode, h, n_patch, cols):
    lines = [f"Mã {row.ticker} ({' / '.join(ALIASES[row.ticker][1:] or ALIASES[row.ticker])})."]
    if mode != "ts_only":
        lines.append("Tin sự kiện trước phiên giao dịch:")
        for it in row.items:
            lead = f" {it['lead'][:200]}" if it["lead"] else ""
            lines.append(f"- [{EVENT_VI[it['event']]}, điểm {it['score']:+d}, {age_vi(it['age'])}] {it['title']}.{lead}")
    if mode in ("full", "ts_only", "shuffle_ts", "shuffle_text"):
        lines.append(f"Diễn biến {SEQ_LEN} phiên gần nhất (giá, khối ngoại, VN-Index): " + TS_TOKEN * n_patch)
    elif mode == "ts_as_text":
        last = dict(zip(cols, row.ts[-1]))
        lines.append("Chỉ số gần nhất: " + ", ".join(f"{c}={last[c]:+.3f}" for c in TS_TEXT_COLS))
    lines.append(f"Sau {h} phiên giao dịch, giá cổ phiếu sẽ cao hơn giá mở cửa hôm nay?")
    msgs = [{"role": "system", "content": "Bạn là chuyên gia phân tích chứng khoán Việt Nam."},
            {"role": "user", "content": "\n".join(lines)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


class Batcher:
    def __init__(self, df, model, mode, h, target, scaler, cols, max_len=768, seed=0):
        df = df.reset_index(drop=True)
        rng = np.random.default_rng(seed)
        ts = np.stack(df["ts"].to_list())
        items = df["items"].to_list()
        if mode == "shuffle_ts":
            ts = ts[rng.permutation(len(ts))]
        if mode == "shuffle_text":
            items = [items[i] for i in rng.permutation(len(items))]
        df = df.assign(items=items, ts=list(ts))
        self.prompts = [make_prompt(model.tok, r, mode, h, model.n_patch, cols) for r in df.itertuples()]
        self.ts = torch.from_numpy((ts - scaler[0]) / scaler[1]).float()
        self.ret = df[f"{target}_{h}"].to_numpy()
        self.raw = df[f"ret_{h}"].to_numpy()
        self.y = torch.from_numpy((self.ret > 0).astype(np.float32))
        self.use_ts = mode in ("full", "ts_only", "shuffle_ts", "shuffle_text")
        self.tok, self.max_len, self.df = model.tok, max_len, df

    def __len__(self):
        return len(self.prompts)

    def get(self, idx, device):
        enc = self.tok([self.prompts[i] for i in idx], return_tensors="pt", padding=True,
                       truncation=True, max_length=self.max_len, padding_side="right")
        ts = self.ts[idx].to(device) if self.use_ts else None
        return enc["input_ids"].to(device), enc["attention_mask"].to(device), ts, self.y[idx].to(device)


@torch.no_grad()
def predict(model, b, device, bs=8):
    model.eval()
    out = []
    for i in range(0, len(b), bs):
        ids, am, ts, _ = b.get(list(range(i, min(i + bs, len(b)))), device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(torch.sigmoid(model(ids, am, ts)).float().cpu())
    return torch.cat(out).numpy()


def evaluate(p, b, lo_thr, hi_thr):
    y = b.ret > 0
    long, short = p >= hi_thr, p <= lo_thr
    trade = long | short
    side = np.where(long, 1, -1)
    res = {"n": len(p), "acc@0.5": float(np.mean((p > 0.5) == y) * 100),
           "auc": float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan"),
           "always_up": float(y.mean() * 100), "n_trades": int(trade.sum()), "n_long": int(long.sum()),
           "n_short": int(short.sum())}
    if trade.any():
        res["trade_win"] = float(np.mean((side[trade] > 0) == y[trade]) * 100)
        res["trade_avg_ret_pct"] = float(np.mean(side[trade] * b.ret[trade]) * 100)
        res["long_win"] = float(np.mean(y[long]) * 100) if long.any() else float("nan")
        res["short_win"] = float(np.mean(~y[short]) * 100) if short.any() else float("nan")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--target", default="ret", choices=["ret", "exc"])
    ap.add_argument("--mode", default="full",
                    choices=["full", "text_only", "ts_only", "ts_as_text", "shuffle_ts", "shuffle_text"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_new", type=float, default=1e-3)
    ap.add_argument("--trade_q", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overfit", type=int, default=0)
    ap.add_argument("--out", default="reports/llm")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = "cuda"
    df = pd.read_pickle("data/text/events.pkl")
    cols = json.load(open("data/text/events_cols.json"))
    tr, va, te, scaler = split_fold(df, a.fold, a.horizon)
    if a.overfit:
        tr = tr.sample(a.overfit, random_state=a.seed)
        va = te = tr
    model = TSLLM(BASE, len(cols), SEQ_LEN).to(device)
    model.llm.print_trainable_parameters()
    btr, bva, bte = [Batcher(d, model, a.mode, a.horizon, a.target, scaler, cols, seed=a.seed) for d in (tr, va, te)]
    print(f"train {len(btr)} val {len(bva)} test {len(bte)} | prompt example:\n{btr.prompts[0][:900]}", flush=True)
    new = list(model.reprog.parameters()) + list(model.head.parameters())
    lora = [p for n, p in model.llm.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": lora, "lr": a.lr}, {"params": new, "lr": a.lr_new}], weight_decay=0.01)
    steps = a.epochs * (len(btr) // (a.bs * a.accum) + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.lr_new], total_steps=steps, pct_start=0.1)
    pos_w = torch.tensor((1 - btr.y.mean()) / btr.y.mean(), device=device)   # balance the up/down prior
    best_auc, best_state = -1, None
    for ep in range(a.epochs):
        model.train()
        perm, t0, losses = np.random.permutation(len(btr)), time.time(), []
        for k, i in enumerate(range(0, len(perm), a.bs)):
            ids, am, ts, y = btr.get(perm[i:i + a.bs].tolist(), device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(ids, am, ts)
            loss = F.binary_cross_entropy_with_logits(logit.float(), y, pos_weight=pos_w) / a.accum
            loss.backward()
            losses.append(loss.item() * a.accum)
            if (k + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(lora + new, 1.0)
                opt.step(); sched.step(); opt.zero_grad()
        pv = predict(model, bva, device)
        auc = evaluate(pv, bva, 0, 1)["auc"]
        print(f"epoch {ep+1}: loss {np.mean(losses):.4f} | val auc {auc:.3f} | {time.time()-t0:.0f}s", flush=True)
        if auc > best_auc or a.overfit:
            best_auc = auc
            best_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    model.load_state_dict(best_state, strict=False)
    pv, pt = predict(model, bva, device), predict(model, bte, device)
    lo_thr, hi_thr = np.quantile(pv, a.trade_q), np.quantile(pv, 1 - a.trade_q)
    res = {"args": vars(a), "val": evaluate(pv, bva, lo_thr, hi_thr), "test": evaluate(pt, bte, lo_thr, hi_thr)}
    tag = f"{a.mode}_f{a.fold}_h{a.horizon}_{a.target}_s{a.seed}" + (f"_overfit{a.overfit}" if a.overfit else "")
    json.dump(res, open(f"{a.out}/{tag}.json", "w"), indent=1)
    np.savez(f"{a.out}/pred_{tag}.npz", p=pt, ret=bte.ret, raw=bte.raw,
             ticker=bte.df["ticker"].to_numpy(), date=bte.df["date"].to_numpy())
    print(json.dumps({"val": res["val"], "test": res["test"]}, indent=1))


if __name__ == "__main__":
    main()
