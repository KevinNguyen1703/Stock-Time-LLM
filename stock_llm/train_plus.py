"""Train/evaluate Time-LLM+ (frozen backbone) on one purged rolling-origin fold of the event dataset.

Ablation axes (one flag each):
  --prompt    none | static (original Time-LLM statistics prompt) | data (the earlier data-derived
              "professor advice" dynamic prompt) | news (C1) | news_shuffled (C1 control)
  --knowledge 0 | 1        C1 knowledge tokens (only used with news prompts)
  --patching  fixed | fft | fft_attn (earlier FFT+attention) | fft_text | bank | bank_text |
              bank_text_aligned (C2 KASD: learnable filter bank, + news/regime hypernetwork, + period-aligned patches)
  --loss      mse | hybrid (earlier hybrid loss) | decomp | dro | dro_rank | full (C3: decomposition, Group DRO
              over market regimes, same-week ranking, selective head with Learn-then-Test trade threshold)
Targets: market part = ret_H - rel_H (peer average), company part = rel_H. Test is evaluated once with
the checkpoint chosen by validation AUC; trade thresholds come from validation.
Training batches are drawn from week blocks (events of the same calendar week) so the ranking loss compares
peers facing the same market.
Usage: python -m stock_llm.train_plus --fold 2 --horizon 30 --prompt news --knowledge 1 --patching bank_text_aligned --loss full
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from stock_llm.data.events import SEQ_LEN, split_fold
from stock_llm.models.timellm_plus import (BANK_MODES, EVENTS, KN_TOKEN, LOSSES, MAX_ITEMS, SOURCES, TS_TOKEN,
                                           GroupDRO, TimeLLMPlus, compute_loss, ltt_threshold, regime_groups)

EVENT_EN = {"earnings": "earnings", "guidance": "plan/AGM", "capital": "capital/dividend", "ownership": "ownership",
            "deal": "deal/project", "legal": "legal", "management": "management", "macro": "macro/sector"}


# ---------------------------------------------------------------- prompts
def window_stats(ts, ci):
    r = ts[:, ci["ret_1"]]
    path = np.cumsum(r)
    x = r - r.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(x, 2 * len(x))) ** 2)[1:len(x) // 2]
    lags = (np.argsort(ac)[::-1][:5] + 1).tolist()
    return path, lags


def static_prompt(row, h, ci):
    """The original Time-LLM prompt: dataset/task description + input statistics."""
    path, lags = window_stats(row.ts, ci)
    return (f"Dataset description: daily data of Vietnamese stock {row.ticker} (price, foreign flows, VN-Index). "
            f"Task description: predict whether the stock will outperform its peers over the next {h} trading days "
            f"given the previous {SEQ_LEN} days. Input statistics: min value {path.min():+.3f}, max value "
            f"{path.max():+.3f}, median value {np.median(path):+.3f}, the trend of input is "
            f"{'upward' if path[-1] > path[0] else 'downward'}, top 5 lags are: {lags}.")


def data_prompt(row, h, ci):
    """The earlier data-derived dynamic prompt ('professor advice' rules on RSI/MACD/BB/ROC)."""
    ts = row.ts
    rsi, rsi5 = ts[-1, ci["rsi_14"]], ts[-6, ci["rsi_14"]]
    macd, macd5 = ts[-1, ci["macd"]], ts[-6, ci["macd"]]
    bb = (ts[-1, ci["bb_pos"]] + 1) / 2
    roc = ts[-1, ci["ret_20"]] * 100
    chg = ts[:, ci["ret_1"]].sum() * 100
    sig, why = [], []
    if rsi > 70: sig.append(-1); why.append(f"RSI at {rsi:.1f} shows overbought conditions")
    elif rsi < 30: sig.append(1); why.append(f"RSI at {rsi:.1f} indicates oversold territory")
    elif rsi > 50 and rsi > rsi5: sig.append(1); why.append(f"RSI trending up at {rsi:.1f}")
    elif rsi <= 50 and rsi <= rsi5: sig.append(-1); why.append("RSI declining below 50")
    if macd > 0 and macd > macd5: sig.append(1); why.append("MACD positive and rising")
    elif macd < 0 and macd <= macd5: sig.append(-1); why.append("MACD negative and falling")
    elif macd < 0 and macd > macd5: sig.append(1); why.append("MACD turning up from negative territory")
    if bb > 0.85: sig.append(-1); why.append("price near upper Bollinger Band")
    elif bb < 0.15: sig.append(1); why.append("price near lower Bollinger Band")
    if abs(roc) > 5: sig.append(1 if roc > 0 else -1); why.append(f"ROC at {roc:.1f}%")
    s = sum(sig)
    view = "bullish" if s > 0 else "bearish" if s < 0 else "neutral"
    return (f"Based on the past {SEQ_LEN} days of {row.ticker}, the technical picture is {view}. "
            f"{'. '.join(why[:2]) or 'Mixed signals present'}. Price moved {chg:+.1f}% over the period. "
            f"Outlook for the next {h} days: {'upward' if s > 0 else 'downward' if s < 0 else 'sideways'}.")


def news_lines(items):
    out = []
    for it in items:
        when = "today" if it["age"] == 0 else f"{it['age']} session{'s' if it['age'] > 1 else ''} ago"
        out.append(f"- [{EVENT_EN[it['event']]}, sentiment {it['score']:+d}, {when}] {it['title']}")
    return out


def build_prompt(tok, row, prompt, knowledge, h, ci, n_ts):
    parts = []
    if prompt in ("static", "data", "news", "news_shuffled"):
        parts.append(static_prompt(row, h, ci))
    if prompt == "data":
        parts.append(data_prompt(row, h, ci))
    parts.append("Input series: " + TS_TOKEN * n_ts)
    if prompt in ("news", "news_shuffled"):
        if knowledge:
            parts.append("News knowledge: " + KN_TOKEN * MAX_ITEMS)
        parts.append("Pre-open news about the company:\n" + "\n".join(news_lines(row.items)))
    parts.append(f"Will {row.ticker} outperform its peers over the next {h} trading days?")
    return tok.apply_chat_template([{"role": "user", "content": "\n".join(parts)}], tokenize=False,
                                   add_generation_prompt=True)


# ---------------------------------------------------------------- data
class Split:
    def __init__(self, df, model, a, scaler, ci, sd, rng, vol_median):
        df = df.reset_index(drop=True)
        items = df["items"].to_list()
        if a.prompt == "news_shuffled":
            items = [items[i] for i in rng.permutation(len(items))]
        news = a.prompt in ("news", "news_shuffled")
        df = df.assign(items=items)
        self.prompts = [build_prompt(model.tok, r, a.prompt, a.knowledge, a.horizon, ci, model.n_ts)
                        for r in df.itertuples()]
        ts = np.stack(df["ts"].to_list())
        self.ts = torch.from_numpy((ts - scaler[0]) / scaler[1]).float()
        kn = np.zeros((len(df), 4, MAX_ITEMS), np.int64)
        mask = np.zeros((len(df), MAX_ITEMS), bool)
        if news:
            for j, its in enumerate(items):
                for k, it in enumerate(its[:MAX_ITEMS]):
                    kn[j, :, k] = [EVENTS.index(it["event"]), it["score"], it["age"],
                                   SOURCES.index(it["source"]) if it["source"] in SOURCES else 0]
                    mask[j, k] = True
        self.kn, self.mask = torch.from_numpy(kn), torch.from_numpy(mask)
        self.rel = df[f"rel_{a.horizon}"].to_numpy()
        self.ret = df[f"ret_{a.horizon}"].to_numpy()
        self.mkt_t = torch.from_numpy(((self.ret - self.rel) / sd[0]).astype(np.float32))
        self.comp_t = torch.from_numpy((self.rel / sd[1]).astype(np.float32))
        self.events = [its[0]["event"] if its else "none" for its in items]
        last = ts[:, -1]
        self.regime = regime_groups(torch.from_numpy(last[:, ci["mkt_ret_20"]]),
                                    torch.from_numpy(last[:, ci["mkt_vol_20"]]), vol_median)
        self.week = pd.to_datetime(df["date"]).dt.to_period("W").astype(str).to_numpy()
        self.df, self.tok = df, model.tok

    def week_batches(self, bs, rng):
        """Shuffle within and across calendar-week blocks; each batch holds events of one week."""
        out = []
        for w in np.unique(self.week):
            idx = rng.permutation(np.flatnonzero(self.week == w))
            out += [idx[i:i + bs].tolist() for i in range(0, len(idx), bs)]
        return [out[i] for i in rng.permutation(len(out))]

    def __len__(self):
        return len(self.prompts)

    def batch(self, idx, device):
        enc = self.tok([self.prompts[i] for i in idx], return_tensors="pt", padding=True, padding_side="right")
        kn = self.kn[idx]
        return (enc["input_ids"].to(device), enc["attention_mask"].to(device), self.ts[idx].to(device),
                (kn[:, 0].to(device), kn[:, 1].to(device), kn[:, 2].to(device), kn[:, 3].to(device),
                 self.mask[idx].to(device)),
                {"mkt": self.mkt_t[idx].to(device), "comp": self.comp_t[idx].to(device),
                 "regime": self.regime[idx].to(device)})


# ---------------------------------------------------------------- eval
def scores(out, loss_mode):
    if loss_mode == "full":
        return torch.sigmoid(out["dir"]) - 0.5
    if loss_mode in ("decomp", "dro", "dro_rank"):
        return out["company"]
    return out["market"] + out["company"]


@torch.no_grad()
def predict(model, s, a, device, bs):
    model.eval()
    sc, conf, w = [], [], []
    for i in range(0, len(s), bs):
        ids, am, ts, kn, _ = s.batch(list(range(i, min(i + bs, len(s)))), device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(ids, am, ts, kn)
        sc.append(scores(out, a.loss).float().cpu()); conf.append(torch.sigmoid(out["conf"]).float().cpu())
        extra = getattr(model.spec, "last_periods", None)
        if extra is None:
            extra = getattr(model.spec, "last_weights", None)
        if extra is not None:
            w.append(extra.float().cpu())
    return torch.cat(sc).numpy(), torch.cat(conf).numpy(), (torch.cat(w).numpy() if w else None)


def evaluate(p, conf, s, thr, conf_thr, ltt_thr="off"):
    y_rel, y_ret = s.rel > 0, s.ret > 0
    lo, hi = thr
    side = np.where(p >= hi, 1, np.where(p <= lo, -1, 0))
    t = side != 0
    res = {"n": len(p), "auc_company": float(roc_auc_score(y_rel, p)), "auc_total": float(roc_auc_score(y_ret, p)),
           "floor_company": float(max(y_rel.mean(), 1 - y_rel.mean()) * 100),
           "n_trades": int(t.sum()),
           "trade_win": float(np.mean((side[t] > 0) == y_rel[t]) * 100) if t.any() else float("nan"),
           "trade_avg_rel_pct": float(np.mean(side[t] * s.rel[t]) * 100) if t.any() else float("nan")}
    for name, th in (("sel", conf_thr), ("ltt", ltt_thr)):
        if isinstance(th, str):
            continue
        if th is None:                       # Learn-then-Test could not certify any threshold: abstain
            res.update({f"{name}_n": 0, f"{name}_win": float("nan"), f"{name}_avg_rel_pct": float("nan")})
            continue
        sel = conf >= th
        d = np.sign(p[sel])
        res.update({f"{name}_n": int(sel.sum()),
                    f"{name}_win": float(np.mean((d > 0) == y_rel[sel]) * 100) if sel.any() else float("nan"),
                    f"{name}_avg_rel_pct": float(np.mean(d * s.rel[sel]) * 100) if sel.any() else float("nan")})
    return res


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="qwen1.5b", choices=["qwen1.5b", "qwen7b"])
    ap.add_argument("--fold", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--prompt", default="news", choices=["none", "static", "data", "news", "news_shuffled"])
    ap.add_argument("--knowledge", type=int, default=1)
    ap.add_argument("--patching", default="bank_text_aligned",
                    choices=["fixed", "fft", "fft_attn", "fft_text", *BANK_MODES])
    ap.add_argument("--loss", default="full", choices=list(LOSSES))
    ap.add_argument("--ltt_alpha", type=float, default=0.45, help="max error rate certified on trades taken")
    ap.add_argument("--ltt_delta", type=float, default=0.1)
    ap.add_argument("--coverage", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--trade_q", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use only this many samples per split")
    ap.add_argument("--out", default="reports/plus")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = "cuda"
    df = pd.read_pickle("data/text/events.pkl")
    cols = json.load(open("data/text/events_cols.json"))
    ci = {c: i for i, c in enumerate(cols)}
    tr, va, te, scaler = split_fold(df, a.fold, a.horizon)
    tr, va, te = [d.dropna(subset=[f"rel_{a.horizon}", f"ret_{a.horizon}"]) for d in (tr, va, te)]
    if a.limit:
        tr, va, te = [d.sample(min(a.limit, len(d)), random_state=a.seed) for d in (tr, va, te)]
    sd = ((tr[f"ret_{a.horizon}"] - tr[f"rel_{a.horizon}"]).std(), tr[f"rel_{a.horizon}"].std())
    model = TimeLLMPlus(a.backbone, len(cols), SEQ_LEN, ci["ret_1"], a.patching, bool(a.knowledge)).to(device)
    params = model.trainable_parameters()
    print(f"trainable params: {sum(p.numel() for p in params):,} (LLM frozen)", flush=True)
    vol_median = float(np.median([t[-1][ci["mkt_vol_20"]] for t in tr["ts"]]))
    s_tr, s_va, s_te = [Split(d, model, a, scaler, ci, sd, rng, vol_median) for d in (tr, va, te)]
    print("train regime counts (down/lowvol, down/highvol, up/lowvol, up/highvol):",
          np.bincount(s_tr.regime.numpy(), minlength=4).tolist(), flush=True)
    dro = GroupDRO(device=device)
    print(f"train {len(s_tr)} val {len(s_va)} test {len(s_te)}\nprompt example:\n"
          + s_tr.prompts[0].replace(TS_TOKEN * model.n_ts, f"{TS_TOKEN}x{model.n_ts}")
          .replace(KN_TOKEN * MAX_ITEMS, f"{KN_TOKEN}x{MAX_ITEMS}")[:1500], flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.05)
    steps = a.epochs * len(s_tr.week_batches(a.bs, np.random.default_rng(0)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    best, best_state = -1.0, None
    for ep in range(a.epochs):
        model.train()
        t0, losses = time.time(), []
        for idx in s_tr.week_batches(a.bs, rng):
            ids, am, ts, kn, y = s_tr.batch(idx, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(ids, am, ts, kn)
            out = {k: v.float() for k, v in out.items()}
            loss = compute_loss(out, y, a.loss, a.coverage, dro)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
            losses.append(loss.item())
        pv, _, _ = predict(model, s_va, a, device, a.bs * 2)
        auc = roc_auc_score(s_va.rel > 0, pv)
        print(f"epoch {ep+1}: loss {np.mean(losses):.4f} | val company-AUC {auc:.3f} | {time.time()-t0:.0f}s", flush=True)
        if auc > best:
            best = auc
            best_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    model.load_state_dict(best_state, strict=False)
    pv, cv, _ = predict(model, s_va, a, device, a.bs * 2)
    pt, ct, wt = predict(model, s_te, a, device, a.bs * 2)
    thr = (np.quantile(pv, a.trade_q), np.quantile(pv, 1 - a.trade_q))
    if a.loss == "full":
        conf_thr = float(np.quantile(cv, 1 - a.coverage))
        ltt = ltt_threshold(cv, (pv > 0) == (s_va.rel > 0), a.ltt_alpha, a.ltt_delta)
    else:
        conf_thr, ltt = "off", "off"
    res = {"args": vars(a), "val": evaluate(pv, cv, s_va, thr, conf_thr, ltt),
           "test": evaluate(pt, ct, s_te, thr, conf_thr, ltt), "ltt_threshold": ltt}
    if wt is not None:   # C2 interpretability: learned band periods (KASD) or band weights (fft*) per event type
        key = "band_periods_by_event" if a.patching in BANK_MODES else "band_weights_by_event"
        res[key] = {e: wt[np.array(s_te.events) == e].mean(0).round(3).tolist() for e in sorted(set(s_te.events))}
    tag = (f"{a.backbone}_f{a.fold}_h{a.horizon}_p-{a.prompt}_k{a.knowledge}_{a.patching}_{a.loss}_s{a.seed}"
           + (f"_limit{a.limit}" if a.limit else ""))
    json.dump(res, open(f"{a.out}/{tag}.json", "w"), indent=1)
    np.savez(f"{a.out}/pred_{tag}.npz", p=pt, conf=ct, rel=s_te.rel, ret=s_te.ret,
             ticker=s_te.df["ticker"].to_numpy(), date=s_te.df["date"].to_numpy())
    print(json.dumps({k: res[k] for k in ("val", "test")}, indent=1))


if __name__ == "__main__":
    main()
