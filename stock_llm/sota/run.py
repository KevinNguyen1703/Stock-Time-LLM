"""Run one SOTA method on one fold of the Time-LLM+ event dataset, with the same evaluation as Time-LLM+.

Trained forecasters (TSLib reference code, third_party/Time-Series-Library, TSLib default hyperparameters):
  patchtst | itransformer | timemixer | timexer           numeric only, "MS" mode (all channels in, ret_1 out)
  tats_<backbone>      TaTS (ICLR'26): daily news embeddings -> MLP -> 96 extra channels (repo defaults);
                       the dataset-specific "prior" column of Time-MMD does not exist here and is not used
  mmtsf_<backbone>     MM-TSFlib (NeurIPS'24): news-item embeddings -> MLP -> avg pool -> text forecast,
                       blended (1-w)*numeric + w*text with w = 0.01, MLP lr 1e-2 (repo defaults)
  restts_<backbone>    REST-TS (arXiv 2606.19413, no public code; re-implemented from the paper): backbone
                       trained first, then a text branch trained only on the backbone's residual
Zero-shot foundation models (no training on this data; 256 days of log-price history, past covariates):
  chronos2 | timesfm3 | timesfm25
Score = predicted 30-day log return. Trades: long top 10% / short bottom 10% of scores, thresholds from the
validation split. Outputs reports/sota/<method>/<tag>.json + pred_<tag>.npz (same fields as train_plus).
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

TSLIB = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "Time-Series-Library")
H = 30
TSLIB_MODELS = {"patchtst": "PatchTST", "itransformer": "iTransformer", "timemixer": "TimeMixer", "timexer": "TimeXer"}
ZERO_SHOT = ("chronos2", "timesfm3", "timesfm25")


# ---------------------------------------------------------------- models
def tslib_config(name, enc_in, seq_len):
    """TSLib run.py defaults; TimeMixer additionally uses its own ETT script settings (it needs down-sampling)."""
    c = dict(task_name="long_term_forecast", seq_len=seq_len, label_len=0, pred_len=H, enc_in=enc_in,
             dec_in=enc_in, c_out=enc_in, d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048,
             moving_avg=25, factor=1, dropout=0.1, embed="timeF", freq="d", activation="gelu",
             channel_independence=1, decomp_method="moving_avg", use_norm=1, down_sampling_layers=0,
             down_sampling_window=1, down_sampling_method=None, patch_len=16, features="MS", top_k=5,
             num_kernels=6, output_attention=False, seg_len=96, num_class=1)
    if name == "TimeMixer":
        c.update(d_model=16, d_ff=32, down_sampling_layers=3, down_sampling_window=2, down_sampling_method="avg")
    return SimpleNamespace(**c)


def tslib_hparams(name):
    if name == "TimeMixer":
        return {"lr": 0.01, "batch": 16, "epochs": 10, "patience": 10}
    return {"lr": 1e-4, "batch": 32, "epochs": 10, "patience": 3}


class Backbone(nn.Module):
    def __init__(self, name, enc_in, seq_len):
        super().__init__()
        sys.path.insert(0, os.path.abspath(TSLIB))
        import importlib
        self.m = importlib.import_module(f"models.{name}").Model(tslib_config(name, enc_in, seq_len))

    def forward(self, x):                        # -> forecast of the last (target) channel, [B, H]
        return self.m(x, None, None, None)[:, -H:, -1]


class TaTS(nn.Module):
    def __init__(self, name, n_num, seq_len, d_llm=1024, text_emb=96):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_llm, d_llm // 8), nn.ReLU(), nn.Dropout(0.3), nn.Linear(d_llm // 8, text_emb))
        self.bb = Backbone(name, n_num + text_emb, seq_len)

    def forward(self, b):
        t = self.mlp(b["text"].float())
        x = torch.cat([b["x"][:, :, :-1], t, b["x"][:, :, -1:]], -1)   # text channels before the target channel
        return self.bb(x)


class MMTSF(nn.Module):
    def __init__(self, name, n_num, seq_len, d_llm=1024, w=0.01):
        super().__init__()
        self.bb = Backbone(name, n_num, seq_len)
        self.mlp = nn.Sequential(nn.Linear(d_llm, d_llm // 8), nn.ReLU(), nn.Dropout(0.3), nn.Linear(d_llm // 8, H))
        self.w = w

    def text(self, b):
        e = self.mlp(b["items"].float())                                  # [B, items, H]
        m = b["mask"].float().unsqueeze(-1)
        return (e * m).sum(1) / m.sum(1).clamp(min=1)                     # avg pooling (repo default)

    def forward(self, b):
        return (1 - self.w) * self.bb(b["x"]) + self.w * self.text(b)


class RESTTS(MMTSF):
    """Backbone forecast + text branch trained on the backbone's residual (stage 2, backbone frozen)."""
    def forward(self, b):
        return self.bb(b["x"]) + self.text(b)


# ---------------------------------------------------------------- training
def batches(split, idx, device):
    out = {k: torch.from_numpy(split[k][idx]).to(device) for k in ("x", "fut", "text", "items", "mask")}
    return out


def predict(model, split, device, fwd, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(split["ret"]), bs):
            out.append(fwd(model, batches(split, np.arange(i, min(i + bs, len(split["ret"]))), device)).float().cpu())
    return torch.cat(out).numpy()


def fit(model, data, hp, device, fwd, params, groups=None, seed=0):
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(groups or params, lr=hp["lr"])
    best, state, bad = 1e18, None, 0
    for ep in range(hp["epochs"]):
        model.train()
        t0, losses = time.time(), []
        perm = rng.permutation(len(data["train"]["ret"]))
        for i in range(0, len(perm), hp["batch"]):
            b = batches(data["train"], perm[i:i + hp["batch"]], device)
            loss = nn.functional.mse_loss(fwd(model, b), b["fut"])
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        v = float(np.mean((predict(model, data["val"], device, fwd) - data["val"]["fut"]) ** 2))
        print(f"  epoch {ep+1}: train {np.mean(losses):.4f} val {v:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if v < best:
            best, bad, state = v, 0, {k: x.detach().clone() for k, x in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    model.load_state_dict(state)


def run_trained(method, data, device, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    kind, _, bb = method.partition("_") if "_" in method else ("", "", method)
    name = TSLIB_MODELS[bb if kind else method]
    n_num, seq_len = data["train"]["x"].shape[2], data["train"]["x"].shape[1]
    hp = tslib_hparams(name)
    if not kind:
        model = Backbone(name, n_num, seq_len).to(device)
        fwd = lambda m, b: m(b["x"])                                       # noqa: E731
        fit(model, data, hp, device, fwd, model.parameters(), seed=seed)
    elif kind == "tats":
        model = TaTS(name, n_num, seq_len).to(device)
        fwd = lambda m, b: m(b)                                            # noqa: E731
        groups = [{"params": model.bb.parameters(), "lr": hp["lr"]}, {"params": model.mlp.parameters(), "lr": 1e-2}]
        fit(model, data, hp, device, fwd, None, groups, seed)
    elif kind == "mmtsf":
        model = MMTSF(name, n_num, seq_len).to(device)
        fwd = lambda m, b: m(b)                                            # noqa: E731
        groups = [{"params": model.bb.parameters(), "lr": hp["lr"]}, {"params": model.mlp.parameters(), "lr": 1e-2}]
        fit(model, data, hp, device, fwd, None, groups, seed)
    elif kind == "restts":
        model = RESTTS(name, n_num, seq_len).to(device)
        print(" stage 1: numeric backbone", flush=True)
        fit(model, data, hp, device, lambda m, b: m.bb(b["x"]), model.bb.parameters(), seed=seed)
        for p in model.bb.parameters():
            p.requires_grad_(False)
        print(" stage 2: text branch on the backbone residual", flush=True)
        fwd = lambda m, b: m(b)                                            # noqa: E731
        fit(model, data, {**hp, "lr": 1e-3}, device, fwd, model.mlp.parameters(), seed=seed)
    else:
        raise ValueError(method)
    return (predict(model, data["val"], device, fwd).sum(1) * data["meta"]["sd_target"],
            predict(model, data["test"], device, fwd).sum(1) * data["meta"]["sd_target"])


# ---------------------------------------------------------------- zero-shot foundation models
def zero_shot_all(method, device, ctx=256):
    """Score every event once (the models are not trained, so scores do not depend on the fold)."""
    path = f"reports/sota/{method}/scores_all.npz"
    if os.path.exists(path):
        z = np.load(path)
        return dict(zip(z["idx"], z["score"]))
    import pandas as pd
    from stock_llm.data.events import load_index, ts_frame
    from stock_llm.data.universe import TICKERS
    df = pd.read_pickle("data/text/events.pkl")
    idx = load_index()
    frames = {t: ts_frame(t, idx) for t in TICKERS}
    targets, covs, keys = [], [], []
    for i, r in df.iterrows():
        f, px = frames[r.ticker]
        p = f.index.get_loc(r.date)
        lo = max(0, p - ctx)
        targets.append(np.log(px["Close"].reindex(f.index).to_numpy()[lo:p]))
        covs.append({c: f[c].to_numpy()[lo:p] for c in f.columns if c != "ret_1"})
        keys.append(i)
    scores = np.zeros(len(keys))
    if method == "chronos2":
        from chronos import Chronos2Pipeline
        pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=device)
        inputs = [{"target": t.astype(np.float32), "past_covariates": {k: v.astype(np.float32) for k, v in c.items()}}
                  for t, c in zip(targets, covs)]
        for i in range(0, len(inputs), 256):
            out = pipe.predict(inputs[i:i + 256], prediction_length=H)
            for j, o in enumerate(out):                       # o: [n_variates=1, n_quantiles, H]; median = middle
                q = o[0]
                scores[i + j] = float(q[q.shape[0] // 2, -1]) - float(targets[i + j][-1])
    elif method == "timesfm3":
        import timesfm
        m = timesfm.TimesFM3Forecaster.from_pretrained("google/timesfm-3.0-pytorch", device=device)
        for i, (t, c) in enumerate(zip(targets, covs)):
            cov = np.stack([v for v in c.values()], 0).astype(np.float32)
            out = m.predict(t.astype(np.float32), H, past_only_covariates=cov)
            pt = np.asarray(out.forecast).reshape(-1)
            scores[i] = float(pt[-1]) - float(t[-1])
    elif method == "timesfm25":
        import timesfm
        m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        m.compile(timesfm.ForecastConfig(max_context=ctx, max_horizon=H, normalize_inputs=True,
                                         use_continuous_quantile_head=True, force_flip_invariance=True,
                                         infer_is_positive=False, fix_quantile_crossing=True))
        for i in range(0, len(targets), 256):
            pt, _ = m.forecast(horizon=H, inputs=[t.astype(np.float32) for t in targets[i:i + 256]])
            for j in range(pt.shape[0]):
                scores[i + j] = float(pt[j, -1]) - float(targets[i + j][-1])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, idx=np.array(keys), score=scores)
    return dict(zip(keys, scores))


# ---------------------------------------------------------------- evaluation (same as train_plus)
def evaluate(pv, pt, val, test, q=0.1):
    lo, hi = np.quantile(pv, q), np.quantile(pv, 1 - q)
    side = np.where(pt >= hi, 1, np.where(pt <= lo, -1, 0))
    t, y = side != 0, test["ret"] > 0
    return side, {"n": len(pt), "auc_total": float(roc_auc_score(y, pt)),
                  "auc_company": float(roc_auc_score(test["rel"] > 0, pt)),
                  "floor": float(max(y.mean(), 1 - y.mean()) * 100), "n_trades": int(t.sum()),
                  "trade_win": float(np.mean((side[t] > 0) == y[t]) * 100) if t.any() else float("nan"),
                  "trade_avg_pct": float(np.mean(side[t] * test["ret"][t]) * 100) if t.any() else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--fold", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test_start", default=None)
    ap.add_argument("--test_end", default=None)
    ap.add_argument("--out", default="reports/sota")
    a = ap.parse_args()
    from stock_llm.sota.data import build_fold
    device = "cuda"
    rng_ = (a.test_start, a.test_end) if a.test_start else None
    splits, meta = build_fold(a.fold, test_range=rng_)
    data = {**splits, "meta": meta}
    print(f"{a.method} fold {a.fold}: train {len(splits['train']['ret'])} val {len(splits['val']['ret'])} "
          f"test {len(splits['test']['ret'])}", flush=True)
    if a.method in ZERO_SHOT:
        s = zero_shot_all(a.method, device)
        pv = np.array([s[i] for i in splits["val"]["idx"]]); pt = np.array([s[i] for i in splits["test"]["idx"]])
    else:
        pv, pt = run_trained(a.method, data, device, a.seed)
    side, res = evaluate(pv, pt, splits["val"], splits["test"])
    tag = f"{a.method}_f{a.fold}_s{a.seed}" + (f"_wf{a.test_start[:7]}" if a.test_start else "")
    out = f"{a.out}/{a.method}"
    os.makedirs(out, exist_ok=True)
    json.dump({"args": vars(a), "test": res}, open(f"{out}/{tag}.json", "w"), indent=1)
    np.savez(f"{out}/pred_{tag}.npz", p=pt, p_total=pt, ret=splits["test"]["ret"], rel=splits["test"]["rel"],
             side_q=side, ticker=splits["test"]["ticker"], date=splits["test"]["date"])
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
