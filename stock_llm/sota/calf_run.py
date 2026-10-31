"""CALF (AAAI'25) on the Time-LLM+ event dataset, with CALF's own model and cross-modal loss (cmLoss).

Runs in its own process because CALF's top-level packages (models/, utils/) clash with TSLib's.
Settings from CALF's long-term forecasting script (ETTh1): GPT-2 with 6 layers, d_model 768, LoRA r 8 /
alpha 32, dropout 0.3, lr 5e-4, batch 256, up to 100 epochs with patience 10; cmLoss weights 1.0 / 0.01 / 1.0
(task / feature / output, l1). The task and output losses are applied to the target channel (ret_1), the
feature-alignment loss is unchanged. Evaluation and outputs are identical to stock_llm.sota.run.
Usage: python -m stock_llm.sota.calf_run --fold 0
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

CALF = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "CALF"))
H = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test_start", default=None)
    ap.add_argument("--test_end", default=None)
    ap.add_argument("--out", default="reports/sota")
    a = ap.parse_args()
    from stock_llm.sota.data import build_fold
    from stock_llm.sota.run import batches, evaluate
    for k in [k for k in sys.modules if k.split(".")[0] in ("models", "utils", "layers")]:
        del sys.modules[k]                      # drop this repo's / TSLib's same-named packages if cached
    # CALF's models/ has no __init__.py (namespace package), so this repo's regular models/ package would win
    # regardless of order: hide the repo root from the import path while CALF's modules load
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    saved = sys.path[:]
    sys.path[:] = [CALF] + [p for p in saved if p not in ("", ".", root)]
    from models.CALF import Model
    from utils.cmLoss import cmLoss
    sys.path[:] = saved
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = "cuda"
    rng_ = (a.test_start, a.test_end) if a.test_start else None
    splits, meta = build_fold(a.fold, test_range=rng_)
    n_ch, seq_len = splits["train"]["x"].shape[2], splits["train"]["x"].shape[1]
    cfg = SimpleNamespace(task_name="long_term_forecast", seq_len=seq_len, pred_len=H, label_len=0, enc_in=n_ch,
                          c_out=n_ch, d_model=768, n_heads=4, d_ff=768, dropout=0.3, gpt_layers=6, r=8,
                          lora_alpha=32, lora_dropout=0.1, num_class=1,
                          word_embedding_path=os.path.join(CALF, "wte_pca_500.pt"))
    model = Model(cfg, device).to(device)
    crit = cmLoss("l1", "l1", "l1", "long_term_forecast", feature_w=0.01, output_w=1.0, task_w=1.0)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=5e-4)

    def fwd(b):
        o = model(b["x"])
        return {**o, "outputs_time": o["outputs_time"][:, -H:, -1:], "outputs_text": o["outputs_text"][:, -H:, -1:]}

    def predict(split):
        model.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(split["ret"]), 256):
                out.append(fwd(batches(split, np.arange(i, min(i + 256, len(split["ret"]))), device))["outputs_time"][:, :, 0].float().cpu())
        return torch.cat(out).numpy()

    rng = np.random.default_rng(a.seed)
    best, state, bad = 1e18, None, 0
    for ep in range(100):
        model.train(); t0, losses = time.time(), []
        perm = rng.permutation(len(splits["train"]["ret"]))
        for i in range(0, len(perm), 256):
            b = batches(splits["train"], perm[i:i + 256], device)
            loss = crit(fwd(b), b["fut"].unsqueeze(-1))
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        v = float(np.mean((predict(splits["val"]) - splits["val"]["fut"]) ** 2))
        print(f"  epoch {ep+1}: train {np.mean(losses):.4f} val {v:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if v < best:
            best, bad, state = v, 0, {k: x.detach().clone() for k, x in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 10:
                break
    model.load_state_dict(state)
    pv = predict(splits["val"]).sum(1) * meta["sd_target"]
    pt = predict(splits["test"]).sum(1) * meta["sd_target"]
    side, res = evaluate(pv, pt, splits["val"], splits["test"])
    tag = f"calf_f{a.fold}_s{a.seed}" + (f"_wf{a.test_start[:7]}" if a.test_start else "")
    out = f"{a.out}/calf"
    os.makedirs(out, exist_ok=True)
    json.dump({"args": vars(a), "test": res}, open(f"{out}/{tag}.json", "w"), indent=1)
    np.savez(f"{out}/pred_{tag}.npz", p=pt, p_total=pt, ret=splits["test"]["ret"], rel=splits["test"]["rel"],
             side_q=side, ticker=splits["test"]["ticker"], date=splits["test"]["date"])
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
