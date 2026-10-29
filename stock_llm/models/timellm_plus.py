"""Time-LLM+ : a frozen-backbone Time-LLM extended with three components.

C1  Event-grounded news prompting: the frozen LLM's structured reading of pre-open news (event type,
    sentiment, age, source) is given both as prompt text and as dedicated knowledge tokens.
C2  Text-conditioned spectral patching: multi-scale patch branches whose lengths match FFT frequency
    bands; the per-band weights start from the sample's spectral energy and are adjusted, together with a
    FiLM modulation of each band, by the news knowledge (so text decides which time scale matters).
C3  Regime-decomposed selective loss: separate heads for the market part and the company-specific part
    of the H-day return, a directional head on the company part, and a confidence head trained with a
    coverage-constrained selective objective (SelectiveNet-style) for abstention.

The LLM is frozen (as in the original Time-LLM); trainable: patch encoders, reprogramming, knowledge
encoder, conditioning, heads. Every component has an "off" switch so each ablation row is one flag.
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from stock_llm.models.ts_llm import EVENT_VI

TS_TOKEN, KN_TOKEN = "<|ts|>", "<|kn|>"
EVENTS = list(EVENT_VI)
SOURCES = ["filing", "press", "market_stock", "market_bank", "market_macro"]
MAX_ITEMS, MAX_AGE = 8, 5
BRANCHES = {"fixed": [8], "fft": [4, 8, 16], "fft_attn": [4, 8, 16], "fft_text": [4, 8, 16]}
BACKBONES = {"qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct", "qwen7b": "Qwen/Qwen2.5-7B-Instruct"}


def n_patches(seq_len, p):
    return (seq_len - p) // (p // 2) + 1


def n_ts_tokens(patching, seq_len):
    return sum(n_patches(seq_len, p) for p in BRANCHES[patching])


class KnowledgeEncoder(nn.Module):
    """Structured news items -> d-dim item embeddings (+ mask) and a pooled knowledge summary."""
    def __init__(self, d):
        super().__init__()
        self.event = nn.Embedding(len(EVENTS), d)
        self.score = nn.Embedding(5, d)          # -2..2
        self.age = nn.Embedding(MAX_AGE + 1, d)
        self.source = nn.Embedding(len(SOURCES), d)
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.null = nn.Parameter(torch.zeros(d))

    def forward(self, ev, sc, age, src, mask):   # each [B, MAX_ITEMS]; mask True = real item
        h = self.mlp(self.event(ev) + self.score(sc + 2) + self.age(age.clamp(max=MAX_AGE)) + self.source(src))
        h = torch.where(mask.unsqueeze(-1), h, self.null)
        m = mask.float().unsqueeze(-1)
        summary = (h * m).sum(1) / m.sum(1).clamp(min=1)
        summary = torch.where(mask.any(1, keepdim=True), summary, self.null.expand_as(summary))
        return h, summary


class SpectralPatching(nn.Module):
    """Multi-scale patch branches -> d-dim tokens, weighted by band energy and (optionally) by text."""
    def __init__(self, n_feat, seq_len, d, mode, target_idx, heads=4, dropout=0.1):
        super().__init__()
        self.mode, self.lens, self.target_idx, self.seq_len = mode, BRANCHES[mode], target_idx, seq_len
        self.embed = nn.ModuleList([nn.Linear(p * n_feat, d) for p in self.lens])
        self.pos = nn.ParameterList([nn.Parameter(torch.zeros(1, n_patches(seq_len, p), d)) for p in self.lens])
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 1)
        nb = len(self.lens)
        if mode == "fft_attn":       # the original FFT+attention idea: weights from the series itself
            self.attn = nn.Sequential(nn.Linear(d * nb, d), nn.GELU(), nn.Linear(d, nb))
        if mode == "fft_text":       # C2: weights and FiLM from the news knowledge summary
            self.cond = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, nb * (1 + 2 * d)))
            nn.init.zeros_(self.cond[-1].weight); nn.init.zeros_(self.cond[-1].bias)  # start as plain FFT
        self.last_weights = None

    def band_energy(self, x):
        """Log spectral energy of the target series in the period band [p, 2p) of each branch."""
        s = x[:, :, self.target_idx]
        power = torch.fft.rfft(s - s.mean(1, keepdim=True), dim=1).abs() ** 2       # [B, T//2+1]
        k = torch.arange(power.shape[1], device=x.device).clamp(min=1)
        period = self.seq_len / k
        e = []
        for i, p in enumerate(self.lens):
            hi = 2 * p if i < len(self.lens) - 1 else self.seq_len + 1
            band = (period >= p) & (period < hi)
            band[0] = False
            e.append(torch.log(power[:, band].sum(1) + 1e-6))
        return torch.stack(e, 1)                                                    # [B, nb]

    def forward(self, x, knowledge=None):
        toks = []
        for i, p in enumerate(self.lens):
            u = x.unfold(1, p, p // 2)                                              # [B, P, F, p]
            toks.append(self.enc(self.embed[i](u.reshape(u.shape[0], u.shape[1], -1)) + self.pos[i]))
        nb = len(self.lens)
        if nb == 1:
            return toks
        logits = self.band_energy(x)
        if self.mode == "fft_attn":
            logits = logits + self.attn(torch.cat([t.mean(1) for t in toks], -1))
        if self.mode == "fft_text":
            c = self.cond(knowledge)
            d = toks[0].shape[-1]
            logits = logits + c[:, :nb]
            film = c[:, nb:].view(-1, nb, 2, d)
            toks = [t * (1 + film[:, i, 0].unsqueeze(1)) + film[:, i, 1].unsqueeze(1) for i, t in enumerate(toks)]
        w = torch.softmax(logits, -1)
        self.last_weights = w.detach()
        return [t * (w[:, i] * nb).view(-1, 1, 1) for i, t in enumerate(toks)]


class Reprogram(nn.Module):
    """Time-LLM reprogramming: d-dim tokens attend to text prototypes (linear maps of word embeddings)."""
    def __init__(self, d, d_llm, word_emb, n_proto=512, heads=8, dropout=0.1):
        super().__init__()
        self.register_buffer("word_emb", word_emb, persistent=False)
        self.to_proto = nn.Linear(word_emb.shape[0], n_proto, bias=False)
        self.q = nn.Linear(d, d_llm)
        self.attn = nn.MultiheadAttention(d_llm, heads, dropout=dropout, batch_first=True)
        self.out = nn.Linear(d_llm, d_llm)

    def forward(self, h):
        proto = self.to_proto(self.word_emb.T.float()).T.unsqueeze(0).expand(h.shape[0], -1, -1)
        a, _ = self.attn(self.q(h), proto, proto)
        return self.out(a)


class TimeLLMPlus(nn.Module):
    def __init__(self, backbone, n_feat, seq_len, target_idx, patching="fft_text", knowledge=True, d=128):
        super().__init__()
        name = BACKBONES[backbone]
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.add_special_tokens({"additional_special_tokens": [TS_TOKEN, KN_TOKEN]})
        self.ts_id, self.kn_id = self.tok.convert_tokens_to_ids([TS_TOKEN, KN_TOKEN])
        self.llm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
        self.llm.requires_grad_(False)
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        emb = self.llm.get_input_embeddings().weight.detach()
        d_llm = emb.shape[1]
        self.patching, self.use_knowledge = patching, knowledge
        self.n_ts = n_ts_tokens(patching, seq_len)
        self.know = KnowledgeEncoder(d)
        self.spec = SpectralPatching(n_feat, seq_len, d, patching, target_idx)
        self.reprog = Reprogram(d, d_llm, emb[:20000].clone())
        self.kn_proj = nn.Linear(d, d_llm)
        self.trunk = nn.Sequential(nn.LayerNorm(d_llm), nn.Dropout(0.1), nn.Linear(d_llm, 256), nn.GELU())
        self.heads = nn.Linear(256, 4)   # [market, company, company-direction logit, confidence logit]

    def trainable_parameters(self):
        return [p for n, p in self.named_parameters() if p.requires_grad and not n.startswith("llm.")]

    def forward(self, input_ids, attention_mask, ts, kn):
        placeholder = (input_ids == self.ts_id) | (input_ids == self.kn_id)
        emb = self.llm.get_input_embeddings()(torch.where(placeholder, 0, input_ids))
        k_items, k_sum = self.know(*kn)
        soft = self.reprog(torch.cat(self.spec(ts, k_sum), 1)).to(emb.dtype)        # [B, n_ts, d_llm]
        m = input_ids == self.ts_id
        emb = emb.masked_scatter(m.unsqueeze(-1), soft.reshape(-1, soft.shape[-1]))
        if self.use_knowledge:
            km = input_ids == self.kn_id
            kt = self.kn_proj(k_items).to(emb.dtype)
            emb = emb.masked_scatter(km.unsqueeze(-1), kt.reshape(-1, kt.shape[-1]))
        out = self.llm(inputs_embeds=emb, attention_mask=attention_mask, output_hidden_states=True)
        h = out.hidden_states[-1]
        last = attention_mask.sum(1) - 1
        o = self.heads(self.trunk(h[torch.arange(h.shape[0]), last].float()))
        return {"market": o[:, 0], "company": o[:, 1], "dir": o[:, 2], "conf": o[:, 3]}


def selective_loss(dir_logit, conf_logit, y, coverage, lam=8.0):
    """SelectiveNet-style: confidence-weighted BCE on the trades taken, plus a coverage penalty."""
    c = torch.sigmoid(conf_logit)
    bce = nn.functional.binary_cross_entropy_with_logits(dir_logit, y, reduction="none")
    risk = (c * bce).sum() / c.sum().clamp(min=1e-6)
    return risk + lam * torch.relu(coverage - c.mean()) ** 2


def compute_loss(out, batch, loss_mode, coverage, dir_weight=0.5):
    """C3 ablation ladder: mse -> hybrid (old) -> decomp -> decomp_dir -> selective."""
    total_true = batch["mkt"] + batch["comp"]
    total_pred = out["market"] + out["company"]
    if loss_mode == "mse":
        return nn.functional.mse_loss(total_pred, total_true)
    if loss_mode == "hybrid":          # the original hybrid loss: MSE + soft directional BCE on the total
        up = (total_true > 0).float()
        return nn.functional.mse_loss(total_pred, total_true) + dir_weight * \
            nn.functional.binary_cross_entropy_with_logits(3 * total_pred, up)   # targets are standardised
    loss = nn.functional.huber_loss(out["market"], batch["mkt"]) + nn.functional.huber_loss(out["company"], batch["comp"])
    if loss_mode == "decomp":
        return loss
    y = (batch["comp"] > 0).float()
    loss = loss + dir_weight * nn.functional.binary_cross_entropy_with_logits(out["dir"], y)
    if loss_mode == "decomp_dir":
        return loss
    return loss + selective_loss(out["dir"], out["conf"], y, coverage)

