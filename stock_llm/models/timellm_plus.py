"""Time-LLM+ : a frozen-backbone Time-LLM extended with three components.

C1  Event-grounded news prompting: the frozen LLM's structured reading of pre-open news (event type,
    sentiment, age, source) is given both as prompt text and as dedicated knowledge tokens.
C2  Knowledge-Conditioned Adaptive Spectral Decomposition (KASD): a learnable Gaussian band-pass filter
    bank in the frequency domain splits the input into K band components; a hypernetwork reads the news
    knowledge and the current market state and shifts each band's centre frequency and width per sample
    (text changes WHICH components are extracted, not how they are weighted); each band is then patched
    with a patch length equal to its own sample-specific period (one patch ~ one cycle), implemented with
    differentiable resampling so the patch length receives gradients; a diversity penalty keeps the bands
    apart. (`fixed`, `fft`, `fft_attn`, `fft_text` remain as the earlier patching variants for ablation.)
C3  Regime-Robust Selective Learning: market / company return decomposition; Group DRO over market
    regimes known at prediction time (VN-Index trend x volatility) so patterns that only hold in one
    regime are suppressed; a cross-sectional pairwise ranking loss among events of the same week; a
    confidence head (SelectiveNet-style) whose trading threshold is set on validation by Learn-then-Test
    conformal risk control (bounded error rate on the trades taken, with stated confidence).

The LLM is frozen (as in the original Time-LLM); trainable: patch/spectral encoders, reprogramming,
knowledge encoder, hypernetwork, heads. Every component has an "off" switch so each ablation row is one flag.
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
BANK_MODES = ("bank", "bank_text", "bank_text_aligned")   # C2: static bank / + hypernetwork / + period-aligned
BANK_PERIODS = (48.0, 20.0, 10.0, 5.0, 2.5)               # initial band centres (days)
BANK_TOKENS, BANK_POINTS, BANK_FIXED_LEN = 12, 8, 8.0
BACKBONES = {"qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct", "qwen7b": "Qwen/Qwen2.5-7B-Instruct"}


def n_patches(seq_len, p):
    return (seq_len - p) // (p // 2) + 1


def n_ts_tokens(patching, seq_len):
    if patching in BANK_MODES:
        return len(BANK_PERIODS) * BANK_TOKENS
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


class AdaptiveSpectral(nn.Module):
    """C2 KASD: learnable Gaussian band-pass bank (+ knowledge/regime hypernetwork) + period-aligned patching."""
    def __init__(self, n_feat, seq_len, d, mode, heads=4, dropout=0.1):
        super().__init__()
        self.mode, self.T, self.K = mode, seq_len, len(BANK_PERIODS)
        f0 = torch.tensor([1.0 / p for p in BANK_PERIODS])
        self.mu_raw = nn.Parameter(torch.logit(f0 / 0.5))                  # mu = 0.5 * sigmoid(.)  (cycles/day)
        self.sig_raw = nn.Parameter(torch.log(torch.expm1(f0 * 0.5)))      # sigma = softplus(.)
        if mode in ("bank_text", "bank_text_aligned"):
            self.hyper = nn.Sequential(nn.Linear(d + n_feat, d), nn.GELU(), nn.Linear(d, 2 * self.K))
            nn.init.zeros_(self.hyper[-1].weight); nn.init.zeros_(self.hyper[-1].bias)   # starts as static bank
        self.embed = nn.Linear(BANK_POINTS * n_feat, d)
        self.band_emb = nn.Parameter(torch.zeros(self.K, d))
        self.pos = nn.Parameter(torch.zeros(BANK_TOKENS, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 1)
        self.last_periods, self.aux = None, torch.tensor(0.0)

    def base_params(self):
        return 0.5 * torch.sigmoid(self.mu_raw), nn.functional.softplus(self.sig_raw) + 1e-3

    def band_params(self, x, k_sum):
        mu, sig = self.base_params()
        mu, sig = mu.expand(x.shape[0], -1), sig.expand(x.shape[0], -1)
        if hasattr(self, "hyper"):
            delta = 0.5 * torch.tanh(self.hyper(torch.cat([k_sum, x[:, -1]], -1)))
            mu, sig = mu * torch.exp(delta[:, :self.K]), sig * torch.exp(delta[:, self.K:])
        return mu.clamp(1.0 / self.T, 0.5), sig

    def diversity(self):
        """Mean pairwise cosine overlap of the (global) band filters on a fine frequency grid."""
        mu, sig = self.base_params()
        f = torch.linspace(0, 0.5, 256, device=mu.device)
        g = torch.exp(-0.5 * ((f[None] - mu[:, None]) / sig[:, None]) ** 2)
        g = g / g.norm(dim=1, keepdim=True)
        ov = g @ g.T
        return (ov.sum() - ov.diagonal().sum()) / (self.K * (self.K - 1))

    def forward(self, x, k_sum):
        B, T, F = x.shape
        mu, sig = self.band_params(x, k_sum)                                       # [B, K]
        freqs = torch.fft.rfftfreq(T, device=x.device)
        G = torch.exp(-0.5 * ((freqs[None, None] - mu[..., None]) / sig[..., None]) ** 2)   # [B, K, Nf]
        X = torch.fft.rfft(x, dim=1)                                               # [B, Nf, F]
        bands = torch.fft.irfft(X[:, None] * G[..., None], n=T, dim=2)              # [B, K, T, F]
        if self.mode == "bank_text_aligned":
            L = (1.0 / mu).clamp(2.0, T / 2)                                       # patch length = band period
        else:
            L = torch.full_like(mu, BANK_FIXED_LEN)
        j = torch.arange(BANK_TOKENS, device=x.device, dtype=x.dtype)
        sp = torch.linspace(0, 1, BANK_POINTS, device=x.device, dtype=x.dtype)
        end = (T - 1) - j[None, None] * (L[..., None] / 2)                          # [B, K, M], stride = L/2
        pos = (end[..., None] - (1 - sp) * L[..., None, None]).clamp(0, T - 1)     # [B, K, M, S]
        i0 = pos.detach().floor().long()
        i1 = (i0 + 1).clamp(max=T - 1)
        w = (pos - i0.to(pos.dtype)).reshape(B, self.K, -1, 1)                     # differentiable in L
        take = lambda idx: torch.gather(bands, 2, idx.reshape(B, self.K, -1, 1).expand(-1, -1, -1, F))  # noqa: E731
        v = (take(i0) * (1 - w) + take(i1) * w).reshape(B, self.K, BANK_TOKENS, BANK_POINTS * F)
        tok = self.embed(v) + self.band_emb[None, :, None] + self.pos[None, None]
        tok = self.enc(tok.reshape(B * self.K, BANK_TOKENS, -1)).reshape(B, self.K * BANK_TOKENS, -1)
        self.last_periods = (1.0 / mu).detach()
        self.aux = self.diversity()
        return [tok]


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
        self.spec = (AdaptiveSpectral(n_feat, seq_len, d, patching) if patching in BANK_MODES
                     else SpectralPatching(n_feat, seq_len, d, patching, target_idx))
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
        aux = getattr(self.spec, "aux", None)
        return {"market": o[:, 0], "company": o[:, 1], "dir": o[:, 2], "conf": o[:, 3],
                "aux": aux if isinstance(aux, torch.Tensor) and aux.requires_grad else torch.zeros((), device=o.device)}


LOSSES = ("mse", "hybrid", "decomp", "dro", "dro_rank", "full")
N_REGIMES = 4


def regime_groups(mkt_ret_20, mkt_vol_20, vol_median):
    """Market regime known at prediction time: VN-Index 20-day trend (up/down) x volatility (high/low)."""
    return 2 * (mkt_ret_20 > 0).long() + (mkt_vol_20 > vol_median).long()


class GroupDRO:
    """Online Group DRO (Sagawa et al., 2020): exponentiated-gradient weights over regime groups."""
    def __init__(self, n_groups=N_REGIMES, eta=0.01, device="cuda"):
        self.q = torch.full((n_groups,), 1.0 / n_groups, device=device)
        self.eta = eta

    def __call__(self, per_sample, groups):
        g_loss = torch.zeros_like(self.q).index_add(0, groups, per_sample)
        g_cnt = torch.zeros_like(self.q).index_add(0, groups, torch.ones_like(per_sample))
        present = g_cnt > 0
        g_mean = torch.where(present, g_loss / g_cnt.clamp(min=1), torch.zeros_like(g_loss))
        self.q = self.q * torch.exp(self.eta * g_mean.detach() * present)
        self.q = self.q / self.q.sum()
        w = self.q * present
        return (w * g_mean).sum() / w.sum()


def rank_loss(score, y):
    """Pairwise logistic ranking of the company part among events in the same batch (same week)."""
    diff_y = y[:, None] - y[None, :]
    mask = diff_y > 0
    if not mask.any():
        return score.sum() * 0
    return nn.functional.softplus(-(score[:, None] - score[None, :]))[mask].mean()


def selective_loss(dir_logit, conf_logit, y, coverage, lam=8.0):
    """SelectiveNet-style: confidence-weighted BCE on the trades taken, plus a coverage penalty."""
    c = torch.sigmoid(conf_logit)
    bce = nn.functional.binary_cross_entropy_with_logits(dir_logit, y, reduction="none")
    risk = (c * bce).sum() / c.sum().clamp(min=1e-6)
    return risk + lam * torch.relu(coverage - c.mean()) ** 2


def compute_loss(out, batch, loss_mode, coverage, dro=None, rank_weight=0.5, dir_weight=0.5, aux_weight=0.05):
    """C3 ablation ladder: mse -> hybrid (earlier) -> decomp -> dro -> dro_rank -> full (+ selective)."""
    total_true = batch["mkt"] + batch["comp"]
    total_pred = out["market"] + out["company"]
    aux = aux_weight * out["aux"]
    if loss_mode == "mse":
        return nn.functional.mse_loss(total_pred, total_true) + aux
    if loss_mode == "hybrid":          # the earlier hybrid loss: MSE + soft directional BCE on the total
        up = (total_true > 0).float()
        return nn.functional.mse_loss(total_pred, total_true) + dir_weight * \
            nn.functional.binary_cross_entropy_with_logits(3 * total_pred, up) + aux   # targets are standardised
    per = nn.functional.huber_loss(out["market"], batch["mkt"], reduction="none") + \
        nn.functional.huber_loss(out["company"], batch["comp"], reduction="none")
    if loss_mode == "decomp":
        return per.mean() + aux
    loss = dro(per, batch["regime"])
    if loss_mode == "dro":
        return loss + aux
    loss = loss + rank_weight * rank_loss(out["company"], batch["comp"])
    if loss_mode == "dro_rank":
        return loss + aux
    y = (batch["comp"] > 0).float()
    return loss + selective_loss(out["dir"], out["conf"], y, coverage) + aux


def ltt_threshold(conf, correct, alpha=0.45, delta=0.1, min_n=20):
    """Learn-then-Test (fixed-sequence) conformal risk control on validation.

    Tests thresholds from most to least conservative; keeps lowering while the null "error rate on the
    trades taken > alpha" is rejected at level delta (binomial tail). Returns the lowest certified
    threshold, or None (abstain) if even the most conservative one cannot be certified."""
    from scipy.stats import binom
    chosen = None
    for q in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
        tau = float(torch.quantile(torch.as_tensor(conf, dtype=torch.float64), q)) if q > 0 else -1.0
        sel = conf >= tau
        n = int(sel.sum())
        if n < min_n:
            continue
        k = int((~correct[sel]).sum())
        if binom.cdf(k, n, alpha) <= delta:
            chosen = tau
        else:
            break
    return chosen
