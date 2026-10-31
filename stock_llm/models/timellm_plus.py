"""Time-LLM+ : a frozen-backbone Time-LLM extended with three components.

C1  Event-grounded news prompting: the frozen LLM's structured reading of pre-open news (event type,
    sentiment, age, source) is given both as prompt text and as dedicated knowledge tokens.
C2  Knowledge-Conditioned Adaptive Spectral Decomposition (KASD): a learnable Gaussian band-pass filter
    bank in the frequency domain splits the input into K band components; a hypernetwork reads the news
    knowledge and the current market state, plus an explicit per-event-type spectral prior, and shifts each
    band's centre frequency and width per sample (text changes WHICH components are extracted, not how they
    are weighted); during training the bands are supervised to explain the spectrum of the realised
    post-event return path (response-spectrum NLL), which teaches which time scales follow which news; each band is then patched
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
N_REGIMES = 4
N_KSTATS = 3   # explicit news statistics for the hypernetwork: net sentiment, mean sentiment, item count
BRANCHES = {"fixed": [8], "fft": [4, 8, 16], "fft_attn": [4, 8, 16], "fft_text": [4, 8, 16]}
BANK_MODES = ("bank", "bank_text", "bank_text_aligned")   # C2: static bank / + hypernetwork / + period-aligned
BANK_PERIODS = (48.0, 20.0, 10.0, 5.0, 2.5)               # initial band centres (days)
BANK_TOKENS, BANK_POINTS, BANK_FIXED_LEN = 12, 8, 8.0
RESPONSE_WEIGHT = 4.0   # response-spectrum NLL inside spec.aux (x aux_weight 0.05 in the loss = 0.2 overall)
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

    def forward(self, x, knowledge=None, k_stats=None, ev=None, mask=None, future=None):
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
        self.f_lo = 1.0 / seq_len                                          # mu in (1/T, 0.5) cycles/day, smoothly
        self.mu_raw = nn.Parameter(torch.logit((f0 - self.f_lo) / (0.5 - self.f_lo)))
        self.sig_raw = nn.Parameter(torch.log(torch.expm1(f0 * 0.5)))      # sigma = softplus(.)
        if mode in ("bank_text", "bank_text_aligned"):
            self.hyper = nn.Sequential(nn.Linear(d + n_feat + N_KSTATS, d), nn.GELU(), nn.Linear(d, 2 * self.K))
            # small non-zero init: a zero-initialised output layer received too little gradient to move
            nn.init.normal_(self.hyper[-1].weight, std=0.02); nn.init.zeros_(self.hyper[-1].bias)
            # explicit event-type spectral prior: one learned (centre, width) shift per event type
            # (last row = "no news"), averaged over the event's news items
            self.event_shift = nn.Embedding(len(EVENTS) + 1, 2 * self.K)
            nn.init.normal_(self.event_shift.weight, std=0.1)
        self.embed = nn.Linear(BANK_POINTS * n_feat, d)
        self.band_emb = nn.Parameter(torch.zeros(self.K, d))
        self.pos = nn.Parameter(torch.zeros(BANK_TOKENS, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 1)
        self.last_periods, self.aux = None, torch.tensor(0.0)

    def base_params(self):
        mu = self.f_lo + (0.5 - self.f_lo) * torch.sigmoid(self.mu_raw)
        return mu, nn.functional.softplus(self.sig_raw) + 1e-3

    def event_prior(self, ev, mask):
        m = mask.float().unsqueeze(-1)
        mean = (self.event_shift(ev) * m).sum(1) / m.sum(1).clamp(min=1)
        none = self.event_shift.weight[-1].expand_as(mean)
        return torch.where(mask.any(1, keepdim=True), mean, none)

    def band_params(self, x, k_sum, k_stats, ev=None, mask=None):
        mu, sig = self.base_params()
        mu, sig = mu.expand(x.shape[0], -1), sig.expand(x.shape[0], -1)
        if hasattr(self, "hyper"):
            raw = self.hyper(torch.cat([k_sum, x[:, -1], k_stats], -1))
            if ev is not None:
                raw = raw + self.event_prior(ev, mask)
            delta = 0.5 * torch.tanh(raw)
            # shift in logit space so the centre stays inside (1/T, 0.5) without a gradient-killing clamp
            z = torch.logit((mu - self.f_lo) / (0.5 - self.f_lo)) + delta[:, :self.K]
            mu, sig = self.f_lo + (0.5 - self.f_lo) * torch.sigmoid(z), sig * torch.exp(delta[:, self.K:])
        return mu, sig

    def diversity(self):
        """Mean pairwise cosine overlap of the (global) band filters on a fine frequency grid."""
        mu, sig = self.base_params()
        f = torch.linspace(0, 0.5, 256, device=mu.device)
        g = torch.exp(-0.5 * ((f[None] - mu[:, None]) / sig[:, None]) ** 2)
        g = g / g.norm(dim=1, keepdim=True)
        ov = g @ g.T
        return (ov.sum() - ov.diagonal().sum()) / (self.K * (self.K - 1))

    def response_nll(self, mu, sig, future):
        """Response-spectrum supervision (training only): the bands, as an equal-weight mixture of normalised
        Gaussians over frequency, should explain the power spectrum of the realised post-event return path.
        A likelihood, so it cannot be satisfied by simply widening every band."""
        H = future.shape[1]
        f = torch.fft.rfftfreq(H, device=future.device)[1:]                        # drop the mean (f = 0)
        P = torch.fft.rfft(future - future.mean(1, keepdim=True), dim=1).abs()[:, 1:] ** 2
        p = P / P.sum(1, keepdim=True).clamp(min=1e-12)                           # response spectrum, sums to 1
        g = torch.exp(-0.5 * ((f[None, None] - mu[..., None]) / sig[..., None]) ** 2)
        g = g / g.sum(-1, keepdim=True).clamp(min=1e-12)                          # each band: a distribution
        mix = g.mean(1)                                                            # [B, Nf]
        return -(p * torch.log(mix + 1e-6)).sum(1).mean()

    def forward(self, x, k_sum, k_stats=None, ev=None, mask=None, future=None):
        B, T, F = x.shape
        mu, sig = self.band_params(x, k_sum, k_stats, ev, mask)                            # [B, K]
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
        if future is not None and hasattr(self, "hyper"):
            self.aux = self.aux + RESPONSE_WEIGHT * self.response_nll(mu, sig, future)
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
        # [market, company, total-direction correction, company-direction correction, confidence]
        self.heads = nn.Linear(256, 5)
        # C3 redesign: confidence head that also sees the market regime (used by loss "sel_regime")
        self.conf_regime = nn.Sequential(nn.Linear(256 + N_REGIMES, 64), nn.GELU(), nn.Linear(64, 1))
        self.use_regime_conf = False
        # evidence-aware selection (loss "sel_evidence"): the confidence head also sees the news-anchor
        # value and strength, so it can learn when the model should act on (or overrule) the news signal
        self.conf_evidence = nn.Sequential(nn.Linear(256 + 2, 64), nn.GELU(), nn.Linear(64, 1))
        self.use_evidence_conf = False
        with torch.no_grad():   # corrections start at exactly 0, so the model starts at the anchor
            self.heads.weight[2:4].zero_(); self.heads.bias[2:4].zero_()

    def trainable_parameters(self):
        return [p for n, p in self.named_parameters() if p.requires_grad and not n.startswith("llm.")]

    def param_groups(self, lr, hyper_mult=5.0):
        hyper = [p for n, p in self.named_parameters() if n.startswith("spec.hyper.")]
        ids = {id(p) for p in hyper}
        rest = [p for p in self.trainable_parameters() if id(p) not in ids]
        return [{"params": rest, "lr": lr}] + ([{"params": hyper, "lr": lr * hyper_mult}] if hyper else [])

    @staticmethod
    def knowledge_stats(kn):
        sc, mask = kn[1].float(), kn[4].float()
        n = mask.sum(1)
        net = (sc * mask).sum(1)
        return torch.stack([net / 4, net / n.clamp(min=1) / 2, n / MAX_ITEMS], 1)

    def forward(self, input_ids, attention_mask, ts, kn, anchor, future=None, regime=None):
        """anchor: [B, 2] fixed-sign baseline logits (total, company) fitted on training data only."""
        # Only the frozen LLM runs in bf16. All trainable modules run in fp32: under bf16 autocast the heads'
        # outputs keep ~3 significant digits and per-event score differences were rounded to a single value.
        placeholder = (input_ids == self.ts_id) | (input_ids == self.kn_id)
        emb = self.llm.get_input_embeddings()(torch.where(placeholder, 0, input_ids))
        with torch.autocast("cuda", enabled=False):
            k_items, k_sum = self.know(*kn)
            soft = self.reprog(torch.cat(self.spec(ts.float(), k_sum, self.knowledge_stats(kn), kn[0], kn[4],
                                                   future if self.training else None), 1))
            kt = self.kn_proj(k_items) if self.use_knowledge else None
        m = input_ids == self.ts_id
        emb = emb.masked_scatter(m.unsqueeze(-1), soft.to(emb.dtype).reshape(-1, soft.shape[-1]))
        if self.use_knowledge:
            km = input_ids == self.kn_id
            emb = emb.masked_scatter(km.unsqueeze(-1), kt.to(emb.dtype).reshape(-1, kt.shape[-1]))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.llm(inputs_embeds=emb, attention_mask=attention_mask, output_hidden_states=True)
        h = out.hidden_states[-1]
        last = attention_mask.sum(1) - 1
        with torch.autocast("cuda", enabled=False):
            z = self.trunk(h[torch.arange(h.shape[0]), last].float())
            o = self.heads(z)
            if self.use_evidence_conf:
                ev_in = torch.stack([anchor[:, 0], anchor[:, 0].abs()], 1)
                o = torch.cat([o[:, :4], self.conf_evidence(torch.cat([z, ev_in], 1))], 1)
            elif self.use_regime_conf and regime is not None:
                onehot = nn.functional.one_hot(regime, N_REGIMES).float()
                o = torch.cat([o[:, :4], self.conf_regime(torch.cat([z, onehot], 1))], 1)
        aux = getattr(self.spec, "aux", None)
        return {"market": o[:, 0], "company": o[:, 1], "corr_total": o[:, 2], "corr_company": o[:, 3],
                "dir_total": anchor[:, 0] + o[:, 2], "dir_company": anchor[:, 1] + o[:, 3], "conf": o[:, 4],
                "aux": aux if isinstance(aux, torch.Tensor) and aux.requires_grad else torch.zeros((), device=o.device)}


LOSSES = ("mse", "hybrid", "decomp", "dro", "dro_rank", "full",
          "sel", "hybrid_sel", "sel_profit", "sel_regime", "sel_evidence",   # C3 redesign (selection-focused)
          "sel_evidence_rel",                                    # + regime-neutral (balanced) direction
          "dir")                                                 # reference: anchored direction head, no selection
SEL_LOSSES = ("sel", "hybrid_sel", "sel_profit", "sel_regime", "sel_evidence", "sel_evidence_rel", "dir")


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


def selective_loss(dir_logit, conf_logit, y, coverage, lam=8.0, weight=None, pos_weight=None):
    """SelectiveNet-style: confidence-weighted BCE on the trades taken, plus a coverage penalty.
    weight (optional, profit-aware selection): per-event importance, e.g. the size of the move."""
    c = torch.sigmoid(conf_logit)
    bce = nn.functional.binary_cross_entropy_with_logits(dir_logit, y, reduction="none", pos_weight=pos_weight)
    if weight is not None:
        c_w = c * weight
        risk = (c_w * bce).sum() / c_w.sum().clamp(min=1e-6)
        return risk + lam * torch.relu(coverage - c.mean()) ** 2
    risk = (c * bce).sum() / c.sum().clamp(min=1e-6)
    return risk + lam * torch.relu(coverage - c.mean()) ** 2


def compute_loss(out, batch, loss_mode, coverage, dro=None, rank_weight=0.5, dir_weight=0.5, aux_weight=0.05,
                 primary="total", shrink=1.0, pos_weight=None):
    """pos_weight (used by *_rel losses): (1 - up_rate) / up_rate of the training set, so the direction head
    is trained on balanced classes and does not absorb the training years' market drift."""
    """C3 ablation ladder: mse -> hybrid (earlier) -> decomp -> dro -> dro_rank -> full (+ selective)."""
    total_true = batch["mkt"] + batch["comp"]
    total_pred = out["market"] + out["company"]
    aux = aux_weight * out["aux"]
    if loss_mode == "mse":
        return nn.functional.mse_loss(total_pred, total_true) + aux
    if loss_mode in SEL_LOSSES:        # C3 redesign: plain return objective + learned trade selection
        up = (total_true > 0).float()
        bce = nn.functional.binary_cross_entropy_with_logits
        if loss_mode == "hybrid_sel":
            base = nn.functional.mse_loss(total_pred, total_true) + dir_weight * bce(3 * total_pred, up)
        else:
            base = nn.functional.mse_loss(total_pred, total_true)
        pw = pos_weight if loss_mode.endswith("_rel") else None
        base = base + dir_weight * bce(out["dir_total"], up, pos_weight=pw) + shrink * (out["corr_total"] ** 2).mean()
        w = None
        if loss_mode in ("sel_profit", "sel_regime", "sel_evidence", "sel_evidence_rel"):   # profit-aware
            w = total_true.abs() / total_true.abs().mean().clamp(min=1e-6)
        if loss_mode == "dir":
            return base + aux
        return base + selective_loss(out["dir_total"], out["conf"], up, coverage, weight=w, pos_weight=pw) + aux
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
    y_tot, y_comp = (total_true > 0).float(), (batch["comp"] > 0).float()
    bce = nn.functional.binary_cross_entropy_with_logits
    loss = loss + dir_weight * (bce(out["dir_total"], y_tot) + bce(out["dir_company"], y_comp))
    loss = loss + shrink * (out["corr_total"] ** 2 + out["corr_company"] ** 2).mean()   # stay near the anchor
    key, y = ("dir_total", y_tot) if primary == "total" else ("dir_company", y_comp)
    return loss + selective_loss(out[key], out["conf"], y, coverage) + aux


def ltt_threshold(conf, correct, alpha=0.45, delta=0.1, min_n=20, weights=None):
    """Learn-then-Test (fixed-sequence) conformal risk control on validation.

    Tests thresholds from most to least conservative; keeps lowering while the null "error rate on the
    trades taken > alpha" is rejected at level delta (binomial tail). Returns the lowest certified
    threshold, or None (abstain) if even the most conservative one cannot be certified.
    With `weights` (e.g. recency weights) the error rate is weighted and the binomial test uses the Kish
    effective sample size, a heuristic shift-aware variant: recent validation days count more."""
    from scipy.stats import binom
    chosen = None
    for q in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
        tau = float(torch.quantile(torch.as_tensor(conf, dtype=torch.float64), q)) if q > 0 else -1.0
        sel = conf >= tau
        n = int(sel.sum())
        if n < min_n:
            continue
        if weights is None:
            k = int((~correct[sel]).sum())
        else:
            w = weights[sel]
            n = int((w.sum() ** 2) / (w ** 2).sum())
            if n < min_n:
                continue
            k = int(round(n * float((w * ~correct[sel]).sum() / w.sum())))
        if binom.cdf(k, n, alpha) <= delta:
            chosen = tau
        else:
            break
    return chosen
