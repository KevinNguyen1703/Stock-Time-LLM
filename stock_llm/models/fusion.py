"""Price-news fusion: patch-transformer price encoder + gated cross-attention over news tokens.

The cross-attention output is scaled by tanh(gate) with gate initialised at 0, so the model starts as
a pure price model and only uses text if training finds it useful (the learned gate is reported).
"""
import torch
import torch.nn as nn

from stock_llm.data.multimodal import LOOKBACK, SOURCES


class GatedXAttn(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.gate_attn = nn.Parameter(torch.zeros(1))
        self.gate_ff = nn.Parameter(torch.zeros(1))

    def forward(self, h, mem, mem_pad):
        a, w = self.attn(self.norm(h), mem, mem, key_padding_mask=mem_pad, need_weights=True)
        h = h + torch.tanh(self.gate_attn) * a
        return h + torch.tanh(self.gate_ff) * self.ff(h), w


class FusionModel(nn.Module):
    def __init__(self, n_feat, seq_len, n_tickers, emb_dim=1024, text="real", patch=8, stride=4,
                 d=64, heads=4, layers=2, dropout=0.2):
        super().__init__()
        self.text, self.patch, self.stride = text, patch, stride
        n_patch = (seq_len - patch) // stride + 1
        self.embed = nn.Linear(patch * n_feat, d)
        self.pos = nn.Parameter(torch.zeros(1, n_patch, d))
        self.ticker = nn.Embedding(n_tickers, d)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.price_enc = nn.TransformerEncoder(enc, layers)
        # news token = proj(embedding) + source + age (+ llm score); a learned null token is always present
        self.news_proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(emb_dim, d))
        self.src = nn.Embedding(len(SOURCES), d)
        self.age = nn.Embedding(LOOKBACK + 1, d)
        self.score = nn.Linear(1, d)
        self.null = nn.Parameter(torch.zeros(1, 1, d))
        self.xattn = nn.ModuleList([GatedXAttn(d, heads, dropout) for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, 2))  # [return, up-logit]

    def news_tokens(self, E, idx, age, src, score):
        pad = idx < 0
        safe = idx.clamp(min=0)
        if self.text == "score_only":
            tok = self.score(score.unsqueeze(-1))
        else:
            tok = self.news_proj(E[safe].float())
            if self.text != "embed_only":
                tok = tok + self.score(score.unsqueeze(-1))
        tok = tok + self.src(src) + self.age(age.clamp(max=LOOKBACK))
        return tok.masked_fill(pad.unsqueeze(-1), 0.0), pad

    def forward(self, x, ticker, E=None, news=None):
        p = x.unfold(1, self.patch, self.stride)
        h = self.embed(p.reshape(p.shape[0], p.shape[1], -1)) + self.pos + self.ticker(ticker).unsqueeze(1)
        h = self.price_enc(h)
        attn = None
        if self.text != "none":
            co, co_pad = self.news_tokens(E, *news["co"])
            mk, mk_pad = self.news_tokens(E, *news["mk"])
            null = self.null.expand(x.shape[0], -1, -1)
            mem = torch.cat([null, co, mk], 1)
            pad = torch.cat([torch.zeros(x.shape[0], 1, dtype=torch.bool, device=x.device), co_pad, mk_pad], 1)
            for blk in self.xattn:
                h, attn = blk(h, mem, pad)
        out = self.head(h.mean(1))
        return out[:, 0], out[:, 1], attn

    def gates(self):
        return {f"layer{i}": (float(torch.tanh(b.gate_attn)), float(torch.tanh(b.gate_ff))) for i, b in enumerate(self.xattn)}
