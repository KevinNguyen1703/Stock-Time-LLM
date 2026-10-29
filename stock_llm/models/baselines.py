"""Compact neural baselines. All map X [B, T, F] (scaled features) -> next-day return [B]."""
import torch
import torch.nn as nn


class LSTMReg(nn.Module):
    def __init__(self, n_feat, seq_len, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.rnn = nn.LSTM(n_feat, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.head(self.rnn(x)[0][:, -1]).squeeze(-1)


class DLinear(nn.Module):
    """Trend/seasonal decomposition by moving average, a linear map over time for each part."""
    def __init__(self, n_feat, seq_len, kernel=25, dropout=0.2):
        super().__init__()
        self.pool = nn.AvgPool1d(kernel, stride=1, padding=kernel // 2, count_include_pad=False)
        self.trend = nn.Linear(seq_len, 1)
        self.season = nn.Linear(seq_len, 1)
        self.mix = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_feat, 1))

    def forward(self, x):
        xt = x.transpose(1, 2)                       # [B, F, T]
        trend = self.pool(xt)
        h = self.trend(trend).squeeze(-1) + self.season(xt - trend).squeeze(-1)   # [B, F]
        return self.mix(h).squeeze(-1)


class PatchTST(nn.Module):
    """Channel-mixing patch transformer (patches span all features)."""
    def __init__(self, n_feat, seq_len, patch=8, stride=4, d_model=64, heads=4, layers=2, dropout=0.2):
        super().__init__()
        self.patch, self.stride = patch, stride
        n_patch = (seq_len - patch) // stride + 1
        self.embed = nn.Linear(patch * n_feat, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_patch, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 1))

    def forward(self, x):
        p = x.unfold(1, self.patch, self.stride)     # [B, n_patch, F, patch]
        p = p.reshape(p.shape[0], p.shape[1], -1)
        h = self.enc(self.embed(p) + self.pos)
        return self.head(h.mean(1)).squeeze(-1)


REGISTRY = {"lstm": LSTMReg, "dlinear": DLinear, "patchtst": PatchTST}
