"""Models: extracted from the StratScout prototype.

All processing is explicit; importing this module does not run a workflow.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureMLP(nn.Module):
    def __init__(self, d_in, hidden, n_classes, *, context):
        self.context = context
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in * 3, hidden),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(True),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(torch.cat([x.mean(1), x.amax(1), x[:, -1]], -1))


class GRUClassifier(nn.Module):
    def __init__(self, d_in, hidden, n_classes, layers=1, *, context):
        self.context = context
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.gru = nn.GRU(d_in, hidden, layers, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden * 2, n_classes))

    def forward(self, x):
        out, _ = self.gru(self.norm(x))
        return self.head(out.mean(1))


class TCNClassifier(nn.Module):
    """Dilated 1-D convolutions over time: cheap, local, and well suited to small datasets."""

    def __init__(self, d_in, hidden, n_classes, *, context):
        self.context = context
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.net = nn.Sequential(
            nn.Conv1d(d_in, hidden, 3, padding=1, dilation=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, 3, padding=2, dilation=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, 3, padding=4, dilation=4),
            nn.BatchNorm1d(hidden),
            nn.ReLU(True),
        )
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden * 2, n_classes))

    def forward(self, x):
        h = self.net(self.norm(x).transpose(1, 2))
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], -1))


class TinyTransformer(nn.Module):
    def __init__(self, d_in, hidden, n_classes, heads=4, layers=2, *, context):
        self.context = context
        super().__init__()
        self.proj = nn.Linear(d_in, hidden)
        self.pos = nn.Parameter(torch.randn(1, self.context.config.seq_len + 1, hidden) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
        enc = nn.TransformerEncoderLayer(
            hidden, heads, hidden * 2, dropout=0.1, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x):
        h = self.proj(x)
        h = torch.cat([self.cls.expand(h.shape[0], -1, -1), h], 1)
        return self.head(self.enc(h + self.pos[:, : h.shape[1]])[:, 0])


MODEL_ZOO = {
    "feature_mlp": FeatureMLP,
    "gru": GRUClassifier,
    "tcn": TCNClassifier,
    "transformer": TinyTransformer,
}
