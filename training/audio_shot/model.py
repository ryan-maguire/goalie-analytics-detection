"""GRU head for audio. Same architecture as temporal_shot/model.py
but the class is duplicated rather than imported — the original
import-via-sys.path approach caused a circular import (Python sees
audio_shot/model.py while loading audio_shot/model.py and tries to
resolve `from model import …` against itself)."""

import torch
import torch.nn as nn


class AudioShotHead(nn.Module):
    def __init__(
        self,
        in_features: int = 25,
        hidden:      int = 64,
        n_layers:    int = 2,
        dropout:     float = 0.25,
    ):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.gru = nn.GRU(
            input_size    = hidden,
            hidden_size   = hidden,
            num_layers    = n_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        out, _ = self.gru(x)
        return self.head(out).squeeze(-1)


def count_params(model: nn.Module, only_trainable: bool = False) -> int:
    return sum(p.numel() for p in model.parameters()
                if not only_trainable or p.requires_grad)
