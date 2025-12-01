"""
PyTorch implementation of a lightweight multi-modal AKI predictor.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class ClinicalEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TimeSeriesEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, features)
        output, (h_n, _) = self.lstm(x)
        # Use last hidden state
        return h_n[-1]


class DummyTextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 10001, embed_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (batch, seq_len)
        embedded = self.embedding(token_ids)
        _, h_n = self.gru(embedded)
        return h_n[-1]


class MultiModalAKIModel(nn.Module):
    def __init__(self, clinical_dim: int, timeseries_dim: int, vocab_size: int = 10001):
        super().__init__()
        self.clinical_encoder = ClinicalEncoder(clinical_dim)
        self.timeseries_encoder = TimeSeriesEncoder(timeseries_dim)
        self.text_encoder = DummyTextEncoder(vocab_size=vocab_size)

        fusion_dim = 64 * 3
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        clinical: torch.Tensor,
        timeseries: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> torch.Tensor:
        clinical_repr = self.clinical_encoder(clinical)
        ts_repr = self.timeseries_encoder(timeseries)
        text_repr = self.text_encoder(text_tokens)

        fused = torch.cat([clinical_repr, ts_repr, text_repr], dim=-1)
        risk = self.fusion(fused)
        return risk.squeeze(-1)


def load_dummy_model(clinical_dim: int, timeseries_dim: int, vocab_size: int = 10001) -> MultiModalAKIModel:
    """
    Creates the model and initializes random weights. The model can be extended to
    load trained weights if available.
    """
    model = MultiModalAKIModel(clinical_dim=clinical_dim, timeseries_dim=timeseries_dim, vocab_size=vocab_size)
    model.eval()
    return model
