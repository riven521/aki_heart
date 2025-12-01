"""
Utility helpers for preprocessing and postprocessing.
"""
from __future__ import annotations

import re
from typing import List

import torch


def tokenize_text(text: str, max_length: int = 64) -> torch.Tensor:
    """
    Simple whitespace and punctuation tokenizer that maps tokens to hashed ids.
    Uses a deterministic hashing to keep vocabulary bounded without requiring
    an external tokenizer or vocabulary file.
    """
    tokens = re.findall(r"\b\w+\b", text.lower())
    token_ids = []
    for tok in tokens[:max_length]:
        # A small hashing trick to map words into a stable bucket space
        token_ids.append((hash(tok) % 10000) + 1)  # reserve 0 for padding
    if len(token_ids) < max_length:
        token_ids.extend([0] * (max_length - len(token_ids)))
    return torch.tensor(token_ids, dtype=torch.long)


def clinical_to_tensor(features: List[float]) -> torch.Tensor:
    return torch.tensor(features, dtype=torch.float32).unsqueeze(0)


def timeseries_to_tensor(series: List[List[float]]) -> torch.Tensor:
    return torch.tensor(series, dtype=torch.float32).unsqueeze(0)


def risk_category(score: float) -> str:
    if score < 0.33:
        return "Low"
    if score < 0.66:
        return "Medium"
    return "High"
