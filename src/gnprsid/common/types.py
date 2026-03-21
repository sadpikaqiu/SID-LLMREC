from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelProfile:
    name: str
    base_model: str
    tokenizer_name: str
    device_map: str
    dtype: str
    max_length: int


@dataclass(frozen=True)
class PredictionRecord:
    sample_id: str
    target: str
    prediction: str
    repr_name: str
    history_source: str
    prompt: str
