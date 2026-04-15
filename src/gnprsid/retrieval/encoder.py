from __future__ import annotations

from pathlib import Path
from typing import Any

from gnprsid.common.runtime import resolve_torch_dtype
from gnprsid.common.tokenizer import build_tokenizer_load_kwargs


def load_encoder(
    model_name_or_path: str | Path,
    dtype_name: str = "auto",
    pooling: str = "mean",
    device_map: str | None = "auto",
    load_in_4bit: bool = False,
):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ImportError("Retrieval encoding requires torch and transformers.") from error

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        **build_tokenizer_load_kwargs(use_fast=True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": resolve_torch_dtype(torch, dtype_name, device_type),
        "low_cpu_mem_usage": True,
    }
    if device_map not in {None, ""}:
        model_kwargs["device_map"] = device_map
    if pooling == "attention":
        model_kwargs["attn_implementation"] = "eager"
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        model_kwargs["dtype"] = torch.float16

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    except TypeError:
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

    if pooling == "attention" and hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    model.eval()
    return tokenizer, model


def _model_device(model):
    return next(model.parameters()).device


def masked_mean_pool(hidden_states, attention_mask):
    import torch

    hidden = hidden_states[0].float()
    mask = attention_mask[0].unsqueeze(-1).float()
    masked_hidden = hidden * mask
    denom = mask.sum(dim=0).clamp_min(1.0)
    feature = masked_hidden.sum(dim=0) / denom
    if not torch.isfinite(feature).all():
        raise ValueError("Mean pooled feature contains non-finite values.")
    norm = torch.linalg.norm(feature).item()
    if norm == 0:
        raise ValueError("Mean pooled feature collapsed to a zero vector.")
    return feature


def attention_pool(hidden_states, attention, attention_mask):
    import torch

    hidden = hidden_states[0].float()
    if not torch.isfinite(hidden).all():
        raise ValueError("Hidden states contain non-finite values.")
    if attention is None:
        raise ValueError("Attention tensors are required for attention pooling.")
    if not torch.isfinite(attention).all():
        raise ValueError("Attention tensor contains non-finite values.")
    averaged_attention = attention.mean(dim=1)[0].float()
    weighted_hidden = torch.matmul(averaged_attention, hidden)
    feature = weighted_hidden.mean(dim=0)
    if not torch.isfinite(feature).all():
        raise ValueError("Attention pooled feature contains non-finite values.")
    norm = torch.linalg.norm(feature).item()
    if norm == 0:
        raise ValueError("Attention pooled feature collapsed to a zero vector.")
    return feature


def encode_text(
    model,
    tokenizer,
    text: str,
    max_length: int,
    pooling: str = "mean",
):
    import torch

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {key: value.to(_model_device(model)) for key, value in inputs.items()}
    need_attentions = pooling == "attention"
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=need_attentions)

    hidden_last = outputs.hidden_states[-1]
    if not torch.isfinite(hidden_last).all():
        raise ValueError("Encoder hidden states contain non-finite values.")

    if pooling == "attention":
        if outputs.attentions is None or outputs.attentions[-1] is None:
            raise RuntimeError("Model did not return attention tensors with attention pooling enabled.")
        feature = attention_pool(hidden_last, outputs.attentions[-1], inputs["attention_mask"])
    else:
        feature = masked_mean_pool(hidden_last, inputs["attention_mask"])
    return feature.detach().cpu().float().numpy()


def tensor_stats(tensor) -> dict[str, object]:
    import torch

    finite_mask = torch.isfinite(tensor)
    total = int(tensor.numel())
    finite = int(finite_mask.sum().item())
    nan_count = int(torch.isnan(tensor).sum().item())
    inf_count = int(torch.isinf(tensor).sum().item())
    return {
        "shape": tuple(int(v) for v in tensor.shape),
        "finite": finite,
        "total": total,
        "finite_ratio": finite / total if total else 1.0,
        "nan": nan_count,
        "inf": inf_count,
    }
