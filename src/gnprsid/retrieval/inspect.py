from __future__ import annotations

from pathlib import Path

from gnprsid.common.config import load_yaml
from gnprsid.common.io import iter_jsonl
from gnprsid.common.profiles import load_model_profile, resolve_project_path
from gnprsid.retrieval.encoder import load_encoder, masked_mean_pool, tensor_stats


def inspect_encoder(
    bank_path: str | Path,
    repr_name: str,
    model_config_path: str | Path | None = None,
    retrieval_config_path: str | Path | None = None,
    sample_count: int = 2,
    model_name_or_path: str | None = None,
) -> list[dict]:
    import torch

    retrieval_cfg = load_yaml(resolve_project_path(retrieval_config_path)) if retrieval_config_path else {}
    model_cfg = load_model_profile(model_config_path or retrieval_cfg.get("model_profile", "qwen2.5-7b-instruct"))
    tokenizer, model = load_encoder(
        model_name_or_path or model_cfg["base_model"],
        dtype_name=str(retrieval_cfg.get("dtype", model_cfg.get("dtype", "auto"))),
        pooling=str(retrieval_cfg.get("pooling", "mean")),
        device_map=str(retrieval_cfg.get("device_map", "auto")),
        load_in_4bit=bool(retrieval_cfg.get("load_in_4bit", False)),
    )
    max_length = int(retrieval_cfg.get("max_length", model_cfg.get("max_length", 2048)))
    pooling = str(retrieval_cfg.get("pooling", "mean"))

    rows = [row for row in iter_jsonl(bank_path) if row["repr"] == repr_name][:sample_count]
    reports = []
    for row in rows:
        inputs = tokenizer(row["key_text"], return_tensors="pt", truncation=True, max_length=max_length)
        model_device = next(model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, output_attentions=(pooling == "attention"))
        hidden_last = outputs.hidden_states[-1][0].float()
        report = {
            "sample_id": row["sample_id"],
            "split": row["split"],
            "uid": row["uid"],
            "token_count": int(inputs["input_ids"].shape[1]),
            "hidden_last": tensor_stats(hidden_last),
        }
        if pooling == "attention":
            report["attention_last"] = tensor_stats(outputs.attentions[-1][0].float())
        mean_feature = masked_mean_pool(outputs.hidden_states[-1], inputs["attention_mask"])
        report["masked_mean_feature"] = tensor_stats(mean_feature)
        report["feature_norm"] = float(torch.linalg.norm(mean_feature).item())
        reports.append(report)
    return reports
