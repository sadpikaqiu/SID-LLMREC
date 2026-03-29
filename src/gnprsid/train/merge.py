from __future__ import annotations

from pathlib import Path

from gnprsid.common.io import ensure_dir, read_json, write_json
from gnprsid.common.profiles import load_model_profile, resolve_project_path
from gnprsid.common.runtime import resolve_torch_dtype


def _resolve_model_source(source: str | Path) -> str:
    source_path = Path(str(source))
    if source_path.is_absolute():
        return str(source_path)
    project_candidate = resolve_project_path(source_path)
    if project_candidate.exists():
        return str(project_candidate)
    return str(source)


def merge_peft_adapter(
    model_config_path: str | Path,
    adapter_path: str | Path,
    output_path: str | Path | None = None,
) -> dict:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ImportError("Merging a PEFT adapter requires torch, transformers, and peft.") from error

    model_cfg = load_model_profile(model_config_path)
    adapter_path = resolve_project_path(adapter_path)
    if output_path is None:
        output_path = adapter_path.parent / "merged"
    output_path = ensure_dir(resolve_project_path(output_path))

    base_model_source = _resolve_model_source(model_cfg["base_model"])
    tokenizer_source = _resolve_model_source(model_cfg.get("tokenizer_name", model_cfg["base_model"]))
    adapter_config_path = adapter_path / "adapter_config.json"
    if adapter_config_path.exists():
        adapter_config = read_json(adapter_config_path)
        base_model_source = _resolve_model_source(adapter_config.get("base_model_name_or_path") or base_model_source)
        if adapter_path.joinpath("tokenizer_config.json").exists():
            tokenizer_source = str(adapter_path)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = resolve_torch_dtype(torch, str(model_cfg.get("dtype", "auto")), device_type)
    device_map = model_cfg.get("device_map")

    model_kwargs = {
        "trust_remote_code": True,
        "dtype": dtype,
    }
    if device_map not in {None, ""}:
        model_kwargs["device_map"] = device_map

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_source,
        **model_kwargs,
    )
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))
    merged_model = peft_model.merge_and_unload()

    merged_model.save_pretrained(str(output_path), safe_serialization=True)
    tokenizer.save_pretrained(str(output_path))

    manifest = {
        "model_config_path": str(resolve_project_path(model_config_path)),
        "base_model": base_model_source,
        "adapter_path": str(adapter_path),
        "output_path": str(output_path),
    }
    write_json(output_path / "merge_manifest.json", manifest)
    return manifest
