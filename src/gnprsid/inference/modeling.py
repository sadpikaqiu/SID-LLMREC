from __future__ import annotations

from pathlib import Path
from typing import Iterable

from gnprsid.common.io import read_json
from gnprsid.common.profiles import load_model_profile
from gnprsid.common.runtime import resolve_torch_dtype


def _build_fallback_chat_prompt(messages: list[dict[str, str]]) -> str:
    chunks = []
    for message in messages:
        role = message["role"].upper()
        chunks.append(f"[{role}]\n{message['content']}")
    chunks.append("[ASSISTANT]\n")
    return "\n\n".join(chunks)


def render_chat_prompts(tokenizer, message_batches: list[list[dict[str, str]]]) -> list[str]:
    prompts = []
    for messages in message_batches:
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt = _build_fallback_chat_prompt(messages)
        else:
            prompt = _build_fallback_chat_prompt(messages)
        prompts.append(prompt)
    return prompts


def load_generation_model(model_config_path: str | Path, checkpoint_path: str | Path | None = None):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ImportError("Local inference requires torch and transformers.") from error

    model_cfg = load_model_profile(model_config_path)
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = resolve_torch_dtype(torch, str(model_cfg.get("dtype", "auto")), device_type)
    device_map = model_cfg.get("device_map")
    model_kwargs = {
        "trust_remote_code": True,
        "dtype": dtype,
    }
    if device_map not in {None, ""}:
        model_kwargs["device_map"] = device_map

    if checkpoint_path and checkpoint_path.joinpath("adapter_config.json").exists():
        try:
            from peft import PeftModel
        except ImportError as error:
            raise ImportError("Loading PEFT adapters requires the 'peft' package.") from error

        adapter_config = read_json(checkpoint_path / "adapter_config.json")
        base_model_source = adapter_config.get("base_model_name_or_path") or model_cfg["base_model"]
        tokenizer_source = (
            checkpoint_path
            if checkpoint_path.joinpath("tokenizer_config.json").exists()
            else model_cfg.get("tokenizer_name", base_model_source)
        )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(base_model_source, **model_kwargs)
        model = PeftModel.from_pretrained(model, str(checkpoint_path))
        model_source = f"{base_model_source} + {checkpoint_path}"
    else:
        model_source = str(checkpoint_path) if checkpoint_path else str(model_cfg["base_model"])
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path if checkpoint_path else model_cfg.get("tokenizer_name", model_cfg["base_model"]),
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path if checkpoint_path else model_cfg["base_model"],
            **model_kwargs,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    return model_cfg, tokenizer, model, model_source


def generate_from_messages(
    model_cfg: dict,
    tokenizer,
    model,
    message_batches: list[list[dict[str, str]]],
    batch_size: int = 1,
) -> list[str]:
    import torch
    from tqdm import tqdm

    generation_cfg = dict(model_cfg.get("generation", {}))
    do_sample = bool(generation_cfg.get("do_sample", False))
    max_new_tokens = int(generation_cfg.get("max_new_tokens", 200))
    prompts = render_chat_prompts(tokenizer, message_batches)
    results: list[str] = []
    model_device = next(model.parameters()).device

    total_batches = (len(prompts) + batch_size - 1) // batch_size
    for start in tqdm(
        range(0, len(prompts), batch_size),
        total=total_batches,
        desc="Generating",
    ):
        prompt_batch = prompts[start : start + batch_size]
        inputs = tokenizer(
            prompt_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(model_cfg.get("max_length", 2048)),
        )
        input_lengths = inputs["attention_mask"].sum(dim=1)
        inputs = {key: value.to(model_device) for key, value in inputs.items()}

        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = float(generation_cfg.get("temperature", 0.7))
            generate_kwargs["top_p"] = float(generation_cfg.get("top_p", 0.9))

        with torch.no_grad():
            outputs = model.generate(**inputs, **generate_kwargs)

        for row_index, output_ids in enumerate(outputs):
            generated_ids = output_ids[int(input_lengths[row_index].item()) :]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            results.append(text)
    return results
