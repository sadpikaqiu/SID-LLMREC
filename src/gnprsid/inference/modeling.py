from __future__ import annotations

from pathlib import Path
from typing import Iterable

from gnprsid.common.io import read_json
from gnprsid.common.profiles import (
    load_model_profile,
    resolve_adapter_base_model_source,
    resolve_model_source,
    resolve_project_path,
)
from gnprsid.common.runtime import resolve_torch_dtype
from gnprsid.common.tokenizer import load_tokenizer_with_fallback as _load_tokenizer_with_fallback


def _build_fallback_chat_prompt(messages: list[dict[str, str]]) -> str:
    chunks = []
    for message in messages:
        role = message["role"].upper()
        chunks.append(f"[{role}]\n{message['content']}")
    chunks.append("[ASSISTANT]\n")
    return "\n\n".join(chunks)


def _resolve_chat_template_kwargs(model_cfg: dict) -> dict:
    kwargs = dict(model_cfg.get("chat_template_kwargs", {}))
    if "enable_thinking" in model_cfg and "enable_thinking" not in kwargs:
        kwargs["enable_thinking"] = bool(model_cfg["enable_thinking"])
    return kwargs


def render_chat_prompts(
    tokenizer,
    message_batches: list[list[dict[str, str]]],
    chat_template_kwargs: dict | None = None,
) -> list[str]:
    prompts = []
    template_kwargs = dict(chat_template_kwargs or {})
    for messages in message_batches:
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **template_kwargs,
                )
            except TypeError as error:
                if template_kwargs and "unexpected keyword argument" in str(error):
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                else:
                    raise
            except Exception:
                prompt = _build_fallback_chat_prompt(messages)
        else:
            prompt = _build_fallback_chat_prompt(messages)
        prompts.append(prompt)
    return prompts


def _build_candidate_sequence_map(tokenizer, allowed_completions: list[str]) -> dict[tuple[int, ...], str]:
    sequence_map: dict[tuple[int, ...], str] = {}
    for text in allowed_completions:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if token_ids:
            sequence_map.setdefault(tuple(token_ids), text)
    if not sequence_map:
        raise ValueError("allowed_completions must contain at least one tokenizable completion")
    return sequence_map


def _build_token_trie(sequences: Iterable[tuple[int, ...]]) -> dict[int | str, dict]:
    root: dict[int | str, dict] = {}
    for sequence in sequences:
        node = root
        for token_id in sequence:
            node = node.setdefault(token_id, {})
        node["__end__"] = {}
    return root


def _lookup_allowed_tokens(
    trie: dict[int | str, dict],
    generated_ids: list[int],
    eos_token_id: int | None,
) -> list[int]:
    node = trie
    for token_id in generated_ids:
        next_node = node.get(token_id)
        if next_node is None:
            return [eos_token_id] if eos_token_id is not None else []
        node = next_node

    allowed = [token_id for token_id in node.keys() if token_id != "__end__"]
    if "__end__" in node and eos_token_id is not None:
        allowed.append(eos_token_id)
    return allowed


def _trim_after_eos(token_ids: list[int], eos_token_id: int | None) -> tuple[int, ...]:
    if eos_token_id is None:
        return tuple(token_ids)
    if eos_token_id in token_ids:
        return tuple(token_ids[: token_ids.index(eos_token_id)])
    return tuple(token_ids)


def _normalize_generation_config(model, generation_cfg: dict) -> None:
    cfg = getattr(model, "generation_config", None)
    if cfg is None:
        return

    do_sample = bool(generation_cfg.get("do_sample", False))
    cfg.do_sample = do_sample
    if do_sample:
        if "temperature" in generation_cfg:
            cfg.temperature = float(generation_cfg["temperature"])
        if "top_p" in generation_cfg:
            cfg.top_p = float(generation_cfg["top_p"])
        if "top_k" in generation_cfg:
            cfg.top_k = int(generation_cfg["top_k"])
    else:
        cfg.temperature = 1.0
        cfg.top_p = 1.0
        cfg.top_k = 50


def _generate_constrained_topk(
    model_cfg: dict,
    tokenizer,
    model,
    prompts: list[str],
    batch_size: int,
    allowed_completions: list[str],
    top_k_sequences: int,
) -> list[str]:
    import torch
    from tqdm import tqdm

    sequence_map = _build_candidate_sequence_map(tokenizer, allowed_completions)
    trie = _build_token_trie(sequence_map.keys())
    eos_token_id = tokenizer.eos_token_id
    max_candidate_len = max(len(sequence) for sequence in sequence_map)
    num_return_sequences = min(top_k_sequences, len(sequence_map))
    num_beams = min(len(sequence_map), max(num_return_sequences * 5, 50))
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
        prompt_padded_length = int(inputs["input_ids"].shape[1])
        inputs = {key: value.to(model_device) for key, value in inputs.items()}

        def prefix_allowed_tokens_fn(batch_id, input_ids):
            generated_ids = input_ids[prompt_padded_length:].tolist()
            allowed = _lookup_allowed_tokens(trie, generated_ids, eos_token_id)
            return allowed or ([eos_token_id] if eos_token_id is not None else [])

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_candidate_len + 1,
                do_sample=False,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
                early_stopping=True,
            )

        batch_size_actual = len(prompt_batch)
        for row_index in range(batch_size_actual):
            ranked: list[str] = []
            seen = set()
            start_index = row_index * num_return_sequences
            end_index = start_index + num_return_sequences
            for output_ids in outputs[start_index:end_index]:
                generated_ids = output_ids[prompt_padded_length:].tolist()
                token_key = _trim_after_eos(generated_ids, eos_token_id)
                candidate = sequence_map.get(token_key)
                if not candidate:
                    candidate = tokenizer.decode(list(token_key), skip_special_tokens=True).strip()
                if candidate and candidate not in seen:
                    seen.add(candidate)
                    ranked.append(candidate)
            results.append(" ".join(ranked[:top_k_sequences]))

    return results


def load_generation_model(model_config_path: str | Path, checkpoint_path: str | Path | None = None):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise ImportError("Local inference requires torch and transformers.") from error

    model_cfg = load_model_profile(model_config_path)
    checkpoint_path = resolve_project_path(checkpoint_path) if checkpoint_path else None
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
        base_model_source = resolve_adapter_base_model_source(
            adapter_config.get("base_model_name_or_path"),
            model_cfg["base_model"],
        )
        tokenizer_source = (
            checkpoint_path
            if checkpoint_path.joinpath("tokenizer_config.json").exists()
            else resolve_model_source(model_cfg.get("tokenizer_name", base_model_source))
        )
        tokenizer = _load_tokenizer_with_fallback(
            AutoTokenizer,
            tokenizer_source,
            resolve_model_source(model_cfg.get("tokenizer_name", base_model_source)),
        )
        model = AutoModelForCausalLM.from_pretrained(base_model_source, **model_kwargs)
        model = PeftModel.from_pretrained(model, str(checkpoint_path))
        model_source = f"{base_model_source} + {checkpoint_path}"
    else:
        base_model_source = resolve_model_source(model_cfg["base_model"])
        tokenizer_source = resolve_model_source(model_cfg.get("tokenizer_name", base_model_source))
        model_source = str(checkpoint_path) if checkpoint_path else base_model_source
        tokenizer = _load_tokenizer_with_fallback(
            AutoTokenizer,
            str(checkpoint_path) if checkpoint_path else tokenizer_source,
            tokenizer_source,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_path) if checkpoint_path else base_model_source,
            **model_kwargs,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    _normalize_generation_config(model, dict(model_cfg.get("generation", {})))
    model.eval()
    return model_cfg, tokenizer, model, model_source


def generate_from_raw_prompts(
    model_cfg: dict,
    tokenizer,
    model,
    prompts: list[str],
    batch_size: int = 1,
    allowed_completions: list[str] | None = None,
    top_k_sequences: int = 10,
) -> list[str]:
    import torch

    generation_cfg = dict(model_cfg.get("generation", {}))
    do_sample = bool(generation_cfg.get("do_sample", False))
    max_new_tokens = int(generation_cfg.get("max_new_tokens", 200))
    if allowed_completions is not None:
        return _generate_constrained_topk(
            model_cfg,
            tokenizer,
            model,
            prompts,
            batch_size=batch_size,
            allowed_completions=allowed_completions,
            top_k_sequences=top_k_sequences,
        )

    results: list[str] = []
    model_device = next(model.parameters()).device

    from tqdm import tqdm

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
        prompt_padded_length = int(inputs["input_ids"].shape[1])
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
            generated_ids = output_ids[prompt_padded_length:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            results.append(text)
    return results


def generate_from_messages(
    model_cfg: dict,
    tokenizer,
    model,
    message_batches: list[list[dict[str, str]]],
    batch_size: int = 1,
    allowed_completions: list[str] | None = None,
    top_k_sequences: int = 10,
) -> list[str]:
    prompts = render_chat_prompts(
        tokenizer,
        message_batches,
        chat_template_kwargs=_resolve_chat_template_kwargs(model_cfg),
    )
    return generate_from_raw_prompts(
        model_cfg,
        tokenizer,
        model,
        prompts,
        batch_size=batch_size,
        allowed_completions=allowed_completions,
        top_k_sequences=top_k_sequences,
    )
