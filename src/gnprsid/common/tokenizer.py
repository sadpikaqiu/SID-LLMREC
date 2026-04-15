from __future__ import annotations

from pathlib import Path
from typing import Any


def build_tokenizer_load_kwargs(**extra_kwargs: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "fix_mistral_regex": True,
    }
    kwargs.update(extra_kwargs)
    return kwargs


def load_tokenizer_with_fallback(
    AutoTokenizer,
    primary_source: str | Path,
    fallback_source: str | Path,
    **extra_kwargs: Any,
):
    tokenizer_kwargs = build_tokenizer_load_kwargs(**extra_kwargs)
    try:
        return AutoTokenizer.from_pretrained(
            primary_source,
            **tokenizer_kwargs,
        )
    except AttributeError as error:
        message = str(error)
        if "has no attribute 'keys'" not in message:
            raise
        return AutoTokenizer.from_pretrained(
            fallback_source,
            **tokenizer_kwargs,
        )
