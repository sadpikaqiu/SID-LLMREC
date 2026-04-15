from pathlib import Path

from gnprsid.common.profiles import resolve_adapter_base_model_source
from gnprsid.common.tokenizer import (
    build_tokenizer_load_kwargs,
    load_tokenizer_with_fallback,
)


class _FakeAutoTokenizer:
    calls = []

    @classmethod
    def from_pretrained(cls, source, **kwargs):
        cls.calls.append((source, kwargs))
        if str(source) == "broken-adapter":
            raise AttributeError("'list' object has no attribute 'keys'")
        return {"source": source}


def test_build_tokenizer_load_kwargs_enable_mistral_regex_fix():
    kwargs = build_tokenizer_load_kwargs(use_fast=True)

    assert kwargs == {
        "trust_remote_code": True,
        "fix_mistral_regex": True,
        "use_fast": True,
    }


def test_tokenizer_falls_back_on_broken_adapter_config():
    _FakeAutoTokenizer.calls = []
    tokenizer = load_tokenizer_with_fallback(_FakeAutoTokenizer, "broken-adapter", "base-tokenizer")
    assert tokenizer == {"source": "base-tokenizer"}
    assert _FakeAutoTokenizer.calls == [
        ("broken-adapter", {"trust_remote_code": True, "fix_mistral_regex": True}),
        ("base-tokenizer", {"trust_remote_code": True, "fix_mistral_regex": True}),
    ]


def test_resolve_adapter_base_model_source_falls_back_from_missing_absolute_path(tmp_path):
    missing_path = tmp_path / "deleted-base-model"
    resolved = resolve_adapter_base_model_source(str(missing_path), "fallback/model")
    assert resolved == "fallback/model"


def test_resolve_adapter_base_model_source_keeps_existing_project_path(tmp_path, monkeypatch):
    adapter_base = tmp_path / "checkpoints" / "merged-base"
    adapter_base.mkdir(parents=True)
    monkeypatch.setattr("gnprsid.common.profiles.project_root", lambda: tmp_path)
    resolved = resolve_adapter_base_model_source(Path("checkpoints/merged-base"), "fallback/model")
    assert resolved == str(adapter_base)
