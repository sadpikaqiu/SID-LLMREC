from pathlib import Path

from gnprsid.common.profiles import resolve_adapter_base_model_source
from gnprsid.inference.modeling import _load_tokenizer_with_fallback as load_inference_tokenizer
from gnprsid.train.merge import _load_tokenizer_with_fallback as load_merge_tokenizer


class _FakeAutoTokenizer:
    calls = []

    @classmethod
    def from_pretrained(cls, source, trust_remote_code=True):
        cls.calls.append((source, trust_remote_code))
        if str(source) == "broken-adapter":
            raise AttributeError("'list' object has no attribute 'keys'")
        return {"source": source}


def test_inference_tokenizer_falls_back_on_broken_adapter_config():
    _FakeAutoTokenizer.calls = []
    tokenizer = load_inference_tokenizer(_FakeAutoTokenizer, "broken-adapter", "base-tokenizer")
    assert tokenizer == {"source": "base-tokenizer"}
    assert _FakeAutoTokenizer.calls == [
        ("broken-adapter", True),
        ("base-tokenizer", True),
    ]


def test_merge_tokenizer_falls_back_on_broken_adapter_config():
    _FakeAutoTokenizer.calls = []
    tokenizer = load_merge_tokenizer(_FakeAutoTokenizer, "broken-adapter", "base-tokenizer")
    assert tokenizer == {"source": "base-tokenizer"}
    assert _FakeAutoTokenizer.calls == [
        ("broken-adapter", True),
        ("base-tokenizer", True),
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
