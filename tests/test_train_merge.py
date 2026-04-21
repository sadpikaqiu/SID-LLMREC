import json
import sys
import types

from gnprsid.train.merge import merge_peft_adapter


class _FakeTokenizer:
    def __init__(self):
        self.saved_to = None

    def save_pretrained(self, path):
        self.saved_to = path


class _FakeMergedModel:
    def __init__(self):
        self.saved_to = None

    def save_pretrained(self, path, safe_serialization=True):
        self.saved_to = (path, safe_serialization)


class _FakePeftWrapper:
    def __init__(self, merged_model):
        self._merged_model = merged_model

    def merge_and_unload(self):
        return self._merged_model


def test_merge_peft_adapter_uses_adapter_base_model_for_tokenizer(monkeypatch, tmp_path):
    adapter_path = tmp_path / "checkpoint-2200"
    adapter_path.mkdir()
    (adapter_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "/resolved/base-model"}),
        encoding="utf-8",
    )
    output_path = tmp_path / "merged"

    tokenizer_calls = {}
    model_calls = {}
    merged_model = _FakeMergedModel()
    fake_tokenizer = _FakeTokenizer()

    monkeypatch.setattr(
        "gnprsid.train.merge.load_model_profile",
        lambda _: {
            "base_model": "/stale/model-path",
            "tokenizer_name": "/stale/model-path",
            "dtype": "bfloat16",
            "device_map": None,
        },
    )
    monkeypatch.setattr("gnprsid.train.merge.resolve_model_source", lambda source: str(source))
    monkeypatch.setattr("gnprsid.train.merge.resolve_project_path", lambda path: path)
    monkeypatch.setattr(
        "gnprsid.train.merge.resolve_adapter_base_model_source",
        lambda preferred, fallback: "/resolved/base-model",
    )

    def fake_load_tokenizer(auto_tokenizer, primary_source, fallback_source, **kwargs):
        tokenizer_calls["primary"] = primary_source
        tokenizer_calls["fallback"] = fallback_source
        return fake_tokenizer

    monkeypatch.setattr("gnprsid.train.merge._load_tokenizer_with_fallback", fake_load_tokenizer)

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"

    class _FakeAutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            model_calls["source"] = source
            model_calls["kwargs"] = kwargs
            return object()

    class _FakeAutoTokenizer:
        pass

    class _FakePeftModel:
        @classmethod
        def from_pretrained(cls, base_model, adapter_dir):
            model_calls["adapter_path"] = adapter_dir
            return _FakePeftWrapper(merged_model)

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoModelForCausalLM=_FakeAutoModelForCausalLM, AutoTokenizer=_FakeAutoTokenizer))
    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(PeftModel=_FakePeftModel))

    result = merge_peft_adapter("configs/models/qwen3_8b.yaml", adapter_path, output_path=output_path)

    assert tokenizer_calls == {
        "primary": "/resolved/base-model",
        "fallback": "/resolved/base-model",
    }
    assert model_calls["source"] == "/resolved/base-model"
    assert result["base_model"] == "/resolved/base-model"
