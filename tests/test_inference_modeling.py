import sys
import types

from gnprsid.inference.modeling import generate_from_messages, load_generation_model


class _FakeTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "kwargs": kwargs,
            }
        )
        return "rendered-prompt"


def test_generate_from_messages_passes_enable_thinking(monkeypatch):
    tokenizer = _FakeTokenizer()
    captured = {}

    def fake_generate_from_raw_prompts(model_cfg, tokenizer, model, prompts, batch_size, allowed_completions, top_k_sequences):
        captured["prompts"] = prompts
        return ["ok"]

    monkeypatch.setattr("gnprsid.inference.modeling.generate_from_raw_prompts", fake_generate_from_raw_prompts)

    result = generate_from_messages(
        model_cfg={"enable_thinking": False, "generation": {}},
        tokenizer=tokenizer,
        model=None,
        message_batches=[[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]],
        batch_size=1,
        allowed_completions=None,
        top_k_sequences=10,
    )

    assert result == ["ok"]
    assert captured["prompts"] == ["rendered-prompt"]
    assert tokenizer.calls[0]["kwargs"]["enable_thinking"] is False


class _FakeLoadedTokenizer:
    def __init__(self):
        self.pad_token = None
        self.eos_token = "<eos>"
        self.padding_side = None


class _FakeLoadedModel:
    def __init__(self):
        self.eval_called = False

    def eval(self):
        self.eval_called = True


def test_load_generation_model_uses_adapter_base_model_for_tokenizer(monkeypatch, tmp_path):
    checkpoint_path = tmp_path / "checkpoint-2200"
    checkpoint_path.mkdir()
    (checkpoint_path / "adapter_config.json").write_text("{}", encoding="utf-8")

    tokenizer_calls = {}
    model_calls = {}
    fake_tokenizer = _FakeLoadedTokenizer()
    fake_model = _FakeLoadedModel()

    monkeypatch.setattr(
        "gnprsid.inference.modeling.load_model_profile",
        lambda _: {
            "base_model": "/stale/model-path",
            "tokenizer_name": "/stale/model-path",
            "dtype": "bfloat16",
            "device_map": None,
            "generation": {},
        },
    )
    monkeypatch.setattr("gnprsid.inference.modeling.resolve_project_path", lambda path: path)
    monkeypatch.setattr(
        "gnprsid.inference.modeling.read_json",
        lambda _: {"base_model_name_or_path": "/resolved/base-model"},
    )
    monkeypatch.setattr(
        "gnprsid.inference.modeling.resolve_adapter_base_model_source",
        lambda preferred, fallback: "/resolved/base-model",
    )

    def fake_load_tokenizer(auto_tokenizer, primary_source, fallback_source, **kwargs):
        tokenizer_calls["primary"] = primary_source
        tokenizer_calls["fallback"] = fallback_source
        return fake_tokenizer

    monkeypatch.setattr("gnprsid.inference.modeling._load_tokenizer_with_fallback", fake_load_tokenizer)
    monkeypatch.setattr("gnprsid.inference.modeling._normalize_generation_config", lambda model, cfg: None)

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
            return fake_model

    class _FakeAutoTokenizer:
        pass

    class _FakePeftModel:
        @classmethod
        def from_pretrained(cls, model, adapter_dir):
            model_calls["adapter_path"] = adapter_dir
            return model

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoModelForCausalLM=_FakeAutoModelForCausalLM, AutoTokenizer=_FakeAutoTokenizer))
    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(PeftModel=_FakePeftModel))

    _, tokenizer, model, model_source = load_generation_model("configs/models/qwen3_8b.yaml", checkpoint_path=checkpoint_path)

    assert tokenizer_calls == {
        "primary": "/resolved/base-model",
        "fallback": "/resolved/base-model",
    }
    assert tokenizer is fake_tokenizer
    assert model is fake_model
    assert fake_model.eval_called is True
    assert model_calls["source"] == "/resolved/base-model"
    assert model_source == f"/resolved/base-model + {checkpoint_path}"
