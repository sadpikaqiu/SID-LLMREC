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
