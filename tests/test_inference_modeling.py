from gnprsid.inference.modeling import generate_from_messages


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
