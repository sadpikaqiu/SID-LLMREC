from pathlib import Path
from types import SimpleNamespace

from gnprsid.common.io import read_json, write_jsonl
from gnprsid.inference.batch import run_batch_inference


def _sample_row(sample_id: str, target: str) -> dict:
    return {
        "sample_id": sample_id,
        "uid": 1,
        "split": "test",
        "repr": "sid",
        "target": target,
        "target_time": "2024-01-01 12:00:00",
        "key_text": "User_1 visited: <a_1><b_2><c_3> at 2024-01-01 10:00:00",
        "input_text": "User_1 visited: <a_1><b_2><c_3> at 2024-01-01 10:00:00. When 2024-01-01 12:00:00 user_1 is likely to visit:",
    }


def test_run_batch_inference_direct_mode_uses_free_generation(monkeypatch, tmp_path):
    bank_path = tmp_path / "retrieval_bank_sid.jsonl"
    write_jsonl(bank_path, [_sample_row("test-00000-u1", "<a_1><b_2><c_3>")])

    fake_paths = SimpleNamespace(artifacts=tmp_path, outputs=tmp_path / "outputs", processed=tmp_path / "processed")
    captured = {}

    monkeypatch.setattr("gnprsid.inference.batch.dataset_paths", lambda dataset: fake_paths)
    monkeypatch.setattr(
        "gnprsid.inference.batch.load_generation_model",
        lambda model_config_path, checkpoint_path=None: ({"generation": {}, "max_length": 128}, None, None, "dummy-model"),
    )

    def fake_generate(model_cfg, tokenizer, model, message_batches, batch_size, allowed_completions, top_k_sequences):
        captured["allowed_completions"] = allowed_completions
        captured["prompt"] = message_batches[0][1]["content"]
        return ["<a_1><b_2><c_3> <a_4><b_5><c_6> <a_7><b_8><c_9> <a_10><b_11><c_12> <a_13><b_14><c_15> <a_16><b_17><c_18> <a_19><b_20><c_21> <a_22><b_23><c_24> <a_25><b_26><c_27> <a_28><b_29><c_30>"]

    monkeypatch.setattr("gnprsid.inference.batch.generate_from_messages", fake_generate)

    output_path = tmp_path / "direct.json"
    payload = run_batch_inference(
        dataset="NYC",
        repr_name="sid",
        history_source="current",
        model_config_path="configs/models/qwen25_7b.yaml",
        retrieval_bank_path=bank_path,
        output_path=output_path,
        decoding_mode="direct",
    )

    assert captured["allowed_completions"] is None
    assert "Return exactly 10 complete semantic IDs" in captured["prompt"]
    assert "Example:" not in captured["prompt"]
    assert "Start the reply immediately with the first semantic ID." in captured["prompt"]
    assert payload["metadata"]["decoding_mode"] == "direct_generation"
    assert payload["metadata"]["candidate_space_size"] == 0


def test_run_batch_inference_default_mode_keeps_candidate_constrained(monkeypatch, tmp_path):
    bank_path = tmp_path / "retrieval_bank_sid.jsonl"
    write_jsonl(
        bank_path,
        [
            _sample_row("test-00000-u1", "<a_1><b_2><c_3>"),
            _sample_row("train-00000-u2", "<a_4><b_5><c_6>"),
        ],
    )

    fake_paths = SimpleNamespace(artifacts=tmp_path, outputs=tmp_path / "outputs", processed=tmp_path / "processed")
    captured = {}

    monkeypatch.setattr("gnprsid.inference.batch.dataset_paths", lambda dataset: fake_paths)
    monkeypatch.setattr(
        "gnprsid.inference.batch.load_generation_model",
        lambda model_config_path, checkpoint_path=None: ({"generation": {}, "max_length": 128}, None, None, "dummy-model"),
    )

    def fake_generate(model_cfg, tokenizer, model, message_batches, batch_size, allowed_completions, top_k_sequences):
        captured["allowed_completions"] = allowed_completions
        captured["prompt"] = message_batches[0][1]["content"]
        return ["<a_1><b_2><c_3> <a_4><b_5><c_6>"]

    monkeypatch.setattr("gnprsid.inference.batch.generate_from_messages", fake_generate)

    payload = run_batch_inference(
        dataset="NYC",
        repr_name="sid",
        history_source="current",
        model_config_path="configs/models/qwen25_7b.yaml",
        retrieval_bank_path=bank_path,
        output_path=tmp_path / "constrained.json",
    )

    assert captured["allowed_completions"] == ["<a_1><b_2><c_3>", "<a_4><b_5><c_6>"]
    assert "Output format:" not in captured["prompt"]
    assert payload["metadata"]["decoding_mode"] == "candidate_constrained_beam_search"
    assert payload["metadata"]["candidate_space_size"] == 2
