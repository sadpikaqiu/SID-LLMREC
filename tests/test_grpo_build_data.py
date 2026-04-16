from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from gnprsid.common.io import write_jsonl
from gnprsid.grpo.build_data import build_grpo_data


def test_build_grpo_data_writes_verl_parquet(monkeypatch, tmp_path):
    processed = tmp_path / "processed"
    artifacts = tmp_path / "artifacts"
    processed.mkdir()
    artifacts.mkdir()

    row = {
        "sample_id": "train-00000-u1",
        "uid": 1,
        "repr": "sid",
        "split": "train",
        "target": "<a_1><b_2><c_3>",
        "target_time": "2024-01-01 12:00:00",
        "key_text": "User_1 visited: <a_1><b_2><c_3> at 2024-01-01 10:00:00",
        "input_text": "User_1 visited: <a_1><b_2><c_3> at 2024-01-01 10:00:00. When 2024-01-01 12:00:00 user_1 is likely to visit:",
    }
    valid_row = dict(row)
    valid_row["sample_id"] = "val-00000-u1"
    valid_row["split"] = "val"

    write_jsonl(processed / "samples_sid_train.jsonl", [row])
    write_jsonl(processed / "samples_sid_val.jsonl", [valid_row])

    fake_paths = SimpleNamespace(processed=processed, artifacts=artifacts)
    monkeypatch.setattr("gnprsid.grpo.build_data.dataset_paths", lambda dataset: fake_paths)
    monkeypatch.setattr(
        "gnprsid.grpo.build_data._load_grpo_tokenizer",
        lambda model_profile: ({"enable_thinking": False}, object()),
    )
    monkeypatch.setattr(
        "gnprsid.grpo.build_data._render_messages",
        lambda messages, model_cfg, tokenizer: "rendered-grpo-prompt",
    )

    result = build_grpo_data("NYC")
    train_df = pd.read_parquet(result["train_path"])
    valid_df = pd.read_parquet(result["valid_path"])

    assert len(train_df) == 1
    assert len(valid_df) == 1
    assert list(train_df.columns) == ["data_source", "prompt", "prompt_messages", "ability", "reward_model", "extra_info"]
    first_prompt = train_df.iloc[0]["prompt"]
    assert first_prompt == "rendered-grpo-prompt"
    prompt_items = list(train_df.iloc[0]["prompt_messages"])
    assert len(prompt_items) == 2
    assert prompt_items[0]["role"] == "system"
    assert "exactly 10 complete semantic IDs" in prompt_items[0]["content"]
    assert "You must return exactly 10 complete semantic IDs" in prompt_items[1]["content"]
    assert "<a_1><b_2><c_3>" in prompt_items[1]["content"]
    assert "Start the reply immediately with the first semantic ID." in prompt_items[1]["content"]
    assert train_df.iloc[0]["reward_model"]["ground_truth"] == "<a_1><b_2><c_3>"
    assert train_df.iloc[0]["extra_info"]["prompt_template_version"] == "v3"
    assert result["model_profile"] == "qwen3-8b-instruct"
