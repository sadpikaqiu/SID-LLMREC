from types import SimpleNamespace

from gnprsid.common.io import iter_jsonl, write_jsonl
from gnprsid.grpo.build_data import build_grpo_data


def test_build_grpo_data_writes_ms_swift_jsonl(monkeypatch, tmp_path):
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

    result = build_grpo_data("NYC")
    train_rows = list(iter_jsonl(result["train_path"]))
    valid_rows = list(iter_jsonl(result["valid_path"]))

    assert len(train_rows) == 1
    assert len(valid_rows) == 1
    assert result["train_path"].endswith("train.jsonl")
    assert result["valid_path"].endswith("valid.jsonl")

    first_row = train_rows[0]
    assert list(first_row.keys()) == [
        "data_source",
        "ability",
        "messages",
        "ground_truth",
        "sample_id",
        "uid",
        "repr",
        "history_source",
        "target_time",
        "prompt_template_version",
    ]
    assert len(first_row["messages"]) == 2
    assert first_row["messages"][0]["role"] == "system"
    assert "exactly 10 complete semantic IDs" in first_row["messages"][0]["content"]
    assert "You must return exactly 10 complete semantic IDs" in first_row["messages"][1]["content"]
    assert first_row["ground_truth"] == "<a_1><b_2><c_3>"
    assert first_row["prompt_template_version"] == "v3"
    assert result["model_profile"] == "qwen3-8b-instruct"
