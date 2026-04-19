from pathlib import Path
from types import SimpleNamespace

from gnprsid.common.config import dump_yaml
from gnprsid.common.io import write_jsonl
from gnprsid.grpo.inspect_sample import inspect_grpo_sample


def test_inspect_grpo_sample_replays_single_row(monkeypatch, tmp_path):
    grpo_dir = tmp_path / "grpo"
    grpo_dir.mkdir()
    grpo_path = grpo_dir / "valid.jsonl"
    write_jsonl(
        grpo_path,
        [
            {
                "data_source": "gnprsid_nyc_sid_current",
                "messages": [
                    {"role": "system", "content": "system text"},
                    {"role": "user", "content": "user text"},
                ],
                "ability": "next_poi_current",
                "ground_truth": "<a_1><b_2><c_3>",
                "sample_id": "val-1",
                "uid": 1,
                "repr": "sid",
                "history_source": "current",
                "target_time": "2024-01-01 12:00:00",
                "prompt_template_version": "v3",
            }
        ],
    )

    config_path = tmp_path / "grpo.yaml"
    dump_yaml(
        config_path,
        {
            "stage": "grpo",
            "backend": "ms-swift",
            "dataset": "NYC",
            "model_profile": "qwen3-8b-instruct",
            "train_path": str(grpo_path),
            "valid_path": str(grpo_path),
            "init_model_path": str(tmp_path / "checkpoint"),
        },
    )

    monkeypatch.setattr(
        "gnprsid.grpo.inspect_sample.resolve_model_profile_path",
        lambda profile: Path(tmp_path / "qwen3_8b.yaml"),
    )
    monkeypatch.setattr(
        "gnprsid.grpo.inspect_sample.load_generation_model",
        lambda model_config_path, checkpoint_path=None: (
            {"generation": {}, "enable_thinking": False},
            object(),
            object(),
            "fake-model",
        ),
    )
    monkeypatch.setattr(
        "gnprsid.grpo.inspect_sample.render_chat_prompts",
        lambda tokenizer, message_batches, chat_template_kwargs=None: ["rendered-prompt"],
    )
    monkeypatch.setattr(
        "gnprsid.grpo.inspect_sample.generate_from_messages",
        lambda model_cfg, tokenizer, model, message_batches, batch_size, allowed_completions, top_k_sequences: [
            "<a_1><b_2><c_3>"
        ],
    )
    monkeypatch.setattr(
        "gnprsid.grpo.inspect_sample.dataset_paths",
        lambda dataset: SimpleNamespace(artifacts=tmp_path / "artifacts"),
    )

    result = inspect_grpo_sample(config_path, split="valid", row_index=0)

    assert result["sample_id"] == "val-1"
    assert result["rendered_prompt"] == "rendered-prompt"
    assert result["prediction"] == "<a_1><b_2><c_3>"
    assert result["parsed_predictions"] == ["<a_1><b_2><c_3>"]
    assert result["parsed_prediction_count"] == 1


def test_inspect_grpo_sample_selects_row_by_sample_id(monkeypatch, tmp_path):
    grpo_dir = tmp_path / "grpo"
    grpo_dir.mkdir()
    grpo_path = grpo_dir / "valid.jsonl"
    write_jsonl(
        grpo_path,
        [
            {
                "data_source": "gnprsid_nyc_sid_current",
                "messages": [
                    {"role": "system", "content": "system A"},
                    {"role": "user", "content": "user A"},
                ],
                "ability": "next_poi_current",
                "ground_truth": "<a_1><b_2><c_3>",
                "sample_id": "val-1",
                "uid": 1,
                "repr": "sid",
                "history_source": "current",
                "target_time": "2024-01-01 12:00:00",
                "prompt_template_version": "v3",
            },
            {
                "data_source": "gnprsid_nyc_sid_current",
                "messages": [
                    {"role": "system", "content": "system B"},
                    {"role": "user", "content": "user B"},
                ],
                "ability": "next_poi_current",
                "ground_truth": "<a_9><b_9><c_9>",
                "sample_id": "val-2",
                "uid": 2,
                "repr": "sid",
                "history_source": "current",
                "target_time": "2024-01-02 12:00:00",
                "prompt_template_version": "v3",
            },
        ],
    )

    config_path = tmp_path / "grpo.yaml"
    dump_yaml(
        config_path,
        {
            "stage": "grpo",
            "backend": "ms-swift",
            "dataset": "NYC",
            "model_profile": "qwen3-8b-instruct",
            "train_path": str(grpo_path),
            "valid_path": str(grpo_path),
            "init_model_path": str(tmp_path / "checkpoint"),
        },
    )

    monkeypatch.setattr(
        "gnprsid.grpo.inspect_sample.resolve_model_profile_path",
        lambda profile: Path(tmp_path / "qwen3_8b.yaml"),
    )
    monkeypatch.setattr(
        "gnprsid.grpo.inspect_sample.load_generation_model",
        lambda model_config_path, checkpoint_path=None: (
            {"generation": {}, "enable_thinking": False},
            object(),
            object(),
            "fake-model",
        ),
    )
    monkeypatch.setattr(
        "gnprsid.grpo.inspect_sample.render_chat_prompts",
        lambda tokenizer, message_batches, chat_template_kwargs=None: ["rendered-prompt"],
    )
    monkeypatch.setattr(
        "gnprsid.grpo.inspect_sample.generate_from_messages",
        lambda model_cfg, tokenizer, model, message_batches, batch_size, allowed_completions, top_k_sequences: [
            "<a_9><b_9><c_9>"
        ],
    )
    monkeypatch.setattr(
        "gnprsid.grpo.inspect_sample.dataset_paths",
        lambda dataset: SimpleNamespace(artifacts=tmp_path / "artifacts"),
    )

    result = inspect_grpo_sample(config_path, split="valid", sample_id="val-2")

    assert result["sample_id"] == "val-2"
    assert result["target"] == "<a_9><b_9><c_9>"
    assert result["prediction"] == "<a_9><b_9><c_9>"
