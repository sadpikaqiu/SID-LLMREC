from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from gnprsid.common.config import dump_yaml
from gnprsid.grpo.inspect_sample import inspect_grpo_sample


def test_inspect_grpo_sample_replays_single_row(monkeypatch, tmp_path):
    grpo_dir = tmp_path / "grpo"
    grpo_dir.mkdir()
    grpo_path = grpo_dir / "valid.parquet"
    pd.DataFrame(
        [
            {
                "data_source": "gnprsid_nyc_sid_current",
                "prompt": [
                    {"role": "system", "content": "system text"},
                    {"role": "user", "content": "user text"},
                ],
                "ability": "next_poi_current",
                "reward_model": {"ground_truth": "<a_1><b_2><c_3>"},
                "extra_info": {"sample_id": "val-1"},
            }
        ]
    ).to_parquet(grpo_path, index=False)

    config_path = tmp_path / "grpo.yaml"
    dump_yaml(
        config_path,
        {
            "stage": "grpo",
            "backend": "verl",
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


def test_inspect_grpo_sample_supports_rendered_prompt_rows(monkeypatch, tmp_path):
    grpo_dir = tmp_path / "grpo"
    grpo_dir.mkdir()
    grpo_path = grpo_dir / "valid.parquet"
    pd.DataFrame(
        [
            {
                "data_source": "gnprsid_nyc_sid_current",
                "prompt": "<|im_start|>system\nsystem text<|im_end|>",
                "prompt_messages": [
                    {"role": "system", "content": "system text"},
                    {"role": "user", "content": "user text"},
                ],
                "ability": "next_poi_current",
                "reward_model": {"ground_truth": "<a_1><b_2><c_3>"},
                "extra_info": {"sample_id": "val-2"},
            }
        ]
    ).to_parquet(grpo_path, index=False)

    config_path = tmp_path / "grpo.yaml"
    dump_yaml(
        config_path,
        {
            "stage": "grpo",
            "backend": "verl",
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

    captured = {}

    def fake_generate_from_raw_prompts(model_cfg, tokenizer, model, prompts, batch_size, allowed_completions, top_k_sequences):
        captured["prompts"] = prompts
        return ["<a_1><b_2><c_3>"]

    monkeypatch.setattr("gnprsid.grpo.inspect_sample.generate_from_raw_prompts", fake_generate_from_raw_prompts)
    monkeypatch.setattr(
        "gnprsid.grpo.inspect_sample.dataset_paths",
        lambda dataset: SimpleNamespace(artifacts=tmp_path / "artifacts"),
    )

    result = inspect_grpo_sample(config_path, split="valid", row_index=0)

    assert captured["prompts"] == ["<|im_start|>system\nsystem text<|im_end|>"]
    assert result["sample_id"] == "val-2"
    assert result["prediction"] == "<a_1><b_2><c_3>"
    assert result["parsed_prediction_count"] == 1
