from pathlib import Path

from gnprsid.common.config import dump_yaml
from gnprsid.train.base import run_training_stage


def test_run_training_stage_grpo_builds_verl_command(monkeypatch, tmp_path):
    output_dir = tmp_path / "grpo-output"
    train_path = tmp_path / "train.parquet"
    valid_path = tmp_path / "valid.parquet"
    init_model_path = tmp_path / "init-model"
    reward_path = tmp_path / "reward.py"

    train_path.write_text("placeholder", encoding="utf-8")
    valid_path.write_text("placeholder", encoding="utf-8")
    init_model_path.mkdir()
    reward_path.write_text("def compute_score(*args, **kwargs):\n    return 0.0\n", encoding="utf-8")

    config_path = tmp_path / "grpo.yaml"
    dump_yaml(
        config_path,
        {
            "stage": "grpo",
            "backend": "verl",
            "dataset": "NYC",
            "model_profile": "qwen2.5-7b-instruct",
            "train_path": str(train_path),
            "valid_path": str(valid_path),
            "init_model_path": str(init_model_path),
            "reward_function_path": str(reward_path),
            "output_dir": str(output_dir),
            "logger": "[console]",
            "lora": {
                "r": 16,
                "alpha": 32,
                "target_modules": ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"],
            },
        },
    )

    captured = {}

    monkeypatch.setattr("gnprsid.train.base.find_spec", lambda name: object() if name == "verl" else None)

    def fake_run(command, check):
        captured["command"] = command
        captured["check"] = check

    monkeypatch.setattr("gnprsid.train.base.subprocess.run", fake_run)

    manifest = run_training_stage(config_path, stage_override="grpo")

    command = captured["command"]
    assert command[:3] == ["python", "-m", "verl.trainer.main_ppo"] or command[1:3] == ["-m", "verl.trainer.main_ppo"]
    assert any(token == "algorithm.adv_estimator=grpo" for token in command)
    assert any(token == f"data.train_files={train_path}" for token in command)
    assert any(token == f"reward.custom_reward_function.path={reward_path}" for token in command)
    assert any(token == f"actor_rollout_ref.model.path={init_model_path}" for token in command)
    assert any(token == "+actor_rollout_ref.model.override_config._attn_implementation=flash_attention_2" for token in command)
    assert any(token == "+actor_rollout_ref.rollout.update_weights_bucket_megabytes=4096" for token in command)
    assert captured["check"] is True
    assert manifest["result"]["output_dir"] == str(output_dir)
