import subprocess
from pathlib import Path

from gnprsid.common.config import dump_yaml
from gnprsid.train.base import _cleanup_grpo_runtime_processes, run_training_stage


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
            "model_profile": "qwen3-8b-instruct",
            "train_path": str(train_path),
            "valid_path": str(valid_path),
            "init_model_path": str(init_model_path),
            "reward_function_path": str(reward_path),
            "output_dir": str(output_dir),
            "logger": "[console]",
            "tensor_model_parallel_size": 2,
            "use_remove_padding": False,
            "trainer_use_legacy_worker_impl": "enable",
            "actor_param_offload": True,
            "actor_optimizer_offload": True,
            "rollout_free_cache_engine": True,
            "rollout_enforce_eager": True,
            "lora": {
                "r": 16,
                "alpha": 32,
                "target_modules": ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"],
            },
        },
    )

    captured = {}

    monkeypatch.setattr("gnprsid.train.base.find_spec", lambda name: object() if name == "verl" else None)

    def fake_run(command, check, env=None):
        captured["command"] = command
        captured["check"] = check
        captured["env"] = env

    monkeypatch.setattr("gnprsid.train.base.subprocess.run", fake_run)

    manifest = run_training_stage(config_path, stage_override="grpo")

    command = captured["command"]
    assert command[:3] == ["python", "-m", "verl.trainer.main_ppo"] or command[1:3] == ["-m", "verl.trainer.main_ppo"]
    assert any(token == "algorithm.adv_estimator=grpo" for token in command)
    assert any(token == f"data.train_files={train_path}" for token in command)
    assert any(token == "++data.apply_chat_template_kwargs.enable_thinking=false" for token in command)
    assert any(token == f"reward.custom_reward_function.path={reward_path}" for token in command)
    assert any(token == f"actor_rollout_ref.model.path={init_model_path}" for token in command)
    assert any(token == "+actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2" for token in command)
    assert any(token == "actor_rollout_ref.model.use_remove_padding=false" for token in command)
    assert any(token == "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096" for token in command)
    assert any(token == "actor_rollout_ref.rollout.tensor_model_parallel_size=2" for token in command)
    assert any(token == "actor_rollout_ref.rollout.free_cache_engine=true" for token in command)
    assert any(token == "actor_rollout_ref.rollout.enforce_eager=true" for token in command)
    assert any(token == "actor_rollout_ref.actor.fsdp_config.param_offload=true" for token in command)
    assert any(token == "actor_rollout_ref.actor.fsdp_config.optimizer_offload=true" for token in command)
    assert any(token == "trainer.use_legacy_worker_impl=enable" for token in command)
    assert captured["env"]["GNPRSID_REWARD_TRACE_DIR"] == str(output_dir / "reward_traces")
    assert captured["env"]["GNPRSID_REWARD_TRACE_GROUP_SIZE"] == "512"
    assert captured["check"] is True
    assert manifest["result"]["output_dir"] == str(output_dir)
    assert manifest["result"]["reward_trace_dir"] == str(output_dir / "reward_traces")


def test_cleanup_grpo_runtime_processes_attempts_ray_and_vllm_cleanup(monkeypatch):
    commands = []

    def fake_which(name):
        mapping = {
            "ray": "/usr/bin/ray",
            "pkill": "/usr/bin/pkill",
        }
        return mapping.get(name)

    def fake_run(command, check, env=None):
        commands.append((command, check, env))

    monkeypatch.setattr("gnprsid.train.base.shutil.which", fake_which)
    monkeypatch.setattr("gnprsid.train.base.subprocess.run", fake_run)

    attempted = _cleanup_grpo_runtime_processes()

    assert attempted == [
        ["/usr/bin/ray", "stop", "--force"],
        ["/usr/bin/pkill", "-f", "VLLM::EngineCore"],
        ["/usr/bin/pkill", "-f", "VLLM::Worker_TP"],
    ]
    assert commands == [
        (["/usr/bin/ray", "stop", "--force"], False, None),
        (["/usr/bin/pkill", "-f", "VLLM::EngineCore"], False, None),
        (["/usr/bin/pkill", "-f", "VLLM::Worker_TP"], False, None),
    ]


def test_run_training_stage_grpo_attempts_cleanup_after_failure(monkeypatch, tmp_path):
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
            "model_profile": "qwen3-8b-instruct",
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

    commands = []
    monkeypatch.setattr("gnprsid.train.base.find_spec", lambda name: object() if name == "verl" else None)

    def fake_which(name):
        mapping = {
            "ray": "/usr/bin/ray",
            "pkill": "/usr/bin/pkill",
        }
        return mapping.get(name)

    def fake_run(command, check, env=None):
        commands.append((command, check, env))
        if command and command[0] != "/usr/bin/ray" and command[0] != "/usr/bin/pkill":
            raise subprocess.CalledProcessError(returncode=1, cmd=command)

    monkeypatch.setattr("gnprsid.train.base.shutil.which", fake_which)
    monkeypatch.setattr("gnprsid.train.base.subprocess.run", fake_run)

    try:
        run_training_stage(config_path, stage_override="grpo")
    except subprocess.CalledProcessError:
        pass
    else:
        raise AssertionError("Expected GRPO backend to re-raise subprocess failure.")

    assert commands[1:] == [
        (["/usr/bin/ray", "stop", "--force"], False, None),
        (["/usr/bin/pkill", "-f", "VLLM::EngineCore"], False, None),
        (["/usr/bin/pkill", "-f", "VLLM::Worker_TP"], False, None),
    ]
