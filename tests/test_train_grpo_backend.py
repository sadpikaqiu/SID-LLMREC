import subprocess

from gnprsid.common.config import dump_yaml, load_yaml
from gnprsid.train.base import _cleanup_grpo_runtime_processes, run_training_stage


def test_run_training_stage_grpo_builds_ms_swift_command(monkeypatch, tmp_path):
    output_dir = tmp_path / "grpo-output"
    train_path = tmp_path / "train.jsonl"
    valid_path = tmp_path / "valid.jsonl"
    init_model_path = tmp_path / "init-model"
    reward_path = tmp_path / "reward.py"

    train_path.write_text('{"messages":[]}\n', encoding="utf-8")
    valid_path.write_text('{"messages":[]}\n', encoding="utf-8")
    init_model_path.mkdir()
    reward_path.write_text("def compute_score(*args, **kwargs):\n    return 0.0\n", encoding="utf-8")

    config_path = tmp_path / "grpo.yaml"
    dump_yaml(
        config_path,
        {
            "stage": "grpo",
            "backend": "ms-swift",
            "dataset": "NYC",
            "model_profile": "qwen3-8b-instruct",
            "train_path": str(train_path),
            "valid_path": str(valid_path),
            "init_model_path": str(init_model_path),
            "reward_function_path": str(reward_path),
            "output_dir": str(output_dir),
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "num_generations": 8,
            "n_gpus_per_node": 8,
            "nnodes": 1,
            "attn_impl": "flash_attention_2",
            "offload_model": True,
            "offload_optimizer": True,
            "vllm_tensor_parallel_size": 1,
            "project_name": "gnprsid-grpo",
            "experiment_name": "nyc-sid-current-qwen3",
            "report_to": "none",
            "lora": {
                "r": 16,
                "alpha": 32,
                "target_modules": ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"],
            },
        },
    )

    captured = {}
    monkeypatch.setattr("gnprsid.train.base.shutil.which", lambda name: "swift" if name == "swift" else None)

    def fake_run(command, check, env=None):
        captured["command"] = command
        captured["check"] = check
        captured["env"] = env

    monkeypatch.setattr("gnprsid.train.base.subprocess.run", fake_run)

    manifest = run_training_stage(config_path, stage_override="grpo")

    command = captured["command"]
    assert command == ["swift", "rlhf", "--config", str(output_dir / "ms_swift_grpo.yaml")]
    assert captured["check"] is True
    assert captured["env"]["GNPRSID_REWARD_TRACE_DIR"] == str(output_dir / "reward_traces")
    assert captured["env"]["GNPRSID_REWARD_TRACE_GROUP_SIZE"] == "8"
    assert captured["env"]["GNPRSID_GRPO_REWARD_PATH"] == str(reward_path)
    assert captured["env"]["GNPRSID_GRPO_REWARD_NAME"] == "compute_score"
    assert captured["env"]["NPROC_PER_NODE"] == "8"
    assert "src" in captured["env"]["PYTHONPATH"]

    generated_cfg = load_yaml(output_dir / "ms_swift_grpo.yaml")
    assert generated_cfg["rlhf_type"] == "grpo"
    assert generated_cfg["model"] == str(init_model_path)
    assert generated_cfg["dataset"] == [str(train_path)]
    assert generated_cfg["val_dataset"] == [str(valid_path)]
    assert generated_cfg["reward_funcs"] == ["gnprsid_top10"]
    assert generated_cfg["attn_impl"] == "flash_attn"
    assert generated_cfg["enable_thinking"] is False
    assert generated_cfg["add_non_thinking_prefix"] is True
    assert generated_cfg["loss_scale"] == "last_round+ignore_empty_think"
    assert generated_cfg["offload_model"] is True
    assert generated_cfg["offload_optimizer"] is True
    assert generated_cfg["target_modules"] == ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]
    assert manifest["result"]["ms_swift_config_path"] == str(output_dir / "ms_swift_grpo.yaml")


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
    train_path = tmp_path / "train.jsonl"
    valid_path = tmp_path / "valid.jsonl"
    init_model_path = tmp_path / "init-model"
    reward_path = tmp_path / "reward.py"

    train_path.write_text('{"messages":[]}\n', encoding="utf-8")
    valid_path.write_text('{"messages":[]}\n', encoding="utf-8")
    init_model_path.mkdir()
    reward_path.write_text("def compute_score(*args, **kwargs):\n    return 0.0\n", encoding="utf-8")

    config_path = tmp_path / "grpo.yaml"
    dump_yaml(
        config_path,
        {
            "stage": "grpo",
            "backend": "ms-swift",
            "dataset": "NYC",
            "model_profile": "qwen3-8b-instruct",
            "train_path": str(train_path),
            "valid_path": str(valid_path),
            "init_model_path": str(init_model_path),
            "reward_function_path": str(reward_path),
            "output_dir": str(output_dir),
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "num_generations": 8,
            "n_gpus_per_node": 8,
            "nnodes": 1,
        },
    )

    commands = []

    def fake_which(name):
        mapping = {
            "swift": "/usr/bin/swift",
            "ray": "/usr/bin/ray",
            "pkill": "/usr/bin/pkill",
        }
        return mapping.get(name)

    def fake_run(command, check, env=None):
        commands.append((command, check, env))
        if command and command[0] == "/usr/bin/swift":
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
