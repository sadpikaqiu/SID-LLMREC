from gnprsid.common.config import dump_yaml
from gnprsid.train.base import (
    TORCHRUN_SKIP_MANIFEST_ENV,
    _alignment_runtime_options,
    run_training_stage,
)


def test_alignment_runtime_options_keep_checkpointing_for_single_process():
    runtime = _alignment_runtime_options({})

    assert runtime["gradient_checkpointing"] is True
    assert runtime["gradient_checkpointing_kwargs"] == {"use_reentrant": False}
    assert runtime["ddp_find_unused_parameters"] is None


def test_alignment_runtime_options_disable_checkpointing_for_multi_process():
    runtime = _alignment_runtime_options({"num_processes": 4})

    assert runtime["gradient_checkpointing"] is False
    assert runtime["gradient_checkpointing_kwargs"] is None
    assert runtime["ddp_find_unused_parameters"] is False


def test_run_training_stage_alignment_uses_torchrun_when_num_processes_set(monkeypatch, tmp_path):
    output_dir = tmp_path / "alignment-output"
    train_path = tmp_path / "train.jsonl"
    valid_path = tmp_path / "valid.jsonl"
    train_path.write_text('{"instruction":"i","input":"q","output":"o"}\n', encoding="utf-8")
    valid_path.write_text('{"instruction":"i","input":"q","output":"o"}\n', encoding="utf-8")

    config_path = tmp_path / "alignment.yaml"
    dump_yaml(
        config_path,
        {
            "stage": "alignment",
            "backend": "trl",
            "dataset": "NYC",
            "model_profile": "qwen2.5-7b-instruct",
            "train_path": str(train_path),
            "valid_path": str(valid_path),
            "output_dir": str(output_dir),
            "batch_size": 16,
            "gradient_accumulation_steps": 2,
            "num_train_epochs": 4,
            "learning_rate": 2.0e-5,
            "cutoff_len": 512,
            "num_processes": 4,
            "lora": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.05,
                "target_modules": ["embed_tokens", "lm_head"],
            },
        },
    )

    captured = {}

    def fake_which(name):
        if name == "torchrun":
            return "/usr/bin/torchrun"
        return None

    monkeypatch.setattr("gnprsid.train.base.shutil.which", fake_which)

    def fake_run(command, check, env=None):
        captured["command"] = command
        captured["check"] = check
        captured["env"] = env

    monkeypatch.setattr("gnprsid.train.base.subprocess.run", fake_run)

    manifest = run_training_stage(config_path, stage_override="alignment")

    command = captured["command"]
    assert command[:3] == ["/usr/bin/torchrun", "--nproc_per_node=4", "--standalone"]
    assert command[3:] == [
        "-m",
        "gnprsid.cli",
        "train",
        "run",
        "--stage",
        "alignment",
        "--config",
        str(config_path),
    ]
    assert captured["env"][TORCHRUN_SKIP_MANIFEST_ENV] == "1"
    assert captured["check"] is True
    assert manifest["result"]["distributed_launch"] is True
    assert manifest["result"]["num_processes"] == 4


def test_run_training_stage_alignment_still_launches_torchrun_when_only_world_size_present(monkeypatch, tmp_path):
    output_dir = tmp_path / "alignment-output"
    train_path = tmp_path / "train.jsonl"
    valid_path = tmp_path / "valid.jsonl"
    train_path.write_text('{"instruction":"i","input":"q","output":"o"}\n', encoding="utf-8")
    valid_path.write_text('{"instruction":"i","input":"q","output":"o"}\n', encoding="utf-8")

    config_path = tmp_path / "alignment.yaml"
    dump_yaml(
        config_path,
        {
            "stage": "alignment",
            "backend": "trl",
            "dataset": "NYC",
            "model_profile": "qwen2.5-7b-instruct",
            "train_path": str(train_path),
            "valid_path": str(valid_path),
            "output_dir": str(output_dir),
            "batch_size": 16,
            "gradient_accumulation_steps": 2,
            "num_train_epochs": 4,
            "learning_rate": 2.0e-5,
            "cutoff_len": 512,
            "num_processes": 4,
            "lora": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.05,
                "target_modules": ["embed_tokens", "lm_head"],
            },
        },
    )

    captured = {}

    def fake_which(name):
        if name == "torchrun":
            return "/usr/bin/torchrun"
        return None

    monkeypatch.setattr("gnprsid.train.base.shutil.which", fake_which)
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("TORCHELASTIC_RUN_ID", raising=False)

    def fake_run(command, check, env=None):
        captured["command"] = command
        captured["check"] = check
        captured["env"] = env

    monkeypatch.setattr("gnprsid.train.base.subprocess.run", fake_run)

    manifest = run_training_stage(config_path, stage_override="alignment")

    assert captured["command"][:3] == ["/usr/bin/torchrun", "--nproc_per_node=4", "--standalone"]
    assert captured["env"][TORCHRUN_SKIP_MANIFEST_ENV] == "1"
    assert manifest["result"]["distributed_launch"] is True


def test_run_training_stage_alignment_does_not_relaunch_when_skip_manifest_env_present(monkeypatch, tmp_path):
    output_dir = tmp_path / "alignment-output"
    train_path = tmp_path / "train.jsonl"
    valid_path = tmp_path / "valid.jsonl"
    base_model_path = tmp_path / "model"
    train_path.write_text('{"instruction":"i","input":"q","output":"o"}\n', encoding="utf-8")
    valid_path.write_text('{"instruction":"i","input":"q","output":"o"}\n', encoding="utf-8")
    base_model_path.mkdir()

    config_path = tmp_path / "alignment.yaml"
    dump_yaml(
        config_path,
        {
            "stage": "alignment",
            "backend": "trl",
            "dataset": "NYC",
            "model_profile": "qwen2.5-7b-instruct",
            "base_model_override": str(base_model_path),
            "train_path": str(train_path),
            "valid_path": str(valid_path),
            "output_dir": str(output_dir),
            "batch_size": 16,
            "gradient_accumulation_steps": 2,
            "num_train_epochs": 4,
            "learning_rate": 2.0e-5,
            "cutoff_len": 512,
            "num_processes": 4,
            "lora": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.05,
                "target_modules": ["embed_tokens", "lm_head"],
            },
        },
    )

    monkeypatch.setenv(TORCHRUN_SKIP_MANIFEST_ENV, "1")

    try:
        run_training_stage(config_path, stage_override="alignment")
    except ImportError as error:
        assert "Alignment training requires" in str(error)
    else:
        raise AssertionError("Expected alignment backend to continue locally instead of relaunching torchrun.")
