import pytest

from gnprsid.common.config import dump_yaml
from gnprsid.train.base import run_training_stage


def test_run_training_stage_warmup_builds_llamafactory_command(monkeypatch, tmp_path):
    output_dir = tmp_path / "warmup-output"
    train_path = tmp_path / "train.jsonl"
    valid_path = tmp_path / "valid.jsonl"
    base_model_path = tmp_path / "alignment-merged"
    train_path.write_text('{"instruction":"i","input":"q","output":"o"}\n', encoding="utf-8")
    valid_path.write_text('{"instruction":"i","input":"q","output":"o"}\n', encoding="utf-8")
    base_model_path.mkdir()

    config_path = tmp_path / "warmup.yaml"
    dump_yaml(
        config_path,
        {
            "stage": "warmup",
            "backend": "llamafactory",
            "dataset": "NYC",
            "model_profile": "qwen2.5-7b-instruct",
            "base_model_override": str(base_model_path.relative_to(tmp_path)),
            "train_path": str(train_path),
            "valid_path": str(valid_path),
            "output_dir": str(output_dir),
            "cutoff_len": 3072,
            "template": "qwen",
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 2,
            "learning_rate": 2.0e-5,
            "num_train_epochs": 3,
            "lora_target": ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"],
        },
    )

    captured = {}
    monkeypatch.setattr("gnprsid.train.base.shutil.which", lambda name: "/usr/bin/llamafactory-cli")
    monkeypatch.setattr("gnprsid.train.base.resolve_project_path", lambda path: tmp_path / path)

    def fake_run(command, check):
        captured["command"] = command
        captured["check"] = check

    monkeypatch.setattr("gnprsid.train.base.subprocess.run", fake_run)

    manifest = run_training_stage(config_path, stage_override="warmup")

    command = captured["command"]
    assert command == ["/usr/bin/llamafactory-cli", "train", str(output_dir / "llamafactory_train.yaml")]
    assert captured["check"] is True
    assert manifest["stage"] == "warmup"
    assert manifest["backend"] == "llamafactory"
    assert manifest["result"]["base_model_source"] == str(base_model_path)
    assert manifest["result"]["dataset_dir"] == str(output_dir / "llamafactory_dataset")


def test_run_training_stage_warmup_rejects_missing_local_base_model(monkeypatch, tmp_path):
    output_dir = tmp_path / "warmup-output"
    train_path = tmp_path / "train.jsonl"
    valid_path = tmp_path / "valid.jsonl"
    train_path.write_text('{"instruction":"i","input":"q","output":"o"}\n', encoding="utf-8")
    valid_path.write_text('{"instruction":"i","input":"q","output":"o"}\n', encoding="utf-8")

    config_path = tmp_path / "warmup.yaml"
    dump_yaml(
        config_path,
        {
            "stage": "warmup",
            "backend": "llamafactory",
            "dataset": "NYC",
            "model_profile": "qwen2.5-7b-instruct",
            "base_model_override": "checkpoints/NYC/alignment/qwen25_7b_phase_b2/merged",
            "train_path": str(train_path),
            "valid_path": str(valid_path),
            "output_dir": str(output_dir),
            "cutoff_len": 3072,
            "template": "qwen",
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 2,
            "learning_rate": 2.0e-5,
            "num_train_epochs": 3,
            "lora_target": ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"],
        },
    )

    monkeypatch.setattr("gnprsid.train.base.shutil.which", lambda name: "/usr/bin/llamafactory-cli")
    monkeypatch.setattr("gnprsid.train.base.resolve_project_path", lambda path: tmp_path / path)

    with pytest.raises(FileNotFoundError, match="Missing local model path"):
        run_training_stage(config_path, stage_override="warmup")
