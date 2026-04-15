from gnprsid.common.config import dump_yaml
from gnprsid.train.base import TORCHRUN_SKIP_MANIFEST_ENV, run_training_stage


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
