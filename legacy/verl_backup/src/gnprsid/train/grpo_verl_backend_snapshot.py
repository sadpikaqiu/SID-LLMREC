from __future__ import annotations

"""Snapshot of the archived verl GRPO backend kept for reference only."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from gnprsid.common.io import ensure_dir
from gnprsid.common.profiles import resolve_project_path
from gnprsid.grpo.reward_trace import TRACE_DIR_ENV, TRACE_GROUP_SIZE_ENV


def _cleanup_grpo_runtime_processes() -> list[list[str]]:
    cleanup_commands: list[list[str]] = []
    ray_cli = shutil.which("ray")
    if ray_cli:
        cleanup_commands.append([ray_cli, "stop", "--force"])

    pkill_cli = shutil.which("pkill")
    if pkill_cli:
        cleanup_commands.extend(
            [
                [pkill_cli, "-f", "VLLM::EngineCore"],
                [pkill_cli, "-f", "VLLM::Worker_TP"],
            ]
        )

    attempted: list[list[str]] = []
    for cleanup_command in cleanup_commands:
        attempted.append(cleanup_command)
        try:
            subprocess.run(cleanup_command, check=False)
        except OSError:
            pass
    return attempted


class GRPOVerlBackendSnapshot:
    backend_name = "verl"

    def build_command(self, cfg: dict[str, Any], output_dir: str | Path) -> tuple[list[str], dict[str, str]]:
        output_dir = ensure_dir(output_dir)
        train_path = resolve_project_path(cfg["train_path"])
        valid_path = resolve_project_path(cfg["valid_path"])
        init_model_path = resolve_project_path(cfg["init_model_path"])
        reward_path = resolve_project_path(cfg["reward_function_path"])
        target_modules = cfg.get("lora", {}).get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"],
        )
        target_modules_expr = "[" + ",".join(str(module) for module in target_modules) + "]"
        reward_trace_dir = ensure_dir(Path(output_dir) / "reward_traces")
        train_batch_size = int(cfg.get("train_batch_size", 64))
        rollout_n = int(cfg.get("rollout_n", 8))

        command = [
            sys.executable,
            "-m",
            "verl.trainer.main_ppo",
            "algorithm.adv_estimator=grpo",
            f"data.train_files={train_path}",
            f"data.val_files={valid_path}",
            f"data.train_batch_size={train_batch_size}",
            f"actor_rollout_ref.model.path={init_model_path}",
            f"actor_rollout_ref.model.lora_rank={int(cfg.get('lora', {}).get('r', 16))}",
            f"actor_rollout_ref.model.lora_alpha={int(cfg.get('lora', {}).get('alpha', 32))}",
            f"actor_rollout_ref.model.target_modules={target_modules_expr}",
            f"+actor_rollout_ref.model.override_config.attn_implementation={cfg.get('attn_implementation', 'flash_attention_2')}",
            f"actor_rollout_ref.actor.optim.lr={float(cfg.get('learning_rate', 1e-6))}",
            f"actor_rollout_ref.rollout.n={rollout_n}",
            f"reward.custom_reward_function.path={reward_path}",
            f"reward.custom_reward_function.name={cfg.get('reward_function_name', 'compute_score')}",
            f"trainer.default_local_dir={output_dir}",
        ]
        env = os.environ.copy()
        env[TRACE_DIR_ENV] = str(reward_trace_dir)
        env[TRACE_GROUP_SIZE_ENV] = str(train_batch_size * rollout_n)
        return command, env
