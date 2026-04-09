from __future__ import annotations

import os
import shutil
import subprocess
import sys
from importlib.util import find_spec
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Type

from gnprsid.common.config import dump_yaml, load_yaml
from gnprsid.common.io import copy_file, ensure_dir, write_json
from gnprsid.common.logging import get_logger
from gnprsid.common.paths import project_root
from gnprsid.common.profiles import load_model_profile, resolve_model_profile_path, resolve_project_path
from gnprsid.grpo.reward_trace import TRACE_DIR_ENV, TRACE_GROUP_SIZE_ENV


logger = get_logger(__name__)


@dataclass
class TrainContext:
    config_path: Path
    stage_config: dict[str, Any]
    model_profile_path: Path
    model_profile: dict[str, Any]
    stage: str
    backend: str
    output_dir: Path


class TrainingBackend(ABC):
    stage: str
    backend_name: str

    def prepare(self, context: TrainContext) -> dict[str, Any]:
        ensure_dir(context.output_dir)
        prepared = {
            "resolved_stage_config": context.stage_config,
            "resolved_model_profile": context.model_profile,
        }
        write_json(context.output_dir / "prepared_config.json", prepared)
        return prepared

    @abstractmethod
    def run(self, context: TrainContext, prepared: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def collect_artifacts(
        self,
        context: TrainContext,
        prepared: dict[str, Any],
        run_result: dict[str, Any],
    ) -> dict[str, Any]:
        manifest = {
            "stage": context.stage,
            "backend": context.backend,
            "config_path": str(context.config_path),
            "model_profile_path": str(context.model_profile_path),
            "output_dir": str(context.output_dir),
            "prepared": prepared,
            "result": run_result,
        }
        write_json(context.output_dir / "run_manifest.json", manifest)
        return manifest


BACKEND_REGISTRY: dict[tuple[str, str], Type[TrainingBackend]] = {}


def register_backend(cls: Type[TrainingBackend]) -> Type[TrainingBackend]:
    BACKEND_REGISTRY[(cls.stage, cls.backend_name)] = cls
    return cls


def _build_context(config_path: str | Path, stage_override: str | None = None) -> TrainContext:
    config_path = resolve_project_path(config_path)
    stage_config = load_yaml(config_path)
    stage = stage_override or stage_config["stage"]
    backend = stage_config["backend"]
    model_profile_path = resolve_model_profile_path(stage_config["model_profile"])
    model_profile = load_model_profile(stage_config["model_profile"])
    output_dir = resolve_project_path(stage_config["output_dir"])
    return TrainContext(
        config_path=config_path,
        stage_config=stage_config,
        model_profile_path=model_profile_path,
        model_profile=model_profile,
        stage=stage,
        backend=backend,
        output_dir=output_dir,
    )


def _resolve_model_source(source: str | Path) -> str:
    source_path = Path(str(source))
    if source_path.is_absolute():
        return str(source_path)
    project_candidate = resolve_project_path(source_path)
    if project_candidate.exists():
        return str(project_candidate)
    return str(source)


@register_backend
class AlignmentTRLBackend(TrainingBackend):
    stage = "alignment"
    backend_name = "trl"

    def run(self, context: TrainContext, prepared: dict[str, Any]) -> dict[str, Any]:
        try:
            import torch
            from datasets import load_dataset
            from peft import LoraConfig
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from trl import SFTConfig, SFTTrainer
        except ImportError as error:
            raise ImportError(
                "Alignment training requires torch, datasets, transformers, peft, and trl."
            ) from error

        from gnprsid.common.runtime import resolve_torch_dtype, set_seed

        cfg = context.stage_config
        model_cfg = context.model_profile
        set_seed(int(cfg.get("seed", 42)))
        if cfg.get("wandb_project"):
            os.environ["WANDB_PROJECT"] = str(cfg["wandb_project"])

        train_path = str(resolve_project_path(cfg["train_path"]))
        valid_path = str(resolve_project_path(cfg["valid_path"]))
        train_data = load_dataset("json", data_files=train_path)["train"]
        eval_data = load_dataset("json", data_files=valid_path)["train"]

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = resolve_torch_dtype(torch, str(cfg.get("dtype", model_cfg.get("dtype", "auto"))), device_type)
        base_model_source = _resolve_model_source(cfg.get("base_model_override", model_cfg["base_model"]))
        tokenizer_source = _resolve_model_source(cfg.get("tokenizer_override", model_cfg.get("tokenizer_name", base_model_source)))
        model = AutoModelForCausalLM.from_pretrained(
            base_model_source,
            trust_remote_code=True,
            dtype=dtype,
        )
        model.gradient_checkpointing_enable()

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        lora_cfg = LoraConfig(
            r=int(cfg["lora"]["r"]),
            lora_alpha=int(cfg["lora"]["alpha"]),
            target_modules=list(cfg["lora"]["target_modules"]),
            lora_dropout=float(cfg["lora"]["dropout"]),
            task_type="CAUSAL_LM",
            bias="none",
        )

        eos_token = tokenizer.eos_token or ""

        def to_prompt_completion(batch):
            prompts = []
            completions = []
            for instruction, input_text, output_text in zip(batch["instruction"], batch["input"], batch["output"]):
                prompts.append(
                    "### Instruction:\n"
                    f"{str(instruction).strip()}\n\n"
                    "### Input:\n"
                    f"{str(input_text).strip()}\n\n"
                    "### Response:\n"
                )
                completions.append(f"{str(output_text).strip()}{eos_token}")
            return {"prompt": prompts, "completion": completions}

        train_data = train_data.map(
            to_prompt_completion,
            batched=True,
            remove_columns=train_data.column_names,
        )
        eval_data = eval_data.map(
            to_prompt_completion,
            batched=True,
            remove_columns=eval_data.column_names,
        )

        bf16 = str(cfg.get("dtype", model_cfg.get("dtype", "auto"))).lower() == "bfloat16"
        training_args = SFTConfig(
            output_dir=str(context.output_dir),
            per_device_train_batch_size=int(cfg["batch_size"]),
            per_device_eval_batch_size=int(cfg["batch_size"]),
            gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
            num_train_epochs=float(cfg["num_train_epochs"]),
            learning_rate=float(cfg["learning_rate"]),
            eval_strategy="steps",
            eval_steps=int(cfg.get("eval_steps", 50)),
            save_steps=int(cfg.get("save_steps", 100)),
            logging_steps=int(cfg.get("logging_steps", 10)),
            warmup_steps=int(cfg.get("warmup_steps", 100)),
            bf16=bf16,
            report_to="wandb" if cfg.get("wandb_project") else "none",
            run_name=str(cfg.get("wandb_run_name", "alignment")),
            max_length=int(cfg["cutoff_len"]),
            completion_only_loss=True,
        )
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=eval_data,
            args=training_args,
            processing_class=tokenizer,
            peft_config=lora_cfg,
        )
        trainer.train()

        final_dir = ensure_dir(context.output_dir / "final")
        trainer.save_model(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        return {
            "base_model_source": base_model_source,
            "train_path": train_path,
            "valid_path": valid_path,
            "final_checkpoint_dir": str(final_dir),
        }


@register_backend
class SFTLLaMAFactoryBackend(TrainingBackend):
    stage = "sft"
    backend_name = "llamafactory"

    def run(self, context: TrainContext, prepared: dict[str, Any]) -> dict[str, Any]:
        cli_path = shutil.which("llamafactory-cli")
        if not cli_path:
            raise FileNotFoundError(
                "Could not find 'llamafactory-cli'. Install LLaMA-Factory in the Linux training environment first."
            )

        cfg = context.stage_config
        model_cfg = context.model_profile
        output_dir = ensure_dir(context.output_dir)
        dataset_dir = ensure_dir(output_dir / "llamafactory_dataset")
        train_src = resolve_project_path(cfg["train_path"])
        valid_src = resolve_project_path(cfg["valid_path"])
        train_dst = copy_file(train_src, dataset_dir / "train.jsonl")
        valid_dst = copy_file(valid_src, dataset_dir / "valid.jsonl")

        dataset_info = {
            "gnprsid_train": {
                "file_name": train_dst.name,
                "formatting": "alpaca",
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output",
                },
            },
            "gnprsid_valid": {
                "file_name": valid_dst.name,
                "formatting": "alpaca",
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "response": "output",
                },
            },
        }
        write_json(dataset_dir / "dataset_info.json", dataset_info)

        run_output_dir = output_dir / "llamafactory_output"
        bf16 = str(model_cfg.get("dtype", "auto")).lower() == "bfloat16"
        train_yaml = {
            "model_name_or_path": cfg.get("base_model_override", model_cfg["base_model"]),
            "stage": "sft",
            "do_train": True,
            "finetuning_type": "lora",
            "lora_target": ",".join(cfg["lora_target"]),
            "template": cfg.get("template", "qwen"),
            "dataset_dir": str(dataset_dir),
            "dataset": "gnprsid_train",
            "eval_dataset": "gnprsid_valid",
            "cutoff_len": int(cfg["cutoff_len"]),
            "output_dir": str(run_output_dir),
            "per_device_train_batch_size": int(cfg["per_device_train_batch_size"]),
            "gradient_accumulation_steps": int(cfg["gradient_accumulation_steps"]),
            "learning_rate": float(cfg["learning_rate"]),
            "num_train_epochs": float(cfg["num_train_epochs"]),
            "bf16": bf16,
            "logging_steps": int(cfg.get("logging_steps", 10)),
            "save_steps": int(cfg.get("save_steps", 100)),
            "eval_steps": int(cfg.get("eval_steps", 100)),
            "save_strategy": "steps",
            "eval_strategy": "steps",
            "overwrite_output_dir": True,
            "report_to": cfg.get("report_to", "none"),
        }
        train_yaml_path = output_dir / "llamafactory_train.yaml"
        dump_yaml(train_yaml_path, train_yaml)

        command = [cli_path, "train", str(train_yaml_path)]
        logger.info("Running LLaMA-Factory command: %s", " ".join(command))
        subprocess.run(command, check=True)
        return {
            "dataset_dir": str(dataset_dir),
            "train_yaml_path": str(train_yaml_path),
            "run_output_dir": str(run_output_dir),
        }


@register_backend
class GRPOVerlBackend(TrainingBackend):
    stage = "grpo"
    backend_name = "verl"

    def run(self, context: TrainContext, prepared: dict[str, Any]) -> dict[str, Any]:
        if find_spec("verl") is None:
            raise ImportError("GRPO training requires the official 'verl' package to be installed.")

        cfg = context.stage_config
        output_dir = ensure_dir(context.output_dir)

        train_path = resolve_project_path(cfg["train_path"])
        valid_path = resolve_project_path(cfg["valid_path"])
        init_model_path = resolve_project_path(cfg["init_model_path"])
        reward_path = resolve_project_path(cfg["reward_function_path"])
        if not train_path.exists():
            raise FileNotFoundError(f"Missing GRPO train parquet: {train_path}")
        if not valid_path.exists():
            raise FileNotFoundError(f"Missing GRPO valid parquet: {valid_path}")
        if not init_model_path.exists():
            raise FileNotFoundError(f"Missing GRPO init model path: {init_model_path}")
        if not reward_path.exists():
            raise FileNotFoundError(f"Missing GRPO reward function file: {reward_path}")

        target_modules = cfg.get("lora", {}).get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"],
        )
        target_modules_expr = "[" + ",".join(str(module) for module in target_modules) + "]"
        attn_implementation = str(cfg.get("attn_implementation", "flash_attention_2"))
        update_weights_bucket_megabytes = int(cfg.get("update_weights_bucket_megabytes", 4096))
        train_batch_size = int(cfg.get("train_batch_size", 64))
        rollout_n = int(cfg.get("rollout_n", 8))
        reward_trace_dir = ensure_dir(output_dir / "reward_traces")

        command = [
            sys.executable,
            "-m",
            "verl.trainer.main_ppo",
            "algorithm.adv_estimator=grpo",
            f"data.train_files={train_path}",
            f"data.val_files={valid_path}",
            f"data.train_batch_size={train_batch_size}",
            f"data.max_prompt_length={int(cfg.get('max_prompt_length', 2048))}",
            f"data.max_response_length={int(cfg.get('max_response_length', 256))}",
            f"data.filter_overlong_prompts={str(bool(cfg.get('filter_overlong_prompts', True))).lower()}",
            f"data.truncation={cfg.get('truncation', 'error')}",
            f"actor_rollout_ref.model.path={init_model_path}",
            "actor_rollout_ref.model.enable_gradient_checkpointing=true",
            "actor_rollout_ref.model.use_remove_padding=true",
            f"actor_rollout_ref.model.lora_rank={int(cfg.get('lora', {}).get('r', 16))}",
            f"actor_rollout_ref.model.lora_alpha={int(cfg.get('lora', {}).get('alpha', 32))}",
            f"actor_rollout_ref.model.target_modules={target_modules_expr}",
            f"+actor_rollout_ref.model.override_config._attn_implementation={attn_implementation}",
            f"actor_rollout_ref.actor.optim.lr={float(cfg.get('learning_rate', 1e-6))}",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={int(cfg.get('ppo_mini_batch_size', train_batch_size))}",
            f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={int(cfg.get('ppo_micro_batch_size_per_gpu', 4))}",
            "actor_rollout_ref.actor.use_kl_loss=true",
            f"actor_rollout_ref.actor.kl_loss_coef={float(cfg.get('kl_loss_coef', 0.001))}",
            f"actor_rollout_ref.actor.kl_loss_type={cfg.get('kl_loss_type', 'low_var_kl')}",
            "actor_rollout_ref.actor.entropy_coeff=0",
            "actor_rollout_ref.actor.fsdp_config.param_offload=false",
            "actor_rollout_ref.actor.fsdp_config.optimizer_offload=false",
            f"actor_rollout_ref.rollout.name={cfg.get('rollout_name', 'vllm')}",
            f"actor_rollout_ref.rollout.n={rollout_n}",
            f"actor_rollout_ref.rollout.gpu_memory_utilization={float(cfg.get('gpu_memory_utilization', 0.8))}",
            f"actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes={update_weights_bucket_megabytes}",
            "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
            f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={int(cfg.get('log_prob_micro_batch_size_per_gpu', 4))}",
            f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={int(cfg.get('ref_log_prob_micro_batch_size_per_gpu', 4))}",
            "actor_rollout_ref.ref.fsdp_config.param_offload=true",
            "algorithm.use_kl_in_reward=false",
            "trainer.critic_warmup=0",
            f"trainer.logger={cfg.get('logger', '[console]')}",
            f"reward.custom_reward_function.path={reward_path}",
            f"reward.custom_reward_function.name={cfg.get('reward_function_name', 'compute_score')}",
            f"trainer.project_name={cfg.get('project_name', 'gnprsid-grpo')}",
            f"trainer.experiment_name={cfg.get('experiment_name', 'nyc-sid-current')}",
            f"trainer.n_gpus_per_node={int(cfg.get('n_gpus_per_node', 1))}",
            f"trainer.nnodes={int(cfg.get('nnodes', 1))}",
            f"trainer.save_freq={int(cfg.get('save_freq', 100))}",
            f"trainer.test_freq={int(cfg.get('test_freq', 100))}",
            f"trainer.total_epochs={int(cfg.get('total_epochs', 3))}",
            f"trainer.default_local_dir={output_dir}",
            "trainer.resume_mode=disable",
        ]

        logger.info("Running verl command: %s", " ".join(str(token) for token in command))
        env = os.environ.copy()
        env[TRACE_DIR_ENV] = str(reward_trace_dir)
        env[TRACE_GROUP_SIZE_ENV] = str(train_batch_size * rollout_n)
        subprocess.run(command, check=True, env=env)
        return {
            "train_path": str(train_path),
            "valid_path": str(valid_path),
            "init_model_path": str(init_model_path),
            "reward_function_path": str(reward_path),
            "reward_trace_dir": str(reward_trace_dir),
            "command": [str(token) for token in command],
            "output_dir": str(output_dir),
        }


def run_training_stage(config_path: str | Path, stage_override: str | None = None) -> dict[str, Any]:
    context = _build_context(config_path, stage_override=stage_override)
    backend_cls = BACKEND_REGISTRY.get((context.stage, context.backend))
    if backend_cls is None:
        raise KeyError(f"No backend registered for stage={context.stage} backend={context.backend}")

    backend = backend_cls()
    prepared = backend.prepare(context)
    result = backend.run(context, prepared)
    manifest = backend.collect_artifacts(context, prepared, result)
    logger.info("Finished training stage=%s backend=%s", context.stage, context.backend)
    return manifest
