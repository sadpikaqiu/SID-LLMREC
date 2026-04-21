from __future__ import annotations

import os
import shutil
import subprocess
import sys
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
from gnprsid.common.tokenizer import build_tokenizer_load_kwargs


logger = get_logger(__name__)

TORCHRUN_SKIP_MANIFEST_ENV = "GNPRSID_SKIP_MANIFEST_WRITE"
MS_SWIFT_REWARD_PATH_ENV = "GNPRSID_GRPO_REWARD_PATH"
MS_SWIFT_REWARD_NAME_ENV = "GNPRSID_GRPO_REWARD_NAME"


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


def _resolve_training_model_source(source: str | Path) -> str:
    source_text = str(source)
    source_path = Path(source_text)
    if source_path.is_absolute():
        if not source_path.exists():
            raise FileNotFoundError(f"Missing local model path: {source_path}")
        return str(source_path)

    project_candidate = resolve_project_path(source_path)
    if project_candidate.exists():
        return str(project_candidate)

    if (
        source_text.startswith(("./", ".\\", "../", "..\\"))
        or "\\" in source_text
        or source_text.count("/") > 1
    ):
        raise FileNotFoundError(f"Missing local model path: {project_candidate}")

    return str(source)


def _resolve_chat_template_kwargs(model_profile: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(model_profile.get("chat_template_kwargs", {}))
    if "enable_thinking" in model_profile and "enable_thinking" not in kwargs:
        kwargs["enable_thinking"] = bool(model_profile["enable_thinking"])
    return kwargs


def _requested_num_processes(cfg: dict[str, Any]) -> int:
    return int(cfg.get("num_processes", 1))


def _is_torchrun_worker() -> bool:
    return (
        os.environ.get(TORCHRUN_SKIP_MANIFEST_ENV) == "1"
        or "LOCAL_RANK" in os.environ
        or "TORCHELASTIC_RUN_ID" in os.environ
    )


def _build_torchrun_prefix(cfg: dict[str, Any]) -> list[str]:
    torchrun_path = shutil.which("torchrun")
    if not torchrun_path:
        raise FileNotFoundError("Could not find 'torchrun'. Install PyTorch distributed tooling first.")

    num_processes = _requested_num_processes(cfg)
    nnodes = int(cfg.get("nnodes", 1))
    command = [torchrun_path, f"--nproc_per_node={num_processes}"]
    if nnodes == 1:
        command.append("--standalone")
    else:
        command.extend(
            [
                f"--nnodes={nnodes}",
                f"--node_rank={int(cfg.get('node_rank', 0))}",
                f"--master_addr={cfg.get('master_addr', '127.0.0.1')}",
                f"--master_port={int(cfg.get('master_port', 29500))}",
            ]
        )
    return command


def _launch_stage_via_torchrun(context: TrainContext) -> dict[str, Any]:
    command = _build_torchrun_prefix(context.stage_config)
    command.extend(
        [
            "-m",
            "gnprsid.cli",
            "train",
            "run",
            "--stage",
            context.stage,
            "--config",
            str(context.config_path),
        ]
    )
    env = os.environ.copy()
    env[TORCHRUN_SKIP_MANIFEST_ENV] = "1"
    logger.info("Running distributed %s command: %s", context.stage, " ".join(command))
    subprocess.run(command, check=True, env=env)
    return {
        "distributed_launch": True,
        "num_processes": _requested_num_processes(context.stage_config),
        "command": command,
    }


def _alignment_runtime_options(cfg: dict[str, Any]) -> dict[str, Any]:
    num_processes = _requested_num_processes(cfg)
    gradient_checkpointing = bool(cfg.get("gradient_checkpointing", num_processes <= 1))
    gradient_checkpointing_kwargs = cfg.get("gradient_checkpointing_kwargs")
    if gradient_checkpointing and gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {"use_reentrant": False}

    ddp_find_unused_parameters = cfg.get("ddp_find_unused_parameters")
    if ddp_find_unused_parameters is None and num_processes > 1:
        ddp_find_unused_parameters = False

    return {
        "gradient_checkpointing": gradient_checkpointing,
        "gradient_checkpointing_kwargs": gradient_checkpointing_kwargs,
        "ddp_find_unused_parameters": ddp_find_unused_parameters,
    }


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
        except OSError as error:
            logger.warning("Failed to execute cleanup command %s: %s", " ".join(cleanup_command), error)
    return attempted


def _prepend_pythonpath(env: dict[str, str], extra_path: Path) -> None:
    existing = env.get("PYTHONPATH", "")
    extra = str(extra_path)
    env["PYTHONPATH"] = extra if not existing else extra + os.pathsep + existing


def _normalize_ms_swift_attn_impl(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"flash_attention_2", "flash-attention-2", "flash_attention2"}:
        return "flash_attn"
    return normalized


def _validate_ms_swift_grpo_shape(cfg: dict[str, Any]) -> None:
    num_processes = int(cfg.get("n_gpus_per_node", 1)) * int(cfg.get("nnodes", 1))
    per_device_train_batch_size = int(cfg.get("per_device_train_batch_size", 1))
    per_device_eval_batch_size = int(cfg.get("per_device_eval_batch_size", per_device_train_batch_size))
    gradient_accumulation_steps = int(cfg.get("gradient_accumulation_steps", 1))
    num_generations = int(cfg.get("num_generations", 8))

    global_train_batch = per_device_train_batch_size * gradient_accumulation_steps * num_processes
    global_eval_batch = per_device_eval_batch_size * num_processes
    if global_train_batch % num_generations != 0:
        raise ValueError(
            "ms-swift GRPO requires global train batch size divisible by num_generations: "
            f"{global_train_batch=} {num_generations=}"
        )
    if global_eval_batch % num_generations != 0:
        raise ValueError(
            "ms-swift GRPO requires global eval batch size divisible by num_generations: "
            f"{global_eval_batch=} {num_generations=}"
        )


@register_backend
class AlignmentTRLBackend(TrainingBackend):
    stage = "alignment"
    backend_name = "trl"

    def run(self, context: TrainContext, prepared: dict[str, Any]) -> dict[str, Any]:
        if _requested_num_processes(context.stage_config) > 1 and not _is_torchrun_worker():
            return _launch_stage_via_torchrun(context)

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
        runtime_options = _alignment_runtime_options(cfg)
        set_seed(int(cfg.get("seed", 42)))
        if cfg.get("wandb_project"):
            os.environ["WANDB_PROJECT"] = str(cfg["wandb_project"])

        train_path = str(resolve_project_path(cfg["train_path"]))
        valid_path = str(resolve_project_path(cfg["valid_path"]))
        train_data = load_dataset("json", data_files=train_path)["train"]
        eval_data = load_dataset("json", data_files=valid_path)["train"]

        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = resolve_torch_dtype(torch, str(cfg.get("dtype", model_cfg.get("dtype", "auto"))), device_type)
        base_model_source = _resolve_training_model_source(cfg.get("base_model_override", model_cfg["base_model"]))
        tokenizer_source = _resolve_model_source(cfg.get("tokenizer_override", model_cfg.get("tokenizer_name", base_model_source)))
        model = AutoModelForCausalLM.from_pretrained(
            base_model_source,
            trust_remote_code=True,
            dtype=dtype,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            **build_tokenizer_load_kwargs(),
        )
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
        training_args_kwargs = {
            "output_dir": str(context.output_dir),
            "per_device_train_batch_size": int(cfg["batch_size"]),
            "per_device_eval_batch_size": int(cfg["batch_size"]),
            "gradient_accumulation_steps": int(cfg["gradient_accumulation_steps"]),
            "num_train_epochs": float(cfg["num_train_epochs"]),
            "learning_rate": float(cfg["learning_rate"]),
            "eval_strategy": "steps",
            "eval_steps": int(cfg.get("eval_steps", 50)),
            "save_steps": int(cfg.get("save_steps", 100)),
            "logging_steps": int(cfg.get("logging_steps", 10)),
            "warmup_steps": int(cfg.get("warmup_steps", 100)),
            "bf16": bf16,
            "report_to": "wandb" if cfg.get("wandb_project") else "none",
            "run_name": str(cfg.get("wandb_run_name", "alignment")),
            "max_length": int(cfg["cutoff_len"]),
            "completion_only_loss": True,
            "gradient_checkpointing": runtime_options["gradient_checkpointing"],
        }
        if runtime_options["gradient_checkpointing_kwargs"] is not None:
            training_args_kwargs["gradient_checkpointing_kwargs"] = runtime_options["gradient_checkpointing_kwargs"]
        if runtime_options["ddp_find_unused_parameters"] is not None:
            training_args_kwargs["ddp_find_unused_parameters"] = runtime_options["ddp_find_unused_parameters"]

        training_args = SFTConfig(**training_args_kwargs)
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
        return _run_llamafactory_backend(context)


@register_backend
class WarmupLLaMAFactoryBackend(TrainingBackend):
    stage = "warmup"
    backend_name = "llamafactory"

    def run(self, context: TrainContext, prepared: dict[str, Any]) -> dict[str, Any]:
        return _run_llamafactory_backend(context)


def _run_llamafactory_backend(context: TrainContext) -> dict[str, Any]:
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
        base_model_source = _resolve_training_model_source(cfg.get("base_model_override", model_cfg["base_model"]))
        train_yaml = {
            "model_name_or_path": base_model_source,
            "stage": "sft",
            "do_train": True,
            "finetuning_type": "lora",
            "use_fast_tokenizer": bool(cfg.get("use_fast_tokenizer", False)),
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
            "base_model_source": base_model_source,
            "dataset_dir": str(dataset_dir),
            "train_yaml_path": str(train_yaml_path),
            "run_output_dir": str(run_output_dir),
            "distributed_launch": False,
            "num_processes": _requested_num_processes(cfg),
            "command": command,
        }


@register_backend
class GRPOMsSwiftBackend(TrainingBackend):
    stage = "grpo"
    backend_name = "ms-swift"

    def run(self, context: TrainContext, prepared: dict[str, Any]) -> dict[str, Any]:
        cfg = context.stage_config
        output_dir = ensure_dir(context.output_dir)
        runtime_dir = ensure_dir(output_dir / ".gnprsid")
        swift_cli = shutil.which("swift")
        if not swift_cli:
            raise FileNotFoundError("Could not find 'swift'. Install ms-swift in the Linux training environment first.")

        train_path = resolve_project_path(cfg["train_path"])
        valid_path = resolve_project_path(cfg["valid_path"])
        init_model_path = resolve_project_path(cfg["init_model_path"])
        reward_path = resolve_project_path(cfg["reward_function_path"])
        plugin_path = resolve_project_path("src/gnprsid/grpo/ms_swift_plugin.py")
        if not train_path.exists():
            raise FileNotFoundError(f"Missing GRPO train dataset: {train_path}")
        if not valid_path.exists():
            raise FileNotFoundError(f"Missing GRPO valid dataset: {valid_path}")
        if not init_model_path.exists():
            raise FileNotFoundError(f"Missing GRPO init model path: {init_model_path}")
        if not reward_path.exists():
            raise FileNotFoundError(f"Missing GRPO reward function file: {reward_path}")
        if not plugin_path.exists():
            raise FileNotFoundError(f"Missing ms-swift reward plugin file: {plugin_path}")

        _validate_ms_swift_grpo_shape(cfg)

        model_cfg = prepared["resolved_model_profile"]
        target_modules = list(
            cfg.get("lora", {}).get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"],
            )
        )
        enable_thinking = _resolve_chat_template_kwargs(model_cfg).get("enable_thinking")
        offload_model = bool(cfg.get("offload_model", cfg.get("actor_param_offload", False)))
        offload_optimizer = bool(cfg.get("offload_optimizer", cfg.get("actor_optimizer_offload", False)))
        reward_trace_dir = ensure_dir(output_dir / "reward_traces")
        ms_swift_config = {
            "rlhf_type": "grpo",
            "model": str(init_model_path),
            "model_type": model_cfg.get("ms_swift_model_type"),
            "template": model_cfg.get("ms_swift_template"),
            "tuner_type": "lora",
            "dataset": [str(train_path)],
            "val_dataset": [str(valid_path)],
            "split_dataset_ratio": 0,
            "output_dir": str(output_dir),
            "add_version": False,
            "external_plugins": [str(plugin_path)],
            "reward_funcs": ["gnprsid_top10"],
            "torch_dtype": str(model_cfg.get("dtype", "bfloat16")),
            "attn_impl": _normalize_ms_swift_attn_impl(str(cfg.get("attn_impl", cfg.get("attn_implementation", "flash_attn")))),
            "max_length": int(cfg.get("max_length", cfg.get("max_prompt_length", 2048))),
            "max_completion_length": int(cfg.get("max_completion_length", cfg.get("max_response_length", 256))),
            "num_generations": int(cfg.get("num_generations", cfg.get("rollout_n", 8))),
            "per_device_train_batch_size": int(cfg.get("per_device_train_batch_size", 1)),
            "per_device_eval_batch_size": int(cfg.get("per_device_eval_batch_size", cfg.get("per_device_train_batch_size", 1))),
            "gradient_accumulation_steps": int(cfg.get("gradient_accumulation_steps", 1)),
            "learning_rate": float(cfg.get("learning_rate", 1e-6)),
            "num_train_epochs": float(cfg.get("num_train_epochs", cfg.get("total_epochs", 3))),
            "beta": float(cfg.get("beta", cfg.get("kl_loss_coef", 0.001))),
            "temperature": float(cfg.get("temperature", 1.0)),
            "top_p": float(cfg.get("top_p", 0.85)),
            "logging_steps": int(cfg.get("logging_steps", 10)),
            "save_steps": int(cfg.get("save_steps", cfg.get("save_freq", 100))),
            "eval_steps": int(cfg.get("eval_steps", cfg.get("test_freq", 100))),
            "save_strategy": "steps",
            "eval_strategy": "steps",
            "save_total_limit": int(cfg.get("save_total_limit", 2)),
            "report_to": str(cfg.get("report_to", "none")),
            "run_name": str(cfg.get("experiment_name", "gnprsid-grpo")),
            "use_vllm": bool(cfg.get("use_vllm", True)),
            "vllm_mode": str(cfg.get("vllm_mode", "colocate")),
            "vllm_gpu_memory_utilization": float(
                cfg.get("vllm_gpu_memory_utilization", cfg.get("gpu_memory_utilization", 0.6))
            ),
            "vllm_tensor_parallel_size": int(
                cfg.get("vllm_tensor_parallel_size", cfg.get("tensor_model_parallel_size", 1))
            ),
            "vllm_max_model_len": int(
                cfg.get(
                    "vllm_max_model_len",
                    int(cfg.get("max_length", cfg.get("max_prompt_length", 2048)))
                    + int(cfg.get("max_completion_length", cfg.get("max_response_length", 256))),
                )
            ),
            "sleep_level": int(cfg.get("sleep_level", 1)),
            "offload_model": offload_model,
            "offload_optimizer": offload_optimizer,
            "deepspeed": cfg.get("deepspeed"),
            "packing": bool(cfg.get("packing", False)),
            "log_completions": bool(cfg.get("log_completions", True)),
            "dataloader_num_workers": int(cfg.get("dataloader_num_workers", 4)),
            "dataset_num_proc": int(cfg.get("dataset_num_proc", 4)),
            "lora_rank": int(cfg.get("lora", {}).get("r", 16)),
            "lora_alpha": int(cfg.get("lora", {}).get("alpha", 32)),
            "target_modules": target_modules,
        }
        if enable_thinking is not None:
            ms_swift_config["enable_thinking"] = bool(enable_thinking)
            if not enable_thinking:
                ms_swift_config["add_non_thinking_prefix"] = True
                ms_swift_config["loss_scale"] = str(cfg.get("loss_scale", "last_round+ignore_empty_think"))
        if cfg.get("warmup_ratio") is not None:
            ms_swift_config["warmup_ratio"] = float(cfg["warmup_ratio"])

        ms_swift_config = {key: value for key, value in ms_swift_config.items() if value is not None}

        ms_swift_config_path = runtime_dir / "ms_swift_grpo.yaml"
        dump_yaml(ms_swift_config_path, ms_swift_config)

        command = [swift_cli, "rlhf", "--config", str(ms_swift_config_path)]
        logger.info("Running ms-swift command: %s", " ".join(str(token) for token in command))
        env = os.environ.copy()
        _prepend_pythonpath(env, project_root() / "src")
        env[TRACE_DIR_ENV] = str(reward_trace_dir)
        env[TRACE_GROUP_SIZE_ENV] = str(ms_swift_config["num_generations"])
        env[MS_SWIFT_REWARD_PATH_ENV] = str(reward_path)
        env[MS_SWIFT_REWARD_NAME_ENV] = str(cfg.get("reward_function_name", "compute_score"))
        env["NPROC_PER_NODE"] = str(int(cfg.get("n_gpus_per_node", 1)))
        env["NNODES"] = str(int(cfg.get("nnodes", 1)))
        env["NODE_RANK"] = str(int(cfg.get("node_rank", 0)))
        env["MASTER_ADDR"] = str(cfg.get("master_addr", "127.0.0.1"))
        env["MASTER_PORT"] = str(int(cfg.get("master_port", 29500)))
        if ms_swift_config["report_to"] == "wandb" and cfg.get("project_name"):
            env["WANDB_PROJECT"] = str(cfg["project_name"])
        try:
            subprocess.run(command, check=True, env=env)
        except subprocess.CalledProcessError:
            attempted_cleanup = _cleanup_grpo_runtime_processes()
            if attempted_cleanup:
                logger.warning(
                    "GRPO training failed; attempted cleanup commands: %s",
                    [" ".join(cleanup_command) for cleanup_command in attempted_cleanup],
                )
            raise
        return {
            "train_path": str(train_path),
            "valid_path": str(valid_path),
            "init_model_path": str(init_model_path),
            "reward_function_path": str(reward_path),
            "reward_trace_dir": str(reward_trace_dir),
            "runtime_dir": str(runtime_dir),
            "ms_swift_config_path": str(ms_swift_config_path),
            "plugin_path": str(plugin_path),
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
    if os.environ.get(TORCHRUN_SKIP_MANIFEST_ENV) == "1":
        logger.info("Finished distributed worker stage=%s backend=%s", context.stage, context.backend)
        return result
    manifest = backend.collect_artifacts(context, prepared, result)
    logger.info("Finished training stage=%s backend=%s", context.stage, context.backend)
    return manifest
