from __future__ import annotations

from pathlib import Path

from gnprsid.common.io import ensure_dir, write_json
from gnprsid.common.profiles import resolve_project_path


def _resolve_actor_dir(checkpoint_path: Path) -> Path:
    if checkpoint_path.name == "actor" and checkpoint_path.is_dir():
        return checkpoint_path
    actor_dir = checkpoint_path / "actor"
    if actor_dir.is_dir():
        return actor_dir
    raise FileNotFoundError(f"Could not locate actor directory under {checkpoint_path}")


def merge_verl_actor(
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    backend: str = "fsdp",
    trust_remote_code: bool = False,
    use_cpu_init: bool = False,
) -> dict:
    try:
        from verl.model_merger.base_model_merger import ModelMergerConfig
        from verl.model_merger.fsdp_model_merger import FSDPModelMerger
    except ImportError as error:
        raise ImportError("Merging a verl actor requires the 'verl' package.") from error

    checkpoint_path = resolve_project_path(checkpoint_path)
    actor_dir = _resolve_actor_dir(checkpoint_path)
    if output_path is None:
        output_path = actor_dir.parent / f"{actor_dir.name}_merged"
    output_path = ensure_dir(resolve_project_path(output_path))

    hf_dir = actor_dir / "huggingface"
    if not hf_dir.is_dir():
        raise FileNotFoundError(f"Expected Hugging Face config files under {hf_dir}")

    if backend != "fsdp":
        raise NotImplementedError("The first GNPR-SID GRPO version only supports backend='fsdp'.")

    config = ModelMergerConfig(
        operation="merge",
        backend=backend,
        target_dir=str(output_path),
        hf_upload_path=None,
        private=False,
        test_hf_dir=None,
        tie_word_embedding=False,
        trust_remote_code=trust_remote_code,
        is_value_model=False,
        local_dir=str(actor_dir),
        hf_model_config_path=str(hf_dir),
        use_cpu_initialization=use_cpu_init,
    )

    merger = FSDPModelMerger(config)
    merger.merge_and_save()
    merger.cleanup()

    manifest = {
        "checkpoint_path": str(checkpoint_path),
        "actor_path": str(actor_dir),
        "output_path": str(output_path),
        "backend": backend,
    }
    write_json(output_path / "merge_manifest.json", manifest)
    return manifest
