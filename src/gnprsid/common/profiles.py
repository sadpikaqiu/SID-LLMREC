from __future__ import annotations

from pathlib import Path
from typing import Any

from gnprsid.common.config import load_yaml
from gnprsid.common.paths import project_root


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return project_root() / path


def _search_yaml_by_name(config_dir: Path, name: str) -> Path:
    candidates = sorted(config_dir.glob("*.yaml")) + sorted(config_dir.glob("*.yml"))
    for candidate in candidates:
        if candidate.stem == name:
            return candidate
        payload = load_yaml(candidate)
        if payload.get("name") == name:
            return candidate
    raise FileNotFoundError(f"Could not resolve config '{name}' under {config_dir}")


def resolve_model_profile_path(profile: str | Path) -> Path:
    profile_path = Path(profile)
    if profile_path.suffix in {".yaml", ".yml"}:
        return resolve_project_path(profile_path)
    return _search_yaml_by_name(project_root() / "configs" / "models", str(profile))


def load_model_profile(profile: str | Path) -> dict[str, Any]:
    return load_yaml(resolve_model_profile_path(profile))
