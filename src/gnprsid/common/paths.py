from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class DatasetPaths:
    dataset: str
    root: Path
    raw: Path
    interim: Path
    processed: Path
    artifacts: Path
    checkpoints: Path
    outputs: Path


def dataset_paths(dataset: str) -> DatasetPaths:
    root = project_root()
    return DatasetPaths(
        dataset=dataset,
        root=root / "data" / dataset,
        raw=root / "data" / dataset / "raw",
        interim=root / "data" / dataset / "interim",
        processed=root / "data" / dataset / "processed",
        artifacts=root / "artifacts" / dataset,
        checkpoints=root / "checkpoints" / dataset,
        outputs=root / "outputs" / dataset,
    )
