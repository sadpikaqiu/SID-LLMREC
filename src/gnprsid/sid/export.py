from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from gnprsid.common.config import load_yaml
from gnprsid.common.io import ensure_dir, write_json
from gnprsid.common.logging import get_logger
from gnprsid.common.runtime import set_seed
from gnprsid.sid.train import EmbeddingDataset
from gnprsid.sid.v2.crqvae import CRQVAE


logger = get_logger(__name__)


def export_sid_from_config(config_path: str | Path, checkpoint_path: str | Path | None = None) -> dict:
    config = load_yaml(config_path)
    train_cfg = config["train"]
    sid_output_dir = ensure_dir(config["sid_output_dir"])
    train_manifest_path = sid_output_dir / "train_manifest.json"
    if checkpoint_path is None:
        if not train_manifest_path.exists():
            raise FileNotFoundError(f"Missing train manifest: {train_manifest_path}")
        checkpoint_path = load_yaml(train_manifest_path) if train_manifest_path.suffix in {".yaml", ".yml"} else None
    if checkpoint_path is None:
        import json

        checkpoint_path = json.loads(train_manifest_path.read_text(encoding="utf-8"))["best_loss_checkpoint"]

    checkpoint_path = Path(checkpoint_path)
    device = torch.device(train_cfg["device"])
    set_seed(config.get("seed", 2024))

    dataset = EmbeddingDataset(config["poi_embedding_path"])
    model = CRQVAE(
        in_dim=dataset.dim,
        num_emb_list=train_cfg["num_emb_list"],
        e_dim=train_cfg["e_dim"],
        layers=train_cfg["layers"],
        dropout_prob=train_cfg["dropout_prob"],
        bn=train_cfg["bn"],
        loss_type=train_cfg["loss_type"],
        quant_loss_weight=train_cfg["quant_loss_weight"],
        beta=train_cfg["beta"],
        kmeans_init=train_cfg["kmeans_init"],
        kmeans_iters=train_cfg["kmeans_iters"],
        sk_epsilons=train_cfg["sk_epsilons"],
        sk_iters=train_cfg["sk_iters"],
        use_linear=train_cfg["use_linear"],
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict"])
    model = model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=train_cfg["num_workers"], pin_memory=True)
    sid_indices = {}
    sid_vectors = {}
    with torch.no_grad():
        for pids, batch in tqdm(loader, desc="Export SID", ncols=100):
            batch = batch.to(device)
            vectors, indices, _ = model.get_indices(batch)
            for idx, pid in enumerate(pids.tolist()):
                sid_indices[pid] = indices[idx].tolist()
                sid_vectors[pid] = vectors[idx].tolist()

    value_counts = Counter(tuple(value) for value in sid_indices.values())
    seen_values = {}
    payload = {}
    rows = []
    for pid in sorted(sid_indices.keys()):
        indices = list(sid_indices[pid])
        key = tuple(indices)
        if value_counts[key] > 1:
            seen_values[key] = seen_values.get(key, -1) + 1
            indices = indices + [seen_values[key]]
        sid_token = "".join(f"<{chr(97 + i)}_{value}>" for i, value in enumerate(indices))
        payload[str(pid)] = {
            "pid": int(pid),
            "sid_indices": indices,
            "sid_token": sid_token,
            "vector": sid_vectors[pid],
        }
        rows.append({"pid": int(pid), "sid_indices": indices, "sid_token": sid_token})

    json_path = sid_output_dir / "pid_to_sid.json"
    csv_path = sid_output_dir / "pid_to_sid.csv"
    write_json(json_path, payload)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pid", "sid_indices", "sid_token"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    stats = {
        "checkpoint_path": str(checkpoint_path),
        "sid_json": str(json_path),
        "sid_csv": str(csv_path),
        "num_pois": len(payload),
        "collision_rate_before_suffix": float((sum(value_counts.values()) - len(value_counts)) / max(sum(value_counts.values()), 1)),
    }
    write_json(sid_output_dir / "sid_export_stats.json", stats)
    logger.info("Exported SID mapping to %s", sid_output_dir)
    return stats
