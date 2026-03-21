from __future__ import annotations

import heapq
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_constant_schedule_with_warmup, get_linear_schedule_with_warmup

from gnprsid.common.config import load_yaml
from gnprsid.common.io import ensure_dir, write_json
from gnprsid.common.logging import get_logger
from gnprsid.common.runtime import set_seed
from gnprsid.sid.embedding import build_poi_embedding_dict, category2vec, region2vec
from gnprsid.sid.v2.crqvae import CRQVAE


logger = get_logger(__name__)


class EmbeddingDataset(Dataset):
    def __init__(self, embedding_pkl: str | Path):
        import pickle

        with Path(embedding_pkl).open("rb") as handle:
            emb_dict = pickle.load(handle)
        self.ids = sorted(emb_dict.keys())
        self.embeddings = np.array([emb_dict[k] for k in self.ids])
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        pid = self.ids[index]
        emb = torch.FloatTensor(self.embeddings[index])
        return pid, emb

    def __len__(self):
        return len(self.ids)


class SIDTrainer:
    def __init__(self, args, model: CRQVAE, data_num: int):
        self.args = args
        self.model = model
        self.device = torch.device(args.device)
        self.model = self.model.to(self.device)
        self.epochs = args.epochs
        self.eval_step = min(args.eval_step, args.epochs)
        self.use_sk = args.use_sk
        self.weight_decay = args.weight_decay
        self.lr = args.lr
        self.learner = args.learner
        self.lr_scheduler_type = args.lr_scheduler_type
        self.warmup_steps = args.warmup_epochs * data_num
        self.max_steps = args.epochs * data_num
        self.save_limit = args.save_limit
        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.checkpoint_dir = ensure_dir(args.ckpt_dir / datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.optimizer = self._build_optimizer()
        self.scheduler = self._get_scheduler()
        self.best_loss_ckpt = self.checkpoint_dir / "best_loss_model.pth"
        self.best_collision_ckpt = self.checkpoint_dir / "best_collision_model.pth"
        self.best_save_heap = []
        self.newest_save_queue = []

    def _build_optimizer(self):
        params = self.model.parameters()
        learner = self.learner.lower()
        if learner == "adam":
            return optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        if learner == "sgd":
            return optim.SGD(params, lr=self.lr, weight_decay=self.weight_decay)
        if learner == "adagrad":
            return optim.Adagrad(params, lr=self.lr, weight_decay=self.weight_decay)
        if learner == "rmsprop":
            return optim.RMSprop(params, lr=self.lr, weight_decay=self.weight_decay)
        return optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

    def _get_scheduler(self):
        if self.lr_scheduler_type.lower() == "linear":
            return get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.max_steps,
            )
        return get_constant_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.warmup_steps,
        )

    def _save_checkpoint(self, epoch: int, loss: float, collision_rate: float, ckpt_path: Path) -> None:
        torch.save(
            {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "best_collision_rate": self.best_collision_rate,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": vars(self.args),
            },
            ckpt_path,
            pickle_protocol=4,
        )

    def _train_epoch(self, data_loader: DataLoader, epoch_idx: int):
        self.model.train()
        total_loss = 0.0
        total_rq = 0.0
        total_recon = 0.0
        for _, (_, batch) in enumerate(tqdm(data_loader, desc=f"Train {epoch_idx}", ncols=100)):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            out, rq_loss, _ = self.model(batch, use_sk=self.use_sk)
            loss, loss_rq, loss_recon = self.model.compute_loss(rq_loss, out, xs=batch)
            if torch.isnan(loss):
                raise ValueError("Training loss is NaN")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            total_rq += loss_rq.item()
            total_recon += loss_recon.item()
        return total_loss, total_rq, total_recon

    @torch.no_grad()
    def _valid_epoch(self, data_loader: DataLoader) -> float:
        self.model.eval()
        indices_set = set()
        num_samples = 0
        for _, (_, batch) in enumerate(tqdm(data_loader, desc="Evaluate", ncols=100)):
            batch = batch.to(self.device)
            num_samples += len(batch)
            _, indices, _ = self.model.get_indices(batch)
            for index in indices:
                indices_set.add("-".join(str(int(item)) for item in index))
        return (num_samples - len(indices_set)) / max(num_samples, 1)

    def fit(self, data_loader: DataLoader) -> dict:
        for epoch_idx in range(self.epochs):
            train_loss, train_rq_loss, train_recon_loss = self._train_epoch(data_loader, epoch_idx)
            logger.info(
                "epoch=%s train_loss=%.4f rq_loss=%.4f recon_loss=%.4f",
                epoch_idx,
                train_loss,
                train_rq_loss,
                train_recon_loss,
            )
            if (epoch_idx + 1) % self.eval_step != 0:
                continue

            collision_rate = self._valid_epoch(data_loader)
            logger.info("epoch=%s collision_rate=%.6f", epoch_idx, collision_rate)
            ckpt_path = self.checkpoint_dir / f"epoch_{epoch_idx + 1:04d}_loss_{train_loss:.4f}_collision_{collision_rate:.4f}.pth"
            self._save_checkpoint(epoch_idx, train_loss, collision_rate, ckpt_path)

            if (epoch_idx + 1) >= 200 and train_loss < self.best_loss:
                self.best_loss = train_loss
                self._save_checkpoint(epoch_idx, train_loss, collision_rate, self.best_loss_ckpt)
            if (epoch_idx + 1) >= 200 and collision_rate < self.best_collision_rate:
                self.best_collision_rate = collision_rate
                self._save_checkpoint(epoch_idx, train_loss, collision_rate, self.best_collision_ckpt)

            now_save = (-collision_rate, ckpt_path)
            if len(self.newest_save_queue) < self.args.save_limit:
                self.newest_save_queue.append(now_save)
                heapq.heappush(self.best_save_heap, now_save)
            else:
                old_save = self.newest_save_queue.pop(0)
                self.newest_save_queue.append(now_save)
                if collision_rate < -self.best_save_heap[0][0]:
                    bad_save = heapq.heappop(self.best_save_heap)
                    heapq.heappush(self.best_save_heap, now_save)
                    if bad_save not in self.newest_save_queue and bad_save[1].exists():
                        bad_save[1].unlink()
                if old_save not in self.best_save_heap and old_save[1].exists():
                    old_save[1].unlink()

        manifest = {
            "checkpoint_dir": str(self.checkpoint_dir),
            "best_loss_checkpoint": str(self.best_loss_ckpt),
            "best_collision_checkpoint": str(self.best_collision_ckpt),
            "best_loss": float(self.best_loss),
            "best_collision_rate": float(self.best_collision_rate),
        }
        return manifest


def train_sid_from_config(config_path: str | Path) -> dict:
    config = load_yaml(config_path)
    train_cfg = config["train"]
    checkpoint_dir = Path(config["checkpoint_dir"])
    poi_embedding_path = Path(config["poi_embedding_path"])
    if not poi_embedding_path.exists():
        category_pkl = category2vec(
            config["poi_info_path"],
            Path(config["poi_embedding_output_dir"]),
            config.get("category_model_name", "all-MiniLM-L6-v2"),
            config.get("category_embedding_dim", 64),
        )
        poi_info_df = __import__("pandas").read_csv(config["poi_info_path"], on_bad_lines="skip", encoding="utf-8")
        region_pkl = None
        if not ({"latitude", "longitude"}.issubset(poi_info_df.columns) or {"Latitude", "Longitude"}.issubset(poi_info_df.columns)):
            region_pkl = region2vec(
                config["poi_info_path"],
                Path(config["poi_embedding_output_dir"]),
                config.get("region_model_name", config.get("category_model_name", "all-MiniLM-L6-v2")),
                config.get("region_embedding_dim", config.get("category_embedding_dim", 64)),
            )
        build_poi_embedding_dict(
            config["poi_info_path"],
            category_pkl,
            config["poi_embedding_output_dir"],
            region_embedding_pkl=region_pkl,
        )

    dataset = EmbeddingDataset(poi_embedding_path)
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
    loader = DataLoader(
        dataset,
        num_workers=train_cfg["num_workers"],
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        pin_memory=True,
    )
    args = SimpleNamespace(**train_cfg, ckpt_dir=checkpoint_dir)
    set_seed(config.get("seed", 2024))
    trainer = SIDTrainer(args, model, len(loader))
    manifest = trainer.fit(loader)
    manifest["config_path"] = str(config_path)
    manifest["poi_embedding_path"] = str(poi_embedding_path)
    write_json(Path(config["sid_output_dir"]) / "train_manifest.json", manifest)
    return manifest
