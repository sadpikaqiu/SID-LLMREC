from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import kmeans, sinkhorn_algorithm


class CosineVectorQuantizer(nn.Module):
    def __init__(
        self,
        n_e: int,
        e_dim: int,
        beta: float = 0.25,
        kmeans_init: bool = False,
        kmeans_iters: int = 10,
        sk_epsilon=None,
        sk_iters: int = 100,
        use_linear: int = 0,
        use_ema: bool = True,
        ema_decay: float = 0.95,
        ema_epsilon: float = 1e-5,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters
        self.use_linear = use_linear
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_epsilon = ema_epsilon

        if use_ema:
            self.register_buffer("cluster_size", torch.zeros(n_e))
            self.register_buffer("ema_w", torch.zeros(n_e, e_dim))
            if use_linear == 1:
                self.use_linear = 0

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

        if use_ema:
            self.embedding.weight.requires_grad_(False)

        if use_linear == 1:
            self.codebook_projection = torch.nn.Linear(self.e_dim, self.e_dim)
            torch.nn.init.normal_(self.codebook_projection.weight, std=self.e_dim ** -0.5)

    def get_codebook(self):
        codebook = self.embedding.weight
        if self.use_linear:
            codebook = self.codebook_projection(codebook)
        return codebook

    @torch.no_grad()
    def init_emb(self, data):
        centers = kmeans(data, self.n_e, self.kmeans_iters)
        self.embedding.weight.data.copy_(centers)
        self.initted = True

    def forward(self, x, use_sk: bool = True):
        batch_size, dim = x.shape
        latent = x.view(batch_size, dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        codebook = self.get_codebook()
        latent_norm = F.normalize(latent, dim=-1, eps=1e-8)
        codebook_dir = F.normalize(codebook, dim=-1, eps=1e-8)
        sim = latent_norm @ codebook_dir.t()

        if use_sk and self.sk_epsilon is not None and self.sk_epsilon > 0:
            distances = 1 - sim
            q = sinkhorn_algorithm(self.center_distance_for_constraint(distances).double(), self.sk_epsilon, self.sk_iters)
            indices = sim.argmax(dim=-1) if torch.isnan(q).any() else q.argmax(dim=-1)
        else:
            indices = sim.argmax(dim=-1)

        direction = F.embedding(indices, codebook_dir)
        scalar = (latent * direction).sum(dim=-1, keepdim=True).clamp(min=0.0)
        proj_vec = scalar * direction

        commitment_loss = F.mse_loss(proj_vec.detach(), latent)
        if self.use_ema:
            loss = self.beta * commitment_loss
        else:
            codebook_loss = F.mse_loss(proj_vec, latent.detach())
            loss = codebook_loss + self.beta * commitment_loss

        x_q = latent + (proj_vec - latent).detach()

        if self.use_ema and self.training:
            with torch.no_grad():
                one_hot = F.one_hot(indices, self.n_e).float()
                batch_cluster_size = one_hot.sum(dim=0)
                self.cluster_size.mul_(self.ema_decay).add_(batch_cluster_size, alpha=1 - self.ema_decay)

                dw = torch.zeros_like(self.ema_w)
                dw.index_add_(0, indices, latent_norm.to(self.ema_w.device))
                self.ema_w.mul_(self.ema_decay).add_(dw, alpha=1 - self.ema_decay)
                n = self.cluster_size.unsqueeze(1).clamp(min=self.ema_epsilon)
                new_codebook = self.ema_w / n
                new_codebook = F.normalize(new_codebook, dim=-1, eps=1e-8)
                self.embedding.weight.data.copy_(new_codebook)

                avg_usage = self.cluster_size.mean()
                dead_threshold = avg_usage * 0.2
                dead_indices = torch.where(self.cluster_size < dead_threshold)[0]
                if dead_indices.numel() > 0:
                    sample_indices = torch.randperm(batch_size, device=latent.device)[: dead_indices.numel()]
                    replace_samples = F.normalize(latent_norm[sample_indices], dim=-1, eps=1e-8)
                    self.embedding.weight.data[dead_indices] = replace_samples.to(self.embedding.weight.device)
                    self.cluster_size[dead_indices] = 1.0
                    self.ema_w[dead_indices] = replace_samples.to(self.ema_w.device)

        return x_q, loss, indices.view(batch_size), scalar.view(batch_size)

    @staticmethod
    def center_distance_for_constraint(distances: torch.Tensor) -> torch.Tensor:
        max_distance = distances.max()
        min_distance = distances.min()
        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        return (distances - middle) / amplitude
