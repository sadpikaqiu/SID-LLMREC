from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .mlp import MLPLayers
from .rq import ResidualVectorQuantizer


class CRQVAE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_emb_list,
        e_dim: int,
        layers,
        dropout_prob: float = 0.0,
        bn: bool = False,
        loss_type: str = "mse",
        quant_loss_weight: float = 0.25,
        beta: float = 0.25,
        kmeans_init: bool = False,
        kmeans_iters: int = 100,
        sk_epsilons=None,
        sk_iters: int = 100,
        use_linear: int = 0,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.encoder = MLPLayers([in_dim] + list(layers) + [e_dim], dropout=dropout_prob, bn=bn)
        self.rq = ResidualVectorQuantizer(
            num_emb_list,
            e_dim,
            beta=beta,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            sk_epsilons=sk_epsilons,
            sk_iters=sk_iters,
            use_linear=use_linear,
        )
        self.decoder = MLPLayers([e_dim] + list(reversed(layers)) + [in_dim], dropout=dropout_prob, bn=bn)

    def forward(self, x, use_sk: bool = True):
        encoded = self.encoder(x)
        quantized, rq_loss, codes = self.rq(encoded, use_sk=use_sk)
        out = self.decoder(quantized)
        return out, rq_loss, codes

    def compute_loss(self, quant_loss, out, xs=None):
        if self.loss_type == "mse":
            loss_recon = F.mse_loss(out, xs, reduction="mean")
        elif self.loss_type == "l1":
            loss_recon = F.l1_loss(out, xs, reduction="mean")
        else:
            raise ValueError("Unsupported loss type")
        loss_total = loss_recon + self.quant_loss_weight * quant_loss
        return loss_total, quant_loss, loss_recon

    @torch.no_grad()
    def get_indices(self, xs, use_sk: bool = False):
        encoded = self.encoder(xs)
        quantized, _, (indices, scalars) = self.rq(encoded, use_sk=use_sk)
        return quantized.cpu(), indices.cpu(), scalars.cpu()
