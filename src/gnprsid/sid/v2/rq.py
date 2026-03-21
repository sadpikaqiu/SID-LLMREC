from __future__ import annotations

import torch
import torch.nn as nn

from .cvq import CosineVectorQuantizer


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        n_e_list,
        e_dim: int,
        sk_epsilons=None,
        beta: float = 0.25,
        kmeans_init: bool = False,
        kmeans_iters: int = 100,
        sk_iters: int = 100,
        use_linear: int = 0,
    ):
        super().__init__()
        self.vq_layers = nn.ModuleList(
            [
                CosineVectorQuantizer(
                    n_e,
                    e_dim,
                    beta=beta,
                    kmeans_init=kmeans_init,
                    kmeans_iters=kmeans_iters,
                    sk_epsilon=sk_epsilon,
                    sk_iters=sk_iters,
                    use_linear=use_linear,
                )
                for n_e, sk_epsilon in zip(n_e_list, sk_epsilons)
            ]
        )

    def forward(self, x, use_sk: bool = True):
        original_shape = x.shape
        if x.ndim == 3:
            batch_size, time_steps, dim = x.shape
            x = x.view(-1, dim)
        elif x.ndim == 2:
            batch_size, dim = x.shape
            time_steps = None
        else:
            raise ValueError("x must be [B, D] or [B, T, D]")

        residual = x
        x_q = torch.zeros_like(x)
        losses = []
        all_indices = []
        all_scalars = []
        for quantizer in self.vq_layers:
            x_res, loss, indices, scalar = quantizer(residual, use_sk=use_sk)
            x_q = x_q + x_res
            residual = residual - x_res
            losses.append(loss)
            all_indices.append(indices)
            all_scalars.append(scalar)

        x_q = x_q.view(original_shape)
        mean_loss = torch.stack(losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)
        all_scalars = torch.stack(all_scalars, dim=-1)

        if time_steps is not None:
            all_indices = all_indices.view(batch_size, time_steps, -1)
            all_scalars = all_scalars.view(batch_size, time_steps, -1)
        else:
            all_indices = all_indices.view(batch_size, -1)
            all_scalars = all_scalars.view(batch_size, -1)

        return x_q, mean_loss, (all_indices, all_scalars)

    @torch.no_grad()
    def get_codebook(self):
        return torch.stack([quantizer.get_codebook() for quantizer in self.vq_layers])
