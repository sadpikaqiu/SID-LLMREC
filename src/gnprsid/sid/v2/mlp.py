from __future__ import annotations

import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torch.nn.init import xavier_normal_


class MLPLayers(nn.Module):
    def __init__(self, layers, dropout: float = 0.0, activation: str = "relu", bn: bool = False):
        super().__init__()
        modules = []
        for idx, (input_size, output_size) in enumerate(zip(layers[:-1], layers[1:])):
            modules.append(nn.Linear(input_size, output_size))
            if bn and idx != len(layers) - 2:
                modules.append(nn.BatchNorm1d(output_size))
            if idx != len(layers) - 2:
                act = activation_layer(activation, output_size)
                if act is not None:
                    modules.append(act)
            modules.append(nn.Dropout(dropout))
        self.mlp_layers = nn.Sequential(*modules)
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)


def activation_layer(activation_name: str = "relu", emb_dim=None):
    if activation_name is None or activation_name.lower() == "none":
        return None
    if activation_name.lower() == "sigmoid":
        return nn.Sigmoid()
    if activation_name.lower() == "tanh":
        return nn.Tanh()
    if activation_name.lower() == "relu":
        return nn.ReLU()
    if activation_name.lower() == "leakyrelu":
        return nn.LeakyReLU()
    raise NotImplementedError(f"activation function {activation_name} is not implemented")


def kmeans(samples: torch.Tensor, num_clusters: int, num_iters: int = 10) -> torch.Tensor:
    centers = KMeans(n_clusters=num_clusters, max_iter=num_iters).fit(samples.cpu().detach().numpy()).cluster_centers_
    return torch.from_numpy(centers).to(samples.device)


@torch.no_grad()
def sinkhorn_algorithm(distances: torch.Tensor, epsilon: float, sinkhorn_iterations: int) -> torch.Tensor:
    distances = torch.clamp(distances, min=-1e3, max=1e3)
    q = torch.exp(-distances / (epsilon + 1e-8))
    q = q / (q.sum(dim=1, keepdim=True) + 1e-8)
    for _ in range(sinkhorn_iterations):
        q = q / (q.sum(dim=0, keepdim=True) + 1e-8)
        q = q / (q.sum(dim=1, keepdim=True) + 1e-8)
    return q
