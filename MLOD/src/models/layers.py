from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.permute(0, 2, 3, 1)
        tensor = F.layer_norm(
            tensor,
            (tensor.shape[-1],),
            self.weight,
            self.bias,
            self.eps,
        )
        return tensor.permute(0, 3, 1, 2).contiguous()


def make_norm_layer(name: str, channels: int) -> nn.Module:
    name = name.lower()
    if name == "bn":
        return nn.BatchNorm2d(channels)
    if name == "ln":
        return LayerNorm2d(channels)
    raise ValueError(f"Unsupported normalization: {name}")


def make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class TwoMLPHead(nn.Module):
    def __init__(self, input_channels: int, representation_size: int) -> None:
        super().__init__()
        self.fc6 = nn.Linear(input_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.flatten(start_dim=1)
        tensor = F.relu(self.fc6(tensor))
        return F.relu(self.fc7(tensor))


class MultiLabelBoxPredictor(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        label_attention_tau: float = 0.5,
        label_attention_learnable: bool = True,
    ) -> None:
        super().__init__()
        self.classifier = nn.Linear(input_channels, num_classes)
        self.box_regressor = nn.Linear(input_channels, 4)
        self.label_attention_tau = label_attention_tau

        attention = torch.zeros(num_classes, num_classes)
        if label_attention_learnable:
            self.label_attention = nn.Parameter(attention)
        else:
            self.register_buffer("label_attention", attention)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, tensor: torch.Tensor):
        logits = self.classifier(tensor)
        deltas = self.box_regressor(tensor)

        if self.label_attention_tau and self.label_attention_tau > 0:
            probabilities = torch.sigmoid(logits)
            logits = logits + self.label_attention_tau * (
                probabilities @ self.label_attention
            )
        return logits, deltas
