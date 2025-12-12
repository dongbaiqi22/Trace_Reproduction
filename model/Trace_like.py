from __future__ import annotations
from typing import Iterable, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    """
    MLP Encoder (for baseline)
    Input:
        x: Tensor of shape [B, T] or [B, 1, T]. T must be fixed
           (you can pad/crop in the Dataset to align lengths).

    Output:
        h: Tensor of shape [B, C], where C is the width of the last layer.
    """
    def __init__(self, input_len: int, hidden: Iterable[int] = (768, 512, 256, 128), dropout: float = 0.0):
        super().__init__()
        layers = []
        in_dim = input_len
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        self.net = nn.Sequential(*layers)
        self.out_dim = in_dim
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)               # [B,T]
        assert x.dim() == 2, f"Expect [B,T], got {tuple(x.shape)}"
        return self.net(x)                 # [B, 128]


class ProjectionHead(nn.Module):
    """
    TRACE projection head：h(128) -> z2d(2)
      - "small": 128 -> 2  （defaul）
      - "large": 128 -> 1024 -> 2 ）
    """
    def __init__(self, in_dim: int = 128, mode: str = "large"):
        super().__init__()
        self.mode = mode
        if mode == "small":
            self.fc1 = nn.Linear(in_dim, 2)
            nn.init.xavier_normal_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        elif mode == "large":
            self.fc1 = nn.Linear(in_dim, 1024)
            self.fc2 = nn.Linear(1024, 2)
            nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu"); nn.init.zeros_(self.fc1.bias)
            nn.init.xavier_normal_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)
        else:
            raise ValueError("mode must be 'small' or 'large'")

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if self.mode == "small":
            return self.fc1(h)                 # [B,2]
        else:
            return self.fc2(F.relu(self.fc1(h), inplace=True))  # [B,2]


class TRACEModel(nn.Module):
    """
    TRACE: straight learn 2D embedding (Cauchy similarity)
    forward(x) -> (h, z2d)
    forward_views(x1,x2) -> ((h1,z1), (h2,z2))
    """
    def __init__(
        self,
        input_len: int,
        encoder_hidden: Iterable[int] = (768, 512, 256, 128),
        encoder_dropout: float = 0.0,
        proj_mode: str = "large",
    ):
        super().__init__()
        self.encoder = MLPEncoder(input_len=input_len, hidden=encoder_hidden, dropout=encoder_dropout)
        assert self.encoder.out_dim == 128
        self.proj = ProjectionHead(in_dim=self.encoder.out_dim, mode=proj_mode)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)         # [B,128]
        z = self.proj(h)            # [B,2]
        return h, z

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encode(x)

    def forward_views(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        return self.encode(x1), self.encode(x2)
