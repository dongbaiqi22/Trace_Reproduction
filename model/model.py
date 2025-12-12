from __future__ import annotations
from typing import Iterable, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """L2 normalize along dim."""
    return x / (x.norm(p=2, dim=dim, keepdim=True) + eps)


class MLPEncoder(nn.Module):
    """
    MLP Encoder (for baseline)
    Input:
        x: Tensor of shape [B, T] or [B, 1, T]. T must be fixed
           (you can pad/crop in the Dataset to align lengths).

    Output:
        h: Tensor of shape [B, C], where C is the width of the last layer.
    """
    def __init__(self, input_len: int, hidden: Iterable[int] = (512, 256, 128), dropout: float = 0.0):
        super().__init__()
        layers = []
        in_dim = input_len
        for idx, h in enumerate(hidden):
            layers += [
                nn.Linear(in_dim, h),
                nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        self.net = nn.Sequential(*layers)
        self.out_dim = in_dim
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.size(1) == 1:   # [B,1,T] -> [B,T]
            x = x.squeeze(1)
        assert x.dim() == 2, f"Expect [B,T], got shape {tuple(x.shape)}"
        return self.net(x)  # [B, out_dim]


class MLPHead(nn.Module):
    """MLP with two layers as head；output is l2-normalized"""
    def __init__(self, in_dim: int, out_dim: int, hidden: Optional[int] = None, norm_out: bool = True):
        super().__init__()
        hidden = hidden or max(64, in_dim)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.norm_out = norm_out
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x), inplace=True)
        x = self.fc2(x)
        return l2norm(x) if self.norm_out else x


class SpatialContrastModel(nn.Module):
    """
    Baseline Model（MLP encoder + heads）
    - encoder: MLPEncoder
    - rep_head: downstream representation (d_repr, L2)
    - proj_head: contrastive projection (d_proj, L2)
    - (optional) vis_head: output 2D visualization

    Usage
        model = SpatialContrastModel(input_len=T, d_repr=64, d_proj=128, use_vis2d=False)
        z_repr, z_proj, z_vis = model(x)               # x: [B,T] or [B,1,T]
        (z1,p1,_), (z2,p2,_) = model.forward_views(x1, x2)
    """
    def __init__(
        self,
        input_len: int,
        d_repr: int = 64,
        d_proj: int = 128,
        encoder_hidden: Iterable[int] = (512, 256, 128),
        encoder_dropout: float = 0.0,
        use_vis2d: bool = False,
    ):
        super().__init__()
        self.encoder = MLPEncoder(input_len=input_len, hidden=encoder_hidden, dropout=encoder_dropout)
        enc_out = self.encoder.out_dim
        self.rep_head = MLPHead(enc_out, d_repr, hidden=d_repr, norm_out=True)
        self.proj_head = MLPHead(d_repr, d_proj, hidden=d_repr, norm_out=True)
        self.use_vis2d = use_vis2d
        if use_vis2d:
            self.vis_head = MLPHead(d_repr, 2, hidden=d_repr, norm_out=False)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        h = self.encoder(x)               # [B, enc_out]
        z_repr = self.rep_head(h)         # [B, d_repr] (L2)
        z_proj = self.proj_head(z_repr)   # [B, d_proj] (L2)
        z_vis = self.vis_head(z_repr) if self.use_vis2d else None
        return z_repr, z_proj, z_vis

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return self.encode(x)

    def forward_views(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
               Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        out1 = self.encode(x1)
        out2 = self.encode(x2)
        return out1, out2
