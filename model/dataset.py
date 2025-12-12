from __future__ import annotations
from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset
import numpy as np

def subset_mean(x_trials: torch.Tensor, k: int, rng: np.random.Generator) -> torch.Tensor:
    R = x_trials.size(0)
    idx = rng.permutation(R)
    S1 = idx[:k]
    S2 = idx[k:2*k] if 2*k <= R else idx[R-k:]
    return x_trials[S1].mean(dim=0), x_trials[S2].mean(dim=0)

class TracePairDataset(Dataset):
    """
    X: [N,R,T]  —— N neurons, R trials
    """
    def __init__(self, X: torch.Tensor, k: Optional[int] = None, epoch_seed: int = 0):
        assert X.dim() == 3
        self.X = X
        self.N, self.R, self.T = X.shape
        self.k = k or max(1, self.R // 2)
        self._rng = np.random.default_rng(int(epoch_seed))

    def set_epoch(self, epoch: int):
        self._rng = np.random.default_rng(int(epoch))

    def __len__(self): return self.N

    def __getitem__(self, idx: int):
        v1, v2 = subset_mean(self.X[idx], self.k, self._rng)  # [T],[T]
        return idx, v1, v2

def collate_views(batch):
    idx = torch.tensor([b[0] for b in batch], dtype=torch.long)   # [B]
    v1  = torch.stack([b[1] for b in batch], dim=0)               # [B,T]
    v2  = torch.stack([b[2] for b in batch], dim=0)               # [B,T]
    return {"idx": idx, "view1": v1, "view2": v2}
