import torch

def _pairwise_sqdist(u: torch.Tensor) -> torch.Tensor:
    s = (u*u).sum(dim=1, keepdim=True)
    D2 = s + s.t() - 2 * (u @ u.t())
    return torch.clamp(D2, min=0.0)

def cauchy_nce_loss(u: torch.Tensor, pos_mask: torch.Tensor) -> torch.Tensor:
    """
    u: [M, 2]  —— 2D  embedding
    pos_mask: [M, M]，0/1 mask；each row i is the positive pair set for example i
             can have multi positive pairs

    q_ij = 1 / (1 + ||u_i - u_j||^2)
    L_i = - log ( sum_{j∈P(i)} q_ij / sum_{j≠i} q_ij )
    """
    D2 = _pairwise_sqdist(u)
    Q  = 1.0 / (1.0 + D2 + 1e-8)
    Q.fill_diagonal_(0.0)

    denom = Q.sum(dim=1)
    num   = (Q * pos_mask).sum(dim=1)
    num   = torch.clamp(num,   min=1e-12)
    denom = torch.clamp(denom, min=1e-12)

    loss = -torch.log(num / denom)
    valid = (pos_mask.sum(dim=1) > 0)
    return loss[valid].mean() if valid.any() else torch.tensor(0.0, device=u.device)
