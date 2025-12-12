from __future__ import annotations
import torch
def batch_pairs_functional(
    batch_idx: torch.Tensor,             # [B]
    coords: torch.Tensor | None = None,  # [N,2/3]
    func_vec: torch.Tensor | None = None,# [N]
    func_mat: torch.Tensor | None = None,# [N,N]
    r_max: float | None = None,
    f_th: float = 0.2,
    require_same_sign: bool = True,
    use_abs_for_mat: bool = False,
):
    device = batch_idx.device
    B = batch_idx.numel()

    any_func  = (func_mat is not None) or (func_vec is not None)
    any_space = (coords is not None) and (r_max is not None)

    if not any_func and not any_space:
        return []
    if func_mat is not None:
        F = func_mat.index_select(0, batch_idx).index_select(1, batch_idx)
        Fm = F.abs() if use_abs_for_mat else F
        func_ok = (Fm >= f_th)
    elif func_vec is not None:
        fv = func_vec[batch_idx]             # [B]
        mag_ok = (fv.abs()[:, None] >= f_th) & (fv.abs()[None, :] >= f_th)
        if require_same_sign:
            sign_ok = (fv[:, None] * fv[None, :]) > 0
        else:
            sign_ok = torch.ones(B, B, dtype=torch.bool, device=device)
        func_ok = mag_ok & sign_ok
    else:
        func_ok = torch.ones(B, B, dtype=torch.bool, device=device)

    if any_space:
        C = coords[batch_idx].float()
        dist = torch.cdist(C, C)
        space_ok = (dist <= r_max)
    else:
        space_ok = torch.ones(B, B, dtype=torch.bool, device=device)

    mask = func_ok & space_ok
    mask.fill_diagonal_(False)

    ii, jj = torch.where(torch.triu(mask, diagonal=1))
    pairs = [(int(i), int(j)) for i, j in zip(ii, jj)]
    return pairs

def lift_pairs_to_2views(pairs_1view: list[tuple[int,int]], B: int):

    lifted = []
    for i, j in pairs_1view:
        lifted += [(i, j), (i+B, j+B)]
        lifted += [(i, j + B), (i + B, j)]

    return lifted

# --- make_pos_mask_two_views ---
def make_pos_mask_two_views(B: int, device=None):
    M = 2 * B
    pos = torch.zeros(M, M, dtype=torch.bool, device=device)
    eye = torch.eye(B, dtype=torch.bool, device=device)
    pos[:B, B:] = eye
    pos[B:, :B] = eye
    pos.fill_diagonal_(False)
    return pos

# --- make_pos_mask_from_pairs ---
def make_pos_mask_from_pairs(pairs: list[tuple[int,int]], M: int, device=None):
    pos = torch.zeros(M, M, dtype=torch.bool, device=device)
    for i, j in pairs:
        if i == j:
            continue
        pos[i, j] = True
        pos[j, i] = True
    pos.fill_diagonal_(False)
    return pos

# --- merge_pos_masks ---
def merge_pos_masks(*masks: torch.Tensor):
    m = torch.zeros_like(masks[0], dtype=torch.bool)
    for x in masks:
        if x is None:
            continue
        m |= x.bool()
    m.fill_diagonal_(False)
    return m