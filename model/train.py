import torch
from torch.utils.data import DataLoader
from .pairing import (
    make_pos_mask_two_views, batch_pairs_functional,
    lift_pairs_to_2views, make_pos_mask_from_pairs, merge_pos_masks
)
from .loss import cauchy_nce_loss
from .dataset import TracePairDataset, collate_views
from .Trace_like import TRACEModel

def train_trace_with_custom_pairs(
    X,                       # [N,R,T] float32
    T,                       # input_len (=T)
    coords=None,             # [N,2/3] or None
    func_vec=None,           # [N]     or None
    func_mat=None,           # [N,N]   or None
    *,
    epochs=50,
    batch_size=1024,
    lr=1e-3,
    k=None,
    r_max=None,
    f_pos_th=0.2,
    same_sign=True,
    proj_mode="large",
    device="cuda",
    grad_clip=5.0,
    log_every=50,
    return_history=True,
    pin_memory=None,
):
    device = torch.device(device)

    if coords is not None:
        coords = coords.to(device)
    if func_vec is not None:
        func_vec = func_vec.to(device)
    if func_mat is not None:
        func_mat = func_mat.to(device)

    if (func_vec is None) and (func_mat is None) and (coords is None):
        print("[warn] coords/func is not provided, only depends TRACE's positive pair of two views")

    ds = TracePairDataset(X, k=k)
    if pin_memory is None:
        pin_memory = (device.type == "cuda")
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=0, collate_fn=collate_views, pin_memory=pin_memory
    )

    model = TRACEModel(input_len=T, proj_mode=proj_mode).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr)

    history = {"loss": [], "epoch_loss": []}
    global_step = 0

    for epoch in range(1, epochs + 1):
        ds.set_epoch(epoch)
        model.train()

        epoch_loss_sum = 0.0
        epoch_batches = 0

        for batch in dl:
            idx = batch["idx"].to(device)      # [B]
            v1  = batch["view1"].to(device)    # [B,T]
            v2  = batch["view2"].to(device)    # [B,T]

            (h1, u1), (h2, u2) = model.forward_views(v1, v2)  # u*: [B,2]
            U = torch.cat([u1, u2], dim=0)                    # [2B,2]
            B = u1.size(0)
            M = 2 * B

            pos_trace = make_pos_mask_two_views(B, device=U.device)

            sf_pairs = batch_pairs_functional(
                batch_idx=idx,
                coords=coords,
                func_vec=func_vec,
                func_mat=func_mat,
                r_max=r_max,
                f_th=f_pos_th,
                require_same_sign=same_sign,
                use_abs_for_mat=False,
            )
            lifted  = lift_pairs_to_2views(sf_pairs, B)
            pos_sf  = make_pos_mask_from_pairs(lifted, M=M, device=U.device)

            pos_mask = merge_pos_masks(pos_trace, pos_sf)

            # Cauchy-InfoNCE
            loss = cauchy_nce_loss(U, pos_mask)

            std = U.std(dim=0)
            loss = loss + 1e-4 * torch.relu(1e-3 - std).mean() + 1e-4 * (U.mean(dim=0)**2).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            epoch_loss_sum += float(loss.detach().cpu())
            epoch_batches += 1
            global_step += 1
            if return_history and (global_step % log_every == 0):
                history["loss"].append((global_step, float(loss.detach().cpu())))

        epoch_avg = epoch_loss_sum / max(1, epoch_batches)
        print(f"[epoch {epoch:03d}] mean loss = {epoch_avg:.6f} (batches={epoch_batches})")
        if return_history:
            history["epoch_loss"].append((epoch, epoch_avg))
    return (model, history) if return_history else model