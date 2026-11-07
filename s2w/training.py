"""Mini-batch construction, losses with false-negative masking, and the training loop with early stopping."""
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .evaluation import evaluate
from .models import DualEncoder, EarlyFusion, multi_positive_infonce
from .utils import DEVICE, gpu_peak_gb, gpu_reset, seed_everything


def pos_keys(d, n_text):
    """Sorted keys screen*n_text + text_id for every (screen, caption text) pair of a split.
    Used to recognise 'negatives' that actually carry a correct caption text."""
    return torch.unique(d["cap2img"] * n_text + d["textid"]), n_text


def batch_captions(d, idx, all_captions: bool):
    """Caption ids for a batch of screens and the local screen each caption belongs to.
    all_captions=True: every caption of each screen (multi-positive); False: one random caption per screen."""
    if not all_captions:
        cap = d["start"][idx] + (torch.rand(len(idx), device=DEVICE) * d["cnt"][idx]).long()
        return cap, torch.arange(len(idx), device=DEVICE)
    cnt = d["cnt"][idx]
    owner = torch.repeat_interleave(torch.arange(len(idx), device=DEVICE), cnt)
    offs = torch.arange(int(cnt.sum()), device=DEVICE) - torch.repeat_interleave(torch.cumsum(cnt, 0) - cnt, cnt)
    return d["start"][idx][owner] + offs, owner


def batch_loss(model, kind, d, idx, cfg, keys):
    """Contrastive: (multi-positive) InfoNCE. Early fusion: BCE on positives and `k_neg` rolled in-batch
    negatives, skipping negatives whose caption text is correct for the screen."""
    xi = d["XI"][idx]
    key_set, n_text = keys
    if kind == "contrastive":
        cap, owner = batch_captions(d, idx, cfg["multi_pos"])
        P = owner[None, :] == torch.arange(len(idx), device=DEVICE)[:, None]
        if cfg["multi_pos"]:
            P = P | torch.isin(idx[:, None] * n_text + d["textid"][cap][None, :], key_set)
        return multi_positive_infonce(model.logits(xi, d["XT"][cap]), P)
    cap, _ = batch_captions(d, idx, False)
    xt = d["XT"][cap]
    loss = F.binary_cross_entropy_with_logits(model(xi, xt), torch.ones(len(idx), device=DEVICE))
    negs = []
    for k in range(1, min(cfg["k_neg"], len(idx) - 1) + 1):
        capk = cap.roll(k, 0)
        valid = ~torch.isin(idx * n_text + d["textid"][capk], key_set)
        if valid.any():
            negs.append(F.binary_cross_entropy_with_logits(model(xi[valid], d["XT"][capk][valid]),
                                                           torch.zeros(int(valid.sum()), device=DEVICE)))
    return loss + (torch.stack(negs).mean() if negs else 0.0)


def train_model(kind, cfg, seed, D, n_text, epochs=60, patience=8, batch=256):
    """Train an EarlyFusion ('early') or DualEncoder ('contrastive') model with AdamW.
    After every epoch: validation loss, validation Recall@10 and Recall@10 on a training subset of the same
    gallery size (for the overfitting gap). Early stopping keeps the epoch with the best validation Recall@10."""
    seed_everything(seed)
    tr, va, ts = D["train"], D["val"], D["train_sub"]
    k_tr, k_va = pos_keys(tr, n_text), pos_keys(va, n_text)
    cls = EarlyFusion if kind == "early" else DualEncoder
    model = cls(tr["XI"].shape[1], tr["XT"].shape[1], p=cfg["p"], p_in=cfg["p_in"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    n, hist, best, bad = len(tr["rows"]), [], dict(score=-1.0, state=None, epoch=0), 0
    gpu_reset()
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        perm, losses = torch.randperm(n, device=DEVICE), []
        for b in range(0, n, batch):
            idx = perm[b:b + batch]
            if len(idx) < 16:
                continue
            loss = batch_loss(model, kind, tr, idx, cfg, k_tr)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            vidx = torch.arange(min(len(va["rows"]), 1024), device=DEVICE)
            vloss = batch_loss(model, kind, va, vidx, cfg, k_va).item()
            mv = evaluate(model.score_matrix(va["XI"], va["XT"]), va)
            mt = evaluate(model.score_matrix(ts["XI"], ts["XT"]), ts)
        hist.append(dict(epoch=ep, train_loss=float(np.mean(losses)), val_loss=vloss,
                         train_R10=mt["avg_R10"], val_R10=mv["avg_R10"]))
        if mv["avg_R10"] > best["score"]:
            best, bad = dict(score=mv["avg_R10"], state=copy.deepcopy(model.state_dict()), epoch=ep), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best["state"])
    model.eval()
    h = pd.DataFrame(hist)
    hb = h.iloc[best["epoch"] - 1]
    return dict(model=model, hist=h, best_epoch=best["epoch"], best_val=best["score"],
                gap=hb["train_R10"] - hb["val_R10"], final_gap=h.iloc[-1]["train_R10"] - h.iloc[-1]["val_R10"],
                train_s=time.perf_counter() - t0, peak_gb=gpu_peak_gb(),
                params=sum(p.numel() for p in model.parameters()))
