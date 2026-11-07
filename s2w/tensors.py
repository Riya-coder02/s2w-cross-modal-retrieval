"""Per-split tensors: screen/caption index bookkeeping, train-only standardisation and positive-pair masks."""
import numpy as np
import torch

from .utils import DEVICE


def gather_split(rows, n_caps, cap_start, cap_textid):
    """Index bookkeeping for a set of screen rows (positions in `meta`).

    rows     screen rows; cap_ids  their caption ids (flat index); cnt/start  captions per screen and offsets;
    cap2img  caption -> local screen index; textid  normalised-text id per caption
    """
    rows = np.asarray(rows)
    cnt, st = n_caps[rows], cap_start[rows]
    cap_ids = np.concatenate([np.arange(s, s + c) for s, c in zip(st, cnt)])
    return dict(rows=rows, cap_ids=cap_ids, cnt=cnt, start=np.concatenate([[0], np.cumsum(cnt)[:-1]]),
                cap2img=np.repeat(np.arange(len(rows)), cnt), textid=cap_textid[cap_ids])


def fit_scaler(X):
    """Mean/std computed on TRAINING data only (no information from val/test leaks into preprocessing)."""
    return X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6


def positives(d, relaxed: bool):
    """Boolean (screens x captions) matrix of correct pairs.
    strict : caption belongs to the screen
    relaxed: screen has a caption with identical normalised text (generic captions such as 'login page')."""
    n_i = len(d["rows"])
    cap2img = torch.as_tensor(d["cap2img"], device=DEVICE)
    if not relaxed:
        return cap2img[None, :] == torch.arange(n_i, device=DEVICE)[:, None]
    uniq, inv = np.unique(d["textid"], return_inverse=True)
    inv = torch.as_tensor(inv, device=DEVICE)
    A = torch.zeros(n_i, len(uniq), dtype=torch.bool, device=DEVICE)
    A[cap2img, inv] = True
    return A[:, inv]


def to_tensors(d, img_feats, txt_feats, sc_img, sc_txt, with_P=True):
    """Standardised features and index tensors on the device. `with_P=False` for the training split
    (its positive masks are never needed and would not fit in memory)."""
    f = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)
    i = lambda a: torch.tensor(a, dtype=torch.long, device=DEVICE)
    t = dict(XI=f((img_feats[d["rows"]] - sc_img[0]) / sc_img[1]),
             XT=f((txt_feats[d["cap_ids"]] - sc_txt[0]) / sc_txt[1]),
             cap2img=i(d["cap2img"]), start=i(d["start"]), cnt=i(d["cnt"]), textid=i(d["textid"]),
             rows=d["rows"], cap_ids=d["cap_ids"])
    if with_P:
        t["P"], t["P_relaxed"] = positives(d, False), positives(d, True)
    return t


def build_split_tensors(split_labels, n_caps, cap_start, cap_textid, img_feats, txt_feats, seed=42):
    """Tensors for train / val / test and a training subset with the same gallery size as validation
    (so the train-vs-validation recall gap compares like with like)."""
    raw = {s: gather_split(np.where(split_labels == s)[0], n_caps, cap_start, cap_textid) for s in ("train", "val", "test")}
    sub = np.sort(np.random.default_rng(seed).choice(raw["train"]["rows"], len(raw["val"]["rows"]), replace=False))
    raw["train_sub"] = gather_split(sub, n_caps, cap_start, cap_textid)
    sc_i = fit_scaler(img_feats[raw["train"]["rows"]])
    sc_t = fit_scaler(txt_feats[raw["train"]["cap_ids"]])
    return {s: to_tensors(raw[s], img_feats, txt_feats, sc_i, sc_t, with_P=(s != "train")) for s in raw}
