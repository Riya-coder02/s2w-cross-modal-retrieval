"""Near-duplicate detection, grouping of apps linked by near-duplicate screens, and leakage-aware splitting."""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .utils import DEVICE


def normalised(feats: np.ndarray) -> torch.Tensor:
    """L2-normalised features on the device (half precision on GPU to save memory)."""
    t = F.normalize(torch.tensor(feats, device=DEVICE), dim=-1)
    return t.half() if DEVICE == "cuda" else t


@torch.no_grad()
def near_duplicate_pairs(Fn: torch.Tensor, threshold=0.95, chunk=2048):
    """All screen pairs (i < j) with cosine similarity above `threshold`. Returns arrays (i, j, sim)."""
    I, J, S_ = [], [], []
    for a in range(0, len(Fn), chunk):
        S = (Fn[a:a + chunk] @ Fn.T).float()
        ii, jj = torch.nonzero(S > threshold, as_tuple=True)
        m = (ii + a) < jj
        I.append((ii[m] + a).cpu()); J.append(jj[m].cpu()); S_.append(S[ii[m], jj[m]].cpu())
    return [torch.cat(x).numpy() for x in (I, J, S_)]


def app_groups(pairs, app_index: np.ndarray, threshold: float) -> np.ndarray:
    """Connected components of the app graph: two apps are linked if any of their screens are near-duplicates.
    Returns a group label per app."""
    pi, pj, ps = pairs
    m = ps > threshold
    ai, aj = app_index[pi[m]], app_index[pj[m]]
    n = int(app_index.max()) + 1
    g = coo_matrix((np.ones(len(ai)), (ai, aj)), shape=(n, n))
    return connected_components(g, directed=False)[1]


def assign_split(group_of_screen: np.ndarray, val_share=0.10, test_share=0.10, seed=42, force_train=()):
    """Shuffle groups and fill test, then validation, up to their target screen counts; the rest is train.
    Groups listed in `force_train` (e.g. one giant template cluster) always go to train."""
    g = np.random.default_rng(seed)
    sizes = pd.Series(group_of_screen).value_counts()
    forced = set(force_train)
    order = [x for x in g.permutation(sizes.index.values) if x not in forced]
    n = len(group_of_screen)
    target, filled = {"test": test_share * n, "val": val_share * n}, {"test": 0, "val": 0}
    lab = {x: "train" for x in sizes.index}
    for grp in order:
        for s in ("test", "val"):
            if filled[s] < target[s]:
                lab[grp] = s
                filled[s] += sizes[grp]
                break
    return np.array([lab[x] for x in group_of_screen])


def grouped_split(pairs, app_index, threshold, max_group_share, **kw):
    """Proposal split: group = connected component at `threshold`; components above `max_group_share` -> train."""
    grp = app_groups(pairs, app_index, threshold)[app_index]
    sizes = pd.Series(grp).value_counts()
    big = sizes.index[sizes > max_group_share * len(grp)].tolist()
    return assign_split(grp, force_train=big, **kw), grp, int(sizes[big].sum()) if big else 0


@torch.no_grad()
def leakage_report(Fn, split, kept_rows, cap_owner, cap_textid, chunk=2048):
    """How much the test set resembles training: nearest-train cosine per test screen and caption overlap."""
    tr = torch.tensor(np.where(split == "train")[0], device=Fn.device)
    te = torch.tensor(np.where(split == "test")[0], device=Fn.device)
    mx = torch.cat([(Fn[te[a:a + chunk]] @ Fn[tr].T).float().max(1).values for a in range(0, len(te), chunk)]).cpu().numpy()
    tr_text = np.unique(cap_textid[np.isin(cap_owner, kept_rows[split == "train"])])
    te_caps = cap_textid[np.isin(cap_owner, kept_rows[split == "test"])]
    return mx, {"test_screens": int(len(te)),
                "near_dup_in_train_>0.95_%": float((mx > 0.95).mean() * 100),
                "near_dup_in_train_>0.98_%": float((mx > 0.98).mean() * 100),
                "median_nearest_train_cos": float(np.median(mx)),
                "test_captions_seen_in_train_%": float(np.isin(te_caps, tr_text).mean() * 100)}
