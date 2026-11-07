"""Retrieval metrics, re-ranking and the cluster bootstrap."""
import numpy as np
import torch


def ranks(S, P):
    """Rank of the best-ranked correct item for every screen (screen->text) and every caption (text->screen).
    Ties with negatives count half, so a constant scorer cannot look good."""
    S, neg = S.float(), ~P
    best_i = S.masked_fill(neg, float("-inf")).max(1).values
    r_i = 1 + ((S > best_i[:, None]) & neg).sum(1) + 0.5 * ((S == best_i[:, None]) & neg).sum(1)
    best_t = S.masked_fill(neg, float("-inf")).max(0).values
    r_t = 1 + ((S > best_t[None, :]) & neg).sum(0) + 0.5 * ((S == best_t[None, :]) & neg).sum(0)
    return r_i.float(), r_t.float()


def summarise(r, prefix):
    return {f"{prefix}_R1": (r <= 1).float().mean().item() * 100, f"{prefix}_R5": (r <= 5).float().mean().item() * 100,
            f"{prefix}_R10": (r <= 10).float().mean().item() * 100, f"{prefix}_MRR": (1 / r).mean().item(),
            f"{prefix}_MedR": r.median().item()}


def evaluate(S, d, keep_ranks=False):
    """Strict metrics (Recall@1/5/10, MRR, median rank) in both directions plus relaxed Recall@10.
    S: score matrix (screens x captions); d: split tensors with masks P and P_relaxed."""
    r_i, r_t = ranks(S, d["P"])
    rr_i, rr_t = ranks(S, d["P_relaxed"])
    m = {**summarise(r_i, "i2t"), **summarise(r_t, "t2i")}
    m["i2t_R10_relaxed"] = (rr_i <= 10).float().mean().item() * 100
    m["t2i_R10_relaxed"] = (rr_t <= 10).float().mean().item() * 100
    m["avg_R10"] = 0.5 * (m["i2t_R10"] + m["t2i_R10"])
    m["avg_R1"] = 0.5 * (m["i2t_R1"] + m["t2i_R1"])
    if keep_ranks:
        m["_ranks"] = {k: v.cpu().numpy() for k, v in
                       {"i2t": r_i, "t2i": r_t, "i2t_relaxed": rr_i, "t2i_relaxed": rr_t}.items()}
    return m


# ----------------------------------------------------------------------------- re-ranking
def _z(S):
    return (S - S.mean()) / (S.std() + 1e-6)


def rerank_scores(S_con, S_early, K, alpha):
    """Contrastive top-K re-scored with a mix of z-scored contrastive and early-fusion scores.
    Returns (matrix for screen->text with top-K per row, matrix for text->screen with top-K per column);
    items outside the top-K keep their contrastive order below every re-ranked item."""
    comb = (1 - alpha) * _z(S_con) + alpha * _z(S_early)
    out = []
    for dim in (1, 0):
        k = min(K, S_con.shape[dim])
        thr = S_con.topk(k, dim=dim).values.select(dim, k - 1).unsqueeze(dim)
        out.append(torch.where(S_con >= thr, 1e3 + comb, _z(S_con)))
    return out


def evaluate_rerank(m_con, m_early, d, K, alpha, keep_ranks=False):
    S_i2t, S_t2i = rerank_scores(m_con.score_matrix(d["XI"], d["XT"]), m_early.score_matrix(d["XI"], d["XT"]), K, alpha)
    a, b = evaluate(S_i2t, d, keep_ranks), evaluate(S_t2i, d, keep_ranks)
    m = {k: v for k, v in a.items() if k.startswith("i2t")}
    m.update({k: v for k, v in b.items() if k.startswith("t2i")})
    m["avg_R10"] = 0.5 * (m["i2t_R10"] + m["t2i_R10"])
    m["avg_R1"] = 0.5 * (m["i2t_R1"] + m["t2i_R1"])
    if keep_ranks:
        m["_ranks"] = {"i2t": a["_ranks"]["i2t"], "t2i": b["_ranks"]["t2i"],
                       "i2t_relaxed": a["_ranks"]["i2t_relaxed"], "t2i_relaxed": b["_ranks"]["t2i_relaxed"]}
    return m


# ----------------------------------------------------------------------------- bootstrap
def hits_at_k(runs, direction, K=10, relaxed=False):
    """Per-query hit indicator averaged over seeds (runs = list of metric dicts with `_ranks`)."""
    key = direction + ("_relaxed" if relaxed else "")
    return np.mean([(m["_ranks"][key] <= K).astype(float) for m in runs], axis=0)


class ClusterBootstrap:
    """Resample test SCREENS with replacement; a screen's captions move with it."""

    def __init__(self, cap2img, n_boot=1000, seed=42, subset=None):
        self.cap2img = np.asarray(cap2img)
        self.n_img = int(self.cap2img.max()) + 1
        self.cnt = np.bincount(self.cap2img, minlength=self.n_img)
        ids = np.arange(self.n_img) if subset is None else np.asarray(subset)
        self.idx = np.random.default_rng(seed).choice(ids, size=(n_boot, len(ids)))
        self.subset = ids

    def point(self, hit_i, hit_t):
        ht = np.bincount(self.cap2img, weights=hit_t, minlength=self.n_img)
        return hit_i[self.subset].mean() * 100, ht[self.subset].sum() / self.cnt[self.subset].sum() * 100

    def samples(self, hit_i, hit_t):
        """Bootstrap distribution of (screen->text, text->screen) Recall in %."""
        ht = np.bincount(self.cap2img, weights=hit_t, minlength=self.n_img)
        return hit_i[self.idx].mean(1) * 100, ht[self.idx].sum(1) / self.cnt[self.idx].sum(1) * 100


def ci95(v):
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
