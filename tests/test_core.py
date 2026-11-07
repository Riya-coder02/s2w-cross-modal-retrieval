"""Unit tests for the core maths (run with `pytest -q`; CPU only, no data download)."""
import numpy as np
import torch
import torch.nn.functional as F

from s2w.evaluation import ClusterBootstrap, ranks, rerank_scores
from s2w.models import EarlyFusion, multi_positive_infonce, ridge_fit
from s2w.splits import app_groups, assign_split
from s2w.data import clean_caps, norm


def test_ranks_match_brute_force_with_ties():
    rng = np.random.default_rng(0)
    for _ in range(100):
        ni, nc = rng.integers(2, 8), rng.integers(2, 15)
        S = torch.tensor(rng.integers(0, 4, (ni, nc)).astype(float))
        P = torch.tensor(rng.random((ni, nc)) < 0.3)
        for i in range(ni):
            if not P[i].any(): P[i, rng.integers(nc)] = True
        for j in range(nc):
            if not P[:, j].any(): P[rng.integers(ni), j] = True
        r_i, r_t = ranks(S, P)
        for i in range(ni):
            b, n = S[i][P[i]].max(), S[i][~P[i]]
            assert abs(r_i[i] - (1 + (n > b).sum() + 0.5 * (n == b).sum())) < 1e-6
        for j in range(nc):
            b, n = S[:, j][P[:, j]].max(), S[:, j][~P[:, j]]
            assert abs(r_t[j] - (1 + (n > b).sum() + 0.5 * (n == b).sum())) < 1e-6


def test_constant_scorer_gets_chance_rank():
    S, P = torch.zeros(4, 4), torch.eye(4, dtype=torch.bool)
    r_i, r_t = ranks(S, P)
    assert torch.allclose(r_i, torch.full((4,), 2.5)) and torch.allclose(r_t, torch.full((4,), 2.5))


def test_multi_positive_infonce_reduces_to_clip_loss():
    L = torch.randn(6, 6)
    ref = 0.5 * (F.cross_entropy(L, torch.arange(6)) + F.cross_entropy(L.T, torch.arange(6)))
    assert torch.allclose(multi_positive_infonce(L, torch.eye(6, dtype=torch.bool)), ref)


def test_extra_positive_lowers_loss():
    L = torch.zeros(3, 3); L[0, 1] = 5.0
    P = torch.eye(3, dtype=torch.bool)
    P2 = P.clone(); P2[0, 1] = True
    assert multi_positive_infonce(L, P2) < multi_positive_infonce(L, P)


def test_early_fusion_factorised_scores_equal_naive():
    m = EarlyFusion(20, 12, d=16).eval()
    XI, XT = torch.randn(7, 20), torch.randn(11, 12)
    naive = torch.stack([m(XI[i:i + 1].expand(11, -1), XT) for i in range(7)])
    assert torch.allclose(naive, m.score_matrix(XI, XT, chunk_pairs=20), atol=1e-5)


def test_ridge_recovers_linear_map():
    X = torch.randn(500, 8); W = torch.randn(8, 3)
    assert torch.allclose(ridge_fit(X, X @ W, 1e-6), W, atol=1e-3)


def test_rerank_keeps_top_k_above_rest():
    S_con, S_e = torch.randn(5, 9), torch.randn(5, 9)
    s_i2t, _ = rerank_scores(S_con, S_e, K=3, alpha=0.5)
    for i in range(5):
        top = set(S_con[i].topk(3).indices.tolist())
        assert set(s_i2t[i].topk(3).indices.tolist()) == top


def test_groups_never_cross_splits():
    app_index = np.array([0, 0, 1, 2, 3, 3, 4, 5])
    pairs = (np.array([0, 3]), np.array([2, 6]), np.array([0.99, 0.97]))   # links apps 0-1 and 2-4
    grp = app_groups(pairs, app_index, 0.95)[app_index]
    split = assign_split(grp, 0.25, 0.25, seed=0)
    for g in np.unique(grp):
        assert len(set(split[grp == g])) == 1


def test_bootstrap_point_equals_mean():
    cap2img = np.array([0, 0, 1, 2, 2, 2])
    bs = ClusterBootstrap(cap2img, n_boot=10)
    hi, ht = np.array([1.0, 0.0, 1.0]), np.array([1, 0, 1, 1, 1, 0], dtype=float)
    assert bs.point(hi, ht) == (hi.mean() * 100, ht.mean() * 100)


def test_caption_cleaning():
    assert norm("  Login-Page! ") == "login page"
    assert clean_caps(["login page", "Login  page", "", "home"]) == ["login page", "home"]
