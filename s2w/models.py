"""Retrieval models: ridge baseline, early-fusion MLP (joint representation), contrastive dual encoder
(coordinated representation) and the multi-positive InfoNCE loss."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- ridge baseline
def ridge_fit(XT, Y, alpha):
    """Closed-form ridge regression W = (X^T X + alpha I)^-1 X^T Y mapping caption features to image features."""
    A, Y = XT.double(), Y.double()
    eye = torch.eye(A.shape[1], device=A.device, dtype=A.dtype)
    return torch.linalg.solve(A.T @ A + alpha * eye, A.T @ Y).float()


def ridge_scores(W, d):
    """Cosine similarity between every screen and every caption's predicted image features."""
    return F.normalize(d["XI"], dim=-1) @ F.normalize(d["XT"] @ W, dim=-1).T


# ----------------------------------------------------------------------------- neural models
def mlp(din, dh, dout, p):
    return nn.Sequential(nn.Linear(din, dh), nn.GELU(), nn.Dropout(p), nn.Linear(dh, dout))


class EarlyFusion(nn.Module):
    """Project both modalities, concatenate, score the pair with an MLP (one logit per screen-caption pair)."""

    def __init__(self, d_img, d_txt, d=512, p=0.3, p_in=0.0):
        super().__init__()
        self.pi = nn.Sequential(nn.Dropout(p_in), nn.Linear(d_img, d), nn.GELU(), nn.Dropout(p))
        self.pt = nn.Sequential(nn.Dropout(p_in), nn.Linear(d_txt, d), nn.GELU(), nn.Dropout(p))
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(p), nn.Linear(d, 1))

    def forward(self, xi, xt):
        return self.head(torch.cat([self.pi(xi), self.pt(xt)], -1)).squeeze(-1)

    @torch.no_grad()
    def score_matrix(self, XI, XT, chunk_pairs=5e5):
        """All-pairs scores. The first head layer is split as W[x_i; x_t] = W_i x_i + W_t x_t, which is exact
        but avoids materialising every concatenated pair."""
        self.eval()
        hi, ht = self.pi(XI), self.pt(XT)
        d = hi.shape[1]
        L1, L2 = self.head[0], self.head[3]
        dt = torch.float16 if XI.is_cuda else torch.float32
        A = (hi @ L1.weight[:, :d].T + L1.bias).to(dt)
        B = (ht @ L1.weight[:, d:].T).to(dt)
        w2, b2 = L2.weight[0].to(dt), L2.bias[0]
        step, out = max(1, int(chunk_pairs // len(B))), []
        for a in range(0, len(A), step):
            h = F.gelu(A[a:a + step, None, :] + B[None, :, :])
            out.append((h @ w2).float() + b2)
        return torch.cat(out)


class DualEncoder(nn.Module):
    """Separate projection heads into a shared L2-normalised space; similarity = scaled dot product (CLIP-style)."""

    def __init__(self, d_img, d_txt, d=512, out=256, p=0.3, p_in=0.0):
        super().__init__()
        self.fi = nn.Sequential(nn.Dropout(p_in), mlp(d_img, d, out, p))
        self.ft = nn.Sequential(nn.Dropout(p_in), mlp(d_txt, d, out, p))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))   # learnable temperature

    def enc_i(self, x):
        return F.normalize(self.fi(x), dim=-1)

    def enc_t(self, x):
        return F.normalize(self.ft(x), dim=-1)

    def logits(self, xi, xt):
        return self.logit_scale.exp().clamp(max=100.0) * self.enc_i(xi) @ self.enc_t(xt).T

    @torch.no_grad()
    def score_matrix(self, XI, XT):
        self.eval()
        return self.enc_i(XI) @ self.enc_t(XT).T


def multi_positive_infonce(logits, P):
    """Symmetric InfoNCE where a row/column may have several positives (boolean mask P):
    loss = -log( sum_pos exp(s) / sum_all exp(s) ), averaged over both directions (MIL-NCE style).
    With P = identity this is exactly the standard CLIP loss."""
    logits = logits.float()
    pos = logits.masked_fill(~P, torch.finfo(logits.dtype).min)
    l_i2t = (logits.logsumexp(1) - pos.logsumexp(1)).mean()
    l_t2i = (logits.logsumexp(0) - pos.logsumexp(0)).mean()
    return 0.5 * (l_i2t + l_t2i)
