"""Frozen feature extraction: ResNet-50 or ViT-B/16 for screens, BERT for captions, plus image quality stats."""
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .utils import DEVICE, amp

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def image_encoder(name: str):
    """Return (frozen backbone without classifier, feature dim). Both use ImageNet weights from torchvision."""
    from torchvision import models
    if name == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        net.fc = nn.Identity()
        return net, 2048
    if name == "vit_b_16":
        net = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        net.heads = nn.Identity()           # [CLS] token embedding
        return net, 768
    raise ValueError(f"unknown image encoder {name}")


def transform(img_size=224):
    """Resize to a square WITHOUT cropping (screens are tall; cropping would cut navigation bars and buttons)."""
    from torchvision import transforms as T
    return T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def dhash(img: Image.Image, size: int = 8) -> int:
    """64-bit difference hash: identical for exact/near-exact duplicate images."""
    a = np.asarray(img.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    return int("".join("1" if b else "0" for b in (a[:, 1:] > a[:, :-1]).flatten()), 2)


class ScreenImages(Dataset):
    """Decodes screenshots; optionally also returns quality stats (size, pixel std, dhash)."""

    def __init__(self, ds_img, rows, img_col="image", img_size=224, with_quality=False):
        self.ds_img, self.rows, self.col = ds_img, [int(r) for r in rows], img_col
        self.tfm, self.q = transform(img_size), with_quality

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        try:
            im = self.ds_img[self.rows[i]][self.col].convert("RGB")
            if not self.q:
                return self.tfm(im), True, 0, 0, 0.0, "0"
            g = np.asarray(im.convert("L").resize((64, 64)), dtype=np.float32)
            return self.tfm(im), True, im.size[0], im.size[1], float(g.std()), str(dhash(im))
        except Exception:
            return torch.zeros(3, 224, 224), False, 0, 0, 0.0, "0"


@torch.no_grad()
def extract_image_features(ds_img, rows, encoder="resnet50", batch_size=64, num_workers=2, with_quality=False):
    """Encode every screen with a frozen backbone. Returns (features float32 [n, d], quality DataFrame)."""
    net, _ = image_encoder(encoder)
    net = net.to(DEVICE).eval()
    dl = DataLoader(ScreenImages(ds_img, rows, with_quality=with_quality), batch_size=batch_size,
                    num_workers=num_workers, pin_memory=(DEVICE == "cuda"))
    feats, info = [], []
    for x, ok, w, h, gstd, dh in tqdm(dl, desc=encoder):
        with amp():
            f = net(x.to(DEVICE, non_blocking=True))
        feats.append(f.float().cpu())
        info += list(zip(ok.tolist(), w.tolist(), h.tolist(), gstd.tolist(), dh))
    del net
    gc.collect()
    return (torch.cat(feats).numpy().astype(np.float32),
            pd.DataFrame(info, columns=["ok", "w", "h", "gray_std", "dhash"]))


@torch.no_grad()
def extract_text_features(texts, tokenizer, max_len=16, batch_size=512, model_name="bert-base-uncased"):
    """Frozen BERT: returns (mean-pooled, [CLS]) features, each float32 [n, 768]."""
    from transformers import AutoModel
    bert = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
    mean_f, cls_f = [], []
    for a in tqdm(range(0, len(texts), batch_size), desc="BERT"):
        enc = tokenizer(texts[a:a + batch_size], padding=True, truncation=True, max_length=max_len,
                        return_tensors="pt").to(DEVICE)
        with amp():
            h = bert(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        mean_f.append(((h * m).sum(1) / m.sum(1)).float().cpu())
        cls_f.append(h[:, 0].float().cpu())
    del bert
    gc.collect()
    return torch.cat(mean_f).numpy().astype(np.float32), torch.cat(cls_f).numpy().astype(np.float32)
