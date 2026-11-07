"""s2w: cross-modal retrieval between mobile app screens and text summaries (RICO-Screen2Words).

Modules
-------
utils       seeding, timing, GPU helpers, the Reporter that writes results/tables/figures
data        loading RICO-Screen2Words, caption cleaning, caption index
features    frozen image (ResNet-50 / ViT-B/16) and text (BERT) feature extraction, image quality stats
splits      near-duplicate detection, app grouping, leakage-aware train/val/test split, leakage report
tensors     per-split tensors, train-only standardisation, positive-pair masks
models      ridge baseline, early-fusion MLP, contrastive dual encoder, multi-positive InfoNCE
training    mini-batch construction, losses with false-negative masking, training loop with early stopping
evaluation  retrieval metrics (Recall@K, MRR, median rank, relaxed), re-ranking, cluster bootstrap
"""
__version__ = "1.0.0"
