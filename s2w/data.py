"""Loading RICO-Screen2Words, caption normalisation/cleaning and the caption index used everywhere else."""
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

IMG_COL, CAP_COL, APP_COL, CAT_COL, ID_COL = "image", "captions", "app_package_name", "category", "screenId"
HEAVY_COLUMNS = ["image_icon", "image_semantic", "view_hierarchy", "semantic_annotations"]


def load_screen2words(dataset_id="rootsautomation/RICO-Screen2Words"):
    """Load all provided splits as one table.

    Returns
    -------
    meta : DataFrame, one row per screen, without image columns or the large view-hierarchy strings
    ds_img : datasets.Dataset with only the screenshot column (decoded lazily, one image at a time)
    original_splits : dict of the dataset's own split sizes (not used for modelling)
    """
    from datasets import load_dataset, concatenate_datasets
    dsd = load_dataset(dataset_id)
    original = {k: len(v) for k, v in dsd.items()}
    ds = concatenate_datasets(list(dsd.values()))
    drop = [c for c in HEAVY_COLUMNS if c in ds.column_names]
    ds_img = ds.select_columns([IMG_COL])
    meta = ds.remove_columns(drop + [IMG_COL]).to_pandas()
    meta["caps_raw"] = meta[CAP_COL].map(lambda x: [str(t) for t in list(x)] if x is not None else [])
    meta["row"] = np.arange(len(meta))
    return meta, ds_img, original


def norm(s: str) -> str:
    """Lower-case, keep letters/digits/spaces, collapse whitespace. Used to detect identical captions."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", str(s).lower())).strip()


def clean_caps(captions):
    """Normalise whitespace, drop empty captions and duplicates (after `norm`) within one screen."""
    seen, out = set(), []
    for c in captions:
        c2 = re.sub(r"\s+", " ", str(c)).strip()
        k = norm(c2)
        if c2 and k and k not in seen:
            seen.add(k)
            out.append(c2)
    return out


def clean_category(series: pd.Series):
    """Category labels containing digits (e.g. '000 - 1', a download-count fragment) or missing become 'Unknown'."""
    bad = series.isna() | series.astype(str).str.contains(r"\d")
    return series.where(~bad, "Unknown"), bad


@dataclass
class CaptionIndex:
    """Flat view of all (cleaned) captions.

    caps            list of caption strings
    owner           caption -> screen row
    textid          caption -> id of its normalised text (identical captions share an id)
    screens_sharing caption -> number of distinct screens in the dataset with the same normalised text
    """
    caps: list
    owner: np.ndarray
    textid: np.ndarray
    screens_sharing: np.ndarray

    @property
    def n_text(self) -> int:
        return int(self.textid.max()) + 1


def build_caption_index(meta: pd.DataFrame) -> CaptionIndex:
    """Clean captions in place (adds `caps`, `n_caps`, `cap_start` to meta) and return the caption index."""
    meta["caps"] = meta["caps_raw"].map(clean_caps)
    meta["n_caps"] = meta["caps"].str.len()
    meta["cap_start"] = np.concatenate([[0], np.cumsum(meta["n_caps"].values)[:-1]])
    caps = [c for cl in meta["caps"] for c in cl]
    owner = np.repeat(np.arange(len(meta)), meta["n_caps"].values)
    normed = [norm(c) for c in caps]
    vocab = {t: i for i, t in enumerate(sorted(set(normed)))}
    textid = np.array([vocab[t] for t in normed])
    sharing = pd.Series(owner).groupby(textid).nunique().reindex(textid).values
    return CaptionIndex(caps, owner, textid, sharing)
