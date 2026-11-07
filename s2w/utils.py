"""Seeding, timing, GPU helpers and the Reporter that stores every reported number, table and figure."""
import json
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def timer():
    """Wall-clock timer that synchronises the GPU. Usage: `with timer() as t: ...; t["s"]`."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    box = {}
    t0 = time.perf_counter()
    try:
        yield box
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        box["s"] = time.perf_counter() - t0


def gpu_reset() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def gpu_peak_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else float("nan")


def amp():
    """Mixed-precision context on GPU, no-op on CPU."""
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available())


def to_native(x):
    """Convert NumPy / torch values to JSON-serialisable Python values (NaN/inf become None)."""
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 4)
    if isinstance(x, torch.Tensor):
        return to_native(x.detach().cpu().numpy())
    if isinstance(x, np.ndarray):
        return to_native(x.tolist())
    if isinstance(x, (list, tuple)):
        return [to_native(v) for v in x]
    if isinstance(x, dict):
        return {str(k): to_native(v) for k, v in x.items()}
    return x


class Reporter:
    """Collects results and writes them to disk.

    results/results.json   every number recorded with `rec`
    results/tables/*.csv   every table saved with `table`
    results/figures/*.png  every figure saved with `fig`
    private_outputs/*.png  figures containing screenshots (never committed)
    """

    def __init__(self, out_dir="results", private_dir="private_outputs"):
        self.out = Path(out_dir)
        self.tab = self.out / "tables"
        self.figdir = self.out / "figures"
        self.priv = Path(private_dir)
        for d in (self.tab, self.figdir, self.priv):
            d.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def rec(self, key, value, show=True):
        self.results[key] = to_native(value)
        if show:
            print(f"[result] {key} = {self.results[key]}")

    def table(self, df, name, caption, show=True):
        df.to_csv(self.tab / f"{name}.csv")
        if show:
            print(caption)
            try:
                from IPython.display import display
                display(df)
            except ImportError:
                print(df)

    def fig(self, fig, name, private=False):
        import matplotlib.pyplot as plt
        path = (self.priv if private else self.figdir) / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        return path

    def save_json(self, name="results.json"):
        with open(self.out / name, "w") as f:
            json.dump(self.results, f, indent=2)
        return self.out / name
