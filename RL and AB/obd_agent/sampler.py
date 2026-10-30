"""Stream a uniform random sample of rows out of the large Open Bandit all.csv."""
import json
import os

import numpy as np
import pandas as pd


def count_rows(src, cache_path=None):
    """Row count of src, cached by (size, mtime) so the 6 GB file is scanned only once."""
    st = os.stat(src)
    key = {"size": st.st_size, "mtime": st.st_mtime}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            c = json.load(f)
        if c.get("key") == key:
            return c["n"]
    n = sum(len(c) for c in pd.read_csv(src, usecols=[0], chunksize=2_000_000))
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"key": key, "n": n}, f)
    return n


def draw_indices(n_total, n_sample, seed):
    return np.sort(np.random.RandomState(seed).choice(n_total, n_sample, replace=False))


def draw_sample(src, n_sample, seed, cache_path=None, chunksize=1_000_000):
    """Return (DataFrame, total_rows). Same seed + same file => identical sample."""
    n_total = count_rows(src, cache_path)
    keep = draw_indices(n_total, min(n_sample, n_total), seed)
    parts, offset = [], 0
    for chunk in pd.read_csv(src, index_col=0, chunksize=chunksize):
        lo, hi = offset, offset + len(chunk)
        idx = keep[(keep >= lo) & (keep < hi)] - lo
        parts.append(chunk.iloc[idx])
        offset = hi
    return pd.concat(parts, ignore_index=True), n_total
