"""Stream the large Open Bandit all.csv and write a reproducible 150k-row sample (seed 42)."""
import sys
import numpy as np
import pandas as pd

src = sys.argv[1]
dst = sys.argv[2]
N = 150_000

n = sum(len(c) for c in pd.read_csv(src, usecols=[0], chunksize=2_000_000))
keep = np.sort(np.random.RandomState(42).choice(n, N, replace=False))
parts, offset = [], 0
for chunk in pd.read_csv(src, index_col=0, chunksize=1_000_000):
    lo, hi = offset, offset + len(chunk)
    idx = keep[(keep >= lo) & (keep < hi)] - lo
    parts.append(chunk.iloc[idx])
    offset = hi
df = pd.concat(parts, ignore_index=True)
df.to_csv(dst, index=False)
print(n, df.shape)
