import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import report  # noqa: E402
import sampler  # noqa: E402
from experiment import run_experiment  # noqa: E402


def make_csv(path, n=3000, seed=0):
    rng = np.random.RandomState(seed)
    d = {"timestamp": range(n), "item_id": rng.randint(0, 5, n), "position": 1,
         "click": rng.binomial(1, 0.05, n), "propensity_score": np.full(n, 0.2)}
    for i in range(2):
        d[f"user_feature_{i}"] = rng.randint(0, 3, n)
    for i in range(5):
        d[f"user-item_affinity_{i}"] = rng.randn(n)
    pd.DataFrame(d).to_csv(path)


def test_sampler_reproducible(tmp_path):
    p = tmp_path / "all.csv"
    make_csv(p)
    a, n = sampler.draw_sample(str(p), 500, 7)
    b, _ = sampler.draw_sample(str(p), 500, 7)
    c, _ = sampler.draw_sample(str(p), 500, 8)
    assert n == 3000 and len(a) == 500
    assert a.equals(b) and not a.equals(c)


def test_experiment_and_report(tmp_path):
    p = tmp_path / "all.csv"
    make_csv(p)
    df, total = sampler.draw_sample(str(p), 2000, 1)
    res = run_experiment(df, 1, n_boot=50)
    assert [r["policy"] for r in res] == ["uniform", "popularity", "linucb", "logistic_ctr"]
    run = dict(run_id="r1", seed=1, sample_size=2000, total_rows=total, click_rate=0.05,
               n_boot=50, runtime_sec=1.0, results=res)
    md = report.build_markdown(run, [])
    assert "first one" in md and "seed" in md.lower()
    md2 = report.build_markdown(run, [dict(run, run_id="r0", seed=0)])
    assert "Prev mean" in md2
    assert "<table" in report.to_html(md2)
