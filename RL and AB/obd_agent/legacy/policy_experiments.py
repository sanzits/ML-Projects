"""A/B-style fixed policies vs contextual (RL-style) policies on the Open Bandit Dataset.

All policies share one protocol: the 150k-row sample (seed 42), a 70/30 train/test split
(seed 42), and off-policy evaluation on the test split over ALL rows with SNIPS
(value = sum(click*w/pi_b) / sum(w/pi_b), w = pi_e(logged item | context)), plus plain IPS,
effective sample size and a 1000-sample bootstrap 95% CI (seed 42).
Deterministic policies have w = 1 when their chosen item equals the logged item, else 0.
Position is ignored: each row is treated as one item recommendation.

Usage: python policy_experiments.py [--data sample_150k.csv] [--out results]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
N_BOOT = 1000


def snips_stats(click, w, pb):
    iw = w / pb
    return iw.sum(), (click * iw).sum(), iw


def evaluate(name, click, w, pb, notes):
    iw_sum, num, iw = snips_stats(click, w, pb)
    snips = num / iw_sum if iw_sum > 0 else float("nan")
    ips = num / len(click)
    ess = float(iw.sum() ** 2 / (iw ** 2).sum()) if iw_sum > 0 else 0.0
    rng = np.random.RandomState(SEED)
    n = len(click)
    boots = []
    for _ in range(N_BOOT):
        i = rng.randint(0, n, n)
        s = iw[i].sum()
        if s > 0:
            boots.append((click[i] * iw[i]).sum() / s)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(policy=name, snips=float(snips), ips=float(ips), ci_low=float(lo),
                ci_high=float(hi), ess=ess, n_test=int(n),
                n_matched=int((w > 0).sum()), notes=notes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "sample_150k.csv"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = pd.read_csv(a.data)
    tr, te = train_test_split(df, test_size=0.3, random_state=SEED)
    tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
    items = np.sort(df.item_id.unique())
    n_items = len(items)
    item_idx = {it: k for k, it in enumerate(items)}
    aff_cols = [c for c in df.columns if c.startswith("user-item_affinity_")]
    user_cols = [c for c in df.columns if c.startswith("user_feature_")]

    click_te = te.click.values.astype(float)
    pb_te = te.propensity_score.values
    logged_te = te.item_id.map(item_idx).values

    results = []

    # 1. Uniform random A/B baseline: pi_e = 1/n_items for every item.
    results.append(evaluate("uniform", click_te, np.full(len(te), 1.0 / n_items), pb_te,
                            "pi_e = 1/n_items"))

    # 2. Popularity A/B arm: always the item with the best train CTR.
    best = tr.groupby("item_id").click.mean().idxmax()
    results.append(evaluate("popularity", click_te, (te.item_id.values == best).astype(float),
                            pb_te, f"always item {best} (best train CTR)"))

    # Shared contextual features: one-hot user features + the 80-dim affinity vector.
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(tr[user_cols])
    sc = StandardScaler().fit(tr[aff_cols])

    def context(d):
        return np.hstack([enc.transform(d[user_cols]), sc.transform(d[aff_cols])])

    X_tr, X_te = context(tr), context(te)

    # 3. LinUCB (disjoint, numpy): IPS-weighted ridge updates per item, greedy at test.
    d = X_tr.shape[1] + 1
    alpha, lam = 0.0, 1.0
    Xb_tr = np.hstack([X_tr, np.ones((len(X_tr), 1))])
    Xb_te = np.hstack([X_te, np.ones((len(X_te), 1))])
    theta = np.zeros((n_items, d))
    a_tr = tr.item_id.map(item_idx).values
    wts = np.minimum(1.0 / tr.propensity_score.values, 50.0)
    for k in range(n_items):
        m = a_tr == k
        if m.sum() == 0:
            continue
        Xk, rk, wk = Xb_tr[m], tr.click.values[m], wts[m]
        A = lam * np.eye(d) + (Xk * wk[:, None]).T @ Xk
        b = (Xk * (wk * rk)[:, None]).sum(0)
        theta[k] = np.linalg.solve(A, b)
    choice = (Xb_te @ theta.T).argmax(1)
    results.append(evaluate("linucb", click_te, (choice == logged_te).astype(float), pb_te,
                            "disjoint LinUCB (greedy, alpha=0), IPS weights clipped at 50, "
                            "greedy deterministic pi_e"))

    # 4. Logistic CTR model: P(click | context, item), greedy over items.
    # Item-specific affinity (column k <-> item k) is used as a feature.
    def stacked(Xc, aff_raw, item_ids):
        oh = np.zeros((len(item_ids), n_items))
        oh[np.arange(len(item_ids)), item_ids] = 1
        own = aff_raw[np.arange(len(item_ids)), item_ids][:, None]
        return np.hstack([Xc, oh, own])

    aff_tr = tr[aff_cols].values
    aff_te = te[aff_cols].values
    aff_scaled_tr = sc.transform(tr[aff_cols])
    aff_scaled_te = sc.transform(te[aff_cols])
    Xs_tr = stacked(X_tr, aff_scaled_tr, a_tr)
    clf = LogisticRegression(C=0.1, max_iter=500, class_weight=None)
    clf.fit(Xs_tr, tr.click.values)
    scores = np.empty((len(te), n_items))
    for k in range(n_items):
        scores[:, k] = clf.predict_proba(stacked(X_te, aff_scaled_te, np.full(len(te), k)))[:, 1]
    choice = scores.argmax(1)
    results.append(evaluate("logistic_ctr", click_te, (choice == logged_te).astype(float), pb_te,
                            "logistic regression (C=0.1) on user one-hot + affinity + item "
                            "one-hot + item's own affinity; greedy argmax over items"))

    for r in results:
        with open(os.path.join(a.out, f"results_{r['policy']}.json"), "w") as f:
            json.dump(r, f, indent=2)
        print(f"{r['policy']:13s} SNIPS={r['snips']:.5f} CI=[{r['ci_low']:.5f},{r['ci_high']:.5f}] "
              f"IPS={r['ips']:.5f} ESS={r['ess']:.1f} matched={r['n_matched']}")


if __name__ == "__main__":
    main()
