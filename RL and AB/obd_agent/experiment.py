"""A/B-style fixed policies vs contextual (RL-style) policies, evaluated off-policy.

Protocol (adapted from legacy/policy_experiments.py): 70/30 train/test split, SNIPS
(sum(click*w/pi_b) / sum(w/pi_b), w = pi_e(logged item | context)), plain IPS, effective
sample size and a bootstrap 95% CI. Every random step is driven by `seed`.
Deterministic policies have w = 1 when their chosen item equals the logged item, else 0.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

N_BOOT = 1000


def evaluate(name, click, w, pb, notes, seed, n_boot=N_BOOT):
    iw = w / pb
    iw_sum, num = iw.sum(), (click * iw).sum()
    snips = num / iw_sum if iw_sum > 0 else float("nan")
    ips = num / len(click)
    ess = float(iw_sum ** 2 / (iw ** 2).sum()) if iw_sum > 0 else 0.0
    rng = np.random.RandomState(seed)
    n = len(click)
    boots = []
    for _ in range(n_boot):
        i = rng.randint(0, n, n)
        s = iw[i].sum()
        if s > 0:
            boots.append((click[i] * iw[i]).sum() / s)
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"),) * 2
    return dict(policy=name, snips=float(snips), ips=float(ips), ci_low=float(lo),
                ci_high=float(hi), ess=ess, n_test=int(n),
                n_matched=int((w > 0).sum()), notes=notes)


DESCRIPTIONS = {
    "uniform": "A/B fixed baseline: shows every item with equal probability, ignoring the user.",
    "popularity": "A/B fixed arm: always shows the single item with the best click rate in training data.",
    "linucb": "Contextual/RL: LinUCB, a per-item linear model of user features that picks the item with the highest predicted click value.",
    "logistic_ctr": "Contextual/RL: logistic-regression click model on user + item features; picks the item with the highest predicted click probability.",
}


def run_experiment(df, seed, n_boot=N_BOOT, log=print):
    step = [0]

    def announce(name):
        step[0] += 1
        log(f"[{step[0]}/4] Evaluating policy: {name} -- {DESCRIPTIONS[name]}")

    tr, te = train_test_split(df, test_size=0.3, random_state=seed)
    tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
    items = np.sort(df.item_id.unique())
    n_items = len(items)
    item_idx = {it: k for k, it in enumerate(items)}
    aff_cols = [c for c in df.columns if c.startswith("user-item_affinity_")]
    user_cols = [c for c in df.columns if c.startswith("user_feature_")]

    click_te = te.click.values.astype(float)
    pb_te = te.propensity_score.values
    logged_te = te.item_id.map(item_idx).values
    a_tr = tr.item_id.map(item_idx).values
    results = []

    announce("uniform")
    results.append(evaluate("uniform", click_te, np.full(len(te), 1.0 / n_items), pb_te,
                            "pi_e = 1/n_items", seed, n_boot))

    announce("popularity")
    best = tr.groupby("item_id").click.mean().idxmax()
    results.append(evaluate("popularity", click_te, (te.item_id.values == best).astype(float),
                            pb_te, f"always item {best} (best train CTR)", seed, n_boot))

    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(tr[user_cols])
    sc = StandardScaler().fit(tr[aff_cols])

    def context(d):
        return np.hstack([enc.transform(d[user_cols]), sc.transform(d[aff_cols])])

    X_tr, X_te = context(tr), context(te)

    announce("linucb")
    # LinUCB (disjoint): IPS-weighted ridge per item, greedy (alpha=0) at test.
    d = X_tr.shape[1] + 1
    lam = 1.0
    Xb_tr = np.hstack([X_tr, np.ones((len(X_tr), 1))])
    Xb_te = np.hstack([X_te, np.ones((len(X_te), 1))])
    theta = np.zeros((n_items, d))
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
                            "greedy deterministic pi_e", seed, n_boot))

    announce("logistic_ctr")
    # Logistic CTR: P(click | context, item), greedy over items.
    aff_s_tr, aff_s_te = sc.transform(tr[aff_cols]), sc.transform(te[aff_cols])

    def stacked(Xc, aff, item_ids):
        oh = np.zeros((len(item_ids), n_items))
        oh[np.arange(len(item_ids)), item_ids] = 1
        own = aff[np.arange(len(item_ids)), item_ids][:, None]
        return np.hstack([Xc, oh, own])

    clf = LogisticRegression(C=0.1, max_iter=500)
    clf.fit(stacked(X_tr, aff_s_tr, a_tr), tr.click.values)
    scores = np.empty((len(te), n_items))
    for k in range(n_items):
        scores[:, k] = clf.predict_proba(
            stacked(X_te, aff_s_te, np.full(len(te), k)))[:, 1]
    choice = scores.argmax(1)
    results.append(evaluate("logistic_ctr", click_te, (choice == logged_te).astype(float),
                            pb_te, "logistic regression (C=0.1) on user one-hot + affinity + "
                            "item one-hot + item's own affinity; greedy argmax",
                            seed, n_boot))
    return results
