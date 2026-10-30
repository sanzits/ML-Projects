# Project Context: Is RL Really Better Than A/B Testing?

## Problem statement

Does a reinforcement-learning (contextual) policy beat A/B-style fixed policies when both are
evaluated offline on logged recommendation data? We cannot rerun the past, so the question is:

> If a different decision policy had been used when the data was logged, would users have clicked more?

This is a counterfactual (off-policy) evaluation problem. A published article ("Is Reinforcement
Learning truly the better option than A/B testing?") reports RL at 0.93 vs 0.22 (uniform A/B) and
0.11 (popularity A/B). That comparison is not like-for-like (see "Method 3"), so this project
re-runs the experiment with sound estimators and uncertainty, and reproduces the original numbers.

## Data: Open Bandit Dataset (ZOZO, "all" campaign)

- Public logged bandit feedback from a fashion e-commerce site. Local file `all.csv`: about 12.36M rows (6.3 GB).
- 80 items, 3 display positions. Click rate is about 0.5%.
- Columns (89 after dropping the index):
  - `timestamp`, `item_id` (action shown), `position`
  - `click` (reward, 0/1)
  - `propensity_score` (probability the logging policy showed this item at this moment)
  - `user_feature_0..3` (hashed categorical user/context attributes)
  - `user-item_affinity_0..79` (precomputed user-item relevance; column k corresponds to item k)
- Samples used (random_state=42): 500k rows (reproduces the article), 150k rows (experiments).

## Methods

### 1. A/B-style fixed policies (non-contextual baselines)
- **Uniform**: pick an item at random (pi_e = 1/80).
- **Popularity**: always show the item with the best train CTR (deterministic), or sample items by historical frequency (article version).

### 2. Contextual / RL-style policies (use the user context)
- **LinUCB**: per-item ridge models on user one-hot + affinity features, trained with IPS-weighted updates, played greedily.
- **Logistic CTR**: one logistic regression P(click | context, item), then pick the argmax item per user.

### 3. The article's "Contextual RL" (reproduced, but flawed)
Keeps rows where the sum of affinity scores is greater than 0 and averages click/propensity over them.
No item is chosen, so it is a row filter, not a policy. Its denominator (about 3% self-selected rows)
differs from the A/B policies (about 1 in 80 matched rows), so the headline gap is not meaningful.
The A/B numbers also come from unseeded random draws and vary widely (uniform A/B: 0.02 to 2.2 across seeds).

### 4. Evaluation protocol for the fair comparison
- 70/30 train/test split (seed 42); evaluate on test only, over all rows.
- **SNIPS**: sum(click * w / pi_b) / sum(w / pi_b), with w = pi_e(logged item | context). Deterministic policies get w = 1 when their pick equals the logged item, else 0.
- Also report plain IPS, effective sample size, and a 1000-sample bootstrap 95% CI.

## Findings so far

| Setup | Uniform A/B | Popularity A/B | Contextual RL |
|---|---|---|---|
| Article, reproduced (500k rows, biased method) | 0.246 +/- 0.17 (article draw: 0.2245) | 0.106 (article: 0.1055) | 0.9257 (exact) |
| SNIPS on test split (150k rows) | 0.0022 | 0.0051 (CI includes 0) | 0.0024 (logistic), 0.0025 (LinUCB) |

Under a consistent estimator, all policies are about 0.002 to 0.005 with overlapping confidence
intervals, so there is no clear winner. Effective sample sizes are small (about 20 for three policies,
about 500 for logistic CTR).

## Next steps / ideas
- Use the full 12M rows, or stochastic policies, to raise effective sample size.
- Add doubly-robust estimation.

## Files (in `obd_experiments/`)
- `make_sample.py` builds the 150k sample.
- `policy_experiments.py` runs the four policies with SNIPS/IPS, ESS and bootstrap CIs.
- `results/` holds one JSON per policy.
- `sample_150k.csv` and `sample_500k.csv` are the samples.
