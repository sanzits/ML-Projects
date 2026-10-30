"""Build the Markdown/HTML report for a run, including variability vs. previous runs."""
import glob
import html
import json
import os

import numpy as np

KIND = {"uniform": "A/B fixed", "popularity": "A/B fixed",
        "linucb": "contextual/RL", "logistic_ctr": "contextual/RL"}


def load_history(runs_dir, exclude_run_id=None):
    out = []
    for p in sorted(glob.glob(os.path.join(runs_dir, "*", "run.json"))):
        with open(p) as f:
            r = json.load(f)
        if r["run_id"] != exclude_run_id:
            out.append(r)
    return out


def ranking(results):
    return [r["policy"] for r in sorted(results, key=lambda r: -np.nan_to_num(r["snips"], nan=-1))]


def overlap(results):
    """True if every pair of policy CIs overlaps."""
    return all(a["ci_low"] <= b["ci_high"] and b["ci_low"] <= a["ci_high"]
               for a in results for b in results)


def history_stats(history, policies):
    stats = {}
    for p in policies:
        v = [r_["snips"] for h in history for r_ in h["results"] if r_["policy"] == p]
        if v:
            stats[p] = dict(n=len(v), mean=float(np.mean(v)),
                            std=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                            min=float(min(v)), max=float(max(v)))
    return stats


def build_markdown(run, history):
    res = run["results"]
    pols = [r["policy"] for r in res]
    L = [f"# OBD A/B vs RL experiment report — {run['run_id']}", "",
         f"- **Seed:** `{run['seed']}` (reproduce: `python run_agent.py --seed {run['seed']} "
         f"--sample-size {run['sample_size']}`)",
         f"- **Sample:** {run['sample_size']:,} rows drawn from {run['total_rows']:,} "
         f"({run['sample_size'] / run['total_rows']:.2%}); click rate {run['click_rate']:.4f}",
         f"- **Test split:** 30% ({res[0]['n_test']:,} rows); bootstrap resamples: {run['n_boot']}",
         f"- **Runtime:** {run['runtime_sec']:.0f}s", "",
         "## Results (this run)", "", "![SNIPS chart](chart.png)", "",
         "| Rank | Policy | Type | SNIPS | 95% CI | IPS | ESS | Matched rows |",
         "|---|---|---|---|---|---|---|---|"]
    by = {r["policy"]: r for r in res}
    for i, p in enumerate(ranking(res), 1):
        r = by[p]
        L.append(f"| {i} | {p} | {KIND.get(p, '')} | {r['snips']:.5f} | "
                 f"[{r['ci_low']:.5f}, {r['ci_high']:.5f}] | {r['ips']:.5f} | "
                 f"{r['ess']:.1f} | {r['n_matched']} |")
    L += ["", "### What the columns mean", "",
          "- **Policy / Type:** the strategy being evaluated. "A/B fixed" policies ignore the "
          "user (uniform random, or always the most popular item); "contextual/RL" policies "
          "pick an item per user using their features.",
          "- **SNIPS:** Self-Normalized Inverse Propensity Scoring, the estimated click rate "
          "the policy would achieve. Higher is better; the ranking is by this column.",
          "- **95% CI:** bootstrap confidence interval for SNIPS. Overlapping intervals mean "
          "the policies can't be told apart.",
          "- **IPS:** plain Inverse Propensity Scoring estimate (not self-normalized); "
          "noisier than SNIPS, shown as a cross-check.",
          "- **ESS:** effective sample size, i.e. how many logged rows effectively "
          "contribute after weighting. Low ESS (roughly under 100) means a noisy estimate.",
          "- **Matched rows:** test rows where the policy's chosen item equals the logged "
          "item, the only rows that carry weight for deterministic policies.", ""]
    for r in res:
        L.append(f"- `{r['policy']}`: {r['notes']}")

    L += ["", "## Comparison with previous runs", ""]
    if not history:
        L.append("No previous runs yet — this is the first one.")
    else:
        st = history_stats(history, pols)
        L += [f"Across {len(history)} previous run(s) (SNIPS):", "",
              "| Policy | Prev mean | Prev std | Prev min | Prev max | This run | z vs prev |",
              "|---|---|---|---|---|---|---|"]
        for p in pols:
            s = st.get(p)
            if not s:
                continue
            z = f"{(by[p]['snips'] - s['mean']) / s['std']:+.2f}" if s["std"] > 0 else "n/a"
            L.append(f"| {p} | {s['mean']:.5f} | {s['std']:.5f} | {s['min']:.5f} | "
                     f"{s['max']:.5f} | {by[p]['snips']:.5f} | {z} |")
        L += ["", "| Run | Seed | Ranking (best → worst) | CIs all overlap |", "|---|---|---|---|"]
        for h in history[-10:] + [run]:
            L.append(f"| {h['run_id']} | {h['seed']} | {' > '.join(ranking(h['results']))} | "
                     f"{'yes' if overlap(h['results']) else 'no'} |")
        first = [ranking(h["results"])[0] for h in history + [run]]
        L += ["", f"Top-ranked policy across all {len(first)} runs: " +
              ", ".join(f"{p} ×{first.count(p)}" for p in sorted(set(first))) +
              ". If the winner changes between fresh samples, the ranking is noise."]
    return "\n".join(L) + "\n"


def to_html(md):
    """Minimal Markdown -> HTML (headings, bullets, tables, bold, code)."""
    import re
    out, in_table, in_list = [], False, False
    for line in md.splitlines():
        e = html.escape(line)
        e = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", e)
        e = re.sub(r"`(.+?)`", r"<code>\1</code>", e)
        if line.startswith("|"):
            if set(line.replace("|", "").strip()) <= set("-"):
                continue
            cells = [c.strip() for c in e.strip("|").split("|")]
            tag = "th" if not in_table else "td"
            if not in_table:
                out.append("<table border='1' cellpadding='4' style='border-collapse:collapse'>")
            in_table = True
            out.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
            continue
        if in_table:
            out.append("</table>")
            in_table = False
        if line.startswith("- "):
            if not in_list:
                out.append("<ul>")
            in_list = True
            out.append(f"<li>{e[2:]}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        m = re.match(r"!\[(.*?)\]\((.*?)\)", line)
        if m:
            out.append(f"<p><img src='{m.group(2)}' alt='{m.group(1)}' style='max-width:100%'></p>")
        elif line.startswith("#"):
            n = len(line) - len(line.lstrip("#"))
            out.append(f"<h{n}>{e[n:].strip()}</h{n}>")
        elif line.strip():
            out.append(f"<p>{e}</p>")
    if in_table:
        out.append("</table>")
    if in_list:
        out.append("</ul>")
    return "<html><body style='font-family:sans-serif'>" + "\n".join(out) + "</body></html>"


def make_chart(run, history, path):
    """PNG: SNIPS with 95% CI for this run (left) and SNIPS per run across history (right)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = run["results"]
    names = [r["policy"] for r in res]
    colors = ["#7f7f7f" if KIND.get(p) == "A/B fixed" else "#1f77b4" for p in names]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1, 1.3]})
    snips = [r["snips"] for r in res]
    err = [[r["snips"] - r["ci_low"] for r in res], [r["ci_high"] - r["snips"] for r in res]]
    a1.bar(names, snips, yerr=err, capsize=5, color=colors)
    for i, r in enumerate(res):
        a1.text(i, r["ci_high"], f"{r['snips']:.4f}\nESS {r['ess']:.0f}", ha="center",
                va="bottom", fontsize=8)
    a1.set_ylabel("SNIPS policy value (95% bootstrap CI)")
    a1.set_title(f"This run (seed {run['seed']})\ngrey = A/B fixed, blue = contextual/RL")
    a1.set_ylim(0, max(r["ci_high"] for r in res) * 1.25)
    a1.grid(axis="y", alpha=0.3)
    a1.tick_params(axis="x", labelsize=8)

    runs = history + [run]
    for p in names:
        ys = [r_["snips"] for h in runs for r_ in h["results"] if r_["policy"] == p]
        a2.plot(range(1, len(ys) + 1), ys, marker="o", label=p)
    a2.set_xlabel("Run #")
    a2.set_ylabel("SNIPS")
    a2.set_title("SNIPS across runs (fresh random sample each)")
    a2.set_xticks(range(1, len(runs) + 1))
    a2.grid(alpha=0.3)
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
