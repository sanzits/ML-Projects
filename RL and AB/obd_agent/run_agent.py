"""One-command run: fresh random sample -> experiment -> report -> email.

    python run_agent.py                      # new random seed, 500k rows, emails the report
    python run_agent.py --no-email           # save report to disk only
    python run_agent.py --seed 123           # reproduce a previous run
"""
import argparse
import json
import os
import secrets
import time
from datetime import datetime, timezone

import report
import sampler
from emailer import load_env, send_report
from experiment import N_BOOT, run_experiment

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "..", "all.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("OBD_DATA", DEFAULT_DATA))
    ap.add_argument("--sample-size", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=None, help="default: fresh random seed")
    ap.add_argument("--runs-dir", default=os.path.join(HERE, "runs"))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--no-email", action="store_true", help="save the report, do not send")
    a = ap.parse_args()

    load_env(os.path.join(HERE, ".env"))
    seed = a.seed if a.seed is not None else secrets.randbelow(2**31 - 1)
    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%SZ") + f"_s{seed}"
    t0 = time.time()

    print(f"[{run_id}] drawing {a.sample_size:,} rows (seed {seed}) from {a.data}")
    df, total = sampler.draw_sample(a.data, a.sample_size, seed,
                                    cache_path=os.path.join(a.runs_dir, ".row_count.json"))
    print(f"sample {df.shape}; split 70/30 train/test, evaluating on the test split")
    results = run_experiment(df, seed, a.n_boot)

    run = dict(run_id=run_id, seed=seed, sample_size=len(df), total_rows=total,
               click_rate=float(df.click.mean()), n_boot=a.n_boot,
               started_at=started.isoformat(), runtime_sec=time.time() - t0, results=results)
    history = report.load_history(a.runs_dir, exclude_run_id=run_id)
    md = report.build_markdown(run, history)

    out = os.path.join(a.runs_dir, run_id)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "run.json"), "w") as f:
        json.dump(run, f, indent=2)
    chart = os.path.join(out, "chart.png")
    report.make_chart(run, history, chart)
    with open(os.path.join(out, "report.md"), "w") as f:
        f.write(md)
    html_body = report.to_html(md)
    with open(os.path.join(out, "report.html"), "w") as f:
        f.write(html_body)
    print(md)
    print(f"saved to {out}")

    if a.no_email:
        print("--no-email: skipping email")
    else:
        send_report(f"OBD experiment report {run_id}", md, html_body,
                    [os.path.join(out, "run.json")], inline_image=chart)
        print(f"emailed to {os.environ['EMAIL_TO']}")


if __name__ == "__main__":
    main()
