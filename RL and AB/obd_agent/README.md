# OBD experiment agent

One command: draws a **fresh random sample** of the Open Bandit Dataset (`all.csv`), runs
the A/B-style policies (uniform, popularity) and the contextual/RL-style policies (LinUCB,
logistic CTR) with off-policy evaluation (SNIPS, IPS, ESS, bootstrap 95% CI), writes a
report that compares against all previous runs, and emails it.

Context: see `../PROJECT_CONTEXT.md`. `legacy/` holds the original fixed-seed scripts.

## Setup
```
pip install -r requirements.txt
cp .env.example .env     # fill in SMTP settings (gitignored)
```
`all.csv` (6.3 GB) is not in git. By default it is read from `../all.csv`; override with
`--data` or `OBD_DATA`.

## Run
```
python run_agent.py                 # new random seed, 500,000 rows, emails the report
python run_agent.py --no-email      # save only
python run_agent.py --seed 123      # reproduce a run (seed is printed in every report)
python run_agent.py --sample-size 200000 --n-boot 500
```
The first run scans `all.csv` once to count rows (cached in `runs/.row_count.json`); a run
takes ~1 minute after that. Runs are manual only (no scheduler).

## Output
`runs/<run_id>/` holds `run.json` (seed, sample size, per-policy metrics), `report.md` and
`report.html` and `chart.png` (SNIPS with CIs for this run, plus SNIPS across runs); the
chart is embedded in the email. Every new report includes mean/std/min/max SNIPS across earlier runs, a
z-score of this run vs. history, and each run's ranking, so you can see whether the
winner is stable or just sampling noise. `runs/` and `.env` are gitignored.

## Tests
`pytest tests` (uses a small synthetic CSV).
