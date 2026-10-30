"""Local dashboard: SNIPS across runs + a Run button.

    python dashboard.py [--port 8765] [--data /path/to/all.csv]

Open http://127.0.0.1:8765. The page refreshes itself while a run is in progress.
"""
import argparse
import glob
import io
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import report

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(HERE, "runs")
STATE = {"proc": None, "log": [], "started": None}
LOCK = threading.Lock()

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>OBD experiment dashboard</title>
<style>body{font-family:sans-serif;margin:24px;max-width:1100px}img{max-width:100%}
table{border-collapse:collapse}td,th{border:1px solid #999;padding:4px 8px}
button{padding:8px 14px;font-size:14px}pre{background:#111;color:#ddd;padding:10px;
max-height:260px;overflow:auto;font-size:12px}</style></head><body>
<h2>OBD A/B vs RL: SNIPS across runs</h2>
<button id="run" onclick="start()">Run new experiment</button>
<label><input type="checkbox" id="email"> email the report</label>
<span id="status"></span>
<p><img id="chart" src="/chart.png" alt="SNIPS across runs"></p>
<h3>Latest run</h3><div id="latest"></div>
<h3>All runs</h3><div id="runs"></div>
<h3>Run log</h3><pre id="log">(no run started from this page)</pre>
<script>
async function refresh(force){
  const s = await (await fetch('/api/status')).json();
  document.getElementById('status').textContent = s.running ? ' running...' : '';
  document.getElementById('run').disabled = s.running;
  document.getElementById('log').textContent = s.log || '(no run started from this page)';
  const sig = s.n_runs + ':' + s.running;
  if (force || sig !== window._sig) {
    window._sig = sig;
    document.getElementById('chart').src = '/chart.png?' + Date.now();
    const r = await (await fetch('/api/runs')).json();
    document.getElementById('latest').innerHTML = r.latest_html;
    document.getElementById('runs').innerHTML = r.runs_html;
  }
}
async function start(){
  await fetch('/api/run', {method:'POST', body: JSON.stringify({email: document.getElementById('email').checked})});
  refresh(true);
}
refresh(true); setInterval(refresh, 3000);
</script></body></html>"""


def load_runs():
    return report.load_history(RUNS_DIR)


def chart_png(runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.5))
    if not runs:
        ax.text(0.5, 0.5, "No runs yet", ha="center", va="center")
        ax.axis("off")
    else:
        for p in [r["policy"] for r in runs[-1]["results"]]:
            ys = [x["snips"] for h in runs for x in h["results"] if x["policy"] == p]
            ax.plot(range(1, len(ys) + 1), ys, marker="o", label=p)
        ax.set_xlabel("Run # (fresh random sample each)")
        ax.set_ylabel("Policy value (SNIPS)")
        ax.set_xticks(range(1, len(runs) + 1))
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    return buf.getvalue()


def table(rows, head):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{h}</tr>{b}</table>"


def runs_payload(runs):
    if not runs:
        return dict(latest_html="<p>No runs yet.</p>", runs_html="")
    last = runs[-1]
    by = {r["policy"]: r for r in last["results"]}
    latest = table(
        [[i, p, f"{by[p]['snips']:.5f}", f"[{by[p]['ci_low']:.5f}, {by[p]['ci_high']:.5f}]",
          f"{by[p]['ips']:.5f}", f"{by[p]['ess']:.1f}"]
         for i, p in enumerate(report.ranking(last["results"]), 1)],
        ["Rank", "Policy", "Policy value (SNIPS)", "95% CI", "IPS", "ESS"])
    latest = f"<p>{last['run_id']} (seed {last['seed']})</p>" + latest
    rows = [[h["run_id"], h["seed"], " &gt; ".join(report.ranking(h["results"])),
             "yes" if report.overlap(h["results"]) else "no"] for h in reversed(runs)]
    return dict(latest_html=latest,
                runs_html=table(rows, ["Run", "Seed", "Ranking", "All CIs overlap"]))


def start_run(email, data):
    with LOCK:
        if STATE["proc"] and STATE["proc"].poll() is None:
            return False
        cmd = [sys.executable, "-u", os.path.join(HERE, "run_agent.py")]
        if data:
            cmd += ["--data", data]
        if not email:
            cmd.append("--no-email")
        STATE["log"] = []
        STATE["proc"] = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True)

        def pump(p):
            for line in p.stdout:
                STATE["log"].append(line.rstrip())
                del STATE["log"][:-200]

        threading.Thread(target=pump, args=(STATE["proc"],), daemon=True).start()
        return True


class Handler(BaseHTTPRequestHandler):
    data = None

    def _send(self, body, ctype, code=200):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            self._send(PAGE, "text/html")
        elif path == "/chart.png":
            self._send(chart_png(load_runs()), "image/png")
        elif path == "/api/runs":
            self._send(json.dumps(runs_payload(load_runs())), "application/json")
        elif path == "/api/status":
            p = STATE["proc"]
            self._send(json.dumps(dict(
                running=bool(p and p.poll() is None),
                log="\n".join(STATE["log"]),
                n_runs=len(glob.glob(os.path.join(RUNS_DIR, "*", "run.json"))))),
                "application/json")
        else:
            self._send("not found", "text/plain", 404)

    def do_POST(self):
        if self.path != "/api/run":
            return self._send("not found", "text/plain", 404)
        n = int(self.headers.get("Content-Length") or 0)
        opts = json.loads(self.rfile.read(n) or b"{}")
        ok = start_run(bool(opts.get("email")), Handler.data)
        self._send(json.dumps({"started": ok}), "application/json")

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--data", default=os.environ.get("OBD_DATA"))
    a = ap.parse_args()
    Handler.data = a.data
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"Dashboard: http://127.0.0.1:{a.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
