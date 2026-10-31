"""
Benchmark the solver on the 13 Toronto instances.

For each instance: simulated annealing with several seeds, and one hill-climbing run as a
baseline, all with the same time budget. Every timetable is re-checked by the independent
evaluator (timetabler/evaluate.py). Results go to benchmarks/results.json.

    python benchmarks/run_benchmark.py                   # 5 seeds x 120 s (about 80 min on 2 cores)
    python benchmarks/run_benchmark.py --quick           # 3 small instances, 2 seeds x 10 s
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from timetabler.benchmarks import TORONTO, load_toronto  # noqa: E402
from timetabler.evaluate import evaluate  # noqa: E402
from timetabler.solver import solve, warm_up  # noqa: E402


def downsample(history, n=80):
    if not history:
        return []
    h = np.array(history)
    grid = np.unique(np.concatenate([np.geomspace(max(h[0, 0], 1e-3), h[-1, 0], n), [h[-1, 0]]]))
    out = []
    for g in grid:
        k = int(np.searchsorted(h[:, 0], g, side="right") - 1)
        k = max(k, 0)
        out.append([round(float(h[k, 0]), 3), round(float(h[k, 2]), 5)])
    return out


def run(job):
    key, method, seed, seconds = job
    warm_up()
    inst = load_toronto(key)
    r = solve(inst, time_limit=seconds, seed=seed, method=method)
    ev = evaluate(inst, r.t)
    ev0 = evaluate(inst, r.start_t)
    print(f"{key:6s} {method:8s} seed {seed}  start {r.start_penalty:9.4f}  final {r.penalty:9.4f}  "
          f"best known {TORONTO[key]['best']:9.4f}  feasible {ev.feasible}", flush=True)
    return {"instance": key, "method": method, "seed": seed, "seconds": seconds,
            "penalty": r.penalty, "eval_penalty": ev.penalty, "feasible": bool(ev.feasible),
            "start_penalty": ev0.penalty, "start_feasible": bool(ev0.feasible),
            "repaired": bool(r.messages), "iterations": int(r.iterations), "accepted": int(r.accepted),
            "history": downsample(r.history), "timetable": [int(x) for x in r.t]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=120)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated instances to (re)run and merge into --out")
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmarks", "results.json"))
    a = ap.parse_args()
    keys = list(TORONTO)
    if a.quick:
        keys, a.seconds, a.seeds = ["hec92", "sta83", "yor83"], 10, 2
    if a.only:
        keys = [k.strip() for k in a.only.split(",") if k.strip()]
    jobs = [(k, "anneal", s, a.seconds) for k in keys for s in range(a.seeds)]
    jobs += [(k, "descent", 0, a.seconds) for k in keys]
    big = {"pur93": 0, "car91": 1, "car92": 2, "uta92": 3, "rye93": 4}
    jobs.sort(key=lambda j: (big.get(j[0], 9), j[1] != "anneal"))
    t0 = time.time()
    with Pool(a.workers) as pool:
        runs = list(pool.imap_unordered(run, jobs))
    if a.only and os.path.exists(a.out):     # replace these instances' runs, keep the rest
        old = json.load(open(a.out))
        runs = [r for r in old["runs"] if r["instance"] not in keys] + runs
        keys = list(old["instances"]) + [k for k in keys if k not in old["instances"]]
    out = {"settings": {"seconds": a.seconds, "seeds": a.seeds, "workers": a.workers,
                        "machine": f"{platform.machine()}, {os.cpu_count()} cores, Python {platform.python_version()}",
                        "wall_minutes": round((time.time() - t0) / 60, 1)},
           "instances": {k: {kk: v for kk, v in TORONTO[k].items() if kk != "md5"} for k in keys},
           "runs": sorted(runs, key=lambda r: (r["instance"], r["method"], r["seed"]))}
    with open(a.out, "w") as f:
        json.dump(out, f)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
