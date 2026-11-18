"""Fit simulated-human think time to real humans (KYPO).

Target: the KYPO gap distribution AFTER QUICK COMMANDS only. A KYPO gap also
contains how long the previous command ran (an nmap scan, a john crack, an
interactive ssh/vim session), and the simulator's commands finish instantly,
so those gaps are excluded. By default only target-host sessions are used
(commands typed on a machine reached over SSH), the closest match to a
honeypot session; --hosts all uses every host.

Method: grid search over the persona think-time ranges
(median_lo, median_hi, sigma_lo, sigma_hi), simulating the TRAINING human
personas on fresh scenarios and comparing quantiles (p10, p25, p50, p75, p90)
in log space. Writes configs/human_timing.json, which generate_sessions.py
--human-timing uses.

Only timing is calibrated. Typos, hint-following, exploration etc. remain the
simulator's documented assumptions (data/v2/DATA_CARD.md).

    python scripts/convert_kypo.py
    python scripts/calibrate_human_timing.py
    python scripts/generate_sessions.py --human-timing configs/human_timing.json --out data/v2b
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from honeyai.sim.actors import ChainPolicy, human_persona, run_session, set_human_timing
from honeyai.sim.environment import TASKS, make_scenario

QS = (10, 25, 50, 75, 90)
# programs whose runtime ends up inside the NEXT gap in KYPO (long-running or interactive)
LONG_RUNNING = {"nmap", "john", "hydra", "fcrackzip", "msfconsole", "msfdb", "ssh", "scp", "wget", "curl",
                "sleep", "apt", "apt-get", "hashcat", "sqlmap", "nikto", "ping", "vim", "vi", "nano", "less",
                "more", "man", "top", "tail", "python", "python3", "gcc", "make", "service", "systemctl"}


def kypo_target(path: str, hosts: str) -> tuple[np.ndarray, int]:
    sessions = json.load(open(path, encoding="utf-8"))
    gaps = []
    for s in sessions:
        if hosts == "target" and not s.get("on_target_host"):
            continue
        st = s["interactions"]
        for a, b in zip(st, st[1:]):
            prog = (a["command"].strip().split() or [""])[0].split("/")[-1]
            if prog not in LONG_RUNNING:
                gaps.append(b["inter_command_gap_s"])
    g = np.array(gaps)
    return np.percentile(g, QS), len(g)


def simulate(params: dict, n_per: int, seed: int) -> np.ndarray:
    set_human_timing(params)
    rng = random.Random(seed)
    gaps = []
    for pid in range(1, 13):                       # training personas SP01-SP12
        for task in TASKS:
            for _ in range(n_per):
                sc = make_scenario(task, rng.randrange(10**9))
                p, timing = human_persona(pid)
                steps = run_session(sc, ChainPolicy(p, timing, random.Random(rng.getrandbits(32))))
                gaps += [st["inter_command_gap_s"] for st in steps[1:]]   # step 0 = time from login
    return np.percentile(gaps, QS)


def loss(sim_q, tgt_q):
    return float(np.mean((np.log(sim_q) - np.log(np.maximum(tgt_q, 0.05))) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kypo", default=os.path.join("external", "kypo", "kypo_sessions.json"))
    ap.add_argument("--hosts", choices=["target", "all"], default="target")
    ap.add_argument("--out", default=os.path.join("configs", "human_timing.json"))
    ap.add_argument("--n-per", type=int, default=2, help="sessions per persona per task per grid point")
    a = ap.parse_args()
    if not os.path.exists(a.kypo):
        sys.exit(f"{a.kypo} not found: run scripts/convert_kypo.py first")

    tgt, n = kypo_target(a.kypo, a.hosts)
    print(f"KYPO target ({a.hosts} hosts, {n} gaps after quick commands): "
          + "  ".join(f"p{q} {v:.1f}s" for q, v in zip(QS, tgt)))
    default = {"median_lo": 3.5, "median_hi": 10.0, "sigma_lo": 0.5, "sigma_hi": 0.9}
    base_q = simulate(default, a.n_per, 1)
    print("v2 default persona timing:   " + "  ".join(f"p{q} {v:.1f}s" for q, v in zip(QS, base_q))
          + f"   loss {loss(base_q, tgt):.3f}")

    grid = [dict(median_lo=ml, median_hi=mh, sigma_lo=sl, sigma_hi=sh)
            for ml, mh, sl, sh in itertools.product([0.5, 1, 2, 3, 4], [4, 6, 8, 10, 12],
                                                    [0.8, 1.0, 1.2, 1.4], [1.6, 1.8, 2.0, 2.2, 2.4])
            if ml < mh and sl <= sh]
    t0 = time.time()
    scored = sorted(((loss(simulate(g, a.n_per, 1), tgt), i, g) for i, g in enumerate(grid)))
    best = scored[0][2]
    # confirm on a fresh simulation seed so the fit isn't tuned to one random draw
    check_q = simulate(best, a.n_per * 2, 2)
    print(f"searched {len(grid)} settings in {time.time() - t0:.0f}s; best: {best}")
    print("calibrated (fresh seed):     " + "  ".join(f"p{q} {v:.1f}s" for q, v in zip(QS, check_q))
          + f"   loss {loss(check_q, tgt):.3f}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"params": best,
                   "target": {"source": "KYPO (Svabensky et al. 2021, CC BY 4.0)", "hosts": a.hosts,
                              "excluded_after": sorted(LONG_RUNNING), "n_gaps": n,
                              "quantiles_s": dict(zip([f"p{q}" for q in QS], map(float, tgt)))},
                   "v2_default_quantiles_s": dict(zip([f"p{q}" for q in QS], map(float, base_q))),
                   "calibrated_quantiles_s": dict(zip([f"p{q}" for q in QS], map(float, check_q))),
                   "loss": {"v2_default": loss(base_q, tgt), "calibrated": loss(check_q, tgt)},
                   "created": time.strftime("%Y-%m-%d")}, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
