"""Generate the v2 controlled dataset from the session simulator.

Writes to data/v2/ (the original data/ files are not touched):
    dataset1_sessions.json        full interleaved traces (same schema as v1 + provenance fields)
    dataset2_tabular_features.csv XGBoost features (honeyai.features, unchanged definitions)
    dataset4_llm_sequences.jsonl  interleaved text (v1 dataset4 format)

Splits are disjoint by construction:
    human     personas SP01-SP12 train | SP13-SP20 test
    scripted  fixed_replay, state_machine, conditional_error_handling, randomised_ordering train
              | human_like_delays, alternative_commands, retry_on_failure test
    AI        sim_agent_a/b x {none, timing, typo, shorter, hesitation} train
              | sim_agent_c x all 7 strategies test (unseen family)
              | sim_agent_a/b x {avoid_systematic_enumeration, combined} test (unseen strategy)
    scenario seeds never repeat, so no test host/task instance appears in training.

Usage:
    python scripts/generate_sessions.py               # default scale (~3,000 sessions)
    python scripts/generate_sessions.py --scale 0.5   # smaller
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from honeyai.features import extract_all
from honeyai.sim.actors import (EVASIONS, HUMAN_TIMING, ChainPolicy, ScriptPolicy, ai_agent, human_persona,
                                run_session, set_human_timing)
from honeyai.sim.environment import TASKS, make_scenario
from honeyai.sim.recording import dedupe, interleaved_text

TRAIN_EVASIONS = ["none", "timing_mimicry", "typo_mimicry", "shorter_commands",
                  "hesitation_redundant_exploration"]
HELDOUT_EVASIONS = ["avoid_systematic_enumeration", "combined"]
TRAIN_SCRIPTS = ["fixed_replay", "state_machine", "conditional_error_handling", "randomised_ordering"]
TEST_SCRIPTS = ["human_like_delays", "alternative_commands", "retry_on_failure"]


def plan(scale: float):
    """Yield (split, actor_label, family, evasion, participant, reps) cells."""
    n = lambda x: max(1, round(x * scale))
    for pid in range(1, 21):
        yield ("train" if pid <= 12 else "test", "human", "sim_human", "none", f"SP{pid:02d}", n(5))
    for fam in TRAIN_SCRIPTS:
        yield ("train", "scripted", fam, "none", None, n(14))
    for fam in TEST_SCRIPTS:
        yield ("test", "scripted", fam, "none", None, n(8))
    for fam in ["sim_agent_a", "sim_agent_b"]:
        for ev in TRAIN_EVASIONS:
            yield ("train", "AI-driven", fam, ev, None, n(7))
        for ev in HELDOUT_EVASIONS:
            yield ("test", "AI-driven", fam, ev, None, n(3))
    for ev in EVASIONS:
        yield ("test", "AI-driven", "sim_agent_c", ev, None, n(4))


def to_ds4(s):
    return {k: s.get(k) for k in ("session_id", "split", "actor_label", "evasion_strategy",
                                  "generator_family", "participant_id", "task", "total_steps")} \
        | {"interleaved_text": interleaved_text(s)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=os.path.join("data", "v2"))
    ap.add_argument("--human-timing", default=None,
                    help="JSON from scripts/calibrate_human_timing.py (default: v2's original persona timing)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.human_timing:
        with open(a.human_timing, encoding="utf-8") as f:
            set_human_timing(json.load(f)["params"])
    print("human persona timing:", HUMAN_TIMING)

    master = random.Random(a.seed)
    sessions, scen_seed = [], a.seed * 1_000_003
    for split, label, fam, ev, pid, reps in plan(a.scale):
        for task in TASKS:
            for _ in range(reps):
                scen_seed += 1
                sc = make_scenario(task, scen_seed)
                rng = random.Random(master.getrandbits(32))
                if label == "human":
                    params, timing = human_persona(int(pid[2:]))
                    pol = ChainPolicy(params, timing, rng)
                elif label == "scripted":
                    pol = ScriptPolicy(fam, rng)
                else:
                    pol = ai_agent(fam, ev, rng)
                steps = run_session(sc, pol)
                sessions.append({
                    "session_id": f"V2-{len(sessions) + 1:05d}",
                    "actor_label": label, "participant_id": pid, "task": task, "split": split,
                    "evasion_strategy": ev, "generator_family": fam,
                    "total_steps": len(steps),
                    "session_duration_s": round(sum(s["inter_command_gap_s"] for s in steps), 2),
                    "data_source": "scripted_client_vs_simulator" if label == "scripted" else "simulated_policy",
                    "scenario_seed": scen_seed, "objective": sc.objective,
                    "task_completed": bool(steps and steps[-1]["task_progress"] >= 1.0),
                    "interactions": steps,
                })

    sessions, dropped = dedupe(sessions)
    print(f"duplicate check: dropped {dropped['dup_within_split']} within-split duplicates, "
          f"{dropped['test_equals_train']} test sessions identical to a training session")
    with open(os.path.join(a.out, "dataset1_sessions.json"), "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=1, ensure_ascii=False)
    rows = extract_all(sessions)
    with open(os.path.join(a.out, "dataset2_tabular_features.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out, "dataset4_llm_sequences.jsonl"), "w", encoding="utf-8") as f:
        for s in sessions:
            f.write(json.dumps(to_ds4(s), ensure_ascii=False) + "\n")

    with open(os.path.join(a.out, "generation_info.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": a.seed, "scale": a.scale, "human_timing": dict(HUMAN_TIMING),
                   "human_timing_file": a.human_timing, "n_sessions": len(sessions), "dedup": dropped}, f, indent=1)
    print(f"wrote {len(sessions)} sessions to {a.out}")
    print("by split/label:", dict(Counter((s["split"], s["actor_label"]) for s in sessions)))
    L = [s["total_steps"] for s in sessions]
    print(f"steps per session: min {min(L)}  mean {sum(L) / len(L):.1f}  max {max(L)}")
    for lab in ["human", "scripted", "AI-driven"]:
        ss = [s for s in sessions if s["actor_label"] == lab]
        print(f"  {lab:10s} mean steps {sum(s['total_steps'] for s in ss) / len(ss):.1f}  "
              f"completed {sum(s['task_completed'] for s in ss) / len(ss):.0%}")


if __name__ == "__main__":
    main()
