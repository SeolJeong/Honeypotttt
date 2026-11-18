"""Record REAL human participant sessions on the simulated host.

Run it in a terminal and hand the keyboard to the participant:
    python scripts/collect_human.py --participant P006 --contributor SL

Records only: commands, simulated outputs, time between prompt and Enter,
task progress, anonymous participant id. No names, no keystrokes, no IP.
Sessions append to contrib/human_<participant>.jsonl.

Check with your lecturer whether ethics approval / a consent form is needed
before recording anyone (spec section 6).
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from honeyai.sim.environment import TASKS, make_scenario
from honeyai.sim.recording import Recorder, append_jsonl

CONSENT = """
HoneyAI study - ICT3214 coursework
You will complete up to 8 short tasks on a simulated Linux server (nothing is real).
We record only the commands you type, the simulated output, and timing.
No name or personal information is stored. You can stop at any time (type 'quit').
Type 'skip' to give up on a task. A task ends automatically once you've found the answer.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", required=True, help="anonymous id, e.g. P006 (never a name)")
    ap.add_argument("--contributor", default="anon")
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join("contrib", f"human_{a.participant}.jsonl")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    print(CONSENT)
    if input("Do you consent to take part? (yes/no) ").strip().lower() not in ("y", "yes"):
        print("Not recorded.")
        return
    tasks = list(a.tasks)
    random.shuffle(tasks)
    for i, task in enumerate(tasks, 1):
        sc = make_scenario(task, random.randrange(10**8, 10**9))
        rec = Recorder(sc)
        print(f"\n=== Task {i}/{len(tasks)} ===\n{sc.objective}\n")
        quit_all = False
        while not rec.done:
            t0 = time.perf_counter()
            try:
                cmd = input(f"deploy@{sc.host}:{rec.sh.cwd.replace('/home/deploy', '~')}$ ")
            except EOFError:
                quit_all = True
                break
            gap = time.perf_counter() - t0
            if cmd.strip() in ("quit", "exit"):
                quit_all = cmd.strip() == "quit"
                break
            if cmd.strip() == "skip":
                break
            if not cmd.strip():
                continue
            o, _ = rec.run(cmd, gap)
            if o:
                print(o)
        if rec.steps:
            s = rec.session(session_id=f"HU-{a.participant}-{task}-{sc.seed}", actor_label="human",
                            participant_id=a.participant, evasion_strategy="none",
                            generator_family="human", data_source="human_participant_vs_simulator",
                            contributor=a.contributor)
            append_jsonl(out, s)
            print("\n[task complete]" if s["task_completed"] else "\n[task ended]")
        if quit_all:
            break
    print(f"\nThanks! Saved to {out}")


if __name__ == "__main__":
    main()
