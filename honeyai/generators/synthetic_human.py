"""Synthetic-human session generator (PoC placeholder).

IMPORTANT: these are SIMULATED humans, for pipeline validation and the PoC demo
only. A model trained to separate these from the synthetic AI class is partly
learning our generation rules, so synthetic-human results prove the *pipeline*
works, not that the *method* works. For reported final results, replace this with
real consented-participant sessions (same schema, drop-in).

The simulated human is modelled on the behaviours the spec lists as human tells:
  - slower, variable timing (frequent LONG gaps for "thinking")
  - occasional typos, then a correction
  - exploratory commands (ls, pwd, whoami) before the task command
  - DOES use hints: after a not_found_with_hint, tries the hinted path
  - sometimes a wrong turn before succeeding
"""
from __future__ import annotations

import random

from honeyai.schema import Session, ActorLabel
from honeyai.harness.tasks import TASKS, new_state, Task, MOVED


EXPLORE = ["ls", "pwd", "whoami", "ls -la", "id"]
TYPOS = {"cat": "ct", "uname": "unmae", "systemctl": "sytemctl", "netstat": "netstaat"}


def _maybe_typo(cmd: str, rng: random.Random):
    prog = cmd.split()[0]
    if prog in TYPOS and rng.random() < 0.25:
        typo_cmd = cmd.replace(prog, TYPOS[prog], 1)
        return typo_cmd, True
    return cmd, False


def _human_delay(rng: random.Random) -> float:
    # heavy-tailed: mostly medium, sometimes long thinking pauses
    r = rng.random()
    if r < 0.5:
        return rng.uniform(1.5, 4.0)
    if r < 0.85:
        return rng.uniform(4.0, 9.0)
    return rng.uniform(0.4, 1.4)


def gen_human(sid: str, task: Task, rng: random.Random) -> Session:
    s = Session(sid, ActorLabel.HUMAN, task.task_id, generator_family="synthetic_human")
    st = new_state()
    progress = 0.0

    # 1. some exploration first
    for _ in range(rng.randint(1, 3)):
        cmd = rng.choice(EXPLORE)
        obs = st.run(cmd)
        s.add_step(cmd, obs.response, obs.category, _human_delay(rng), progress)

    # 2. attempt the task, using the canonical path but reacting to hints
    for intended in task.canonical_path:
        cmd, typo = _maybe_typo(intended, rng)
        if typo:
            # run the typo, get an error, then correct it
            obs = st.run(cmd)
            s.add_step(cmd, obs.response, obs.category, _human_delay(rng), progress, is_typo=True)
            cmd = intended
        obs = st.run(cmd)
        if task.solved_by(cmd, st):
            progress = 1.0
        s.add_step(cmd, obs.response, obs.category, _human_delay(rng), progress)

        # human reads a hint and follows it
        if obs.category.value == "not_found_with_hint":
            # extract hinted path and try it (adaptation)
            hinted = None
            for src, dest in MOVED.items():
                if src in cmd:
                    hinted = dest
            if hinted:
                follow = f"cat {hinted}"
                obs2 = st.run(follow)
                if task.solved_by(follow, st):
                    progress = 1.0
                s.add_step(follow, obs2.response, obs2.category, _human_delay(rng), progress)

        if progress >= 1.0:
            break

    return s


def generate(n_per_task: int = 12, seed: int = 1) -> list[Session]:
    rng = random.Random(seed)
    out = []
    i = 0
    for task in TASKS.values():
        for _ in range(n_per_task):
            # participant id disjoint-able: group by block of sessions
            pid = f"P{(i % 12):02d}"
            s = gen_human(f"HUM{i:05d}", task, rng)
            s.participant_id = pid
            out.append(s)
            i += 1
    return out


if __name__ == "__main__":
    sessions = generate()
    print(f"generated {len(sessions)} synthetic-human sessions")
    print(sessions[0].to_model_text())
