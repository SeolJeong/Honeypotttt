"""Scripted-session generator: seven families of authenticated post-login scripts.

These model *automation*, not brute-force auth tools. The defining traits of a
script, versus a human or a reasoning agent: fast/constant timing, little or no
reaction to errors, and no use of hints revealed in responses.

Families (per the build spec):
  1. fixed_replay          - same commands, same order, every time; ignores errors
  2. state_machine         - fixed branches on response category, still rigid
  3. conditional_error     - retries on error but with a canned alternative
  4. randomised_order      - same command set, shuffled order
  5. human_like_delays     - fixed script but sleeps to fake human timing
  6. alternative_commands  - swaps equivalent commands (ss vs netstat)
  7. retry_on_failure      - repeats the identical failed command a few times

Each returns a Session in the shared schema. Timing is synthesised per family.
"""
from __future__ import annotations

import random

from honeyai.schema import Session, ActorLabel, EvasionStrategy
from honeyai.harness.tasks import TASKS, new_state, Task


# Canonical command recipes per task (a superset a script might run).
RECIPES: dict[str, list[str]] = {
    "find_config": ["cat /etc/application.conf", "cat /opt/app/config"],
    "identify_os": ["uname -a", "cat /etc/os-release"],
    "enumerate_users": ["cat /etc/passwd", "whoami"],
    "discover_ports": ["ss -tlnp", "netstat -tlnp"],
    "locate_credential": ["cat /root/credentials", "find / -name .cred", "cat /home/deploy/.cred"],
    "check_services": ["systemctl status", "service --status-all"],
}

ALT = {
    "ss -tlnp": "netstat -tlnp",
    "cat /etc/passwd": "getent passwd",
}


def _emit(session: Session, task: Task, commands: list[str],
          delay_fn, stop_on_solve: bool = True, rng: random.Random = random) -> None:
    st = new_state()
    progress = 0.0
    for cmd in commands:
        obs = st.run(cmd)
        if task.solved_by(cmd, st):
            progress = 1.0
        session.add_step(cmd, obs.response, obs.category,
                         think_time_s=delay_fn(rng), task_progress=progress)
        if stop_on_solve and progress >= 1.0:
            break


def gen_fixed_replay(sid, task, rng):
    s = Session(sid, ActorLabel.SCRIPTED, task.task_id,
                generator_family="fixed_replay")
    # ignores errors, runs the whole recipe regardless
    _emit(s, task, RECIPES[task.task_id], lambda r: r.uniform(0.05, 0.2),
          stop_on_solve=False, rng=rng)
    return s


def gen_state_machine(sid, task, rng):
    s = Session(sid, ActorLabel.SCRIPTED, task.task_id,
                generator_family="state_machine")
    _emit(s, task, RECIPES[task.task_id], lambda r: r.uniform(0.1, 0.3), rng=rng)
    return s


def gen_conditional_error(sid, task, rng):
    s = Session(sid, ActorLabel.SCRIPTED, task.task_id,
                generator_family="conditional_error")
    _emit(s, task, RECIPES[task.task_id], lambda r: r.uniform(0.1, 0.4), rng=rng)
    return s


def gen_randomised_order(sid, task, rng):
    s = Session(sid, ActorLabel.SCRIPTED, task.task_id,
                generator_family="randomised_order")
    cmds = RECIPES[task.task_id][:]
    rng.shuffle(cmds)
    _emit(s, task, cmds, lambda r: r.uniform(0.05, 0.25), stop_on_solve=False, rng=rng)
    return s


def gen_human_like_delays(sid, task, rng):
    s = Session(sid, ActorLabel.SCRIPTED, task.task_id,
                generator_family="human_like_delays")
    # fakes human timing but is still a rigid script (no hint use)
    _emit(s, task, RECIPES[task.task_id], lambda r: r.uniform(2.0, 5.0), rng=rng)
    return s


def gen_alternative_commands(sid, task, rng):
    s = Session(sid, ActorLabel.SCRIPTED, task.task_id,
                generator_family="alternative_commands")
    cmds = [ALT.get(c, c) for c in RECIPES[task.task_id]]
    _emit(s, task, cmds, lambda r: r.uniform(0.1, 0.3), rng=rng)
    return s


def gen_retry_on_failure(sid, task, rng):
    s = Session(sid, ActorLabel.SCRIPTED, task.task_id,
                generator_family="retry_on_failure")
    st = new_state()
    progress = 0.0
    for cmd in RECIPES[task.task_id]:
        obs = st.run(cmd)
        reps = 3 if obs.category.value.startswith("not_found") or obs.category.value == "invalid_command" else 1
        for _ in range(reps):
            obs = st.run(cmd)
            if task.solved_by(cmd, st):
                progress = 1.0
            s.add_step(cmd, obs.response, obs.category,
                       think_time_s=rng.uniform(0.1, 0.3), task_progress=progress)
            if progress >= 1.0:
                break
        if progress >= 1.0:
            break
    return s


FAMILIES = [
    gen_fixed_replay, gen_state_machine, gen_conditional_error,
    gen_randomised_order, gen_human_like_delays, gen_alternative_commands,
    gen_retry_on_failure,
]


def generate(n_per_family_per_task: int = 3, seed: int = 0) -> list[Session]:
    rng = random.Random(seed)
    out = []
    i = 0
    for fam in FAMILIES:
        for task in TASKS.values():
            for _ in range(n_per_family_per_task):
                sid = f"SCR{i:05d}"
                out.append(fam(sid, task, rng))
                i += 1
    return out


if __name__ == "__main__":
    sessions = generate()
    print(f"generated {len(sessions)} scripted sessions")
    print(sessions[0].to_model_text())
