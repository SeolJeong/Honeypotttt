"""Build the full dataset: scripted + human + AI sessions.

Uses REAL AI sessions when present, else the simulated stand-in:
  - if any data/sessions/ai_*.jsonl files exist (from generate_ai_sessions.py),
    those real sessions are used for the AI class.
  - otherwise the simulated AI generator fills in (PoC fallback).

Human sessions are still synthetic here; drop real participant JSONL into
data/sessions/human_*.jsonl and they will be picked up the same way.

Run:  python scripts/build_dataset.py
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from honeyai.generators import scripted, synthetic_human, ai_driven
from honeyai.schema import write_sessions, read_sessions, Session, Step
from honeyai.schema import (ActorLabel, ResponseCategory, ThinkTimeBin,
                            EvasionStrategy)

SESS_DIR = os.path.join("data", "sessions")
OUT = os.path.join(SESS_DIR, "poc_dataset.jsonl")


def _load_jsonl_as_sessions(pattern: str) -> list[Session]:
    """Load real session files (written by the generators) back into Session
    objects so they merge cleanly with freshly generated ones."""
    out = []
    for path in sorted(glob.glob(pattern)):
        if os.path.abspath(path) == os.path.abspath(OUT):
            continue  # never fold the combined output back in
        for d in read_sessions(path):
            s = Session(
                session_id=d["session_id"],
                actor_label=ActorLabel(d["actor_label"]),
                task_id=d["task_id"],
                evasion_strategy=EvasionStrategy(d.get("evasion_strategy", "none")),
                generator_family=d.get("generator_family", ""),
                participant_id=d.get("participant_id", ""),
            )
            for st in d["steps"]:
                s.steps.append(Step(
                    step=st["step"], command=st["command"],
                    system_response=st["system_response"],
                    response_category=ResponseCategory(st["response_category"]),
                    think_time_s=st["think_time_s"],
                    think_time_bin=ThinkTimeBin(st["think_time_bin"]),
                    task_progress=st.get("task_progress", 0.0),
                    is_typo=st.get("is_typo", False),
                ))
            out.append(s)
    return out


def main():
    os.makedirs(SESS_DIR, exist_ok=True)

    scr = scripted.generate(n_per_family_per_task=3, seed=0)

    # Human: real files if present, else synthetic
    hum = _load_jsonl_as_sessions(os.path.join(SESS_DIR, "human_*.jsonl"))
    human_src = "REAL" if hum else "SYNTHETIC"
    if not hum:
        hum = synthetic_human.generate(n_per_task=18, seed=1)

    # AI: real files if present, else simulated
    ai = _load_jsonl_as_sessions(os.path.join(SESS_DIR, "ai_*.jsonl"))
    ai_src = "REAL" if ai else "SIMULATED"
    if not ai:
        ai = ai_driven.generate(n_per_task=18, seed=2, evasion_mix=True)

    alls = scr + hum + ai
    write_sessions(alls, OUT)
    print(f"scripted : {len(scr)}")
    print(f"human    : {len(hum)}  ({human_src})")
    print(f"ai-driven: {len(ai)}  ({ai_src})")
    print(f"total    : {len(alls)} -> {OUT}")
    if ai_src == "REAL":
        fams = sorted({s.generator_family for s in ai})
        print(f"AI families: {fams}")


if __name__ == "__main__":
    main()
