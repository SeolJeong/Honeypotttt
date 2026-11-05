"""Shared helpers for recording real sessions (LLM agents, human participants)
against the simulator, and for validating/merging contributed files."""
from __future__ import annotations

import json
import time

from honeyai.sim.actors import MAX_STEPS, gap_bin
from honeyai.sim.environment import TASKS, Scenario, Shell, progress

LABELS = {"human", "scripted", "AI-driven"}
CATEGORIES = {"success", "partial_result", "not_found_with_hint", "not_found_no_hint",
              "permission_denied", "command_not_found"}
REQUIRED = ["session_id", "actor_label", "task", "evasion_strategy", "generator_family",
            "data_source", "interactions"]
STEP_REQUIRED = ["step", "command", "system_response", "response_category", "inter_command_gap_s"]


def _edit_distance(a: str, b: str) -> int:
    if abs(len(a) - len(b)) > 3:
        return 99
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def mark_typos(steps: list[dict]) -> None:
    """Heuristic has_typo for real sessions: a failed command that is followed by
    a near-identical command (edit distance 1-2) is treated as a typo."""
    for i, st in enumerate(steps):
        st.setdefault("has_typo", False)
        if i + 1 < len(steps) and st["response_category"] in ("command_not_found", "not_found_no_hint"):
            d = _edit_distance(st["command"], steps[i + 1]["command"])
            if 1 <= d <= 2:
                st["has_typo"] = True


class Recorder:
    """Wraps a Shell, timestamps every command and builds a dataset1 session."""

    def __init__(self, sc: Scenario):
        self.sc, self.sh = sc, Shell(sc)
        self.steps, self.responses, self.t = [], [], 0.0

    @property
    def done(self) -> bool:
        return progress(self.sc, self.responses) >= 1.0 or len(self.steps) >= MAX_STEPS

    def run(self, command: str, gap_s: float, **extra) -> tuple[str, str]:
        out, cat = self.sh.run(command)
        gap = round(max(0.05, gap_s), 2)
        self.t += gap
        self.responses.append(out)
        self.steps.append({
            "step": len(self.steps), "command": command, "has_typo": False,
            "system_response": out, "response_category": cat,
            "inter_command_gap_s": gap, "think_time_bin": gap_bin(gap),
            "cumulative_time_s": round(self.t, 2),
            "task_progress": round(progress(self.sc, self.responses), 2),
            **extra,
        })
        return out, cat

    def session(self, **meta) -> dict:
        mark_typos(self.steps)
        return {
            "session_id": meta.pop("session_id"),
            "task": self.sc.task, "split": meta.pop("split", "unassigned"),
            "total_steps": len(self.steps),
            "session_duration_s": round(self.t, 2),
            "scenario_seed": self.sc.seed, "objective": self.sc.objective,
            "task_completed": bool(self.steps and self.steps[-1]["task_progress"] >= 1.0),
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **meta, "interactions": self.steps,
        }


def validate(sess: dict) -> list[str]:
    errs = [f"missing {k}" for k in REQUIRED if k not in sess]
    if sess.get("actor_label") not in LABELS:
        errs.append(f"bad actor_label {sess.get('actor_label')!r}")
    if sess.get("task") not in TASKS:
        errs.append(f"bad task {sess.get('task')!r}")
    if sess.get("actor_label") == "human" and not sess.get("participant_id"):
        errs.append("human session without participant_id")
    steps = sess.get("interactions") or []
    if not steps:
        errs.append("no interactions")
    for st in steps:
        for k in STEP_REQUIRED:
            if k not in st:
                errs.append(f"step {st.get('step')}: missing {k}")
        if st.get("response_category") not in CATEGORIES:
            errs.append(f"step {st.get('step')}: bad category {st.get('response_category')!r}")
    return errs


def append_jsonl(path: str, sess: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(sess, ensure_ascii=False) + "\n")


# ── duplicate / leakage check ───────────────────────────────────────────

def _gap_bin_upper(s):
    return "SHORT" if s < 2 else "MEDIUM" if s < 6 else "LONG" if s < 15 else "VERY_LONG"


def interleaved_text(sess: dict) -> str:
    """The dataset4 text a model sees (gap bin, command, category, response)."""
    parts = []
    for st in sess["interactions"]:
        parts.append(f"[GAP {_gap_bin_upper(st['inter_command_gap_s'])}] CMD: {st['command']}")
        parts.append(f"RESP [{st['response_category']}]: {st['system_response']}")
    return "\n".join(parts)


def dedupe(sessions: list[dict]) -> tuple[list[dict], dict]:
    """Drop sessions whose model-visible text repeats.

    - exact duplicates within the same split: keep the first
    - test sessions identical to ANY training session: dropped (train/test leakage)
    Order of `sessions` is preserved. Returns (kept, counts).
    """
    train_txt = {interleaved_text(s) for s in sessions if s.get("split") == "train"}
    seen = {"train": set(), "test": set()}
    kept, counts = [], {"dup_within_split": 0, "test_equals_train": 0}
    for s in sessions:
        t = interleaved_text(s)
        sp = s.get("split", "unassigned")
        if sp == "test" and t in train_txt:
            counts["test_equals_train"] += 1
            continue
        bucket = seen.setdefault(sp, set())
        if t in bucket:
            counts["dup_within_split"] += 1
            continue
        bucket.add(t)
        kept.append(s)
    return kept, counts
