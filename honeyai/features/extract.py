"""Tabular feature extraction (for the XGBoost baseline, Model 1).

Turns a session (dict form, as read from JSONL) into ~20-30 aggregate features.
These are deliberately the kind of surface + behavioural features prior work uses,
so Model 1 is a fair feature-based baseline. The adaptation-related features
(hint_follow_rate, progress_after_hint) are the ones we expect to matter most and
that commands-only baselines miss.
"""
from __future__ import annotations

import statistics as st
from collections import Counter


FEATURE_NAMES = [
    "n_steps", "duration_s", "mean_gap", "std_gap", "min_gap", "max_gap",
    "frac_instant", "frac_short", "frac_medium", "frac_long",
    "mean_cmd_len", "std_cmd_len", "unique_cmd_ratio", "repeat_rate",
    "n_distinct_programs", "error_rate", "not_found_rate", "invalid_rate",
    "typo_rate", "explore_rate", "hint_count", "hint_follow_rate",
    "final_progress", "solved", "progress_after_failure",
]

EXPLORE_PROGS = {"ls", "pwd", "whoami", "id", "cd", "env", "history"}


def _program(cmd: str) -> str:
    return cmd.strip().split()[0] if cmd.strip() else ""


def extract(session: dict) -> dict:
    steps = session.get("steps", [])
    n = len(steps)
    if n == 0:
        return {k: 0.0 for k in FEATURE_NAMES}

    gaps = [s["think_time_s"] for s in steps]
    cmd_lens = [len(s["command"]) for s in steps]
    progs = [_program(s["command"]) for s in steps]
    bins = [s["think_time_bin"] for s in steps]
    cats = [s["response_category"] for s in steps]

    n_err = sum(1 for c in cats if c in ("not_found", "permission_denied", "invalid_command"))
    n_not_found = sum(1 for c in cats if c.startswith("not_found"))
    n_invalid = sum(1 for c in cats if c == "invalid_command")
    n_typo = sum(1 for s in steps if s.get("is_typo"))
    n_explore = sum(1 for p in progs if p in EXPLORE_PROGS)

    # adaptation: after a hinted not-found, did the next step use the hint?
    hint_count = 0
    hint_follow = 0
    for i, s in enumerate(steps):
        if s["response_category"] == "not_found_with_hint":
            hint_count += 1
            if i + 1 < n:
                nxt = steps[i + 1]["command"]
                # hint followed if the next command references a path from the response
                if "/opt/app/config" in nxt or "/home/deploy/.cred" in nxt:
                    hint_follow += 1

    # progress recovery after any failure
    prog_after_fail = 0.0
    for i, s in enumerate(steps[:-1]):
        if s["response_category"] in ("not_found", "invalid_command", "permission_denied"):
            prog_after_fail = max(prog_after_fail, steps[i + 1]["task_progress"])

    final_progress = steps[-1]["task_progress"]
    bin_counts = Counter(bins)

    return {
        "n_steps": float(n),
        "duration_s": float(sum(gaps)),
        "mean_gap": st.mean(gaps),
        "std_gap": st.pstdev(gaps) if n > 1 else 0.0,
        "min_gap": min(gaps),
        "max_gap": max(gaps),
        "frac_instant": bin_counts.get("instant", 0) / n,
        "frac_short": bin_counts.get("short", 0) / n,
        "frac_medium": bin_counts.get("medium", 0) / n,
        "frac_long": bin_counts.get("long", 0) / n,
        "mean_cmd_len": st.mean(cmd_lens),
        "std_cmd_len": st.pstdev(cmd_lens) if n > 1 else 0.0,
        "unique_cmd_ratio": len(set(s["command"] for s in steps)) / n,
        "repeat_rate": 1.0 - (len(set(s["command"] for s in steps)) / n),
        "n_distinct_programs": float(len(set(progs))),
        "error_rate": n_err / n,
        "not_found_rate": n_not_found / n,
        "invalid_rate": n_invalid / n,
        "typo_rate": n_typo / n,
        "explore_rate": n_explore / n,
        "hint_count": float(hint_count),
        "hint_follow_rate": (hint_follow / hint_count) if hint_count else 0.0,
        "final_progress": final_progress,
        "solved": 1.0 if final_progress >= 1.0 else 0.0,
        "progress_after_failure": prog_after_fail,
    }


def extract_matrix(sessions: list[dict]):
    """Return (X, y, feature_names) with X as list-of-lists, y as label strings."""
    X, y = [], []
    for s in sessions:
        feats = extract(s)
        X.append([feats[k] for k in FEATURE_NAMES])
        y.append(s["actor_label"])
    return X, y, FEATURE_NAMES
