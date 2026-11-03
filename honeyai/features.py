"""Part 4: Tabular feature extraction from dataset1 sessions.

Reads dataset1_sessions.json and computes 19 numeric features per session,
producing a CSV that matches dataset2_tabular_features.csv. Every formula
below was reverse-verified against the team's existing dataset2 (exact match
on all 208 sessions), so this extractor can be trusted to reproduce the
feature definitions already used for Model 1.

Features (what they measure and why they help):

Timing — humans are slow and variable; scripts are fast and regular:
  session_duration_s      total time from first to last command
  inter_cmd_mean_s        average think time between commands
  inter_cmd_std_s         variability of think time (humans vary more)
  min_gap_s               fastest gap (scripts have near-zero)
  max_gap_s               longest pause (humans take breaks)

Command complexity — AI tends to use longer, more systematic commands:
  cmd_len_mean            average character length of commands
  cmd_len_std             variability of command length
  unique_cmd_ratio        fraction of distinct commands (scripts repeat more)

Error behaviour — dead ends vs guided responses are different signals:
  typo_count              total commands flagged as typos
  typo_rate               typo_count / command_count
  error_count             dead-end responses only: command_not_found,
                          not_found_no_hint, permission_denied.
                          not_found_with_hint is NOT counted here — the shell
                          gave a path suggestion, so it isn't a true dead end.
  error_rate              error_count / command_count

Adaptation — does the actor act on a hint the system gave it? This is the
central "evasion-aware" signal: an AI mimicking a human still has to decide
whether to use the hint, and that choice shows up here even under evasion.
  not_found_with_hint_count  how many times the system offered a path hint
  adaptation_event_count  how many of those hints were immediately followed
                          by a successful command (consecutive repeated
                          hints are treated as one group, resolved by
                          whichever response comes after the group)
  error_recovery_rate     adaptation_event_count / not_found_with_hint_count
                          — the fraction of hints actually acted on
                          (0.0 when no hints were given)

Task progress:
  final_task_progress     how far the actor got (0.0 to 1.0)
  interaction_depth       total number of steps (same as command_count)

Sequence pattern:
  sequence_entropy        Shannon entropy (base 2) of the full-command
                          distribution — scripts repeat exact commands
                          (low entropy), humans/AI vary arguments more
                          (higher entropy)

Usage:
    python scripts/extract_features.py
"""
from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter

DATA_DIR = "data"
DS1 = os.path.join(DATA_DIR, "dataset1_sessions.json")
DS2_OUT = os.path.join(DATA_DIR, "dataset2_tabular_features.csv")

# dead-end responses only — not_found_with_hint is handled separately as the
# adaptation signal, not as an error
ERROR_CATEGORIES = {"command_not_found", "not_found_no_hint", "permission_denied"}


def _shannon_entropy(counts: list[int]) -> float:
    """Shannon entropy of a frequency distribution (base 2)."""
    total = sum(counts)
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    return round(ent, 4)


def _adaptation_count(steps: list[dict]) -> int:
    """Count hint groups that were immediately followed by a success.

    Consecutive not_found_with_hint responses are treated as one group (the
    actor is still working the same dead end); the group counts as an
    adaptation event if the next *different* response is 'success'.
    """
    count = 0
    i, n = 0, len(steps)
    while i < n:
        if steps[i]["response_category"] == "not_found_with_hint":
            j = i
            while j < n and steps[j]["response_category"] == "not_found_with_hint":
                j += 1
            if j < n and steps[j]["response_category"] == "success":
                count += 1
            i = j
        else:
            i += 1
    return count


def extract_one(sess: dict) -> dict:
    """Extract features for one session. Returns a flat dict."""
    steps = sess["interactions"]
    n = len(steps)

    # timing
    gaps = [s["inter_command_gap_s"] for s in steps if s.get("inter_command_gap_s") is not None]
    duration = sess.get("session_duration_s", sum(gaps))
    gap_mean = sum(gaps) / len(gaps) if gaps else 0.0
    gap_std = (sum((g - gap_mean) ** 2 for g in gaps) / len(gaps)) ** 0.5 if gaps else 0.0
    gap_min = min(gaps) if gaps else 0.0
    gap_max = max(gaps) if gaps else 0.0

    # command complexity
    cmds = [s["command"] for s in steps]
    lengths = [len(c) for c in cmds]
    len_mean = sum(lengths) / len(lengths) if lengths else 0.0
    len_std = (sum((l - len_mean) ** 2 for l in lengths) / len(lengths)) ** 0.5 if lengths else 0.0
    unique_ratio = len(set(cmds)) / len(cmds) if cmds else 0.0

    # typos
    typo_count = sum(1 for s in steps if s.get("has_typo") in (True, "True", "true"))
    typo_rate = typo_count / n if n else 0.0

    # dead-end errors
    error_count = sum(1 for s in steps if s.get("response_category") in ERROR_CATEGORIES)
    error_rate = error_count / n if n else 0.0

    # adaptation: hints actually acted on
    hint_count = sum(1 for s in steps if s.get("response_category") == "not_found_with_hint")
    adapt_count = _adaptation_count(steps)
    recovery_rate = adapt_count / hint_count if hint_count else 0.0

    # task progress
    progresses = [s.get("task_progress", 0.0) for s in steps]
    final_progress = progresses[-1] if progresses else 0.0

    # sequence entropy over full commands
    cmd_counts = list(Counter(cmds).values())
    entropy = _shannon_entropy(cmd_counts)

    return {
        "session_id": sess["session_id"],
        "actor_label": sess["actor_label"],
        "evasion_strategy": sess.get("evasion_strategy", "none"),
        "generator_family": sess.get("generator_family", ""),
        "task": sess.get("task", ""),
        "split": sess.get("split", ""),
        "session_duration_s": round(duration, 2),
        "command_count": n,
        "inter_cmd_mean_s": round(gap_mean, 3),
        "inter_cmd_std_s": round(gap_std, 3),
        "min_gap_s": round(gap_min, 2),
        "max_gap_s": round(gap_max, 2),
        "cmd_len_mean": round(len_mean, 1),
        "cmd_len_std": round(len_std, 1),
        "unique_cmd_ratio": round(unique_ratio, 2),
        "typo_count": typo_count,
        "typo_rate": round(typo_rate, 2),
        "error_count": error_count,
        "error_rate": round(error_rate, 2),
        "error_recovery_rate": round(recovery_rate, 1),
        "adaptation_event_count": adapt_count,
        "not_found_with_hint_count": hint_count,
        "final_task_progress": round(final_progress, 2),
        "interaction_depth": n,
        "sequence_entropy": round(entropy, 3),
    }


def extract_all(sessions: list[dict]) -> list[dict]:
    return [extract_one(s) for s in sessions]


def main():
    with open(DS1, encoding="utf-8") as f:
        sessions = json.load(f)
    print(f"loaded {len(sessions)} sessions from {DS1}")

    rows = extract_all(sessions)

    cols = list(rows[0].keys())
    with open(DS2_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows x {len(cols)} columns to {DS2_OUT}")

    print("\nsample row:")
    for k, v in rows[0].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
