"""Part 2: Replace placeholder responses in the team dataset.

Reads dataset1_sessions.json, rewrites `success` and `partial_result`
responses with realistic shell output (from honeyai.responses), then
regenerates dataset4_llm_sequences.jsonl to match.

Usage:
    python scripts/fix_responses.py

Reads from data/ and writes the updated files back in place.
"""
from __future__ import annotations

import json
import os
import sys

# allow importing honeyai package from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from honeyai.responses import simulate

DATA_DIR = "data"
DS1 = os.path.join(DATA_DIR, "dataset1_sessions.json")
DS4 = os.path.join(DATA_DIR, "dataset4_llm_sequences.jsonl")

# --- GAP bins (must match the team's dataset4 convention) ---
GAP_BINS = [
    (2.0,  "SHORT"),
    (6.0,  "MEDIUM"),
    (15.0, "LONG"),
]

def gap_bin(seconds: float) -> str:
    for threshold, label in GAP_BINS:
        if seconds < threshold:
            return label
    return "VERY_LONG"


def fix_dataset1(sessions: list[dict]) -> int:
    """Rewrite placeholder responses in place. Returns count of changes."""
    changed = 0
    for sess in sessions:
        for step in sess["interactions"]:
            cat = step["response_category"]
            if cat not in ("success", "partial_result"):
                continue
            new_resp = simulate(step["command"], cat)
            if new_resp and new_resp != step["system_response"]:
                step["system_response"] = new_resp
                changed += 1
    return changed


def rebuild_interleaved(sess: dict) -> str:
    """Rebuild the interleaved text for one session (dataset4 format)."""
    parts = []
    for step in sess["interactions"]:
        gap = step.get("inter_command_gap_s", 0.0)
        gbin = gap_bin(gap)
        parts.append(f"[GAP {gbin}] CMD: {step['command']}")
        parts.append(f"RESP [{step['response_category']}]: {step['system_response']}")
    return "\n".join(parts)


def rebuild_dataset4(sessions: list[dict]) -> list[dict]:
    """Regenerate dataset4 records from updated dataset1."""
    lookup = {s["session_id"]: s for s in sessions}
    records = []
    for sess in sessions:
        records.append({
            "session_id": sess["session_id"],
            "split": sess["split"],
            "actor_label": sess["actor_label"],
            "evasion_strategy": sess.get("evasion_strategy", "none"),
            "generator_family": sess.get("generator_family", ""),
            "task": sess.get("task", ""),
            "total_steps": sess["total_steps"],
            "interleaved_text": rebuild_interleaved(sess),
        })
    return records


def main():
    # load dataset1
    with open(DS1, encoding="utf-8") as f:
        sessions = json.load(f)
    print(f"loaded {len(sessions)} sessions from {DS1}")

    # fix responses
    n = fix_dataset1(sessions)
    print(f"replaced {n} placeholder responses")

    # write updated dataset1
    with open(DS1, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2, ensure_ascii=False)
    print(f"wrote {DS1}")

    # rebuild and write dataset4
    ds4 = rebuild_dataset4(sessions)
    with open(DS4, "w", encoding="utf-8") as f:
        for rec in ds4:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {DS4} ({len(ds4)} records)")

    # quick sanity check: no more placeholders
    remaining = 0
    for sess in sessions:
        for step in sess["interactions"]:
            if step["response_category"] in ("success", "partial_result"):
                if step["system_response"].startswith("[OK]") or \
                   step["system_response"].startswith("[PARTIAL]"):
                    remaining += 1
    print(f"\nplaceholders remaining: {remaining}")
    if remaining == 0:
        print("all good — no placeholders left")
    else:
        print(f"WARNING: {remaining} placeholders still present")


if __name__ == "__main__":
    main()
