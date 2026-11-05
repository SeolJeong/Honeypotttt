"""Validate and merge contributed session files into one dataset.

    python scripts/merge_sessions.py contrib/*.jsonl \\
        --base data/v2/dataset1_sessions.json \\
        --test-participants P004 P005 P009 \\
        --test-families llama3.1-8b \\
        --out data/v3

Split rules for sessions whose split is 'unassigned' (all contributed ones):
    human     -> test if participant_id in --test-participants
    AI-driven -> test if generator_family in --test-families, or evasion in --test-evasions
    scripted  -> test if generator_family in --test-families
Sessions from --base keep their split. --drop-simulated removes the simulated
human/AI policy sessions (keeps scripted), for a real-actors-only dataset.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from honeyai.features import extract_all
from honeyai.sim.recording import dedupe, validate
from scripts.generate_sessions import to_ds4


def load_any(path):
    with open(path, encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(l) for l in f if l.strip()]
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--base", default=None)
    ap.add_argument("--test-participants", nargs="*", default=[])
    ap.add_argument("--test-families", nargs="*", default=[])
    ap.add_argument("--test-evasions", nargs="*", default=["avoid_systematic_enumeration", "combined"])
    ap.add_argument("--drop-simulated", action="store_true")
    ap.add_argument("--keep-incomplete", action="store_true",
                    help="keep agent sessions that ended on an API error/refusal (dropped by default)")
    ap.add_argument("--out", default=os.path.join("data", "v3"))
    a = ap.parse_args()

    sessions, bad, seen, dropped = [], 0, set(), 0
    sources = ([a.base] if a.base else []) + a.files
    for path in sources:
        for s in load_any(path):
            if s.get("incomplete") and not a.keep_incomplete:
                dropped += 1
                continue
            errs = validate(s)
            if errs:
                bad += 1
                print(f"  skip {s.get('session_id')} ({path}): {'; '.join(errs[:3])}")
                continue
            if s["session_id"] in seen:
                continue
            seen.add(s["session_id"])
            if a.drop_simulated and s["data_source"] == "simulated_policy":
                continue
            if s.get("split", "unassigned") == "unassigned":
                lab = s["actor_label"]
                test = (lab == "human" and s.get("participant_id") in a.test_participants) or \
                       (lab != "human" and s["generator_family"] in a.test_families) or \
                       (lab == "AI-driven" and s["evasion_strategy"] in a.test_evasions)
                s["split"] = "test" if test else "train"
            sessions.append(s)

    sessions, dup = dedupe(sessions)
    if dup["dup_within_split"] or dup["test_equals_train"]:
        print(f"duplicate check: dropped {dup['dup_within_split']} within-split duplicates, "
              f"{dup['test_equals_train']} test sessions identical to a training session")

    # leakage checks
    tr_p = {s.get("participant_id") for s in sessions if s["split"] == "train" and s["actor_label"] == "human"}
    te_p = {s.get("participant_id") for s in sessions if s["split"] == "test" and s["actor_label"] == "human"}
    if tr_p & te_p:
        print(f"WARNING participants in both splits: {sorted(tr_p & te_p)}")
    for lab in ("AI-driven", "scripted"):
        tr = {s["generator_family"] for s in sessions if s["split"] == "train" and s["actor_label"] == lab}
        te = {s["generator_family"] for s in sessions if s["split"] == "test" and s["actor_label"] == lab}
        if not (te - tr):
            print(f"WARNING no {lab} family is held out of training (test families {sorted(te)})")

    os.makedirs(a.out, exist_ok=True)
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

    print(f"\nmerged {len(sessions)} sessions ({bad} invalid, {dropped} incomplete skipped) -> {a.out}")
    for k, v in sorted(Counter((s["split"], s["actor_label"], s["data_source"]) for s in sessions).items()):
        print(f"  {k[0]:5s} {k[1]:10s} {k[2]:32s} {v}")


if __name__ == "__main__":
    main()
