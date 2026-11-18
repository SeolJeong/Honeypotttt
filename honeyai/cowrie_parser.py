"""Part 9b: Cowrie JSON log parser.

Converts raw Cowrie JSON log files into HoneyAI dataset1 session format
so the trained models can classify wild (unlabelled) sessions.

Cowrie emits one JSON object per line. Key event types:
  cowrie.session.connect   - new SSH connection
  cowrie.login.success     - successful authentication (enters Stage 2)
  cowrie.login.failed      - failed auth attempt (Stage 1 noise)
  cowrie.command.input     - attacker typed a command
  cowrie.command.failed    - command not found
  cowrie.session.closed    - session ended

Usage:
    python scripts/parse_cowrie.py path/to/cowrie.json
    python scripts/parse_cowrie.py path/to/cowrie.json --classify
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime

DATA_DIR = "data"
WILD_OUT = os.path.join(DATA_DIR, "wild_sessions.json")

# commands to ignore (Cowrie internal / not real attacker input)
IGNORE_COMMANDS = {"", "exit", "logout", "quit"}

# maximum gap (seconds) before splitting into a new session
MAX_GAP_S = 600  # 10 minutes idle = new session

# minimum commands to keep a session (too short = not useful)
MIN_COMMANDS = 2

# maximum steps per session (match training data limit)
MAX_STEPS = 16


def parse_timestamp(ts_str):
    """Parse Cowrie timestamp to datetime. Handles multiple formats."""
    for fmt in [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
    ]:
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    return None


def load_cowrie_log(path):
    """Load a Cowrie JSON log file (one JSON object per line)."""
    events = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                events.append(event)
            except json.JSONDecodeError:
                print(f"  warning: skipped malformed line {line_num}")
    return events


def group_by_session(events):
    """Group events by Cowrie session ID."""
    sessions = defaultdict(list)
    for ev in events:
        sid = ev.get("session")
        if sid:
            sessions[sid].append(ev)
    # sort each session's events by timestamp
    for sid in sessions:
        sessions[sid].sort(key=lambda e: e.get("timestamp", ""))
    return dict(sessions)


def has_successful_auth(events):
    """Check if a session includes a successful login (passes Stage 1)."""
    return any(e.get("eventid") == "cowrie.login.success" for e in events)


def extract_commands(events):
    """Extract command events with timestamps from a session's events."""
    cmds = []
    for ev in events:
        eid = ev.get("eventid", "")
        if eid in ("cowrie.command.input", "cowrie.command.failed"):
            cmd = ev.get("input", ev.get("message", "")).strip()
            if cmd and cmd.lower() not in IGNORE_COMMANDS:
                ts = parse_timestamp(ev.get("timestamp", ""))
                cmds.append({
                    "command": cmd,
                    "timestamp": ts,
                    "eventid": eid,
                })
    return cmds


def detect_typo(cmd, next_cmd):
    """Simple heuristic: a command followed by a near-identical one is likely a typo correction."""
    if not next_cmd:
        return False
    # check if commands differ by 1-2 characters (Levenshtein-like)
    if abs(len(cmd) - len(next_cmd)) <= 2:
        diffs = sum(1 for a, b in zip(cmd, next_cmd) if a != b)
        diffs += abs(len(cmd) - len(next_cmd))
        if 0 < diffs <= 2:
            return True
    return False


def classify_response(eventid, cmd):
    """Assign a response category based on Cowrie event type."""
    if eventid == "cowrie.command.failed":
        prog = cmd.split()[0] if cmd.split() else cmd
        return "command_not_found"
    return "success"


def think_time_bin(seconds):
    """Convert gap in seconds to a timing bin."""
    if seconds < 2:
        return "short"
    if seconds < 6:
        return "medium"
    if seconds < 15:
        return "long"
    return "very_long"


def split_long_sessions(cmds):
    """Split a command list into sub-sessions on long gaps or MAX_STEPS."""
    if not cmds:
        return []
    segments = []
    current = [cmds[0]]
    for i in range(1, len(cmds)):
        gap = 0
        if cmds[i]["timestamp"] and cmds[i - 1]["timestamp"]:
            gap = (cmds[i]["timestamp"] - cmds[i - 1]["timestamp"]).total_seconds()
        if gap > MAX_GAP_S or len(current) >= MAX_STEPS:
            segments.append(current)
            current = [cmds[i]]
        else:
            current.append(cmds[i])
    if current:
        segments.append(current)
    return segments


def commands_to_session(cmds, session_id, src_ip=None):
    """Convert a list of command dicts into a dataset1-format session."""
    steps = []
    cumulative = 0.0
    for i, cmd_ev in enumerate(cmds):
        gap = 0.0
        if i > 0 and cmd_ev["timestamp"] and cmds[i - 1]["timestamp"]:
            gap = (cmd_ev["timestamp"] - cmds[i - 1]["timestamp"]).total_seconds()
            gap = max(0.0, gap)
        cumulative += gap

        has_typo = False
        if i < len(cmds) - 1:
            has_typo = detect_typo(cmd_ev["command"], cmds[i + 1]["command"])

        resp_cat = classify_response(cmd_ev["eventid"], cmd_ev["command"])

        steps.append({
            "step": i,
            "command": cmd_ev["command"],
            "has_typo": has_typo,
            "system_response": "(wild data - no captured output)",
            "response_category": resp_cat,
            "inter_command_gap_s": round(gap, 2),
            "think_time_bin": think_time_bin(gap),
            "cumulative_time_s": round(cumulative, 2),
            "task_progress": round((i + 1) / len(cmds), 2),
        })

    return {
        "session_id": session_id,
        "actor_label": "unknown",
        "participant_id": None,
        "task": "wild",
        "split": "wild",
        "evasion_strategy": "unknown",
        "generator_family": "wild",
        "total_steps": len(steps),
        "session_duration_s": round(cumulative, 2),
        "data_source": "cowrie_wild",
        "src_ip": src_ip,
        "interactions": steps,
    }


def parse_cowrie_file(path):
    """Parse a Cowrie log file into HoneyAI sessions."""
    print(f"loading {path}...")
    events = load_cowrie_log(path)
    print(f"  {len(events)} events loaded")

    grouped = group_by_session(events)
    print(f"  {len(grouped)} raw sessions")

    sessions = []
    wild_id = 1
    skipped_noauth = 0
    skipped_short = 0

    for cowrie_sid, sess_events in grouped.items():
        # Stage 1: only keep sessions with successful auth
        if not has_successful_auth(sess_events):
            skipped_noauth += 1
            continue

        # extract commands
        cmds = extract_commands(sess_events)
        if len(cmds) < MIN_COMMANDS:
            skipped_short += 1
            continue

        # get source IP
        src_ip = None
        for ev in sess_events:
            if ev.get("src_ip"):
                src_ip = ev["src_ip"]
                break

        # split long sessions
        segments = split_long_sessions(cmds)
        for seg in segments:
            if len(seg) < MIN_COMMANDS:
                continue
            sid = f"WILD-{wild_id:05d}"
            wild_id += 1
            sessions.append(commands_to_session(seg, sid, src_ip))

    print(f"  {len(sessions)} interactive sessions extracted")
    print(f"  {skipped_noauth} skipped (no successful auth)")
    print(f"  {skipped_short} skipped (fewer than {MIN_COMMANDS} commands)")
    return sessions


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Parse Cowrie logs into HoneyAI sessions")
    ap.add_argument("logfile", help="Path to Cowrie JSON log file")
    ap.add_argument("--output", "-o", default=WILD_OUT,
                    help=f"Output path (default: {WILD_OUT})")
    ap.add_argument("--classify", action="store_true",
                    help="Run trained XGBoost on parsed sessions")
    args = ap.parse_args()

    sessions = parse_cowrie_file(args.logfile)

    if not sessions:
        print("no sessions to write")
        return

    # write sessions
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {len(sessions)} sessions to {args.output}")

    # basic stats
    gaps = []
    for s in sessions:
        for st in s["interactions"]:
            if st["inter_command_gap_s"] > 0:
                gaps.append(st["inter_command_gap_s"])
    if gaps:
        import statistics
        print(f"\ntiming summary:")
        print(f"  median gap: {statistics.median(gaps):.2f}s")
        print(f"  mean gap: {statistics.mean(gaps):.2f}s")
        print(f"  p10/p90: {sorted(gaps)[len(gaps)//10]:.2f}s / {sorted(gaps)[9*len(gaps)//10]:.2f}s")
        print(f"  steps per session: {statistics.mean([s['total_steps'] for s in sessions]):.1f}")

    # optional classification
    if args.classify:
        print("\n--- classifying with XGBoost ---")
        try:
            from honeyai.features import extract_one
            from honeyai.models.xgb_baseline import load_dataset, FEATURE_COLS, LABEL_NAMES
            import numpy as np
            from collections import Counter
            from xgboost import XGBClassifier

            # train on full training set
            data = load_dataset()
            X_train, y_train = data["X_train"], data["y_train"]
            counts = Counter(y_train)
            total = len(y_train)
            n_cls = len(counts)
            weights = {c: total / (n_cls * cnt) for c, cnt in counts.items()}
            sample_w = np.array([weights[lb] for lb in y_train])

            clf = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, eval_metric="mlogloss",
            )
            clf.fit(X_train, y_train, sample_weight=sample_w)

            # extract features and predict
            for s in sessions:
                feats = extract_one(s)
                X = np.array([[float(feats[c]) for c in FEATURE_COLS]])
                pred = clf.predict(X)[0]
                proba = clf.predict_proba(X)[0]
                label = LABEL_NAMES[pred]
                conf = proba[pred]
                print(f"  {s['session_id']}: {label} (confidence: {conf:.2f}, "
                      f"steps: {s['total_steps']}, duration: {s['session_duration_s']:.1f}s)")

            pred_counts = Counter()
            for s in sessions:
                feats = extract_one(s)
                X = np.array([[float(feats[c]) for c in FEATURE_COLS]])
                pred = clf.predict(X)[0]
                pred_counts[LABEL_NAMES[pred]] += 1
            print(f"\npredicted distribution: {dict(pred_counts)}")

        except ImportError as e:
            print(f"  could not classify: {e}")
            print("  make sure xgboost is installed and honeyai package is accessible")


if __name__ == "__main__":
    main()