"""Parse a Cowrie JSON log into HoneyAI sessions (unlabelled wild traffic).

Cowrie writes one JSON event per line. A single SSH session spans several events
(connect, login attempts, each typed command, close), tied together by a shared
`session` id. We reconstruct interactive sessions - those where the attacker
actually typed commands - into the HoneyAI schema.

IMPORTANT: wild traffic is UNLABELLED. We set actor_label = "unknown"; never treat
these as ground-truth human/scripted/AI. Use them for distribution comparison,
qualitative case studies, ATT&CK case studies and the demo - not reported accuracy.

What we can and cannot recover from Cowrie JSON:
  - commands: yes (cowrie.command.input events, in order)
  - timing:   yes (event timestamps -> inter-command think time)
  - responses: Cowrie does not log a clean per-command response, so system_response
    is left empty and response_category = "empty". The command sequence and timing
    are the usable signal for wild data.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from honeyai.schema import (Session, ActorLabel, ResponseCategory,
                            write_sessions)


def _parse_ts(ts: str) -> float:
    """Cowrie timestamp (ISO8601, e.g. 2026-10-29T09:45:12.123456Z) -> epoch sec."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(ts).timestamp()
    except ValueError:
        return 0.0


def parse_log(path: str, min_commands: int = 1) -> list[Session]:
    """Read a Cowrie JSON log and return interactive sessions in HoneyAI schema."""
    # gather events per session id
    by_session: dict[str, dict] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = ev.get("session")
            if not sid:
                continue
            rec = by_session.setdefault(sid, {"commands": [], "src_ip": ev.get("src_ip", ""),
                                              "connect_ts": None})
            eid = ev.get("eventid", "")
            ts = _parse_ts(ev.get("timestamp", "")) if ev.get("timestamp") else 0.0
            if eid == "cowrie.session.connect":
                rec["connect_ts"] = ts
                rec["src_ip"] = ev.get("src_ip", rec["src_ip"])
            elif eid == "cowrie.command.input":
                rec["commands"].append((ts, ev.get("input", "")))

    # build sessions from those with enough commands
    sessions = []
    for i, (sid, rec) in enumerate(by_session.items()):
        cmds = [c for c in rec["commands"] if c[1].strip()]
        if len(cmds) < min_commands:
            continue
        cmds.sort(key=lambda t: t[0])
        s = Session(
            session_id=f"WILD{i:06d}",
            actor_label=ActorLabel.UNKNOWN,
            task_id="wild",
            generator_family="cowrie-wild",
            participant_id=rec["src_ip"],
        )
        prev = rec["connect_ts"] or (cmds[0][0] if cmds else 0.0)
        for ts, cmd in cmds:
            gap = max(0.0, ts - prev) if (ts and prev) else 0.0
            prev = ts
            s.add_step(cmd, system_response="", response_category=ResponseCategory.EMPTY,
                       think_time_s=gap, task_progress=0.0)
        sessions.append(s)
    return sessions


def main():
    ap = argparse.ArgumentParser(description="Parse Cowrie JSON log into HoneyAI sessions.")
    ap.add_argument("logfile", help="path to cowrie.json")
    ap.add_argument("--out", default=os.path.join("data", "sessions", "wild_cowrie.jsonl"))
    ap.add_argument("--min-commands", type=int, default=1,
                    help="keep sessions with at least this many typed commands")
    args = ap.parse_args()

    sessions = parse_log(args.logfile, args.min_commands)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_sessions(sessions, args.out)

    total_cmds = sum(len(s.steps) for s in sessions)
    uniq_ips = len({s.participant_id for s in sessions})
    print(f"interactive sessions: {len(sessions)}")
    print(f"total commands      : {total_cmds}")
    print(f"unique source IPs   : {uniq_ips}")
    print(f"written             : {args.out}")
    if sessions:
        print("\nexample session (first one):")
        print(sessions[0].to_model_text()[:400])


if __name__ == "__main__":
    main()
