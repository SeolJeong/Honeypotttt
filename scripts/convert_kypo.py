"""Convert the KYPO shell-command dataset (real humans) into HoneyAI sessions.

KYPO: Švábenský et al., "Dataset of Shell Commands Used by Participants of
Hands-on Cybersecurity Training", Data in Brief 38 (2021),
doi:10.1016/j.dib.2021.107398. Zenodo record 8136017, CC BY 4.0.
Download: https://zenodo.org/records/8136017/files/data.zip?download=1

What KYPO has: real trainees' commands with timestamps, host and working dir.
What it does NOT have: command outputs. So:
  - sessions are written with system_response = null and
    response_category = null (has_responses = false)
  - only COMMANDS-ONLY models can be evaluated on them
    (train_qlora.py --no-responses, text_baseline.py --no-responses,
    or a timing/command feature model); a model that reads responses
    would be judged on input it never saw
  - this is an EXTERNAL real-human check, never training data

Segmentation (a trainee file covers a whole multi-hour exercise):
  - a new session starts when the host changes (e.g. ssh from the attacker box
    to a target) or after an idle gap > --idle-gap seconds
  - each session is cut into windows of at most --max-steps commands,
    matching the 1-16 step sessions the models are trained on
  - the first command of every segment is dropped from the window because its
    gap (time since login) is unknown; every kept command has a real gap

Hosts: 'attacker' is the trainee's own Kali machine. 'server', 'web',
'client', 'laboratory' are target machines reached over SSH, the closest
match to a honeypot session; those sessions get on_target_host = true.
IP addresses are dropped.

Usage:
    python scripts/convert_kypo.py                               # external/kypo/data.zip
    python scripts/convert_kypo.py --zip path/to/data.zip --out external/kypo/kypo_sessions.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import zipfile
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from honeyai.sim.actors import gap_bin

TARGET_HOSTS = {"server", "web", "client", "laboratory"}


def _ts(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


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


def load_trainees(zip_path: str, include_msf: bool) -> dict[str, list[dict]]:
    out = {}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.endswith(".json"):
                continue
            recs = []
            for line in z.read(name).decode("utf-8", "replace").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if not r.get("cmd"):
                    continue                          # tag-only records
                if r.get("cmd_type") != "bash-command" and not include_msf:
                    continue
                recs.append(r)
            recs.sort(key=lambda r: r["timestamp_str"])
            if recs:
                out[name] = recs
    return out


def segments(recs: list[dict], idle_gap: float) -> list[list[dict]]:
    segs, cur = [], []
    for r in recs:
        if cur and (r.get("hostname") != cur[-1].get("hostname")
                    or _ts(r["timestamp_str"]) - _ts(cur[-1]["timestamp_str"]) > idle_gap):
            segs.append(cur)
            cur = []
        cur.append(r)
    if cur:
        segs.append(cur)
    return segs


def to_sessions(trainees: dict, idle_gap: float, max_steps: int, min_steps: int) -> list[dict]:
    sessions = []
    for name, recs in sorted(trainees.items()):
        parts = name.split("/")
        training, event = parts[0], parts[1] if len(parts) > 2 else ""
        sandbox = recs[0].get("sandbox_id", "x")
        pid = f"KYPO-{training.replace(' ', '_')}-{event[:10]}-sb{sandbox}"
        for si, seg in enumerate(segments(recs, idle_gap)):
            # commands with a real preceding gap
            steps = []
            for prev, r in zip(seg, seg[1:]):
                steps.append((r, _ts(r["timestamp_str"]) - _ts(prev["timestamp_str"])))
            for wi in range(0, len(steps), max_steps):
                win = steps[wi:wi + max_steps]
                if len(win) < min_steps:
                    continue
                t, inter = 0.0, []
                for i, (r, gap) in enumerate(win):
                    gap = round(max(0.0, gap), 2)
                    t += gap
                    nxt = win[i + 1][0]["cmd"] if i + 1 < len(win) else None
                    d = _edit_distance(r["cmd"], nxt) if nxt else 99
                    inter.append({
                        "step": i, "command": r["cmd"],
                        "has_typo": 1 <= d <= 2,          # heuristic: retyped almost the same command next
                        "system_response": None, "response_category": None,
                        "inter_command_gap_s": gap, "think_time_bin": gap_bin(gap),
                        "cumulative_time_s": round(t, 2), "task_progress": None,
                        "working_dir": r.get("wd"), "kypo_cmd_type": r.get("cmd_type"),
                    })
                host = win[0][0].get("hostname")
                sessions.append({
                    "session_id": f"{pid}-s{si}-w{wi // max_steps}",
                    "actor_label": "human", "participant_id": pid,
                    "task": f"kypo:{training}", "split": "external_kypo",
                    "evasion_strategy": "none", "generator_family": "kypo_human",
                    "data_source": "kypo_real_human", "has_responses": False,
                    "host": host, "on_target_host": host in TARGET_HOSTS,
                    "total_steps": len(inter),
                    "session_duration_s": round(t, 2),
                    "interactions": inter,
                })
    return sessions


def to_ds4(s: dict) -> dict:
    """Commands-only dataset4 line (no RESP lines: KYPO has no outputs)."""
    text = "\n".join(f"[GAP {st['think_time_bin'].upper()}] CMD: {st['command']}" for st in s["interactions"])
    return {k: s[k] for k in ("session_id", "split", "actor_label", "evasion_strategy", "generator_family",
                              "participant_id", "task", "total_steps")} | {
        "interleaved_text": text, "has_responses": False, "on_target_host": s["on_target_host"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=os.path.join("external", "kypo", "data.zip"))
    ap.add_argument("--out", default=os.path.join("external", "kypo", "kypo_sessions.json"))
    ap.add_argument("--idle-gap", type=float, default=600, help="seconds of inactivity that end a session")
    ap.add_argument("--max-steps", type=int, default=16)
    ap.add_argument("--min-steps", type=int, default=2)
    ap.add_argument("--include-msf", action="store_true", help="also keep Metasploit console commands")
    a = ap.parse_args()
    if not os.path.exists(a.zip):
        sys.exit(f"{a.zip} not found. Download: https://zenodo.org/records/8136017/files/data.zip?download=1")

    trainees = load_trainees(a.zip, a.include_msf)
    sessions = to_sessions(trainees, a.idle_gap, a.max_steps, a.min_steps)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=1, ensure_ascii=False)
    ds4 = os.path.splitext(a.out)[0] + "_ds4_commands_only.jsonl"
    with open(ds4, "w", encoding="utf-8") as f:
        for s in sessions:
            f.write(json.dumps(to_ds4(s), ensure_ascii=False) + "\n")

    gaps = [st["inter_command_gap_s"] for s in sessions for st in s["interactions"]]
    q = statistics.quantiles(gaps, n=10)
    print(f"trainees {len(trainees)} -> {len(sessions)} sessions "
          f"({sum(s['on_target_host'] for s in sessions)} on target hosts), "
          f"{len(gaps)} commands")
    print(f"steps/session median {statistics.median([s['total_steps'] for s in sessions])}; "
          f"hosts {dict(Counter(s['host'] for s in sessions))}")
    print(f"gap s: p10 {q[0]:.1f}  p50 {statistics.median(gaps):.1f}  p90 {q[8]:.1f}")
    print(f"wrote {a.out} and {ds4}")


if __name__ == "__main__":
    main()
