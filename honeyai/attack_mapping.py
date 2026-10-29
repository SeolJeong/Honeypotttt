"""Command -> MITRE ATT&CK mapping (the "type of attack" layer).

Rule-based mapper from shell commands to ATT&CK techniques/tactics. This is a
triage/dashboard enrichment, NOT a second ML task: it tells an analyst *what* an
actor is doing, alongside HoneyAI's ML answer to *who* the actor is.

It works on both generated sessions and real Cowrie commands, so the same labels
appear in the dashboard and in wild-traffic case studies. Mappings follow common
Cowrie-to-ATT&CK conventions; extend PATTERNS as you see new commands in the wild.
"""
from __future__ import annotations

import re
from collections import Counter

# Each entry: (regex on the command, technique id, technique name, tactic)
PATTERNS: list[tuple[str, str, str, str]] = [
    # Discovery
    (r"\buname\b|\b/etc/os-release\b|\bhostnamectl\b|\blsb_release\b",
     "T1082", "System Information Discovery", "Discovery"),
    (r"\b/etc/passwd\b|\bgetent\b|\bwho\b|\bw\b|\blastlog\b|\bcat /etc/group\b",
     "T1087", "Account Discovery", "Discovery"),
    (r"\bss\b|\bnetstat\b|\blsof -i\b",
     "T1049", "System Network Connections Discovery", "Discovery"),
    (r"\bifconfig\b|\bip a\b|\bip addr\b|\barp\b|\broute\b",
     "T1016", "System Network Configuration Discovery", "Discovery"),
    (r"\bsystemctl\b|\bservice\b|\bps\b|\btop\b|\bps aux\b",
     "T1057", "Process / Service Discovery", "Discovery"),
    (r"\bfind\b|\blocate\b|\bls\b|\bdir\b",
     "T1083", "File and Directory Discovery", "Discovery"),
    # Credential access
    (r"\.cred\b|credentials|\.aws|id_rsa|\.ssh|password|shadow|api_token|\.env\b",
     "T1552", "Unsecured Credentials", "Credential Access"),
    # Collection / config
    (r"\bcat\b.*\bconfig\b|application\.conf|\.conf\b",
     "T1005", "Data from Local System", "Collection"),
    # Execution / ingress (downloaders)
    (r"\bwget\b|\bcurl\b|\btftp\b|\bftpget\b",
     "T1105", "Ingress Tool Transfer", "Command and Control"),
    (r"\bchmod\b\s+\+?x|\./[\w./-]+\s*$|\bbash -c\b|\bsh -c\b|\bperl\b|\bpython\b -c",
     "T1059", "Command and Scripting Interpreter", "Execution"),
    # Persistence
    (r"\bcrontab\b|\b/etc/rc\.local\b|\bsystemctl enable\b|authorized_keys|\buseradd\b|\badduser\b",
     "T1098", "Account Manipulation / Persistence", "Persistence"),
    # Defense evasion
    (r"\bhistory -c\b|\brm -rf\b|\bunset HISTFILE\b|\b>\s*/var/log",
     "T1070", "Indicator Removal", "Defense Evasion"),
    # Impact
    (r"\bcrypto|xmrig|miner|\bdd if=|mkfs|:(){:|wipe\b",
     "T1496", "Resource Hijacking / Impact", "Impact"),
]

COMPILED = [(re.compile(p, re.IGNORECASE), tid, name, tac) for p, tid, name, tac in PATTERNS]


def map_command(command: str) -> tuple[str, str, str] | None:
    """Return (technique_id, technique_name, tactic) for a command, or None."""
    for rx, tid, name, tac in COMPILED:
        if rx.search(command):
            return tid, name, tac
    return None


def map_session(session: dict) -> dict:
    """Summarise the ATT&CK techniques/tactics seen across a session.

    Returns per-step labels plus session-level counts and a primary technique.
    """
    per_step = []
    tech_counter: Counter = Counter()
    tactic_counter: Counter = Counter()
    for s in session["steps"]:
        m = map_command(s["command"])
        if m:
            tid, name, tac = m
            per_step.append({"step": s["step"], "technique": tid,
                             "name": name, "tactic": tac})
            tech_counter[f"{tid} {name}"] += 1
            tactic_counter[tac] += 1
    primary = tech_counter.most_common(1)[0][0] if tech_counter else "Unclassified"
    return {
        "per_step": per_step,
        "techniques": dict(tech_counter),
        "tactics": dict(tactic_counter),
        "primary_technique": primary,
        "n_mapped": sum(tech_counter.values()),
    }


if __name__ == "__main__":
    import os
    from honeyai.schema import read_sessions
    sessions = read_sessions(os.path.join("data", "sessions", "poc_dataset.jsonl"))
    agg: Counter = Counter()
    for s in sessions:
        agg[map_session(s)["primary_technique"]] += 1
    print("primary ATT&CK technique across PoC sessions:")
    for tech, n in agg.most_common():
        print(f"  {n:>4}  {tech}")
