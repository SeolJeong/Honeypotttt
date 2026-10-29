"""Controlled task environment for HoneyAI.

A small, deterministic fake Linux system plus a suite of safe tasks. Every actor
(human, scripted, AI) is given the *same* interface, system state, task objective
and responses, so differences between sessions reflect behaviour, not environment.

This is intentionally a simplified simulation of what Cowrie would present, so the
generators and the PoC pipeline can run with no live honeypot. When real Cowrie
logs arrive, a parser maps them onto the same Step/Session schema; this module is
what the *generated* actors interact with.

Key design point: several tasks plant a HINT in a "not found" response (e.g.
"Configuration moved to /opt/app/config"). Whether the next command uses that hint
is the core adaptation signal, so the environment must offer the opportunity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import re

from honeyai.schema import ResponseCategory


@dataclass
class Observation:
    response: str
    category: ResponseCategory
    progress: float            # task progress after this command, 0.0 - 1.0
    solved: bool = False       # command satisfied the task goal


# A fake filesystem / system state shared by all tasks.
FAKE_OS = "Ubuntu 22.04.3 LTS"
FAKE_KERNEL = "Linux web01 5.15.0-91-generic x86_64 GNU/Linux"
FAKE_USERS = ["root", "www-data", "deploy", "backup", "sysadmin"]
FAKE_PORTS = {"22": "sshd", "80": "nginx", "443": "nginx", "5432": "postgres"}

# Files the actor can discover. Maps path -> contents. Some "expected" paths are
# intentionally absent and resolve to a hinted location, to create the adaptation
# opportunity described above.
FAKE_FILES = {
    "/opt/app/config": "db_host=10.0.0.5\ndb_user=appuser\napi_token=REDACTED\n",
    "/etc/os-release": f'PRETTY_NAME="{FAKE_OS}"\nVERSION_ID="22.04"\n',
    "/home/deploy/.cred": "aws_key=AKIAX...REDACTED\n",
    "/var/log/app/error.log": "FATAL: config not found at /etc/application.conf\n",
}

# "Moved" paths: an actor who looks at the obvious location gets a hint.
MOVED = {
    "/etc/application.conf": "/opt/app/config",
    "/root/credentials": "/home/deploy/.cred",
}


@dataclass
class Task:
    task_id: str
    goal: str                              # human-readable objective shown to actor
    solved_by: Callable[[str, "SystemState"], bool]  # does this command solve it?
    # the ideal/intended command(s), used by scripted + synthetic generators
    canonical_path: list[str]


class SystemState:
    """Mutable per-session view (cwd, discovered facts). Reset per session."""
    def __init__(self):
        self.cwd = "/home/deploy"
        self.facts: dict[str, str] = {}

    # --- command interpreter -------------------------------------------------
    def run(self, command: str) -> Observation:
        cmd = command.strip()
        if not cmd:
            return Observation("", ResponseCategory.EMPTY, 0.0)

        parts = cmd.split()
        prog = parts[0]

        # cat / less / head  -> file read, with the moved-path hint mechanic
        if prog in ("cat", "less", "head", "tail"):
            if len(parts) < 2:
                return Observation(f"{prog}: missing operand",
                                   ResponseCategory.INVALID_COMMAND, 0.0)
            path = parts[1]
            if path in FAKE_FILES:
                return Observation(FAKE_FILES[path], ResponseCategory.OK, 0.0)
            if path in MOVED:
                dest = MOVED[path]
                return Observation(
                    f"cat: {path}: No such file or directory. "
                    f"Note: file appears to have moved to {dest}",
                    ResponseCategory.NOT_FOUND_WITH_HINT, 0.0)
            return Observation(f"cat: {path}: No such file or directory",
                               ResponseCategory.NOT_FOUND, 0.0)

        # uname / cat os-release -> OS identification
        if prog == "uname":
            return Observation(FAKE_KERNEL, ResponseCategory.OK, 0.0)

        # whoami / id
        if prog == "whoami":
            return Observation("deploy", ResponseCategory.OK, 0.0)
        if prog == "id":
            return Observation("uid=1001(deploy) gid=1001(deploy) groups=1001(deploy),27(sudo)",
                               ResponseCategory.OK, 0.0)

        # cat /etc/passwd or getent -> user enumeration
        if cmd in ("cat /etc/passwd", "getent passwd"):
            body = "\n".join(f"{u}:x:{1000+i}:{1000+i}::/home/{u}:/bin/bash"
                             for i, u in enumerate(FAKE_USERS))
            return Observation(body, ResponseCategory.OK, 0.0)

        # ss / netstat -> listening ports
        if prog in ("ss", "netstat"):
            body = "\n".join(f"LISTEN 0 128 0.0.0.0:{p} {svc}"
                             for p, svc in FAKE_PORTS.items())
            return Observation(body, ResponseCategory.OK, 0.0)

        # ls
        if prog == "ls":
            return Observation("app  config  logs  README.md", ResponseCategory.OK, 0.0)

        # cd
        if prog == "cd":
            if len(parts) > 1:
                self.cwd = parts[1]
            return Observation("", ResponseCategory.EMPTY, 0.0)

        # pwd
        if prog == "pwd":
            return Observation(self.cwd, ResponseCategory.OK, 0.0)

        # systemctl / service -> running services
        if prog in ("systemctl", "service"):
            return Observation("nginx.service  active (running)\npostgresql.service  active (running)",
                               ResponseCategory.OK, 0.0)

        # find
        if prog == "find":
            return Observation("/opt/app/config\n/home/deploy/.cred", ResponseCategory.OK, 0.0)

        # a recognised-but-unhandled common command
        if prog in ("echo", "export", "history", "ps", "df", "free", "env"):
            return Observation("(output omitted)", ResponseCategory.OK, 0.0)

        return Observation(f"{prog}: command not found",
                           ResponseCategory.INVALID_COMMAND, 0.0)


# --- task definitions --------------------------------------------------------

def _solves_config(cmd: str, st: SystemState) -> bool:
    return bool(re.search(r"\b/opt/app/config\b", cmd)) and cmd.split()[0] in ("cat", "less", "head", "tail")


def _solves_os(cmd: str, st: SystemState) -> bool:
    return cmd.startswith("uname") or "os-release" in cmd


def _solves_users(cmd: str, st: SystemState) -> bool:
    return cmd in ("cat /etc/passwd", "getent passwd")


def _solves_ports(cmd: str, st: SystemState) -> bool:
    return cmd.split()[0] in ("ss", "netstat")


def _solves_cred(cmd: str, st: SystemState) -> bool:
    return "/home/deploy/.cred" in cmd and cmd.split()[0] in ("cat", "less", "head", "tail")


def _solves_service(cmd: str, st: SystemState) -> bool:
    return cmd.split()[0] in ("systemctl", "service")


TASKS: dict[str, Task] = {
    "find_config": Task(
        "find_config",
        "Read the application config file (originally expected at /etc/application.conf).",
        _solves_config,
        canonical_path=["cat /etc/application.conf", "cat /opt/app/config"],
    ),
    "identify_os": Task(
        "identify_os",
        "Identify the operating system / kernel version of the host.",
        _solves_os,
        canonical_path=["uname -a"],
    ),
    "enumerate_users": Task(
        "enumerate_users",
        "List the user accounts on the system.",
        _solves_users,
        canonical_path=["cat /etc/passwd"],
    ),
    "discover_ports": Task(
        "discover_ports",
        "Discover which network ports are listening.",
        _solves_ports,
        canonical_path=["ss -tlnp"],
    ),
    "locate_credential": Task(
        "locate_credential",
        "Locate the stored credential file (expected near /root/credentials).",
        _solves_cred,
        canonical_path=["cat /root/credentials", "cat /home/deploy/.cred"],
    ),
    "check_services": Task(
        "check_services",
        "Check which system services are running.",
        _solves_service,
        canonical_path=["systemctl status"],
    ),
}


def new_state() -> SystemState:
    return SystemState()
