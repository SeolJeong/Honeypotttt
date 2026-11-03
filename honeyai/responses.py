"""Realistic shell-response simulator.

The team dataset ships placeholder responses for successful commands
(`[OK] <cmd> executed successfully.`). That gives the language model no semantic
signal to learn from. This module maps a command + its response category to
believable shell output, so a session reads like a real Linux shell.

Only `success` and `partial_result` are rewritten. The other categories
(`not_found_with_hint`, `permission_denied`, `command_not_found`,
`not_found_no_hint`) already carry realistic, session-specific text (e.g. the
hint path), so they are left untouched.

simulate(command, category) -> str
"""
from __future__ import annotations

# --- canned system facts (one consistent fake host) ------------------------
HOST = "web01"
KERNEL = f"Linux {HOST} 5.15.0-91-generic #101-Ubuntu SMP x86_64 GNU/Linux"
USERS = ["root", "www-data", "deploy", "backup", "sysadmin"]
PORTS = [("22", "sshd"), ("80", "nginx"), ("443", "nginx"), ("5432", "postgres")]

FILES = {
    "/etc/os-release": ('NAME="Ubuntu"\nVERSION="22.04.3 LTS (Jammy Jellyfish)"\n'
                        'ID=ubuntu\nVERSION_ID="22.04"'),
    "/etc/passwd": "\n".join(
        f"{u}:x:{1000+i}:{1000+i}::/home/{u}:/bin/bash" for i, u in enumerate(USERS)),
    "/opt/app/config": ("db_host=10.0.0.5\ndb_port=5432\ndb_user=appuser\n"
                        "api_token=d41d8cd98f00b204e980\nlog_level=info"),
    "/etc/application.conf": ("db_host=10.0.0.5\ndb_port=5432\ndb_user=appuser\n"
                             "api_token=d41d8cd98f00b204e980\nlog_level=info"),
}


def _program(cmd: str) -> str:
    return cmd.strip().split()[0] if cmd.strip() else ""


def _arg(cmd: str) -> str:
    parts = cmd.strip().split()
    return parts[1] if len(parts) > 1 else ""


def _success(cmd: str) -> str:
    p = _program(cmd)
    arg = _arg(cmd)

    if p == "uname":
        # one flag -> short, many flags / -a -> full string
        flags = [t for t in cmd.split()[1:] if t.startswith("-")]
        return KERNEL if ("-a" in cmd or len("".join(flags)) > 2) else "Linux"

    if p == "lsb_release":
        return ("Distributor ID:\tUbuntu\nDescription:\tUbuntu 22.04.3 LTS\n"
                "Release:\t22.04\nCodename:\tjammy")

    if p in ("cat", "head", "tail", "less", "more"):
        if arg in FILES:
            return FILES[arg]
        if "authorized_keys" in arg or ".ssh" in arg:
            return "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQD... deploy@web01"
        if arg.endswith((".conf", ".cfg", ".ini", "config")):
            return FILES["/opt/app/config"]
        return f"(contents of {arg})"

    if p == "ls":
        if "-l" in cmd or "-la" in cmd or "-al" in cmd:
            return ("total 20\ndrwxr-xr-x 4 deploy deploy 4096 Oct 31 09:12 app\n"
                    "-rw-r--r-- 1 deploy deploy  220 Oct 31 09:12 config\n"
                    "drwxr-xr-x 2 deploy deploy 4096 Oct 31 09:12 logs\n"
                    "-rw-r--r-- 1 deploy deploy  512 Oct 31 09:12 README.md")
        return "app  config  logs  README.md"

    if p == "find":
        return "/opt/app/config\n/home/deploy/.ssh/authorized_keys\n/var/log/app/error.log"

    if p == "grep":
        return f"{arg}:1:matched line containing the pattern"

    if p in ("ps",):
        return ("  PID TTY          TIME CMD\n  912 ?        00:00:01 sshd\n"
                " 1033 ?        00:00:04 nginx\n 1200 ?        00:00:09 postgres")

    if p in ("netstat", "ss"):
        return "\n".join(f"LISTEN 0 128 0.0.0.0:{port} 0.0.0.0:* users:(({svc}))"
                         for port, svc in PORTS)

    if p == "id":
        return "uid=1001(deploy) gid=1001(deploy) groups=1001(deploy),27(sudo)"

    if p == "which":
        tgt = arg or "app"
        return f"/usr/local/bin/{tgt}"

    if p in ("systemctl", "service"):
        return ("nginx.service      loaded active running\n"
                "postgresql.service loaded active running\n"
                "ssh.service        loaded active running")

    if p in ("who", "w"):
        return "deploy   pts/0        Oct 31 09:00 (10.0.0.9)"

    if p == "echo":
        return cmd.split(" ", 1)[1] if " " in cmd else ""

    if p == "app":
        return "app version 2.4.1 (build 8891)"

    return f"{p}: completed"


def simulate(command: str, category: str) -> str:
    """Return realistic output for a command. Only success/partial_result are
    generated here; callers keep the original text for other categories."""
    if category == "success":
        return _success(command)
    if category == "partial_result":
        body = _success(command)
        first = "\n".join(body.splitlines()[:3])
        return first + "\n... (output truncated at 4096 bytes)"
    # categories with already-realistic text: signal "keep original"
    return ""
