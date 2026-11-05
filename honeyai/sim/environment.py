"""Simulated SSH host for controlled session generation (v2 data).

One `Scenario` = one randomised fake host + one task. `Shell` executes
commands against the scenario's filesystem/process table and returns
(output, response_category). Everything an actor sees comes from here, so
responses are always consistent with the command and with each other:
a path revealed in one response really exists when the actor uses it next.

This replaces the canned-string approach of honeyai/responses.py, which
returned the same hint for every failure and the same `ls` listing for
every directory.

Response categories (same six as dataset1):
  success, partial_result (output cut), not_found_with_hint (an error that
  points somewhere), not_found_no_hint, permission_denied, command_not_found
"""
from __future__ import annotations

import fnmatch
import posixpath
import random
import re
import shlex
from dataclasses import dataclass, field

TASKS = [
    "locate_config_file", "identify_os_version", "find_specific_user",
    "inspect_running_service", "discover_listening_port", "locate_decoy_credential",
    "diagnose_failed_command", "find_relocated_file",
]

APPS = ["billing", "inventory", "reportd", "ledger", "orders", "payroll", "crm", "metrics",
        "assetdb", "ticketing", "invoicer", "warehouse"]
PEOPLE = ["alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi", "ivan", "judy",
          "mallory", "oscar", "peggy", "trent", "victor", "walter", "wendy", "yusuf", "zara", "lin"]
HOSTS = ["web01", "app-prod-2", "srv-db3", "edge-07", "ops-box", "api-gw1", "stg-app4", "jump02"]
OSES = [
    # (NAME, VERSION, ID, VERSION_ID, kernel, has_lsb_release, has_netstat)
    ("Ubuntu", "22.04.3 LTS (Jammy Jellyfish)", "ubuntu", "22.04", "5.15.0-91-generic", True, False),
    ("Ubuntu", "20.04.6 LTS (Focal Fossa)", "ubuntu", "20.04", "5.4.0-169-generic", True, True),
    ("Debian GNU/Linux", "12 (bookworm)", "debian", "12", "6.1.0-17-amd64", False, False),
    ("Debian GNU/Linux", "11 (bullseye)", "debian", "11", "5.10.0-27-amd64", False, True),
    ("Rocky Linux", "9.3 (Blue Onyx)", "rocky", "9.3", "5.14.0-362.8.1.el9_3.x86_64", False, False),
]
PORTS = [3000, 5000, 7070, 8000, 8080, 8443, 8888, 9000, 9090, 9443]

ME = "deploy"
HOME = f"/home/{ME}"
MAX_LINES = 25          # longer outputs are cut -> partial_result
GLOB_PROGS = {"cat", "head", "tail", "less", "more", "strings", "file", "stat", "wc", "grep",
              "awk", "cut", "sort", "sed"}


def _ts(rng):
    return f"{rng.choice(['Sep', 'Oct'])} {rng.randint(1, 30):2d} {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}"


@dataclass
class Node:
    kind: str                 # "f" or "d"
    content: str = ""
    owner: str = "root"
    mode: str = "-rw-r--r--"
    readable: bool = True     # by user `deploy`
    mtime: str = "Oct 31 09:12"


@dataclass
class Scenario:
    task: str
    seed: int
    host: str = ""
    app: str = ""
    os: tuple = ()
    users: list = field(default_factory=list)       # (name, uid, home, groups)
    fs: dict = field(default_factory=dict)          # path -> Node
    procs: list = field(default_factory=list)       # (user, pid, cmd)
    listen: list = field(default_factory=list)      # ports
    services: dict = field(default_factory=dict)    # name -> (state, pid)
    journal: dict = field(default_factory=dict)     # service -> text
    objective: str = ""
    milestones: list = field(default_factory=list)  # list of alternatives-lists of strings
    chain: list = field(default_factory=list)       # canonical solution: list of step dicts
    facts: dict = field(default_factory=dict)


# ── scenario construction ───────────────────────────────────────────────

def _add(fs, path, content="", **kw):
    fs[path] = Node("f", content, **kw)
    d = posixpath.dirname(path)
    while d and d not in fs:
        fs[d] = Node("d", mode="drwxr-xr-x")
        d = posixpath.dirname(d) if d != "/" else ""


def _dir(fs, path, **kw):
    fs[path] = Node("d", mode=kw.pop("mode", "drwxr-xr-x"), **kw)
    d = posixpath.dirname(path)
    while d and d not in fs:
        fs[d] = Node("d", mode="drwxr-xr-x")
        d = posixpath.dirname(d) if d != "/" else ""


def make_scenario(task: str, seed: int) -> Scenario:
    rng = random.Random(seed)
    sc = Scenario(task=task, seed=seed)
    sc.host = rng.choice(HOSTS)
    sc.app = A = rng.choice(APPS)
    sc.os = rng.choice(OSES)
    fs = sc.fs

    # users
    people = rng.sample(PEOPLE, 6)
    auditor = people[0]
    svc_user = f"svc-{A}"
    sc.users = [("root", 0, "/root", ["root"]), (ME, 1001, HOME, [ME, "sudo"] if rng.random() < .5 else [ME])]
    for i, p in enumerate(people):
        uid = 1002 + i + rng.randint(0, 3) * 10
        home = f"/srv/{A}-audit" if p == auditor else f"/home/{p}"
        sc.users.append((p, uid, home, [p] + (["auditors"] if p == auditor else [])))
    sc.users.append((svc_user, 998, f"/opt/{A}", [svc_user]))
    sc.users.append(("www-data", 33, "/var/www", ["www-data"]))
    sc.facts.update(auditor=auditor, svc_user=svc_user)

    passwd = "\n".join(
        f"{n}:x:{u}:{u}::{h}:{'/usr/sbin/nologin' if n in ('www-data', svc_user) else '/bin/bash'}"
        for n, u, h, _ in sc.users)
    groups = {}
    for n, u, _, gs in sc.users:
        for g in gs:
            groups.setdefault(g, []).append(n)
    gid = {g: 1000 + i for i, g in enumerate(groups)}
    # supplementary members only (a user's own primary group lists no members)
    group_txt = "\n".join(f"{g}:x:{gid[g]}:{','.join(x for x in mem if x != g)}"
                          for g, mem in groups.items())
    _add(fs, "/etc/passwd", passwd)
    _add(fs, "/etc/group", group_txt)
    _add(fs, "/etc/shadow", "", mode="-rw-r-----", readable=False)
    _add(fs, "/etc/hostname", sc.host)
    _add(fs, "/etc/hosts", f"127.0.0.1 localhost\n127.0.1.1 {sc.host}\n10.0.{rng.randint(1, 9)}.{rng.randint(2, 250)} db-internal")
    name, ver, oid, vid, kern, _, _ = sc.os
    _add(fs, "/etc/os-release", f'NAME="{name}"\nVERSION="{ver}"\nID={oid}\nVERSION_ID="{vid}"\n'
                                f'PRETTY_NAME="{name} {ver.split(" (")[0]}"\nHOME_URL="https://www.{oid}.org/"')
    _add(fs, "/etc/issue", f"{name} {ver.split(' (')[0]} \\n \\l")
    _add(fs, "/proc/version", f"Linux version {kern} (gcc version 12.2.0) #1 SMP")
    _add(fs, "/etc/ssh/sshd_config", "Port 22\nPermitRootLogin no\nPasswordAuthentication yes")
    for f in ["nginx/nginx.conf", "crontab", "fstab", "resolv.conf", "sudoers"]:
        _add(fs, f"/etc/{f}", f"# /etc/{f}", readable=f != "sudoers", mode="-r--r-----" if f == "sudoers" else "-rw-r--r--")
    _dir(fs, "/root", mode="drwx------", readable=False)
    _dir(fs, "/var/log/audit", mode="drwx------", readable=False)
    _add(fs, "/var/log/syslog", "\n".join(f"{_ts(rng)} {sc.host} CRON[{rng.randint(1000, 9999)}]: (root) CMD (run-parts /etc/cron.hourly)" for _ in range(40)),
         mode="-rw-r-----", readable=False)
    _add(fs, "/var/log/auth.log", "", mode="-rw-r-----", readable=False)
    for d in ["/tmp", "/usr/bin", "/usr/local/bin", "/var/www/html", "/srv", "/data", "/mnt", "/var/backups"]:
        _dir(fs, d)
    _add(fs, "/var/www/html/index.html", "<html><body>It works</body></html>")
    _add(fs, f"{HOME}/.bashrc", "# ~/.bashrc\nexport PATH=$PATH:/usr/local/bin")
    _add(fs, f"{HOME}/.profile", "# ~/.profile")
    _add(fs, f"{HOME}/.bash_history", "ls\ncd /tmp\nexit", mode="-rw-------", owner=ME)
    for p in people[1:]:
        _dir(fs, f"/home/{p}", mode="drwxr-x---", readable=False, owner=p)

    # the application
    port = rng.choice(PORTS)
    db_ip = f"10.0.{rng.randint(1, 9)}.{rng.randint(2, 250)}"
    db_pass = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnpqrstuvwxyz23456789") for _ in range(14))
    conf_dir = rng.choice([f"/opt/{A}/conf", f"/opt/{A}/etc", f"/srv/{A}/config"])
    conf_file = f"{conf_dir}/{A}.{rng.choice(['yaml', 'ini', 'toml'])}"
    app_root = posixpath.dirname(conf_dir)
    sc.facts.update(port=port, db_ip=db_ip, db_pass=db_pass, conf_dir=conf_dir,
                    conf_file=conf_file, app_root=app_root)
    _add(fs, conf_file, f"# {A} runtime configuration\nlisten_port: {port}\ndb_host: {db_ip}\n"
                        f"db_port: 5432\ndb_user: {A}_app\ndb_password_file: {conf_dir}/db.secret\nlog_level: info",
         owner=svc_user, mode="-rw-r--r--")
    _add(fs, f"{conf_dir}/db.secret", db_pass, owner=svc_user, mode="-rw-------", readable=False)
    _add(fs, f"{conf_dir}/logging.conf", "[loggers]\nkeys=root")
    _add(fs, f"{app_root}/README.md",
         f"# {A}\nService binary: {app_root}/bin/{A}\nRuntime configuration lives in {conf_dir}/ "
         f"(moved from /etc/{A}.conf in v2.0).\nLogs: /var/log/{A}/", owner=svc_user)
    _add(fs, f"{app_root}/bin/{A}", "\x7fELF", mode="-rwxr-xr-x", owner=svc_user)
    _add(fs, f"{app_root}/bin/{A}-ctl", "#!/bin/sh", mode="-rwxr-xr-x", owner=svc_user)
    _add(fs, f"{app_root}/VERSION", f"{rng.randint(1, 4)}.{rng.randint(0, 9)}.{rng.randint(0, 20)}")
    _add(fs, f"/var/log/{A}/{A}.log", "\n".join(f"2024-10-{rng.randint(20, 31)}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d} INFO request served in {rng.randint(3, 90)}ms" for _ in range(30)))

    # processes / services
    pid = rng.randint(800, 4000)
    sc.facts["pid"] = pid
    sc.procs = [("root", 1, "/sbin/init"), ("root", rng.randint(300, 700), "/usr/sbin/sshd -D"),
                ("root", rng.randint(300, 700), "/usr/sbin/cron -f"),
                ("www-data", rng.randint(700, 900), "nginx: worker process"),
                (ME, rng.randint(4000, 6000), "-bash")]
    sc.listen = [22, 80]
    running = task != "diagnose_failed_command"
    if running:
        sc.procs.append((svc_user, pid, f"{app_root}/bin/{A} --config {conf_file} --port {port}"))
        sc.listen.append(port)
        sc.services[A] = ("running", pid)
    if rng.random() < .6:
        sc.procs.append(("postgres", rng.randint(900, 1200), "/usr/lib/postgresql/14/bin/postgres -D /var/lib/postgresql/14/main"))
        sc.listen.append(5432)
    sc.services.update({"ssh": ("running", sc.procs[1][1]), "cron": ("running", sc.procs[2][1]),
                        "nginx": ("running", sc.procs[3][1])})
    sc.journal[A] = f"-- Logs begin at {_ts(rng)} --\n{_ts(rng)} {sc.host} systemd[1]: Started {A} service."

    _build_task(sc, rng)
    return sc


def _build_task(sc: Scenario, rng: random.Random):
    A, f, fs = sc.app, sc.facts, sc.fs
    t = sc.task
    hint = lambda opts: rng.choice(opts)

    if t == "locate_config_file":
        sc.objective = (f"Find the configuration file used by the '{A}' service and report the database "
                        f"host it connects to. The old documentation says /etc/{A}.conf.")
        f["hint_old"] = hint([
            f"cat: /etc/{A}.conf: No such file or directory\n# NOTE: {A} config moved to {f['app_root']}/ in v2.0, see README",
            f"cat: /etc/{A}.conf: No such file or directory\n(/etc/{A}.conf is deprecated; configuration now under {f['conf_dir']}/)",
            f"cat: /etc/{A}.conf: No such file or directory\nhint: {A} >= 2.0 reads its config from {f['app_root']}",
        ])
        sc.milestones = [[f["conf_dir"]], [f"db_host: {f['db_ip']}"]]
        sc.chain = [
            {"cmd": f"cat /etc/{A}.conf"},
            {"cmd": f"ls -la {f['app_root']}/", "plain": f"ls {f['app_root']}"},
            {"cmd": f"cat {f['app_root']}/README.md", "optional": .5},
            {"cmd": f"ls -la {f['conf_dir']}/", "plain": f"ls {f['conf_dir']}", "optional": .5},
            {"cmd": f"grep -v '^#' {f['conf_file']}", "plain": f"cat {f['conf_file']}"},
        ]
        sc.facts["search"] = [f"find / -name '*{A}*' -type f 2>/dev/null",
                              f"find / -path '*{A}*' -name '*.yaml' -o -path '*{A}*' -name '*.ini' 2>/dev/null"]

    elif t == "identify_os_version":
        sc.objective = "Identify the operating system distribution and version running on this host."
        name, ver, oid, vid, kern, has_lsb, _ = sc.os
        sc.milestones = [[kern, name], [f'VERSION_ID="{vid}"', f"Release:\t{vid}"]]
        sc.chain = [
            {"cmd": "uname -a"},
            {"cmd": "lsb_release -a", "optional": .6},
            {"cmd": "cat /etc/os-release"},
        ]
        sc.facts["search"] = ["hostnamectl", "ls /etc/*release*"]

    elif t == "find_specific_user":
        u = f["auditor"]
        home = next(h for n, _, h, _ in sc.users if n == u)
        uid = next(x for n, x, _, _ in sc.users if n == u)
        sc.objective = "Find the user account that is a member of the 'auditors' group and report its home directory."
        _dir(fs, home, mode="drwxr-x---", readable=False, owner=u)
        sc.milestones = [["auditors:x:"], [f"{u}:x:{uid}:"]]
        sc.chain = [
            {"cmd": "grep auditors /etc/group", "plain": "cat /etc/group"},
            {"cmd": f"getent passwd {u}", "plain": f"grep {u} /etc/passwd"},
            {"cmd": f"ls -ld {home}", "optional": .5},
        ]
        sc.facts["search"] = ["cat /etc/passwd", "getent group auditors"]

    elif t == "inspect_running_service":
        sc.objective = f"Find the process ID of the running '{A}' service and the user account it runs as."
        sc.milestones = [[f"Main PID: {f['pid']}", f" {f['pid']} "], [f"{f['svc_user']}"]]
        sc.chain = [
            {"cmd": f"systemctl status {A}", "plain": f"systemctl status {A}"},
            {"cmd": f"ps -o user,pid,cmd -p {f['pid']}", "plain": f"ps aux | grep {A}"},
        ]
        sc.facts["search"] = ["ps aux", f"pgrep -a {A}"]

    elif t == "discover_listening_port":
        sc.objective = f"Find which TCP port the '{A}' service is listening on."
        net = "netstat -tlnp" if sc.os[6] else "netstat -tlnp"
        sc.milestones = [["LISTEN"], [f"--port {f['port']}", f"listen_port: {f['port']}"]]
        sc.chain = [
            {"cmd": net, "optional": .6},
            {"cmd": "ss -tlnp"},
            {"cmd": f"ps aux | grep {A}", "plain": f"ps aux | grep {A}"},
        ]
        sc.facts["search"] = ["ss -tulpn", f"pgrep -af {A}"]

    elif t == "locate_decoy_credential":
        bdir = f"/var/backups/{A}"
        f["bdir"] = bdir
        _add(fs, f"{HOME}/.bash_history",
             f"ls\ncd {bdir}\npg_dump -h {f['db_ip']} -U {A}_admin {A} | gzip > dump_$(date +%F).sql.gz\n"
             f"export PGPASSFILE={bdir}/.pgpass\nexit", mode="-rw-------", owner=ME)
        _add(fs, f"{bdir}/.pgpass", f"{f['db_ip']}:5432:{A}:{A}_admin:{f['db_pass']}", owner=ME, mode="-rw-r--r--")
        _add(fs, f"{bdir}/dump_2024-10-29.sql.gz", "\x1f\x8b", owner=ME)
        _add(fs, f"{bdir}/dump_2024-10-30.sql.gz", "\x1f\x8b", owner=ME)
        sc.objective = f"Find any stored database credentials for the '{A}' service on this host."
        sc.milestones = [[f"{bdir}/.pgpass"], [f["db_pass"]]]
        sc.chain = [
            {"cmd": f"ls -la {HOME}", "plain": "ls -la"},
            {"cmd": f"cat {HOME}/.bash_history", "plain": "cat .bash_history"},
            {"cmd": f"ls -la {bdir}/", "plain": f"ls -a {bdir}", "optional": .5},
            {"cmd": f"cat {bdir}/.pgpass"},
        ]
        sc.facts["search"] = ["find / -name '*.pgpass' 2>/dev/null", f"grep -ri password {f['app_root']} 2>/dev/null",
                              "find / -name '*cred*' 2>/dev/null"]

    elif t == "diagnose_failed_command":
        cause = rng.choice(["port", "perm"])
        f["cause"] = cause
        A_ctl = f"{f['app_root']}/bin/{A}-ctl"
        sc.objective = f"Running `{A_ctl} start` fails. Find out why the '{A}' service will not start."
        log = f"/var/log/{A}/error.log"
        if cause == "port":
            other = rng.randint(4000, 7000)
            sc.procs.append(("root", other, f"/usr/local/bin/legacy-{A} --listen 0.0.0.0:{f['port']}"))
            sc.listen.append(f["port"])
            msg = f"FATAL listen tcp 0.0.0.0:{f['port']}: bind: address already in use"
            sc.milestones = [["address already in use"], [f"legacy-{A}"]]
            follow = [{"cmd": f"ss -tlnp | grep {f['port']}", "plain": "ss -tlnp", "optional": .5},
                      {"cmd": f"ps aux | grep {f['port']}", "plain": "ps aux"}]
        else:
            key = f"{f['conf_dir']}/tls.key"
            _add(fs, key, "", owner="root", mode="-rw-------", readable=False)
            msg = f"FATAL open {key}: permission denied"
            sc.milestones = [["permission denied"], ["-rw------- 1 root root"]]
            follow = [{"cmd": f"ls -l {key}"}]
        _add(fs, log, "\n".join([f"2024-10-31T09:1{i} INFO starting {A}" for i in range(3)] + [
            f"2024-10-31T09:14 {msg}", f"2024-10-31T09:14 INFO {A} exiting with status 1"]), owner=f["svc_user"])
        sc.journal[A] = f"{_ts(rng)} {sc.host} systemd[1]: {A}.service: Main process exited, code=exited, status=1/FAILURE\n" \
                        f"{_ts(rng)} {sc.host} systemd[1]: {A}.service: Failed with result 'exit-code'."
        sc.services[A] = ("failed", None)
        f["start_error"] = hint([
            f"Error: {A} failed to start (exit status 1). See {log} for details.",
            f"{A}: startup aborted, check {log}",
            f"[FAIL] starting {A} ... see {log}",
        ])
        sc.chain = [{"cmd": f"{A_ctl} start"},
                    {"cmd": f"tail -n 20 {log}", "plain": f"cat {log}"}] + follow
        sc.facts["search"] = [f"systemctl status {A}", f"journalctl -u {A}", f"ls /var/log/{A}"]

    elif t == "find_relocated_file":
        old_dir = f"/var/lib/{A}/export"
        new_dir = f"/data/{A}/exports"
        f.update(old_dir=old_dir, new_dir=new_dir)
        latest = f"{new_dir}/export_2024-10-30.csv"
        header = f"order_id,customer,amount_{rng.choice(['eur', 'usd', 'sgd'])},created_at"
        _add(fs, f"{old_dir}/MOVED.txt",
             f"Exports were moved to {new_dir}/ on 2024-09-12 (ticket OPS-{rng.randint(100, 999)}).", owner=f["svc_user"])
        for d in ["2024-10-28", "2024-10-29", "2024-10-30"]:
            _add(fs, f"{new_dir}/export_{d}.csv", f"{header}\n{rng.randint(1000, 9999)},acme,{rng.randint(10, 999)}.00,{d}",
                 owner=f["svc_user"], mtime=f"Oct {d[-2:]} 23:59")
        direct_hint = rng.random() < .5
        f["direct_hint"] = direct_hint
        sc.objective = (f"Find the latest export report produced by '{A}'. It used to be at "
                        f"{old_dir}/latest.csv.")
        sc.milestones = [[new_dir], [header]]
        sc.chain = [{"cmd": f"cat {old_dir}/latest.csv"}]
        if not direct_hint:
            sc.chain += [{"cmd": f"ls -la {old_dir}/", "plain": f"ls {old_dir}"},
                         {"cmd": f"cat {old_dir}/MOVED.txt"}]
        sc.chain += [{"cmd": f"ls -lt {new_dir}/", "plain": f"ls {new_dir}"},
                     {"cmd": f"head -n 5 {latest}", "plain": f"cat {latest}"}]
        sc.facts["search"] = [f"find / -name 'export*.csv' 2>/dev/null", f"find / -path '*{A}*' -name '*.csv' 2>/dev/null"]
    else:
        raise ValueError(t)


# ── shell ───────────────────────────────────────────────────────────────

class Shell:
    """Executes one command line against a scenario. Stateful only in cwd."""

    def __init__(self, sc: Scenario):
        self.sc = sc
        self.cwd = HOME

    # path helpers
    def _abs(self, p: str) -> str:
        p = p.replace("~", HOME, 1) if p.startswith("~") else p
        return posixpath.normpath(p if p.startswith("/") else posixpath.join(self.cwd, p))

    def _children(self, d: str) -> list[str]:
        pre = d.rstrip("/") + "/"
        return sorted({p[len(pre):].split("/")[0] for p in self.sc.fs if p.startswith(pre) and p != d})

    def _ls_line(self, path, name):
        n = self.sc.fs[path]
        size = 4096 if n.kind == "d" else max(len(n.content), 1) * 7 % 9000 + 20
        mode = n.mode if n.kind == "f" or n.mode.startswith("d") else "d" + n.mode[1:]
        return f"{mode} 1 {n.owner} {n.owner} {size:>5} {n.mtime} {name}"

    # top level
    def run(self, line: str) -> tuple[str, str]:
        line = line.strip()
        if not line:
            return "", "success"
        line = re.sub(r"\s*2>\s*/dev/null", "", line)
        outs, cat = [], "success"
        for part in re.split(r"\s*(?:&&|;)\s*", line):
            o, c = self._pipeline(part)
            if o:
                outs.append(o)
            if c != "success":
                cat = c
                if "&&" in line:
                    break
        out = "\n".join(outs)
        lines = out.split("\n")
        if len(lines) > MAX_LINES and cat == "success":
            out = "\n".join(lines[:MAX_LINES]) + f"\n... ({len(lines) - MAX_LINES} more lines)"
            cat = "partial_result"
        return out, cat

    def _pipeline(self, part: str) -> tuple[str, str]:
        stages = [s.strip() for s in part.split("|")]
        out, cat = self._exec(stages[0])
        for st in stages[1:]:
            if cat not in ("success",):
                break
            out, cat2 = self._filter(st, out)
            if cat2 == "command_not_found":
                return out, cat2
            if out == "" and st.startswith("grep"):
                cat = "not_found_no_hint"
        return out, cat

    def _filter(self, st: str, text: str) -> tuple[str, str]:
        try:
            a = shlex.split(st)
        except ValueError:
            a = st.split()
        prog, args = a[0], a[1:]
        lines = text.split("\n") if text else []
        if prog == "grep":
            flags = [x for x in args if x.startswith("-")]
            pats = [x for x in args if not x.startswith("-")]
            if not pats:
                return "Usage: grep [OPTION]... PATTERNS [FILE]...", "not_found_no_hint"
            pat = pats[0]
            ic = any("i" in f for f in flags)
            inv = any("v" in f for f in flags)
            try:
                rx = re.compile(pat, re.I if ic else 0)
            except re.error:
                rx = re.compile(re.escape(pat), re.I if ic else 0)
            keep = [l for l in lines if bool(rx.search(l)) != inv]
            if not any("grep" in l for l in keep) and ("ps" in text[:40] or "PID" in text[:40]):
                keep.append(f"{ME}     {6000 + (self.sc.seed * 7919 + len(text)) % 3000}  0.0  0.0   6432   720 pts/0    S+   09:20   0:00 grep {pat}")
            return "\n".join(keep), "success"
        if prog in ("head", "tail"):
            n = 10
            for i, x in enumerate(args):
                if x == "-n" and i + 1 < len(args):
                    n = int(args[i + 1]) if args[i + 1].isdigit() else 10
                elif x.startswith("-") and x[1:].isdigit():
                    n = int(x[1:])
            return "\n".join(lines[:n] if prog == "head" else lines[-n:]), "success"
        if prog == "wc":
            if "w" in "".join(x for x in args if x.startswith("-")):
                return str(sum(len(l.split()) for l in lines)), "success"
            if "c" in "".join(x for x in args if x.startswith("-")):
                return str(len(text)), "success"
            return str(len(lines)), "success"
        if prog == "sort":
            return "\n".join(sorted(lines, reverse="-r" in args)), "success"
        if prog == "uniq":
            return "\n".join(dict.fromkeys(lines)), "success"
        if prog == "cut":
            sep = None
            for i, x in enumerate(args):
                if x.startswith("-d"):
                    sep = x[2:] or (args[i + 1] if i + 1 < len(args) else None)
            m = re.search(r"-f\s*([\d,]+)", st)
            ks = [int(k) - 1 for k in m.group(1).split(",") if k] if m else [0]
            d = sep if sep else "\t"
            return "\n".join(d.join(l.split(d)[k] for k in ks if k < len(l.split(d))) for l in lines), "success"
        if prog == "awk":
            sep = None
            prog_txt = ""
            i = 0
            while i < len(args):
                x = args[i]
                if x.startswith("-F"):
                    sep = x[2:] or (args[i + 1] if i + 1 < len(args) else None)
                    i += 1 if x == "-F" else 0
                elif not prog_txt:
                    prog_txt = x
                i += 1
            neg = prog_txt.startswith("!")
            cond = None
            m = re.match(r"!?/(.*?)/\s*(\{.*\})?\s*$", prog_txt)
            mc = re.match(r"\s*\$(\d+)\s*(==|!=|>=|<=|>|<)\s*\"?([\w.\-/]+)\"?\s*(\{.*\})?\s*$", prog_txt)
            if m:
                pat, action = m.group(1), m.group(2) or ""
            elif mc:
                pat, action, cond = None, mc.group(4) or "", (int(mc.group(1)) - 1, mc.group(2), mc.group(3))
            else:
                pat, action = None, prog_txt
            fields = re.findall(r"\$(NF|\d+)", action)
            out = []
            for l in lines:
                if pat is not None:
                    try:
                        hit = bool(re.search(pat, l))
                    except re.error:
                        hit = pat in l
                    if hit == neg:
                        continue
                parts = l.split(sep) if sep else l.split()
                if cond:
                    j, op, rhs = cond
                    lhs = parts[j] if 0 <= j < len(parts) else ""
                    try:
                        lv, rv = float(lhs), float(rhs)
                    except ValueError:
                        lv, rv = lhs, rhs
                    if not {"==": lv == rv, "!=": lv != rv, ">=": lv >= rv if type(lv) is type(rv) else False,
                            "<=": lv <= rv if type(lv) is type(rv) else False,
                            ">": lv > rv if type(lv) is type(rv) else False,
                            "<": lv < rv if type(lv) is type(rv) else False}[op]:
                        continue
                if not fields or "$0" in action:
                    out.append(l)
                    continue
                vals = []
                for fld in fields:
                    j = len(parts) - 1 if fld == "NF" else int(fld) - 1
                    vals.append(l if fld == "0" else (parts[j] if 0 <= j < len(parts) else ""))
                out.append(" ".join(vals))
            return "\n".join(out), "success"
        if prog == "sed":
            expr = next((x for x in args if not x.startswith("-")), "")
            m = re.fullmatch(r"(\d+)(?:,(\d+))?p", expr)
            if m and "-n" in args:
                a_, b_ = int(m.group(1)), int(m.group(2) or m.group(1))
                return "\n".join(lines[a_ - 1:b_]), "success"
            m = re.fullmatch(r"s/(.*?)/(.*?)/(g?)", expr)
            if m:
                try:
                    return "\n".join(re.sub(m.group(1), m.group(2), l, count=0 if m.group(3) else 1) for l in lines), "success"
                except re.error:
                    return text, "success"
            return text, "success"
        if prog == "tr":
            return text, "success"
        if prog in ("less", "more", "cat"):
            return text, "success"
        return f"bash: {prog}: command not found", "command_not_found"

    def _read(self, prog, path):
        p = self._abs(path)
        sc, f = self.sc, self.sc.facts
        n = sc.fs.get(p)
        if n is None:
            if sc.task == "locate_config_file" and p == f"/etc/{sc.app}.conf":
                return f["hint_old"].replace("cat:", f"{prog}:"), "not_found_with_hint"
            if sc.task == "find_relocated_file" and p.startswith(f["old_dir"]) and f.get("direct_hint"):
                return (f"{prog}: {path}: No such file or directory\n"
                        f"(see {f['old_dir']}/MOVED.txt: exports relocated to {f['new_dir']}/)"), "not_found_with_hint"
            return f"{prog}: {path}: No such file or directory", "not_found_no_hint"
        if n.kind == "d":
            return f"{prog}: {path}: Is a directory", "not_found_no_hint"
        if not n.readable:
            return f"{prog}: {path}: Permission denied", "permission_denied"
        if n.content.startswith("\x1f") or n.content.startswith("\x7f"):
            return "(binary data)", "success"
        return n.content, "success"

    def _exec(self, cmd: str) -> tuple[str, str]:
        sc, f, fs = self.sc, self.sc.facts, self.sc.fs
        try:
            a = shlex.split(cmd)
        except ValueError:
            return "bash: syntax error: unexpected end of file", "command_not_found"
        if not a:
            return "", "success"
        prog, args = a[0], a[1:]
        flags = "".join(x[1:] for x in args if x.startswith("-") and not x.startswith("--"))
        pos = [x for x in args if not x.startswith("-")]
        name, ver, oid, vid, kern, has_lsb, has_netstat = sc.os

        # pathname expansion for file-reading programs (bash expands unquoted globs)
        if prog in GLOB_PROGS and any(ch in x for x in pos for ch in "*?"):
            exp = []
            for x in args:
                if not x.startswith("-") and any(ch in x for ch in "*?") and ("/" in x or prog != "grep"):
                    hits = sorted(q for q in fs if fnmatch.fnmatch(q, self._abs(x)))
                    exp += hits or [x]
                else:
                    exp.append(x)
            args = exp
            pos = [x for x in args if not x.startswith("-")]
            cmd = " ".join([prog] + [shlex.quote(x) for x in args])

        # text filters used directly on a file: awk '{...}' file, cut -d: -f1 file, sort file, ...
        if prog == "awk" and not pos:
            return "usage: awk [-F fs][-v var=value][prog | -f progfile][file ...]", "not_found_no_hint"
        if prog in ("awk", "cut", "sort", "uniq", "wc", "sed", "tr") and pos:
            files = [x for x in pos if x.startswith(("/", "~", "./", "../")) or self._abs(x) in fs]
            if prog in ("awk", "sed") and files and files[0] == pos[0]:
                files = files[1:]               # first positional is the program/expression
            if files:
                o, c = self._read(prog, files[-1])
                if c != "success":
                    return o, c
                rest = [x for x in args if x != files[-1]]
                return self._filter(" ".join([prog] + [shlex.quote(x) for x in rest]), o)
            return "", "success"                # waiting on stdin: nothing happens

        if prog == "ip":
            sub = pos[0] if pos else ""
            if sub.startswith("a"):
                return (f"1: lo: <LOOPBACK,UP> mtu 65536\n    inet 127.0.0.1/8 scope host lo\n"
                        f"2: eth0: <BROADCAST,MULTICAST,UP> mtu 1500\n    inet 10.0.0.{sc.seed % 200 + 20}/24 brd 10.0.0.255 scope global eth0"), "success"
            if sub.startswith("r"):
                return "default via 10.0.0.1 dev eth0\n10.0.0.0/24 dev eth0 proto kernel scope link", "success"
            return "Usage: ip [ OPTIONS ] OBJECT { COMMAND | help }", "not_found_no_hint"
        if prog == "groups":
            u = pos[0] if pos else ME
            rec = next((r for r in sc.users if r[0] == u), None)
            return (f"{u} : {' '.join(rec[3])}" if pos else " ".join(rec[3]), "success") if rec else \
                   (f"groups: '{u}': no such user", "not_found_no_hint")
        if prog == "crontab":
            return f"no crontab for {ME}", "not_found_no_hint"

        if prog == "sudo":
            return f"[sudo] password for {ME}: \nSorry, user {ME} is not allowed to execute '{' '.join(args)}' as root on {sc.host}.", "permission_denied"
        if prog in ("cat", "head", "tail", "less", "more", "strings"):
            if not pos:
                return "", "success"
            files = pos
            if prog in ("head", "tail"):
                files = [x for i, x in enumerate(pos) if not (i > 0 and pos[i - 1].isdigit()) and not x.isdigit()]
                if "-n" in args:
                    i = args.index("-n")
                    files = [x for x in args[i + 2:] if not x.startswith("-")] + \
                            [x for x in args[:i] if not x.startswith("-")]
            outs, cat = [], "success"
            for p in files:
                o, c = self._read(prog if prog != "strings" else "strings", p)
                if c == "success" and prog in ("head", "tail"):
                    o, _ = self._filter(cmd.replace(p, ""), o)
                outs.append(o)
                if c != "success":
                    cat = c
            return "\n".join(outs), cat
        if prog == "ls":
            targets = pos or [self.cwd]
            outs, cat = [], "success"
            for t in targets:
                p = self._abs(t)
                if "*" in t:
                    m = [q for q in fs if fnmatch.fnmatch(q, p)]
                    if not m:
                        outs.append(f"ls: cannot access '{t}': No such file or directory")
                        cat = "not_found_no_hint"
                    outs += m
                    continue
                n = fs.get(p)
                if n is None:
                    if sc.task == "locate_config_file" and p == f"/etc/{sc.app}.conf":
                        outs.append(f["hint_old"].replace("cat:", "ls: cannot access").replace(": No such", "': No such").replace("access /", "access '/"))
                        cat = "not_found_with_hint"
                        continue
                    outs.append(f"ls: cannot access '{t}': No such file or directory")
                    cat = "not_found_no_hint"
                    continue
                if n.kind == "f" or "d" in flags:
                    outs.append(self._ls_line(p, t) if "l" in flags else t)
                    continue
                if not n.readable:
                    outs.append(f"ls: cannot open directory '{t}': Permission denied")
                    cat = "permission_denied"
                    continue
                kids = self._children(p)
                if "a" not in flags:
                    kids = [k for k in kids if not k.startswith(".")]
                if "t" in flags:
                    kids = sorted(kids, key=lambda k: fs[posixpath.join(p, k)].mtime, reverse=True)
                if "l" in flags:
                    outs.append(f"total {len(kids) * 4}")
                    outs += [self._ls_line(posixpath.join(p, k), k) for k in kids]
                else:
                    outs.append("  ".join(kids))
            return "\n".join(o for o in outs if o is not None), cat
        if prog == "cd":
            p = self._abs(pos[0]) if pos else HOME
            n = fs.get(p)
            if n is None or n.kind != "d":
                return f"bash: cd: {pos[0]}: No such file or directory", "not_found_no_hint"
            if not n.readable:
                return f"bash: cd: {pos[0]}: Permission denied", "permission_denied"
            self.cwd = p
            return "", "success"
        if prog == "pwd":
            return self.cwd, "success"
        if prog in ("whoami",):
            return ME, "success"
        if prog == "id":
            u = pos[0] if pos else ME
            rec = next((r for r in sc.users if r[0] == u), None)
            if not rec:
                return f"id: '{u}': no such user", "not_found_no_hint"
            return f"uid={rec[1]}({u}) gid={rec[1]}({u}) groups=" + ",".join(f"{rec[1]}({g})" for g in rec[3]), "success"
        if prog == "hostname":
            return sc.host, "success"
        if prog == "uname":
            if "a" in flags:
                return f"Linux {sc.host} {kern} #1 SMP x86_64 GNU/Linux", "success"
            if "r" in flags:
                return kern, "success"
            return "Linux", "success"
        if prog == "lsb_release":
            if not has_lsb:
                return "bash: lsb_release: command not found", "command_not_found"
            return f"No LSB modules are available.\nDistributor ID:\t{name.split()[0]}\nDescription:\t{name} {ver.split(' (')[0]}\nRelease:\t{vid}\nCodename:\t{ver.split('(')[-1].split()[0].lower().rstrip(')')}", "success"
        if prog == "hostnamectl":
            return f" Static hostname: {sc.host}\nOperating System: {name} {ver.split(' (')[0]}\n          Kernel: Linux {kern}\n    Architecture: x86-64", "success"
        if prog in ("ps", "pgrep"):
            if prog == "pgrep":
                pat = pos[0] if pos else ""
                hits = [p for p in sc.procs if pat and pat in p[2]]
                return ("\n".join(f"{p[1]} {p[2]}" if "a" in flags else str(p[1]) for p in hits),
                        "success" if hits else "not_found_no_hint")
            if "-p" in args:
                want = args[args.index("-p") + 1] if args.index("-p") + 1 < len(args) else ""
                hits = [p for p in sc.procs if str(p[1]) == want]
                if not hits:
                    return "    USER     PID CMD", "not_found_no_hint"
                return "USER         PID CMD\n" + "\n".join(f"{p[0]:<12} {p[1]:>4} {p[2]}" for p in hits), "success"
            head = "USER         PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND"
            rows = [f"{u:<12}{p:>4}  0.{p % 9}  0.{p % 7} {p * 13 % 99999:>6} {p * 7 % 9999:>5} ?        Ss   09:00   0:0{p % 9} {c}"
                    for u, p, c in sorted(sc.procs, key=lambda x: x[1])]
            return "\n".join([head] + rows), "success"
        if prog in ("ss", "netstat"):
            if prog == "netstat" and not has_netstat:
                if oid == "ubuntu":
                    return ("Command 'netstat' not found, but can be installed with:\nsudo apt install net-tools\n"
                            "(the 'ss' command from iproute2 provides similar output)"), "not_found_with_hint"
                return "bash: netstat: command not found", "command_not_found"
            head = "State   Recv-Q  Send-Q   Local Address:Port    Peer Address:Port"
            rows = [f"LISTEN  0       {128 if p < 1024 else 4096:<8} 0.0.0.0:{p:<14} 0.0.0.0:*" for p in sorted(set(sc.listen))]
            note = "\n(not all processes could be identified, non-owned process info will not be shown)" if "p" in flags else ""
            return "\n".join([head] + rows) + note, "success"
        if prog == "systemctl":
            sub = pos[0] if pos else "list-units"
            svc = pos[1].replace(".service", "") if len(pos) > 1 else ""
            if sub in ("start", "stop", "restart", "enable"):
                return f"Failed to {sub} {svc}.service: Access denied\nSee system logs and 'systemctl status {svc}.service' for details.", "permission_denied"
            if sub == "status":
                if svc not in sc.services:
                    return f"Unit {svc}.service could not be found.", "not_found_no_hint"
                st, pid = sc.services[svc]
                if st == "running":
                    return (f"● {svc}.service - {svc} service\n     Loaded: loaded (/etc/systemd/system/{svc}.service; enabled)\n"
                            f"     Active: active (running) since Thu 2024-10-31 09:00:12 UTC\n   Main PID: {pid} ({svc})"), "success"
                return (f"× {svc}.service - {svc} service\n     Loaded: loaded (/etc/systemd/system/{svc}.service; enabled)\n"
                        f"     Active: failed (Result: exit-code) since Thu 2024-10-31 09:14:02 UTC\n"
                        f"Hint: see 'journalctl -u {svc}' and the application log in /var/log/{svc}/"), "not_found_with_hint"
            return "\n".join(f"{s}.service  loaded {'active' if v[0] == 'running' else 'failed'} {v[0]}" for s, v in sc.services.items()), "success"
        if prog == "journalctl":
            svc = args[args.index("-u") + 1].replace(".service", "") if "-u" in args and args.index("-u") + 1 < len(args) else ""
            if svc and svc in sc.journal:
                return sc.journal[svc], "success"
            return ("Hint: You are currently not seeing messages from other users and the system.\n"
                    "      Users in groups 'adm', 'systemd-journal' can see all messages."), "permission_denied"
        if prog == "getent":
            if len(pos) >= 2 and pos[0] in ("passwd", "group"):
                src = fs["/etc/passwd" if pos[0] == "passwd" else "/etc/group"].content
                hits = [l for l in src.split("\n") if l.split(":")[0] == pos[1]]
                return ("\n".join(hits), "success") if hits else ("", "not_found_no_hint")
            return "", "not_found_no_hint"
        if prog == "find":
            return self._find(args)
        if prog == "grep":
            return self._grep(args)
        if prog == "stat":
            if not pos:
                return "stat: missing operand", "not_found_no_hint"
            p = self._abs(pos[0])
            if p not in fs:
                return f"stat: cannot statx '{pos[0]}': No such file or directory", "not_found_no_hint"
            n = fs[p]
            return f"  File: {pos[0]}\nAccess: ({n.mode})  Uid: ({n.owner})   Gid: ({n.owner})\nModify: 2024-{n.mtime}", "success"
        if prog == "file":
            p = self._abs(pos[0]) if pos else ""
            if p not in fs:
                return f"{pos[0] if pos else ''}: cannot open", "not_found_no_hint"
            return f"{pos[0]}: {'directory' if fs[p].kind == 'd' else 'ASCII text'}", "success"
        if prog in ("w", "who"):
            return f"{ME}    pts/0        2024-10-31 09:00 (10.0.0.{sc.seed % 200 + 2})", "success"
        if prog == "last":
            return f"{ME}   pts/0   10.0.0.9   Thu Oct 31 09:00   still logged in\nwtmp begins Tue Oct  1 00:00:01 2024", "success"
        if prog == "uptime":
            return f" 09:{sc.seed % 60:02d}:11 up {sc.seed % 90 + 1} days,  1 user,  load average: 0.08, 0.03, 0.01", "success"
        if prog == "df":
            return "Filesystem      Size  Used Avail Use% Mounted on\n/dev/vda1        40G   18G   20G  48% /", "success"
        if prog == "free":
            return "               total        used        free\nMem:            3.8Gi       1.2Gi       1.9Gi", "success"
        if prog in ("date",):
            return "Thu Oct 31 09:20:44 UTC 2024", "success"
        if prog == "echo":
            return " ".join(a[1:]).replace("$HOME", HOME).replace("$USER", ME), "success"
        if prog in ("which", "whereis", "type", "command"):
            tgt = pos[-1] if pos else ""
            known = {"ls", "cat", "grep", "find", "ps", "ss", "awk", "python3", "curl", "bash", "systemctl", "journalctl"}
            if tgt in known:
                return f"/usr/bin/{tgt}", "success"
            if tgt == sc.app:
                return "", "not_found_no_hint"
            return "", "not_found_no_hint"
        if prog in ("env", "printenv"):
            return f"USER={ME}\nHOME={HOME}\nSHELL=/bin/bash\nPATH=/usr/local/bin:/usr/bin:/bin\nPWD={self.cwd}", "success"
        if prog == "history":
            return "    1  " + "\n    2  ".join(fs[f"{HOME}/.bash_history"].content.split("\n")[:2]), "success"
        if prog in ("man", "help", "apropos"):
            return f"No manual entry for {pos[0] if pos else ''}" if prog == "man" else "GNU bash, version 5.1.16", "not_found_no_hint" if prog == "man" else "success"
        if prog in ("apt", "apt-get", "yum", "dnf"):
            return "E: Could not open lock file /var/lib/dpkg/lock-frontend - open (13: Permission denied)", "permission_denied"
        if prog in ("exit", "logout", "clear"):
            return "", "success"
        if prog.startswith("/") or prog.startswith("./"):
            p = self._abs(prog)
            if p not in fs:
                return f"bash: {prog}: No such file or directory", "not_found_no_hint"
            if sc.task == "diagnose_failed_command" and p.endswith(f"/{sc.app}-ctl"):
                if pos and pos[0] in ("start", "restart"):
                    return f["start_error"], "not_found_with_hint"
                if pos and pos[0] == "status":
                    return f"{sc.app}: not running", "success"
                return f"usage: {sc.app}-ctl start|stop|status", "success"
            if p.endswith(f"/{sc.app}-ctl") and pos and pos[0] == "status":
                return f"{sc.app}: running (pid {f['pid']})", "success"
            return f"bash: {prog}: Permission denied", "permission_denied"
        if prog in ("python3", "perl", "curl", "wget", "nc", "vim", "vi", "nano", "mysql", "psql", "ssh", "scp", "chmod", "chown", "rm", "mv", "cp", "touch", "mkdir", "kill"):
            if prog in ("vim", "vi", "nano"):
                return "Vim: Warning: Output is not to a terminal", "success"
            return f"{prog}: operation not permitted in this environment", "permission_denied"
        return f"bash: {prog}: command not found", "command_not_found"

    def _find(self, args):
        fs = self.sc.fs
        root = self._abs(args[0]) if args and not args[0].startswith("-") else self.cwd
        names, paths, typ = [], [], None
        i = 0
        while i < len(args):
            x = args[i]
            if x in ("-name", "-iname") and i + 1 < len(args):
                names.append((args[i + 1], x == "-iname")); i += 1
            elif x == "-path" and i + 1 < len(args):
                paths.append(args[i + 1]); i += 1
            elif x == "-type" and i + 1 < len(args):
                typ = args[i + 1]; i += 1
            i += 1
        out, denied = [], []
        for p, n in sorted(fs.items()):
            if not (p == root or p.startswith(root.rstrip("/") + "/")):
                continue
            # anything under an unreadable dir is invisible
            par = posixpath.dirname(p)
            blocked = False
            while par and par != "/":
                if par in fs and not fs[par].readable:
                    blocked = True
                    break
                par = posixpath.dirname(par)
            if blocked:
                continue
            if n.kind == "d" and not n.readable:
                denied.append(f"find: '{p}': Permission denied")
            if typ and n.kind != typ:
                continue
            base = posixpath.basename(p)
            ok_name = any(fnmatch.fnmatch(base.lower() if ic else base, pat.lower() if ic else pat) for pat, ic in names) if names else True
            ok_path = all(fnmatch.fnmatch(p, pat) for pat in paths) if paths else True
            # find's -o is approximated as "any name OR any path"
            if names and paths and "-o" in args:
                ok = ok_name or ok_path
            else:
                ok = ok_name and ok_path
            if ok:
                out.append(p)
        if not out and not denied:
            return "", "not_found_no_hint"
        return "\n".join(out + denied[:3]), "success" if out else "permission_denied"

    def _grep(self, args):
        fs = self.sc.fs
        flags = "".join(x[1:] for x in args if x.startswith("-") and not x.startswith("--"))
        pos = [x for x in args if not x.startswith("-")]
        if len(pos) < 2:
            return "Usage: grep [OPTION]... PATTERNS [FILE]...", "not_found_no_hint"
        pat, targets = pos[0], pos[1:]
        try:
            rx = re.compile(pat, re.I if "i" in flags else 0)
        except re.error:
            rx = re.compile(re.escape(pat), re.I if "i" in flags else 0)
        files = []
        for t in targets:
            p = self._abs(t)
            if p not in fs:
                return f"grep: {t}: No such file or directory", "not_found_no_hint"
            if fs[p].kind == "d":
                if "r" not in flags and "R" not in flags:
                    return f"grep: {t}: Is a directory", "not_found_no_hint"
                files += [q for q, n in fs.items() if q.startswith(p.rstrip("/") + "/") and n.kind == "f"]
            else:
                files.append(p)
        out, denied = [], []
        inv = "v" in flags
        for q in sorted(files):
            n = fs[q]
            if not n.readable:
                denied.append(f"grep: {q}: Permission denied")
                continue
            for l in n.content.split("\n"):
                if bool(rx.search(l)) != inv:
                    out.append(f"{q}:{l}" if len(files) > 1 else l)
        if out:
            return "\n".join(out), "success"
        if denied:
            return "\n".join(denied[:3]), "permission_denied"
        return "", "not_found_no_hint"


def progress(sc: Scenario, responses: list[str]) -> float:
    seen = "\n".join(responses)
    hit = sum(1 for alts in sc.milestones if any(a in seen for a in alts))
    return hit / len(sc.milestones)
