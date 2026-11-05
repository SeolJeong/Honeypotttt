"""Actor policies that drive the simulated shell (honeyai.sim.environment).

Three kinds of actor, and they are NOT equally "real":

  scripted   Real scripts. Fixed command lists with the control flow of each
             family (replay, state machine, error handling, ...). Their
             traffic is genuinely what such scripts produce against this host.

  human      SIMULATED. A parametric policy (persona) standing in for people.
  AI-driven  SIMULATED. A parametric policy standing in for LLM agents.

The human and AI policies encode explicit assumptions about how those actors
behave (listed in ASSUMPTIONS below and in data/v2/DATA_CARD.md). Any model
trained on them partly learns these assumptions back. They are a bootstrap
for building and debugging the pipeline; headline claims need real human
participants (scripts/collect_human.py) and real LLM agents
(scripts/run_llm_agent.py).

All policies move through the scenario's canonical solution chain. Chain
step k+1 always uses something revealed by step k's response, so "follow
the chain right after a hint" is what adaptation means here. Actors differ
in how reliably/quickly they follow it, what they do instead, how they type,
and how long they take.
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass

from honeyai.sim.environment import Scenario, Shell, make_scenario, progress

MAX_STEPS = 16

ASSUMPTIONS = {
    "human": [
        "follows a revealed hint with persona-specific probability 0.45-0.85; otherwise repeats the "
        "failed command, explores nearby, asks for help (man), or tries sudo",
        "think time is lognormal per persona (median 3.5-10 s) and grows with output length and after errors",
        "typos at 3-18% of commands, corrected on the next command ~85% of the time, retyped quickly",
        "often navigates with cd + relative paths; uses plain command forms",
        "sometimes verifies after finishing; sometimes gives up near the step limit",
    ],
    "AI-driven": [
        "follows a revealed hint with probability 0.90-0.95 (0.75 when avoiding systematic enumeration); "
        "otherwise runs a targeted search (find/grep) that also reaches the answer",
        "latency is lognormal around a family median (1.4-3.2 s) plus a small output-length term",
        "uses absolute paths and polished forms (pipes, flags, 2>/dev/null); stops as soon as the task is done",
        "evasion strategies modify this policy: human-like timing that does NOT depend on output length or "
        "errors; injected typos that are ALWAYS corrected on the next command; shorter/plain commands; "
        "redundant exploration; no find/grep -r enumeration",
    ],
}

GENERIC_DETOURS = ["pwd", "ls", "ls -la", "whoami", "id", "hostname", "uptime", "w", "df -h",
                   "ls /opt", "ls /srv", "ls /var/log", "cat /etc/hostname", "history", "ls /etc | head",
                   "ps aux | head", "ls /home", "date", "free -m", "ls /tmp"]
GUESS_DIRS = ["ls /opt", "ls /srv", "ls /var/lib", "ls /data", "ls /etc", "ls /var/backups", "ls /var/log"]


# ── shared helpers ──────────────────────────────────────────────────────

def lognormal(rng, median, sigma):
    return median * math.exp(rng.gauss(0, sigma))


def make_typo(cmd: str, rng: random.Random) -> str:
    """One realistic keyboard slip: swap/drop/double a char or a lost space."""
    for _ in range(10):
        toks = cmd.split(" ")
        i = 0 if rng.random() < .45 or len(toks) == 1 else rng.randrange(1, len(toks))
        w = toks[i]
        if len(w) < 2:
            continue
        op = rng.choice(["swap", "drop", "double", "space"])
        j = rng.randrange(len(w) - 1)
        if op == "swap":
            w2 = w[:j] + w[j + 1] + w[j] + w[j + 2:]
        elif op == "drop":
            w2 = w[:j] + w[j + 1:]
        elif op == "double":
            w2 = w[:j] + w[j] + w[j:]
        else:
            if i + 1 < len(toks):
                toks[i] = w + toks[i + 1]
                del toks[i + 1]
                out = " ".join(toks)
                if out != cmd:
                    return out
            continue
        if w2 != w:
            toks[i] = w2
            return " ".join(toks)
    return cmd + cmd[-1]


def shorten(cmd: str) -> str:
    """Strip polish: redirections, pipes, most flags."""
    cmd = re.sub(r"\s*2>\s*/dev/null", "", cmd)
    cmd = cmd.split("|")[0].strip()
    toks = cmd.split()
    keep = [toks[0]] + [t for t in toks[1:] if not t.startswith("-") or t in ("-a", "-l")]
    if toks[0] == "grep" and len(keep) >= 3:
        return " ".join(keep)
    if toks[0] in ("grep",) and "-v" in toks:
        return "cat " + toks[-1]
    return " ".join(keep)


def cd_style(cmd: str) -> list[str] | None:
    """`cat /a/b/c` -> ['cd /a/b', 'cat c']  (human navigation habit)."""
    toks = cmd.split()
    if len(toks) < 2 or toks[0] not in ("cat", "ls", "head", "tail", "less") or "|" in cmd:
        return None
    path = toks[-1]
    if not path.startswith("/") or path.count("/") < 2:
        return None
    path = path.rstrip("/")
    d, base = path.rsplit("/", 1)
    if toks[0] == "ls":
        return [f"cd {path}", " ".join(toks[:-1])]
    return [f"cd {d or '/'}", " ".join(toks[:-1] + [base])]


def gap_bin(s):
    return "short" if s < 2 else "medium" if s < 6 else "long" if s < 15 else "very_long"


# ── session runner ──────────────────────────────────────────────────────

def run_session(sc: Scenario, policy) -> list[dict]:
    sh = Shell(sc)
    steps, responses, t = [], [], 0.0
    while len(steps) < MAX_STEPS:
        nxt = policy.next(sc, steps, progress(sc, responses))
        if nxt is None:
            break
        cmd, typo = nxt
        prev = steps[-1] if steps else None
        gap = round(max(0.05, policy.gap(prev, cmd, typo, steps)), 2)
        out, cat = sh.run(cmd)
        responses.append(out)
        t += gap
        steps.append({
            "step": len(steps), "command": cmd, "has_typo": typo,
            "system_response": out, "response_category": cat,
            "inter_command_gap_s": gap, "think_time_bin": gap_bin(gap),
            "cumulative_time_s": round(t, 2), "task_progress": round(progress(sc, responses), 2),
        })
        policy.observe(steps[-1])
    return steps


# ── chain-following policy (simulated humans and AI agents) ─────────────

@dataclass
class ChainParams:
    p_adapt: float            # follow the next chain step when it was just revealed
    p_detour: float           # spontaneous off-task command
    p_typo: float
    p_fix_typo: float         # correct a typo on the very next command
    p_cd: float               # use cd + relative path
    p_plain: float            # plain vs polished command form
    p_repeat_fail: float      # retype a failed command unchanged
    p_help: float
    p_sudo: float
    p_verify: float           # one extra command after finishing
    p_optional: float         # include optional chain steps
    miss_mode: str            # "search" | "guess" | "human"
    shorten: bool = False
    redundant: float = 0.0    # repeat a previous command / pwd / ls (hesitation)
    give_up_after: int = MAX_STEPS


class ChainPolicy:
    def __init__(self, params: ChainParams, timing, rng: random.Random, style=None):
        self.p, self.timing, self.rng, self.style = params, timing, rng, style
        self.queue: list[tuple[str, bool]] = []
        self.i = 0
        self.plan = None
        self.pending_fix = None
        self.verified = False
        self.searched = 0

    def _render(self, step) -> str:
        r, p = self.rng, self.p
        cmd = step.get("plain", step["cmd"]) if r.random() < p.p_plain else step["cmd"]
        if self.style:
            cmd = self.style(cmd, r)
        if p.shorten:
            cmd = shorten(cmd)
        return cmd

    def _emit(self, cmd: str) -> tuple[str, bool]:
        r, p = self.rng, self.p
        if r.random() < p.p_cd:
            parts = cd_style(cmd)
            if parts:
                self.queue += [(c, False) for c in parts[1:]]
                cmd = parts[0]
        if r.random() < p.p_typo:
            bad = make_typo(cmd, r)
            if bad != cmd:
                if r.random() < p.p_fix_typo:
                    self.pending_fix = cmd
                return bad, True
        return cmd, False

    def next(self, sc: Scenario, steps, prog):
        r, p = self.rng, self.p
        if self.plan is None:
            self.plan = [s for s in sc.chain if r.random() < (p.p_optional if "optional" in s else 1.0)]
        if self.pending_fix:
            cmd, self.pending_fix = self.pending_fix, None
            return cmd, False
        if self.queue:
            return self.queue.pop(0)
        if prog >= 1.0:
            if not self.verified and r.random() < p.p_verify:
                self.verified = True
                return self._emit(r.choice([steps[-1]["command"], "ls", "pwd", f"echo done"]))
            return None
        if len(steps) >= p.give_up_after:
            return None
        if steps and r.random() < p.redundant:
            return self._emit(r.choice([steps[-1]["command"], "pwd", "ls", r.choice(steps)["command"]]))
        if r.random() < p.p_detour:
            return self._emit(r.choice(GENERIC_DETOURS))

        last = steps[-1] if steps else None
        if self.i >= len(self.plan):          # plan exhausted but goal unmet: search/explore
            return self._miss(sc, last)
        if last and self.i > 0 and r.random() > p.p_adapt:
            return self._miss(sc, last)
        cmd = self._render(self.plan[self.i])
        self.i += 1
        return self._emit(cmd)

    def _miss(self, sc, last):
        r, p = self.rng, self.p
        if p.miss_mode == "search" and sc.facts.get("search") and self.searched < 2:
            self.searched += 1
            return self._emit(r.choice(sc.facts["search"]))
        if p.miss_mode == "human":
            failed = last and last["response_category"] != "success"
            x = r.random()
            if failed and x < p.p_repeat_fail:
                return last["command"], last["has_typo"]
            if x < p.p_repeat_fail + p.p_help:
                prog = last["command"].split()[0] if last else "ls"
                return self._emit(r.choice([f"man {prog}", f"{prog} --help"]))
            if x < p.p_repeat_fail + p.p_help + p.p_sudo and last:
                return self._emit("sudo " + last["command"])
            if sc.facts.get("search") and r.random() < .35:
                return self._emit(r.choice(sc.facts["search"]))
            return self._emit(r.choice(GUESS_DIRS + GENERIC_DETOURS[:6]))
        return self._emit(r.choice(GUESS_DIRS))

    def gap(self, prev, cmd, typo, steps):
        return self.timing(prev, cmd, typo, steps, self.rng)

    def observe(self, step):
        pass


# ── humans (simulated personas) ─────────────────────────────────────────

# Per-persona think time: median ~ U(median_lo, median_hi), sigma ~ U(sigma_lo, sigma_hi).
# Defaults reproduce data/v2. scripts/calibrate_human_timing.py fits them to real
# humans (KYPO) and writes configs/human_timing.json; generate_sessions.py
# --human-timing loads that file.
HUMAN_TIMING = {"median_lo": 3.5, "median_hi": 10.0, "sigma_lo": 0.5, "sigma_hi": 0.9}


def set_human_timing(params: dict) -> None:
    HUMAN_TIMING.update({k: float(params[k]) for k in ("median_lo", "median_hi", "sigma_lo", "sigma_hi")})


def human_persona(pid: int):
    r = random.Random(10_000 + pid)
    params = ChainParams(
        p_adapt=r.uniform(.45, .85), p_detour=r.uniform(.08, .3), p_typo=r.uniform(.03, .18),
        p_fix_typo=.85, p_cd=r.uniform(.1, .6), p_plain=r.uniform(.5, .95),
        p_repeat_fail=r.uniform(.05, .25), p_help=r.uniform(0, .1), p_sudo=r.uniform(0, .15),
        p_verify=r.uniform(.1, .4), p_optional=r.uniform(.4, .9), miss_mode="human",
        give_up_after=r.randint(11, MAX_STEPS))
    med = r.uniform(HUMAN_TIMING["median_lo"], HUMAN_TIMING["median_hi"])
    sig = r.uniform(HUMAN_TIMING["sigma_lo"], HUMAN_TIMING["sigma_hi"])

    def timing(prev, cmd, typo, steps, rng):
        g = lognormal(rng, med, sig)
        if prev is None:
            return g * 1.5
        if prev["has_typo"] and prev["response_category"] != "success":
            return g * .35                    # retyping a slip is quick
        if prev["response_category"] != "success":
            g *= 1.4
        g *= 1 + min(prev["system_response"].count("\n"), 25) * .04   # reading time
        return g

    return params, timing


# ── AI agents (simulated families) + evasion strategies ─────────────────

def _style_c(cmd, rng):
    """Held-out family's own idioms (test only)."""
    cmd = re.sub(r"^cat (\S+)$", r"head -n 50 \1", cmd)
    cmd = cmd.replace("ls -la", "ls -lah").replace("grep -v '^#' ", "awk '!/^#/' ")
    cmd = re.sub(r"^ps aux \| grep (\S+)$", r"pgrep -af \1", cmd)
    return cmd


AI_FAMILIES = {
    # name: (p_adapt, latency median, sigma, p_optional, p_verify, style)
    "sim_agent_a": (.95, 2.0, .35, .3, .05, None),
    "sim_agent_b": (.92, 3.2, .3, .9, .45, None),
    "sim_agent_c": (.90, 1.4, .45, .5, .2, _style_c),
}
EVASIONS = ["none", "timing_mimicry", "typo_mimicry", "shorter_commands",
            "hesitation_redundant_exploration", "avoid_systematic_enumeration", "combined"]


def ai_agent(family: str, evasion: str, rng: random.Random):
    p_adapt, lat, sig, p_opt, p_ver, style = AI_FAMILIES[family]
    ev = {evasion} if evasion != "combined" else set(EVASIONS[1:6])
    params = ChainParams(
        p_adapt=p_adapt if "avoid_systematic_enumeration" not in ev else .75,
        p_detour=.02, p_typo=.15 if "typo_mimicry" in ev else 0.0, p_fix_typo=1.0,
        p_cd=.25 if "shorter_commands" in ev else .02, p_plain=.9 if "shorter_commands" in ev else .1,
        p_repeat_fail=0, p_help=0, p_sudo=0, p_verify=p_ver, p_optional=p_opt,
        miss_mode="guess" if "avoid_systematic_enumeration" in ev else "search",
        shorten="shorter_commands" in ev,
        redundant=.3 if "hesitation_redundant_exploration" in ev else 0.0)
    h_med = rng.uniform(4, 8)

    def timing(prev, cmd, typo, steps, rng_):
        if "timing_mimicry" in ev:
            return lognormal(rng_, h_med, .55)        # human-looking, but blind to content
        g = lognormal(rng_, lat, sig)
        if prev is not None:
            g += len(prev["system_response"]) / 2500    # reading tokens
        return g

    return ChainPolicy(params, timing, rng, style=style)


# ── scripted clients (real scripts) ─────────────────────────────────────

def task_script(sc: Scenario) -> list[str]:
    """What a task-aware attack script would run. It knows the app name and the
    paths in the task description, never anything from responses."""
    A = sc.app
    t = sc.task
    base = {
        "locate_config_file": [f"cat /etc/{A}.conf", "ls -la /etc", "find / -name '*.conf' 2>/dev/null",
                               f"cat /opt/{A}/{A}.conf", f"ls /opt/{A}", f"cat /opt/{A}/conf/{A}.yaml",
                               "grep -r db_host /etc /opt 2>/dev/null"],
        "identify_os_version": ["uname -a", "cat /etc/os-release", "lsb_release -a", "cat /etc/issue",
                                "hostnamectl"],
        "find_specific_user": ["cat /etc/passwd", "cat /etc/group", "grep auditors /etc/group", "ls -la /home",
                               "getent group auditors", "id"],
        "inspect_running_service": ["ps aux", f"systemctl status {A}", f"pgrep -a {A}", "ps -ef",
                                    f"cat /etc/systemd/system/{A}.service"],
        "discover_listening_port": ["netstat -tlnp", "ss -tlnp", "ss -tulpn", f"ps aux | grep {A}",
                                    f"cat /etc/{A}.conf"],
        "locate_decoy_credential": ["cat ~/.bash_history", "find / -name '*.pgpass' 2>/dev/null",
                                    "find / -name '*cred*' 2>/dev/null", "ls -la /var/backups", "cat ~/.pgpass",
                                    "grep -ri password /etc 2>/dev/null", "cat /etc/shadow"],
        "diagnose_failed_command": [f"/opt/{A}/bin/{A}-ctl start", f"systemctl status {A}", f"journalctl -u {A}",
                                    f"cat /var/log/{A}/error.log", "ss -tlnp", "ps aux", "df -h"],
        "find_relocated_file": [f"cat /var/lib/{A}/export/latest.csv", f"ls /var/lib/{A}/export",
                                "find / -name '*.csv' 2>/dev/null", f"ls /data/{A}", "ls -la /data",
                                f"cat /var/lib/{A}/export/MOVED.txt"],
    }[t]
    return base


ALT_TOOLS = [(r"^cat ", "head -n 100 "), (r"^ls -la ", "ls -1a "), (r"^ps aux$", "ps -ef"),
             (r"^netstat ", "ss "), (r"-name ", "-iname "), (r"^grep -r ", "grep -R ")]
SCRIPT_FAMILIES = ["fixed_replay", "state_machine", "conditional_error_handling", "randomised_ordering",
                   "human_like_delays", "alternative_commands", "retry_on_failure"]
FALLBACK = {"netstat": "ss -tlnp", "lsb_release": "cat /etc/os-release", "hostnamectl": "uname -a"}


class ScriptPolicy:
    def __init__(self, family: str, rng: random.Random):
        self.f, self.rng = family, rng
        self.todo = None
        self.retries = 0
        self.skip_next = False

    def next(self, sc, steps, prog):
        r = self.rng
        if self.todo is None:
            cmds = task_script(sc)
            if r.random() < .5:
                cmds = ["id", "uname -a"][:r.randint(1, 2)] + cmds
            if self.f == "randomised_ordering":
                r.shuffle(cmds)
            if self.f == "alternative_commands":
                out = []
                for c in cmds:
                    for pat, rep in ALT_TOOLS:
                        c = re.sub(pat, rep, c)
                    out.append(c)
                cmds = out
            self.todo = cmds
        last = steps[-1] if steps else None
        if last and last["response_category"] != "success":
            if self.f == "retry_on_failure" and self.retries < 2:
                self.retries += 1
                return last["command"], False
            if self.f == "conditional_error_handling":
                prog_ = last["command"].split()[0]
                if prog_ in FALLBACK and FALLBACK[prog_] not in [s["command"] for s in steps]:
                    return FALLBACK[prog_], False
                m = re.search(r"(/[\w.\-/]+)", last["command"])
                if m and not last["command"].startswith("find"):
                    fb = f"find / -name '{m.group(1).rstrip('/').split('/')[-1]}' 2>/dev/null"
                    if fb not in [s["command"] for s in steps]:
                        return fb, False
            if self.f == "state_machine" and self.todo and not self.skip_next:
                self.skip_next = True
                self.todo.pop(0)          # skip the dependent next step
        else:
            self.skip_next = False
        self.retries = 0 if (last is None or last["response_category"] == "success") else self.retries
        if not self.todo:
            return None
        return self.todo.pop(0), False

    def gap(self, prev, cmd, typo, steps):
        r = self.rng
        if self.f == "human_like_delays":
            return lognormal(r, 5.0, .45)
        if self.f == "retry_on_failure" and prev and prev["command"] == cmd:
            return 1.0 * (2 ** self.retries) + r.uniform(0, .1)
        lo, hi = {"fixed_replay": (.05, .3), "state_machine": (.1, .6),
                  "conditional_error_handling": (.1, .8), "randomised_ordering": (.05, .5),
                  "alternative_commands": (.2, 1.0), "retry_on_failure": (.1, .4)}[self.f]
        return r.uniform(lo, hi)

    def observe(self, step):
        pass
