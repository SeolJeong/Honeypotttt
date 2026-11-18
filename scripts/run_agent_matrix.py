"""Run many LLM agents (e.g. via OpenRouter) x evasion strategies x tasks.

    python scripts/run_agent_matrix.py --list-models llama          # find real model ids
    python scripts/run_agent_matrix.py --config configs/agents.toml --dry-run
    python scripts/run_agent_matrix.py --config configs/agents.toml
    python scripts/run_agent_matrix.py --config configs/agents.toml --only llama3.1-8b

- Checks every model id against the endpoint's /models list before starting.
- Resumable: each (family, evasion, task, rep) has a fixed key and scenario
  seed; anything already in the output file is skipped, so just re-run after
  a crash, rate limit or Ctrl-C.
- --dry-run prints the plan and a ROUGH cost estimate from /models pricing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
import urllib.request
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from honeyai.sim.actors import EVASIONS
from honeyai.sim.environment import TASKS
from honeyai.sim.recording import append_jsonl
from scripts.run_llm_agent import resolve_key, run_one

# rough per-session token use for the cost estimate (about 8 turns; the history grows each turn)
EST_PROMPT_TOKENS = 12_000
EST_COMPLETION_TOKENS = 600


def fetch_models(base_url: str, key: str | None) -> dict:
    req = urllib.request.Request(base_url.rstrip("/") + "/models",
                                 headers={"Authorization": f"Bearer {key}"} if key else {})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    return {m["id"]: m for m in data.get("data", [])}


def load_config(path: str) -> list[dict]:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    d = raw.get("defaults", {})
    agents = []
    for a in raw.get("agent", []):
        c = {**d, **a}
        ev = c.get("evasions", "all")
        c["evasions"] = EVASIONS if ev == "all" else list(ev)
        bad = [e for e in c["evasions"] if e not in EVASIONS]
        if bad:
            sys.exit(f"{c['family']}: unknown evasions {bad}; choose from {EVASIONS}")
        c["tasks"] = TASKS if c.get("tasks", "all") == "all" else list(c["tasks"])
        c.setdefault("reps", 1)
        c.setdefault("temperature", 0.7)
        c.setdefault("contributor", "anon")
        c.setdefault("api_key_env", "OPENROUTER_API_KEY")
        c.setdefault("base_url", "https://openrouter.ai/api/v1")
        for k in ("family", "model"):
            if not c.get(k):
                sys.exit(f"every [[agent]] needs '{k}'")
        agents.append(c)
    fams = [a["family"] for a in agents]
    if len(fams) != len(set(fams)):
        sys.exit("family labels must be unique")
    return agents


def done_keys(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    keys = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                s = json.loads(line)
            except json.JSONDecodeError:
                continue
            if s.get("matrix_key") and not s.get("incomplete"):
                keys.add(s["matrix_key"])
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join("configs", "agents.toml"))
    ap.add_argument("--only", nargs="*", help="run only these family labels")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list-models", nargs="?", const="", metavar="FILTER",
                    help="list model ids (and $/M tokens) from the endpoint, optionally filtered")
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1", help="for --list-models")
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY", help="for --list-models")
    a = ap.parse_args()

    if a.list_models is not None:
        models = fetch_models(a.base_url, os.environ.get(a.api_key_env))
        for mid, m in sorted(models.items()):
            if a.list_models.lower() in mid.lower():
                p = m.get("pricing", {})
                pin = float(p.get("prompt", 0) or 0) * 1e6
                pout = float(p.get("completion", 0) or 0) * 1e6
                print(f"{mid:<60s} in ${pin:7.3f}/M  out ${pout:7.3f}/M  ctx {m.get('context_length')}")
        return

    agents = load_config(a.config)
    if a.only:
        agents = [x for x in agents if x["family"] in a.only]

    # validate model ids + rough cost, per endpoint
    total_cost, cache = 0.0, {}
    for c in agents:
        key = resolve_key(c["api_key_env"])
        if c["base_url"] not in cache:
            try:
                cache[c["base_url"]] = fetch_models(c["base_url"], key)
            except Exception as e:
                print(f"WARNING could not list models at {c['base_url']} ({e}); skipping id check")
                cache[c["base_url"]] = None
        models = cache[c["base_url"]]
        n = len(c["evasions"]) * len(c["tasks"]) * c["reps"]
        out = os.path.join("contrib", f"agent_{c['family']}.jsonl")
        remaining = n - len(done_keys(out))
        line = f"{c['family']:<22s} {c['model']:<50s} {n:4d} sessions ({max(0, remaining)} to do)"
        if models is not None:
            if c["model"] not in models:
                sys.exit(f"\nmodel id not found at {c['base_url']}: {c['model']!r}\n"
                         f"find the right one with: python scripts/run_agent_matrix.py --list-models "
                         f"{c['model'].split('/')[-1][:12]}")
            p = models[c["model"]].get("pricing", {})
            cost = max(0, remaining) * (EST_PROMPT_TOKENS * float(p.get("prompt", 0) or 0)
                                        + EST_COMPLETION_TOKENS * float(p.get("completion", 0) or 0))
            total_cost += cost
            line += f"  ~${cost:.2f}"
        print(line)
    print(f"rough total for remaining sessions: ~${total_cost:.2f} "
          f"(estimate assumes ~{EST_PROMPT_TOKENS} prompt + {EST_COMPLETION_TOKENS} completion tokens/session)")
    if a.dry_run:
        return

    for c in agents:
        key = resolve_key(c["api_key_env"])
        out = os.path.join("contrib", f"agent_{c['family']}.jsonl")
        os.makedirs("contrib", exist_ok=True)
        done = done_keys(out)
        i = 0
        for ev in c["evasions"]:
            for task in c["tasks"]:
                for rep in range(c["reps"]):
                    mkey = f"{c['family']}|{ev}|{task}|{rep}"
                    if mkey in done:
                        continue
                    seed = 10**8 + zlib.crc32(mkey.encode()) % (9 * 10**8)
                    s = run_one(c, task, ev, seed, key,
                                session_id=f"AG-{c['family']}-{ev}-{task}-r{rep}",
                                extra_meta={"matrix_key": mkey})
                    append_jsonl(out, s)
                    i += 1
                    flag = f"  INCOMPLETE ({(s['incomplete_reason'] or '')[:70]})" if s["incomplete"] else ""
                    print(f"[{c['family']} {i}] {ev:<34s} {task:<26s} steps={s['total_steps']:<2d} "
                          f"done={s['task_completed']}{flag}", flush=True)
        print(f"{c['family']}: finished -> {out}")


if __name__ == "__main__":
    main()
