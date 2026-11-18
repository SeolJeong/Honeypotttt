"""Off-the-shelf LLMs as classifiers (no training): the "why fine-tune?" baseline.

Sends each test session (the same dataset4 text Model 3 reads) to an
OpenAI-compatible chat API, e.g. OpenRouter, and asks for a probability per
class. Writes predictions that scripts/compare_models.py reads with --others.

  zero-shot:  class definitions only
  few-shot:   + K labelled TRAINING sessions per class (fixed seed), never test ones

The prompt is fixed in this file. Don't tune it on test results; if you change
it, change it for every model and say so.

    python scripts/zero_shot_llm.py --model openai/gpt-4o-mini --name gpt4o-mini-zs --dry-run
    python scripts/zero_shot_llm.py --model openai/gpt-4o-mini --name gpt4o-mini-zs
    python scripts/zero_shot_llm.py --model openai/gpt-4o-mini --name gpt4o-mini-fs3 --few-shot 3
    python scripts/compare_models.py --llm qlora_patched_results.json \\
        --others gpt4o-mini-zs=runs/zero_shot/gpt4o-mini-zs.jsonl

Caveats to report:
  - once real agent sessions are in the data (v3), an LLM judging sessions made
    by its own model family may recognise its own style. Prefer judges from
    families that generated no data, or report the overlap
  - parse failures get uniform probabilities and are counted (parse_fail_rate)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_agent_matrix import fetch_models
from scripts.run_llm_agent import chat, resolve_key

LABELS = ["human", "scripted", "AI-driven"]
_STEP = re.compile(r"^\[GAP ", re.M)

SYSTEM = (
    "You are a security analyst. You will see one SSH honeypot session: each step shows the time gap "
    "before the command ([GAP SHORT|MEDIUM|LONG|VERY_LONG]), the command, and the system's response.\n"
    "Classify who produced the commands:\n"
    "- human: a person typing; may pause, mistype, explore, repeat things.\n"
    "- scripted: a pre-written program; follows a fixed plan and does not adapt to what responses say.\n"
    "- AI-driven: an autonomous LLM agent; reasons about each response and adapts (for example follows a "
    "hint in the output), and may deliberately imitate a human.\n"
    'Reply with JSON only, probabilities that sum to 1: {"human": p, "scripted": p, "AI-driven": p}'
)


def steps_of(text: str, commands_only: bool) -> list[str]:
    starts = [m.start() for m in _STEP.finditer(text)] or [0]
    chunks = [text[a:b].rstrip("\n") for a, b in zip(starts, starts[1:] + [len(text)])]
    return [c.split("\n", 1)[0] for c in chunks] if commands_only else chunks


def render(rec: dict, commands_only: bool, k: int | None, max_chars: int) -> str:
    st = steps_of(rec["interleaved_text"], commands_only)
    st = st[:k] if k else st
    txt = "\n".join(st)
    return txt if len(txt) <= max_chars else "...[earlier steps truncated]\n" + txt[-max_chars:]


def parse_probs(reply: str):
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", reply):
        try:
            obj, _ = dec.raw_decode(reply, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            try:
                v = [max(0.0, float(obj.get(l, 0))) for l in LABELS]
            except (TypeError, ValueError):
                continue
            if sum(v) > 0:
                return [x / sum(v) for x in v]
    low = reply.lower()                      # last resort: a bare label
    hits = [l for l in LABELS if l.lower() in low]
    if len(hits) == 1:
        return [1.0 if l == hits[0] else 0.0 for l in LABELS]
    return None


def load(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def stratified(recs, n, seed):
    if not n or n >= len(recs):
        return recs
    rng = random.Random(seed)
    by = {}
    for r in recs:
        by.setdefault(r["actor_label"], []).append(r)
    out = []
    for lab, rs in sorted(by.items()):
        rng.shuffle(rs)
        out += rs[:max(1, round(n * len(rs) / len(recs)))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("data", "dataset4_llm_sequences.jsonl"))
    ap.add_argument("--external", default=None, help="e.g. external/kypo/kypo_sessions_ds4_commands_only.jsonl "
                                                      "(real humans; use with --commands-only)")
    ap.add_argument("--model", required=True, help="e.g. openai/gpt-4o-mini (OpenRouter id)")
    ap.add_argument("--name", required=True, help="label used in results, e.g. gpt4o-mini-zs")
    ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    ap.add_argument("--provider-order", nargs="*", default=None)
    ap.add_argument("--few-shot", type=int, default=0, help="labelled TRAIN examples per class")
    ap.add_argument("--commands-only", action="store_true")
    ap.add_argument("--k", type=int, default=None, help="only the first k commands (early attribution)")
    ap.add_argument("--n", type=int, default=None, help="stratified subset of the test set (cost control)")
    ap.add_argument("--max-chars", type=int, default=6000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    recs = load(a.data)
    test = stratified([r for r in recs if r["split"] == "test"], a.n, 0)
    if a.external:
        test = load(a.external) if not a.n else stratified(load(a.external), a.n, 0)
    shots = []
    if a.few_shot:
        rng = random.Random(1)
        train = [r for r in recs if r["split"] == "train"]
        for lab in LABELS:
            pool = [r for r in train if r["actor_label"] == lab]
            shots += [(r, lab) for r in rng.sample(pool, a.few_shot)]
    msgs_prefix = [{"role": "system", "content": SYSTEM}]
    for r, lab in shots:
        msgs_prefix += [{"role": "user", "content": render(r, a.commands_only, 8, 1500)},
                        {"role": "assistant", "content": json.dumps({l: (1.0 if l == lab else 0.0) for l in LABELS})}]

    prefix_name = f"k={a.k}" if a.k else "full"
    out = a.out or os.path.join("runs", "zero_shot", f"{a.name}{'-ext' if a.external else ''}.jsonl")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(out):
        done = {(r["session_id"], r.get("prefix")) for r in load(out)}
    todo = [r for r in test if (r["session_id"], prefix_name) not in done]

    shot_chars = sum(len(m["content"]) for m in msgs_prefix)
    est_in = sum(shot_chars + len(render(r, a.commands_only, a.k, a.max_chars)) for r in todo) / 3.2
    print(f"{a.name}: {len(test)} sessions, {len(todo)} to do | few-shot {a.few_shot}/class | "
          f"commands-only {a.commands_only} | est. {est_in / 1e6:.2f}M input tokens")
    key = resolve_key(a.api_key_env)
    if a.dry_run:
        try:
            m = fetch_models(a.base_url, key).get(a.model)
            if m is None:
                sys.exit(f"model id not found: {a.model}")
            p = m.get("pricing", {})
            cost = est_in * float(p.get("prompt", 0) or 0) + len(todo) * 40 * float(p.get("completion", 0) or 0)
            print(f"rough cost: ~${cost:.2f}")
        except Exception as e:
            print(f"(could not price: {e})")
        return

    fails = 0
    with open(out, "a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            msgs = msgs_prefix + [{"role": "user", "content": render(r, a.commands_only, a.k, a.max_chars)}]
            try:
                reply, lat, served, provider = chat(a.base_url, key, a.model, msgs, 0.0,
                                                    provider_order=a.provider_order)
            except RuntimeError as e:
                print(f"  API error on {r['session_id']}: {e}; stopping (re-run to resume)")
                break
            probs = parse_probs(reply)
            fail = probs is None
            fails += fail
            probs = probs or [1 / 3] * 3
            f.write(json.dumps({"session_id": r["session_id"], "prefix": prefix_name, "true_label": r["actor_label"],
                                "probs": dict(zip(LABELS, [round(x, 4) for x in probs])), "parse_fail": fail,
                                "model": a.model, "served_model": served, "latency_s": round(lat, 3),
                                "few_shot": a.few_shot, "commands_only": a.commands_only}) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} done ({fails} parse failures so far)", flush=True)
    rows = load(out)
    pf = sum(r["parse_fail"] for r in rows) / max(1, len(rows))
    acc = sum(max(r["probs"], key=r["probs"].get) == r["true_label"] for r in rows) / max(1, len(rows))
    print(f"wrote {out}: {len(rows)} predictions | accuracy {acc:.3f} | parse_fail_rate {pf:.3f}")
    print(f"next: python scripts/compare_models.py --llm <Model 3 results> --others {a.name}={out}")


if __name__ == "__main__":
    main()
