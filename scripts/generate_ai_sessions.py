"""Generate REAL AI-driven sessions by driving the task harness with a live LLM.

You run this with your own API key; it replaces the simulated AI placeholders.
Keys are read from the environment, never passed on the command line.

Examples
--------
OpenAI (gpt-4o-mini), non-evasive, 3 sessions per task:
    export OPENAI_API_KEY=sk-...
    python scripts/generate_ai_sessions.py --provider openai --model gpt-4o-mini \
        --per-task 3 --out data/sessions/ai_openai.jsonl

Anthropic (haiku), with combined evasion (AI imitating a human):
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/generate_ai_sessions.py --provider anthropic \
        --model claude-3-5-haiku-20241022 --evasion combined \
        --per-task 3 --out data/sessions/ai_claude_evasive.jsonl

Local, free (Ollama):
    python scripts/generate_ai_sessions.py --provider ollama --model llama3.2 \
        --per-task 3 --out data/sessions/ai_ollama.jsonl

Tip for the experiment: generate >= 2 model families for training and hold >= 1
family out for testing; generate each evasion strategy as its own file, and keep
at least one strategy entirely out of training.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from honeyai.harness.tasks import TASKS
from honeyai.generators.ai_driven import gen_ai_live
from honeyai.schema import write_sessions, EvasionStrategy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["openai", "anthropic", "ollama"], required=True)
    ap.add_argument("--model", default=None, help="model id (provider default if omitted)")
    ap.add_argument("--evasion", default="none",
                    choices=[e.value for e in EvasionStrategy])
    ap.add_argument("--per-task", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    family = f"{args.provider}:{args.model or 'default'}"
    evasion = EvasionStrategy(args.evasion)
    sessions, i = [], 0
    for task in TASKS.values():
        for _ in range(args.per_task):
            sid = f"AIL{i:05d}"
            print(f"  [{i}] {task.task_id} via {family} evasion={evasion.value} ...",
                  flush=True)
            try:
                s = gen_ai_live(sid, task, model_family=family, evasion=evasion,
                                max_steps=args.max_steps,
                                provider=args.provider, model=args.model)
                sessions.append(s)
            except Exception as e:  # keep going; one bad call shouldn't kill the run
                print(f"      skipped: {e}")
            i += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_sessions(sessions, args.out)
    print(f"\nwrote {len(sessions)} sessions -> {args.out}")


if __name__ == "__main__":
    main()
