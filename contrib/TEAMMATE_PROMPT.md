# HoneyAI: help collect session data (teammate brief)

> Paste this whole file to a teammate, or into their AI assistant along with the repo. Everything runs against a **simulated** Linux host written in Python. No real server is touched, and no attack tooling is involved.

---

Hi! We need **real** sessions for the HoneyAI benchmark. Our current dataset (`data/v2`) uses *simulated* humans and AI agents, which is fine for building the pipeline but not for the results we report. Please collect one or both of the following and send me the `.jsonl` files from `contrib/`.

## Setup (5 minutes)
```bash
git clone <repo> && cd Honeypotttt     # or unzip the copy I sent
python3 --version                      # 3.10+; no extra packages needed for collection
```

## Job 1: real LLM-agent sessions (most needed)
Each teammate takes **a different model family**, so we can hold one family out for testing:

| teammate | model | `--family` label (exact) | how | cost |
|---|---|---|---|---|
| A | `llama3.1:8b` via **Ollama** (local) | `llama3.1-8b` | `ollama pull llama3.1:8b` | free |
| B | `gpt-4o-mini` via OpenAI API | `gpt4o-mini` | `export OPENAI_API_KEY=...` | a few dollars at most (check the pricing page) |
| C | `mistral:7b` via Ollama | `mistral-7b` | `ollama pull mistral:7b` | free |

Please don't use Qwen models as agents. We use Qwen as a detector baseline, so it must not also generate the data.

Run all 7 evasion strategies, 2 sessions per task, which gives 112 sessions per model:
```bash
# Ollama (A, C)
python3 scripts/run_llm_agent.py --base-url http://localhost:11434/v1 --api-key-env NONE \
    --model llama3.1:8b --family llama3.1-8b --evasion all --reps 2 --contributor <your initials>

# OpenAI (B)
python3 scripts/run_llm_agent.py --model gpt-4o-mini --family gpt4o-mini \
    --evasion all --reps 2 --contributor <your initials>
```
- Output goes to `contrib/agent_<family>.jsonl`. It appends, so if the run stops you can simply run it again.
- If you have time or budget, run `--reps 4` for more data.
- Keep `--family` exactly as in the table, so files from different people merge correctly.

## Job 2: real human participants (hardest to get, most valuable)
Each of us please recruit **2–3 people**: classmates, friends, family. Anyone who can type commands works, and mixed Linux experience is good.

```bash
python3 scripts/collect_human.py --participant P0XX --contributor <your initials>
```
- **Participant IDs:** use your block: P1 P006–P009 · P2 P010–P013 · P3 P014–P017 · P4 P018–P021 · P5 P022–P025. **Never use names.**
- Read the consent text to them. They type `yes`, then do up to 8 short tasks (about 15–25 minutes in total).
- Don't help them with commands. If they're stuck, they can type `skip`.
- **Ethics:** don't start until I confirm the lecturer has OK'd collecting participant data (spec section 6).

## Rules (these keep the results valid)
1. **Don't edit the `.jsonl` files**, and don't write, fake or "fix" sessions by hand or with an AI. Every human session must come from a real person typing, and every agent session from a real model run.
2. One participant ID per person, ever. Don't reuse IDs across teammates.
3. Don't run the human collector yourself as a "participant". We built the tasks, so we know the answers.
4. Send me: the `contrib/*.jsonl` files, and which model and version you used (e.g. "Ollama llama3.1:8b, 2026-11-05").

## What I do with it
```bash
python3 scripts/merge_sessions.py contrib/*.jsonl --base data/v2/dataset1_sessions.json \
    --test-participants <held-out IDs> --test-families mistral-7b --out data/v3
```
The merge validates every file, assigns train and test splits (participant-disjoint and family-disjoint), and rebuilds all three dataset files.

Thanks!
