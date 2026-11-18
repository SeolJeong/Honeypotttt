# HoneyAI in Docker: data collection with OpenRouter

The container is for **data collection, merging and the CPU baselines**. **GPU training of the QLoRA model is not done here.** Use Colab or the 3070 for that (`notebooks/qlora_colab.ipynb`).

The repo is mounted into the container, so everything you collect (`contrib/`, `data/`) stays on your own disk, even if you delete the container.

## What was tested (2026-11-03) and what wasn't
- **Tested inside a container** built from this Dockerfile's steps, against a fake OpenRouter server:
  - model listing
  - dry run with cost estimate
  - a 429 rate-limit retry
  - a model that refuses (session saved as `incomplete`)
  - commands containing braces (`awk '{print $1}'`)
  - resume after a stop
  - the human recorder with piped input
  - merge (incomplete sessions dropped, leakage warnings)
  - data generation and the text baseline
- **Not tested:**
  - **Real OpenRouter.** It was unreachable from where this was built, so the model IDs in `configs/agents.example.toml` are **unverified examples**. The runner checks them against OpenRouter's live list and refuses to start if one is wrong.
  - Docker Desktop on Windows/macOS.
  - The exact `FROM python:3.12-slim` pull. The test pulled the same image through a registry mirror because of a Docker Hub rate limit.

## 1. Setup (once)
1. Install **Docker Desktop** (Windows/macOS) or Docker Engine (Linux).
2. In the repo folder:
   ```bash
   cp .env.example .env          # then put your OpenRouter key in .env
   cp configs/agents.example.toml configs/agents.toml
   docker compose build
   ```
   `.env` is git-ignored; never commit it or paste the key anywhere else.
3. If `data/v2` is missing, generate it:
   ```bash
   docker compose run --rm honeyai python scripts/generate_sessions.py
   ```

## 2. Choose models
List real model IDs, with prices per million tokens:
```bash
docker compose run --rm honeyai python scripts/run_agent_matrix.py --list-models llama
docker compose run --rm honeyai python scripts/run_agent_matrix.py --list-models mistral
docker compose run --rm honeyai python scripts/run_agent_matrix.py --list-models gpt-4o
```
Then edit `configs/agents.toml`. Each `[[agent]]` is **one model family**. Rules:
- **At least 3 families:** at least 2 for training and at least 1 held out for testing. Pick the held-out one now and write it down.
- **No Qwen models as agents.** Qwen is your detector and zero-shot baseline.
- `family` labels are permanent. Don't rename them after collecting.
- Set `provider_order` per family if you can. It pins the upstream provider, so the recorded timing reflects one consistent provider instead of OpenRouter's routing (see §5).
- Don't change a family's settings partway through collection.

## 3. Pilot, then run
```bash
# plan + rough cost (no API calls except the model list)
docker compose run --rm honeyai python scripts/run_agent_matrix.py --dry-run

# pilot ONE family, then read a few sessions in contrib/agent_<family>.jsonl
docker compose run --rm honeyai python scripts/run_agent_matrix.py --only llama3.1-8b

# everything
docker compose run --rm honeyai python scripts/run_agent_matrix.py
```
- **It's resumable.** Re-run the same command after a crash, rate limit or Ctrl-C, and finished sessions are skipped. Incomplete ones (API error or refusal) are retried.
- The cost estimate is **rough**: it assumes about 12k prompt tokens and 600 completion tokens per session. Check your real spend on the OpenRouter dashboard after the pilot.
- **Watch for `INCOMPLETE (unparseable_or_refused)`.** If one family refuses most sessions, drop it rather than rewording prompts just for that model; family settings must stay comparable.

## 4. Humans, merge, baselines
```bash
# human participant (interactive; hand them the keyboard)
docker compose run --rm honeyai python scripts/collect_human.py --participant P006 --contributor P1

# merge everything (incomplete sessions are dropped by default)
docker compose run --rm honeyai python scripts/merge_sessions.py contrib/*.jsonl \
    --base data/v2/dataset1_sessions.json \
    --test-participants P006 P010 P014 P018 P022 --test-families <held-out family> --out data/v3

# CPU baselines on the merged data
docker compose run --rm honeyai python scripts/text_baseline.py --data data/v3/dataset1_sessions.json
```
The `contrib/*.jsonl` wildcard is expanded by *your* shell. In Windows PowerShell, list the files explicitly or run the merge inside `docker compose run --rm honeyai sh -c "python scripts/merge_sessions.py contrib/*.jsonl ..."`.

## 5. Timing caveat with APIs
Recorded gap = API latency + any `wait_seconds` the agent requested. Through OpenRouter, the latency includes network time, queueing and the provider's speed. That means **different families can differ in timing just because of their provider**, which is a property of the infrastructure, not of AI behaviour. Mitigations:
1. Pin `provider_order` per family.
2. Collect each family in a few sessions spread over different times of day.
3. In the report, compare models with **and without** timing (`--no-timing` ablation). If the LLM's advantage only shows up with timing included, say so.

## 6. Back up
`contrib/` is the only copy of your real data. Copy it somewhere else every evening.
