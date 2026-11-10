# HoneyAI dataset v3 (built 2026-11-08)

**Purpose:** a clean evaluation set for the final report. v2b's test set was used for model development
(see `docs/TEST_RESULTS_SABRINA.md`), so v2b test scores are biased upward. **Every v3 test session is new.**

## Contents (2,818 sessions; no real humans)

| Split | Class | Source | Sessions |
|---|---|---|---|
| train | human | simulated personas SP21–SP32 | 480 |
| train | scripted | 4 scripted families | 442 |
| train | AI-driven | simulated agents `sim_agent_a`, `sim_agent_b` | 559 |
| train | AI-driven | **real LLM agents:** `llama3.1-8b`, `mistral-small-3.2`, `deepseek-v3.2` | 224 |
| test | human | simulated personas SP33–SP40 | 320 |
| test | scripted | 3 held-out scripted families | 188 |
| test | AI-driven | simulated agents `a`/`b` (held-out strategies), `sim_agent_c` (held-out family) | 318 |
| test | AI-driven | **real LLM agents, held-out families:** `gemma-3-27b` (100), `claude-haiku-5.5` (100), plus the training families' held-out strategies (87) | 287 |

Files: `dataset1_sessions.json` (full traces), `dataset2_tabular_features.csv` (XGBoost features),
`dataset4_llm_sequences.jsonl` (LLM input). They have the same schema as v2b.

## How it was built

1. **Simulated part:**
   ```
   python scripts/generate_sessions.py --seed 2027 --human-timing configs/human_timing.json --id-prefix V3- --persona-offset 20 --out data/v3_sim
   ```
   It has the same generator families as v2b, but a new seed and new scenarios. The human personas (SP21–SP40) are not shared with v2/v2b.
2. **Real agents:**
   ```
   python scripts/run_agent_matrix.py --config configs/agents.toml
   ```
   - Each of the 5 models ran 7 evasion strategies × 8 tasks × 2 repetitions, via OpenRouter, at temperature 0.7 with a reply limit of 1,000 tokens. Total cost: $0.22.
   - The agents only ever talk to the Python simulator. No command runs on a real machine.
   - Family roles (train or test) were fixed in `configs/agents.toml` before any session was recorded.
   - No Qwen agents, because Qwen is the detector.
3. **Merge:**
   ```
   python scripts/merge_sessions.py contrib/agent_*.jsonl --base data/v3_sim/dataset1_sessions.json --test-families gemma-3-27b claude-haiku-5.5 --exclude-seen data/v2b/dataset1_sessions.json data/v2/dataset1_sessions.json --out data/v3
   ```
   - 4 incomplete agent sessions were skipped (Llama produced unparseable JSON).
   - 3 test sessions identical to a v2/v2b session were dropped.
   - 16 within-split duplicates were dropped.
   - 29 test sessions identical to a training session were dropped. Most were very short agent sessions, for example a single `cat /etc/os-release`.

## Real-agent behaviour (all 560 recorded sessions)

| Family | Mean commands | Task solved | Median gap, no evasion | Median gap, `timing_mimicry` |
|---|---|---|---|---|
| llama3.1-8b | 8.7 | 53% | 0.95 s | 21.0 s |
| mistral-small-3.2 | 5.8 | 69% | 0.64 s | 6.1 s |
| deepseek-v3.2 | 5.8 | 75% | 0.85 s | 3.2 s |
| gemma-3-27b (test) | 5.4 | 47% | 0.86 s | 6.3 s |
| claude-haiku-5.5 (test) | 2.9 | 88% | 0.82 s | 5.2 s |

Simulated humans have a median gap of 5.5 s. When told to mimic human timing, real agents move into the human range.

**Timing caveat:** an agent's gap is its API latency plus any wait it asked for. API latency depends on the provider and
the network, not just the model.

## Rules for using v3
- **Choose settings on validation (from `train`) only.** Score the test set **once** per final configuration.
- **Report the test results in three parts:**
  - all sessions;
  - **real-agent held-out families only** (Gemma and Claude): this is the cleanest subset, since nobody designed anything against these families;
  - simulated sessions.
- **The simulated test families have been seen before.** They are the same generator families as v2b's test set (new sessions, but the families were seen during development). Say so whenever you report the simulated part.
- **Frozen v4** (`8f433ab`) and **Model 3** (option 1) are each run once on v3.

## Real humans
v3 contains no real human sessions. The team's own collection was dropped (TEST_PLAN §4C). The real-human checks
are the two public KYPO sets in `external/`, used test-only by the commands-only models (TEST_PLAN §4D).

## Known simulator limitation
`ls` prints a whole directory on one line, so `ls | grep x` returns the whole line. This is unchanged from v2b
and affects every model equally.
