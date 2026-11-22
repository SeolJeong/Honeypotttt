# HoneyAI

Evasion-aware attribution of SSH honeypot sessions: was the actor a **human**, a **script**, or an **AI agent**, even
when the AI is deliberately imitating a human?

ICT3214 Security Analytics, Coursework 2 (Group 18).

| Document | What it contains |
|---|---|
| **[`docs/USER_MANUAL.md`](docs/USER_MANUAL.md)** | How to install, run and reproduce everything (start here) |
| [`docs/TEST_RESULTS_SABRINA.md`](docs/TEST_RESULTS_SABRINA.md) | Every Model 3 run, the paired comparisons, the comparison with outside solutions, and the real-human checks |
| [`docs/TEST_PLAN.md`](docs/TEST_PLAN.md) | Test cases, benchmarking protocol, and the decisions declared before each result was seen |
| [`data/v3/DATA_CARD.md`](data/v3/DATA_CARD.md), [`docs/DATA_CARDS.md`](docs/DATA_CARDS.md), [`docs/DATA_PLAN.md`](docs/DATA_PLAN.md) | How the datasets were built, and their rules of use |
| [`docs/DOCKER.md`](docs/DOCKER.md), [`docs/HUMAN_COLLECTION.md`](docs/HUMAN_COLLECTION.md) | Collecting real LLM-agent and human sessions |

---

## What it does

```
Raw Cowrie log ──► Stage 1: authentication-noise filter (logistic regression; drops brute-force login spam)
                          │ authenticated, interactive sessions only
                          ▼
                   Stage 2: three-class attribution (human / scripted / AI-driven)
                     ├─ Model 1: XGBoost on 19 per-session features
                     ├─ Model 2: GRU on the step-by-step command/response trace
                     └─ Model 3: QLoRA-fine-tuned Qwen2.5-0.5B-Instruct on the session transcript
                          │
                          ▼
                   Streamlit dashboard: model comparison, evasion analysis, session inspector, wild data
```

**Model 3** reads each session as text: the gap before each command, the command and the server's response. It is
fine-tuned with QLoRA (4-bit base model, LoRA rank 16), and it scores the three labels instead of generating text,
so its confidence is a real probability. It can also re-score a session after every new command, for early
attribution.

---

## Results (final evaluation: v3 test set, 1,113 fresh sessions)

Macro-F1 on full sessions, as the per-seed mean ± std over 3 seeds; the 3-seed ensemble is in brackets. The
significance tests are paired bootstraps with Holm correction (`scripts/compare_models.py`).

| Model | Macro-F1 | Against Model 3 |
|---|---|---|
| **Model 3** (pre-specified) | **0.769 ± 0.069** (0.796) | — |
| v4 (post hoc: Model 3 + timing-gap augmentation + partial-session training) | 0.896 ± 0.006 (0.899) | v4 better: +0.103 |
| TF-IDF + logistic regression (standard text classifier) | 0.818 | no significant difference |
| GRU (Model 2) | 0.767 | no significant difference |
| XGBoost (Model 1) | 0.648 | Model 3 better: +0.148 |
| Timing classifier (fitted Palisade-style signal) | 0.622 | Model 3 better: +0.174 |
| Original team QLoRA script (`scripts/train_qlora_local.py`) | 0.545 ± 0.004 (37% unparseable outputs) | Model 3 better: +0.258 |
| Qwen2.5-7B-Instruct, zero-shot (no training) | 0.322 | Model 3 better: +0.475 |

**Against outside solutions:**
- **Published timing rule (Palisade, "an agent replies in under 1.7 s"):** at the same false-alarm rate on simulated humans, Model 3 catches 90% of the held-out real LLM agents (Gemma, Claude); the rule catches 64%.
- **Agents told to fake human timing:** the rule catches none of them; Model 3 catches 85%.
- **Asking an existing, 14× larger LLM:** it performs at chance level.

**Where Model 3 is not better:**
- **Text classifier:** it is statistically tied with a standard TF-IDF classifier.
- **Real people:** on two real-human datasets (KYPO, KYPO2; commands only), it flags 23–34% of real people's sessions as AI, against 3–7% for the timing and TF-IDF methods. v4 flags about 9–11%.

**The post-hoc v4** removes a timing shortcut found during development. On fresh data it beats every baseline, each
given the same augmentation. It is reported as a post-hoc finding; the pre-specified Model 3 keeps its label.

**Evasion:**
- **The hardest strategy is `combined`** (every mimicry trick at once): 0.53–0.60 recall overall, for every model.
- **Real LLM agents using it are still caught 83–97% of the time.** The collapse is a property of the simulated combined agent.
- **Server responses add no measurable accuracy on full sessions** for any model: the commands-only ablation. Model 3 does use them early in a session (after 3 commands).

**Live honeypot:** 12 interactive sessions from the team's Cowrie honeypot. All are bots on inspection. XGBoost
mislabels the ones that paused, e.g. the same SSH-key implant bot is labelled "scripted" when it runs in under 1 s
and "human" when it pauses for 8.4 s.

The full numbers, confidence intervals and caveats are in [`docs/TEST_RESULTS_SABRINA.md`](docs/TEST_RESULTS_SABRINA.md).

---

## Quick start

Needs Python 3.10–3.12. Steps 1–4 run on CPU; Model 3 training needs an NVIDIA GPU with 8 GB or more.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1                 # Linux/macOS: source .venv/bin/activate
pip install numpy scikit-learn xgboost torch streamlit pandas jupyter

python scripts/run_evasion_experiments.py                              # 1. Part 8 on v3 (about 1.5 min)
streamlit run honeyai/dashboard.py                                     # 2. dashboard, http://localhost:8501
python scripts/parse_cowrie.py external/cowrie/cowrie.json --classify  # 3. live honeypot sessions
python -m honeyai.stage1_filter                                        # 4. Stage 1 filter
```

**Expected:**
1. QLoRA 0.796, GRU about 0.774, XGBoost about 0.645.
2. The caption reads `Test data: data/v3/...`.
3. 12 sessions are labelled.
4. Macro-F1 1.000.

**For Model 3 on a GPU:** `pip install -r requirements-llm.txt`, open `notebooks/qlora_generative_patched v3.ipynb`
(committed with the Model 3 settings), and Run All. Settings for every other run are in USER_MANUAL §8.3.

**Baselines and the statistical comparison on v3:**

```powershell
python scripts/run_baselines.py --data data/v3/dataset1_sessions.json --kypo external/kypo/kypo_sessions.json --out runs/baselines_v3.json
python scripts/compare_models.py --data data/v3/dataset1_sessions.json --baselines runs/baselines_v3_predictions.json --llm Test-Results-Sabrina/v3_model3_opt1.json --llm-name "Model 3"
```

---

## Data

| Set | Location | Sessions (train / test) | Use |
|---|---|---|---|
| **v3** | `data/v3/` | 1,705 / 1,113 | **Final evaluation.** Every test session is new. 287 of the 605 AI test sessions come from real LLM agents. |
| v2b | `data/` | 1,482 / 830 | Development. Its test set was used to make decisions, so its scores are biased upward. |
| KYPO | `external/kypo/` | test only, 1,414 | Real people (security trainees), commands only. Also used to calibrate simulated human timing. |
| KYPO2 | `external/kypo2/` | test only, 215 | Real people, commands only, fully independent |
| Cowrie | `external/cowrie/cowrie.json` | unlabelled | Live honeypot traffic: case studies and demo only |

**v3 splits are group-disjoint** (nothing in the test set was seen in training):

| | Train | Test (held out) |
|---|---|---|
| Humans (simulated, KYPO-calibrated timing) | personas SP21–SP32 | SP33–SP40 |
| Scripts | 4 families | 3 different families (`human_like_delays`, `alternative_commands`, `retry_on_failure`) |
| Simulated AI agents | `sim_agent_a`, `sim_agent_b`, 5 evasion strategies | `sim_agent_c`; plus `avoid_systematic_enumeration` and `combined` |
| Real LLM agents (OpenRouter) | Llama 3.1-8B, Mistral Small 3.2, DeepSeek V3.2 | **Gemma 3-27B, Claude Haiku 5.5** (unseen families); plus the held-out strategies of the training families |

**Files in each set:**

| File | Content | Used by |
|---|---|---|
| `dataset1_sessions.json` | full traces: command, simulated response, response category, gap, typo flag, task progress | GRU, baselines, Part 8, dashboard |
| `dataset2_tabular_features.csv` | 19 features: timing (6), command (3), error (4), adaptation (3), sequence (3) | XGBoost |
| `dataset4_llm_sequences.jsonl` | the text Model 3 reads | Model 3 |
| `dataset3_auth_noise.csv` (v2b) | 1,000 login records | Stage 1 |

---

## Repository structure

```
Honeypotttt/
├── data/                        v2b (development) + v3/ (final) datasets; qlora_predictions.json files for Part 8 / dashboard
├── external/                    KYPO, KYPO2 (real humans) and the Cowrie log
├── honeyai/
│   ├── sim/                     session simulator (fake Linux host, human / script / AI actors)
│   ├── llm/                     LLM text format and evaluation metrics
│   ├── models/                  xgb_baseline.py (Model 1), gru_sequence.py (Model 2), qlora_*.py
│   ├── experiments/             evasion_analysis.py (Part 8)
│   ├── features.py              19 tabular features
│   ├── stage1_filter.py         Stage 1 filter
│   ├── cowrie_parser.py         Cowrie log → sessions (Part 9)
│   └── dashboard.py             Streamlit dashboard (Part 9)
├── notebooks/                   Model 3: qlora_generative_patched v3.ipynb (reported model), v4.ipynb (post hoc), earlier versions
├── scripts/                     entry points: baselines, comparisons, data generation and collection, KYPO, Part 8/9
├── Test-Results-Sabrina/        every saved LLM result file used in the reported numbers
├── docs/                        manual, test plan, test results, data cards, collection guides
├── configs/                     real-agent list, calibrated human timing
├── contrib/                     raw recorded agent and human sessions
├── requirements-llm.txt         Model 3 packages
└── Dockerfile, docker-compose.yml   data-collection container (CPU)
```

**Main scripts** (all run from the repository root; details in the manual):

| Script | Purpose |
|---|---|
| `scripts/run_baselines.py` | Timing, TF-IDF, XGBoost and GRU baselines (+ commands-only variants) on one dataset; optional `--gap-aug` |
| `scripts/compare_models.py` | Paired bootstrap, Holm and McNemar comparison of an LLM against the baselines |
| `scripts/run_evasion_experiments.py` | Part 8 evasion experiments |
| `honeyai/dashboard.py` (via `streamlit run`) | Part 9 dashboard |
| `scripts/parse_cowrie.py` | Parse a Cowrie log; `--classify` labels sessions with XGBoost |
| `scripts/convert_kypo.py` | Convert KYPO / KYPO2 to sessions |
| `scripts/train_xgboost.py`, `scripts/train_gru.py`, `scripts/extract_features.py` | Models 1 and 2 on v2b |
| `scripts/generate_sessions.py`, `scripts/run_agent_matrix.py`, `scripts/collect_human.py`, `scripts/merge_sessions.py` | Building datasets |
| `scripts/train_llm.py` | Command-line version of the Model 3 notebook (the notebooks produced the reported results) |
| `scripts/train_qlora_local.py` | The original team QLoRA script (reference; see results §3.16) |

---

## Limitations

- **Simulated humans:** the human sessions in v2b and v3 are simulated (only their timing is calibrated to real people). Real-human evidence is commands-only (KYPO, KYPO2), on different tasks.
- **Untested on real people:** the full models, which read server responses, are untested on real people.
- **False alarms on real people:** Model 3 raises more of them than simple timing or text methods, and its rate depends strongly on the training seed.
- **Simulated agents:** which evasion strategies succeed depends on the simulator's assumptions. Real LLM agents behaved differently from the simulated combined agent.
- **v4 is post hoc:** it was designed after seeing development test errors. It is confirmed on fresh data, but is not the pre-specified model.
- **Wild data:** small (12 sessions, about 5 hours) and unlabelled.

---

## Data sources

- **KYPO:** V. Švábenský et al., "Dataset of shell commands used by participants of hands-on cybersecurity training", *Data in Brief* 38, 2021. doi:10.1016/j.dib.2021.107398. Zenodo 8136017, CC BY 4.0.
- **KYPO2:** R. Ošlejšek, M. Macák, K. Dočkalová Burská, "Hands-on cybersecurity training behavior data for process mining", *Data in Brief* 52, 2024. doi:10.1016/j.dib.2023.109956. Zenodo 10298187, CC BY 4.0.
- **Live honeypot:** Cowrie on the team's VPS. Unlabelled; used only for case studies and the demo, never for training or reported accuracy.
- **Real LLM-agent sessions:** recorded through OpenRouter against the Python simulator only. No command was run on a real machine.
