# HoneyAI user manual

**Version:** 2026-11-22. It matches `main` after pull requests #2 and #3, with the results doc named `docs/TEST_RESULTS_SABRINA.md` and the v3 notebook set to Model 3.
**Project:** ICT3214 Security Analytics, Coursework 2 (Group 18).

This manual explains how to install HoneyAI, run every part of it, reproduce the reported results and read the
output. It is written for three kinds of reader:

| Reader | Start at |
|---|---|
| **Marker / reviewer** who wants to see the results without a GPU | §4 Quick start, then §13 Reproducing the reported results |
| **Team member** who needs to retrain or re-evaluate a model | §3 Installation, §5 Data, then the section for that part |
| **Honeypot operator** who wants to classify live sessions | §11 Dashboard and Cowrie parser |

The results themselves are in `docs/TEST_RESULTS_SABRINA.md`. The test protocol and every pre-declared
decision are in `docs/TEST_PLAN.md`. This manual does not repeat their analysis, only the numbers you need to
check that a run worked.

---

## Contents

1. What HoneyAI does
2. What is in the repository
3. Requirements and installation
4. Quick start (no GPU needed)
5. Data
6. Stage 1: authentication-noise filter
7. Models 1 and 2 (XGBoost, GRU) and the other baselines
8. Model 3: the QLoRA LLM
9. Comparing models
10. Part 8: evasion experiments
11. Part 9: dashboard and Cowrie parser
12. Collecting new data
13. Reproducing the reported results
14. Troubleshooting
15. Limitations to keep in mind
16. Glossary

**Conventions.**
- **Commands are for Windows PowerShell,** run from the repository root (`C:\Side-Quest\Honeypotttt` on the development PC). On Linux or macOS, use `/` instead of `\` and `source .venv/bin/activate` to activate the environment.
- **File names with spaces must be quoted,** e.g. `"notebooks/qlora_generative_patched v3.ipynb"`.
- **"Expected"** means what the run produced on the development PC. Seeded GPU runs can differ slightly between machines (§14).

---

## 1. What HoneyAI does

HoneyAI looks at an SSH honeypot session (the commands an attacker typed, the server's responses and the time
between commands) and decides who produced it:

| Class | Meaning |
|---|---|
| **human** | a person typing |
| **scripted** | a pre-written program (bot, scanner, exploit kit) |
| **AI-driven** | an autonomous LLM agent, possibly deliberately imitating a human ("evasion") |

**Pipeline:**

```
Raw Cowrie log ──► Stage 1: authentication-noise filter (drops brute-force login spam)
                          │ successfully authenticated, interactive sessions only
                          ▼
                   Stage 2: three-class attribution
                     ├─ Model 1: XGBoost on 19 per-session features
                     ├─ Model 2: GRU on the step-by-step trace
                     └─ Model 3: QLoRA-fine-tuned Qwen2.5-0.5B-Instruct on the text transcript
                          │
                          ▼
                   Streamlit dashboard (comparison, evasion analysis, session inspector, wild data)
```

**Model 3 comes in several forms; make sure you know which one you are looking at:**

| Name | What it is | Notebook | Status |
|---|---|---|---|
| **Model 3** | Pre-specified final model: full sessions only, "shift" validation | `qlora_generative_patched v3.ipynb` | **The reported Model 3** (TEST_PLAN §4B) |
| v4 | Model 3 + timing-gap augmentation + partial-session training | `qlora_generative_patched v4.ipynb` | Post-hoc variant, frozen at commit `8f433ab` and confirmed on fresh data |
| Commands-only versions | The same two models retrained without server responses | same notebooks, `COMMANDS_ONLY = True` | Ablation and real-human check |
| Teammate recipe | The original `scripts/train_qlora_local.py` | script | Reference only |

**Headline numbers** (v3 test set, 1,113 sessions, macro-F1; per-seed mean ± std over 3 seeds unless stated):

| Model | Macro-F1 |
|---|---|
| **Model 3** | **0.769 ± 0.069** (3-seed ensemble 0.796) |
| v4 (post hoc) | 0.896 ± 0.006 (ensemble 0.899) |
| TF-IDF text baseline | 0.818 |
| GRU (Model 2) | 0.767 (ensemble) |
| XGBoost (Model 1) | 0.648 (ensemble) |
| Timing baseline | 0.622 |
| Teammate recipe | 0.545 ± 0.004 (37% unparseable outputs) |
| Qwen2.5-7B zero-shot | 0.322 |

---

## 2. What is in the repository

```
Honeypotttt/
├── data/
│   ├── dataset1..4 ...            v2b development data (2,312 sessions). Its test set was used during development.
│   ├── qlora_predictions.json     early single-seed Model 3 run (v2b, 0.724). Used only when HONEYAI_DATA_DIR=data
│   ├── wild_sessions.json         12 parsed sessions from the live Cowrie honeypot (written by the parser)
│   └── v3/                        FINAL data: 2,818 sessions, fresh test set, 287 real LLM-agent test sessions
│       ├── dataset1_sessions.json, dataset2_tabular_features.csv, dataset4_llm_sequences.jsonl
│       ├── qlora_predictions.json final Model 3 (3-seed ensemble) for Part 8 and the dashboard
│       └── DATA_CARD.md           how v3 was built and the rules for using it
├── external/                      real-world data (never trained on)
│   ├── cowrie/cowrie.json         raw log from the team's live honeypot (about 3 h, 5 Nov 2026 UTC)
│   ├── kypo/                      KYPO real humans (1,414 sessions)
│   └── kypo2/                     KYPO2 real humans (215 sessions) + the re-zipped source used to convert them
├── honeyai/                       Python package
│   ├── sim/                       session simulator (fake Linux host, actors, recording)
│   ├── llm/                       LLM text format + evaluation metrics
│   ├── models/                    xgb_baseline.py (Model 1), gru_sequence.py (Model 2), qlora_*.py
│   ├── experiments/evasion_analysis.py   Part 8
│   ├── features.py                19 tabular features
│   ├── stage1_filter.py           Stage 1 filter
│   ├── cowrie_parser.py           Part 9: Cowrie log → sessions
│   └── dashboard.py               Part 9: Streamlit dashboard
├── notebooks/                     Model 3 notebooks (v1, v2, v3 = Model 3, v4 = post hoc) + KYPO calibration
├── scripts/                       command-line entry points (§6–§12)
├── Test-Results-Sabrina/          every saved Model 3 / LLM result file (§13)
├── docs/                          TEST_PLAN, test results, data cards, Docker and data-collection guides, this manual
├── configs/                       agent list (configs/agents*.toml), calibrated human timing
├── contrib/                       raw recorded agent / human sessions
├── requirements-llm.txt           packages for Model 3
└── Dockerfile, docker-compose.yml data-collection container (CPU only)
```

**Not in git** (created when you run things): `.venv/`, `runs/` (model outputs, saved adapters), `.env` (API keys).
`external/` is git-ignored, but every dataset the results use is force-added: KYPO, KYPO2 (converted files and
sources) and the Cowrie log.

**Which data each part uses by default:**

| Part | Default data | Change it with |
|---|---|---|
| Model 3 / v4 notebooks | whatever `DATA_PATH` says in the settings cell (§8.2) | the settings cell |
| Part 8, dashboard | **v3** (`data/v3`) | `$env:HONEYAI_DATA_DIR = "data"` for v2b |
| `run_baselines.py`, `compare_models.py` | v2b (`data/`) | `--data data/v3/dataset1_sessions.json` |
| `train_xgboost.py`, `train_gru.py`, `extract_features.py`, Stage 1 | v2b (`data/`), hard-coded | not configurable; use `run_baselines.py` for v3 |
| `parse_cowrie.py --classify` | trains XGBoost on v2b | not configurable |

---

## 3. Requirements and installation

### 3.1 Hardware and software

| Part | Needs |
|---|---|
| Stage 1, Models 1–2, baselines, comparisons, Part 8, dashboard, Cowrie parser | Any PC with Python 3.10–3.12, about 4 GB RAM. CPU only. |
| Model 3 training or scoring | NVIDIA GPU with **8 GB VRAM or more**, CUDA. Tested on an RTX 3070 Ti (8 GB), Windows 11. Google Colab with a T4 also works. |
| Zero-shot Qwen2.5-7B | The same GPU, about **15 GB free disk** for the model download |
| Disk | About 2 GB for the repo and environment, plus about 1 GB per Model 3 run (adapters) |

### 3.2 Install (Windows, PowerShell)

```powershell
cd C:\Side-Quest\Honeypotttt
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy scikit-learn xgboost torch streamlit pandas jupyter
```

**If PowerShell refuses to run `Activate.ps1`,** run this once and activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**For Model 3 on a GPU,** install the CUDA build of PyTorch first (pick the command for your CUDA version at
pytorch.org), then:

```powershell
pip install -r requirements-llm.txt
python -c "import torch; print(torch.cuda.is_available())"     # must print True
```

**Check the installation:**

```powershell
python -c "import numpy, sklearn, xgboost, torch, streamlit; print('all good')"
```

### 3.3 Google Colab (alternative for Model 3)

1. Upload the repository, or clone it.
2. Open the notebook.
3. Select **Runtime → Change runtime type → T4 GPU**.

The first notebook cell installs `peft` and `bitsandbytes`. Paths are relative to the repository root: the
notebook moves there automatically when opened from `notebooks/`.

### 3.4 Every new terminal

```powershell
cd C:\Side-Quest\Honeypotttt
.\.venv\Scripts\Activate.ps1
```

The prompt then starts with `(.venv)`.

---

## 4. Quick start (no GPU needed)

These four steps show the final results using saved files. They take about 5 minutes on a normal PC.

```powershell
# 1. Part 8 evasion experiments on v3 with the final Model 3 (about 1.5 min)
python scripts/run_evasion_experiments.py

# 2. Dashboard (opens http://localhost:8501; first load about 1 min)
streamlit run honeyai/dashboard.py

# 3. Parse and classify the live honeypot log (seconds)
python scripts/parse_cowrie.py external/cowrie/cowrie.json --classify

# 4. Stage 1 filter (seconds)
python -m honeyai.stage1_filter
```

**Expected:**
1. `QLoRA (loaded): test macro-F1 = 0.796`, XGBoost about 0.645 and GRU about 0.774.
2. The dashboard caption reads `Test data: data/v3/...`, and the QLoRA card shows 0.796.
3. `wrote 12 sessions to data\wild_sessions.json`, and 12 labels.
4. Stage 1 macro-F1 1.000.

For the statistical comparisons (§9) you first need the baseline predictions (`run_baselines.py`, about 30 minutes
on CPU).

---

## 5. Data

### 5.1 Session datasets

| Set | Folder | Sessions (train / test) | Use |
|---|---|---|---|
| v2b | `data/` | 1,482 / 830 | Development. Its test set was used to make decisions, so its scores are biased upward. |
| **v3** | `data/v3/` | 1,705 / 1,113 | **Final evaluation.** Every test session is new. 287 of the 605 AI test sessions come from real LLM agents (Llama 3.1-8B, Mistral Small 3.2, DeepSeek V3.2 for training families; Gemma 3-27B and Claude Haiku 5.5 held out). |

The splits are group-disjoint: test humans, scripted families, AI families and two evasion strategies never appear in
training. The details are in `data/v3/DATA_CARD.md`.

**Files in each set:**

| File | Content | Used by |
|---|---|---|
| `dataset1_sessions.json` | Full traces: per step the command, the simulated response, the response category, the gap since the previous command, typo flag, task progress | GRU, baselines, Part 8, dashboard |
| `dataset2_tabular_features.csv` | 19 numeric features per session (timing, command, error, adaptation, sequence) | XGBoost |
| `dataset4_llm_sequences.jsonl` | The text Model 3 reads, e.g. `[GAP SHORT] CMD: ls /tmp` then `RESP [success]: ...` | Model 3 notebooks |
| `dataset3_auth_noise.csv` (v2b only) | 1,000 login-attempt records | Stage 1 |

**Labels and metadata fields:**
- `actor_label`: human / scripted / AI-driven
- `evasion_strategy`: none, timing_mimicry, typo_mimicry, shorter_commands, hesitation_redundant_exploration, avoid_systematic_enumeration, combined
- `generator_family`: which simulator, script family or LLM produced the session
- `split`: train / test

### 5.2 KYPO (real humans, test only)

KYPO contains real trainees' commands from cybersecurity training (Švábenský et al., *Data in Brief* 38, 2021,
CC BY 4.0). It has **no server responses**, so only commands-only models can be scored on it. The converted files
and the original download (`external/kypo/data.zip`) are in the repo. To rebuild the converted files:

```powershell
python scripts/convert_kypo.py              # uses external/kypo/data.zip
# (if data.zip is missing: Invoke-WebRequest -Uri "https://zenodo.org/records/8136017/files/data.zip?download=1" -OutFile external\kypo\data.zip)
```

**Expected:**
- `trainees 267 -> 1414 sessions (357 on target hosts), 14658 commands`
- gaps p10 0.9 / p50 11.9 / p90 129.9 s

The output is `external/kypo/kypo_sessions.json` (used by the baselines) and
`kypo_sessions_ds4_commands_only.jsonl` (used by the LLM).

**Caveat:** KYPO target-host sessions were used to calibrate the simulated humans' timing, so KYPO is not fully
independent on timing. KYPO2 is.

### 5.3 KYPO2 (second, fully independent real-human set)

KYPO2 is from Ošlejšek et al., *Data in Brief* 52 (2024), Zenodo record 10298187, CC BY 4.0. Only `data1` is used
(52 trainees); `data2` duplicates KYPO.

The converted files used for every reported KYPO2 result are in the repo:
- `external/kypo2/kypo2_sessions.json` (baselines);
- `external/kypo2/kypo2_sessions_ds4_commands_only.jsonl` (LLM);
- `external/kypo2/data1_commands.zip`, the re-zipped source they were converted from.

To reproduce the committed files exactly, convert the committed source:

```powershell
python scripts/convert_kypo.py --zip external/kypo2/data1_commands.zip --out external/kypo2/kypo2_sessions.json
```

To rebuild from the original Zenodo download instead:

```powershell
mkdir external\kypo2
Invoke-WebRequest -Uri "https://zenodo.org/records/10298187/files/data1.zip?download=1" -OutFile external\kypo2\data1.zip
python -c "import zipfile; s=zipfile.ZipFile('external/kypo2/data1.zip'); o=zipfile.ZipFile('external/kypo2/data1_commands.zip','w',zipfile.ZIP_DEFLATED); [o.writestr('Locust 3302/2022-11 data1/'+n.rsplit('/',1)[1], s.read(n)) for n in s.namelist() if n.startswith('data1/command_histories/') and n.endswith('.json')]; o.close()"
python scripts/convert_kypo.py --zip external/kypo2/data1_commands.zip --out external/kypo2/kypo2_sessions.json
```

**Expected:**
- `trainees 52 -> 215 sessions (12 on target hosts), 2601 commands`
- gaps p10 1.0 / p50 14.2 / p90 130.4 s

**Note on session IDs:** they contain the folder name used inside the re-zipped file. The committed files and the LLM
result files use `KYPO-Locust_3302_insider-2022-11-28-...`. A rebuild with the one-liner above gives the same 215
sessions with a different ID prefix; match them on the trailing `sbNNN-sN-wN` part, or convert the committed
`data1_commands.zip` to get identical IDs.

### 5.4 Cowrie wild data

`external/cowrie/cowrie.json` is the raw log of the team's live honeypot: 13,315 events, 1,708 raw sessions and 122
source IPs, recorded 5 Nov 2026 00:02–03:14 UTC. It is
**unlabelled**: it is used for case studies and the demo, never for training or accuracy (§11.3).

---

## 6. Stage 1: authentication-noise filter

Separates brute-force login noise from real interactive sessions, using logistic regression on 3 features: whether
login succeeded, the number of attempts, and the mean time between attempts.

```powershell
python -m honeyai.stage1_filter                                          # train + evaluate on dataset3
python -m honeyai.stage1_filter --predict data/dataset3_auth_noise.csv   # classify a CSV with the same columns
```

**Expected:** macro-F1 1.000. The classes are easily separable by design, so this stage is a filter, not a
research result.

---

## 7. Models 1 and 2 (XGBoost, GRU) and the other baselines

### 7.1 Individual training scripts (v2b)

```powershell
python scripts/extract_features.py   # regenerate data/dataset2 from data/dataset1
python scripts/train_xgboost.py      # Model 1: grouped CV + test, feature importance
python scripts/train_gru.py          # Model 2: grouped CV + multi-seed test
```

**Expected (v2b):** XGBoost CV about 0.964, test about 0.69; GRU CV about 0.919, test about 0.71. These scripts
read `data/` (v2b) and cannot be pointed at v3.

### 7.2 All baselines on one dataset: `run_baselines.py`

This is the script behind every reported baseline number. It trains these models with their default settings, on
the same data and split, with the same metrics as Model 3:

| Name | Model |
|---|---|
| `timing` | depth-3 decision tree on mean / min / std gap (the data-driven version of the Palisade timing rule) |
| `text` | TF-IDF + logistic regression on exactly the transcript Model 3 reads |
| `text-cmds` | the same, without responses |
| `xgboost` | Model 1 |
| `gru` | Model 2 |
| `gru-cmds` | Model 2 without responses |

```powershell
# v3, as reported (about 20–40 min on CPU, mostly the GRUs)
python scripts/run_baselines.py --data data/v3/dataset1_sessions.json --kypo external/kypo/kypo_sessions.json --out runs/baselines_v3.json

# v3 with the same timing-gap augmentation as v4 (the fair comparison for v4)
python scripts/run_baselines.py --data data/v3/dataset1_sessions.json --kypo external/kypo/kypo_sessions.json --gap-aug 0.5 --out runs/baselines_v3_aug.json
```

**Outputs:**
- `runs/baselines_v3.json` (summary);
- `runs/baselines_v3_predictions.json` (per-session probabilities, needed by `compare_models.py`).

**Expected (v3, plain):** timing 0.622, text 0.818, text-cmds 0.777, XGBoost about 0.649, GRU about 0.763, GRU-cmds
about 0.759. Augmented: text-cmds 0.858, text 0.854, timing 0.747.
- The deterministic models (timing, text) reproduce exactly.
- XGBoost and the GRUs reproduce within ±0.02.

### 7.3 Text baseline on its own

```powershell
python scripts/text_baseline.py                  # full input
python scripts/text_baseline.py --no-responses   # commands only
```

---

## 8. Model 3: the QLoRA LLM

### 8.1 How Model 3 works

- **Base model:** `Qwen/Qwen2.5-0.5B-Instruct`, loaded in 4-bit (NF4).
- **Fine-tuning:** LoRA rank 16 on all linear layers, learning rate 1e-4, at most 4 epochs, with early stopping on a validation set taken from `train`.
- **Shift validation:** the validation set holds out one AI evasion strategy, one scripted family and whole human participants, so it mirrors the test shift.
- **Input:** each session is a chat transcript of gap tokens, commands and responses, up to 768 tokens. Longer sessions lose their **oldest** commands, so the answer is never cut off.
- **Prediction:** the model scores all three possible answers and turns the scores into probabilities. Prediction therefore never fails to parse, and the confidence is meaningful.
- **Partial sessions:** each test session is also scored after 1, 3 and 5 commands, for early attribution.
- **Seeds:** each configuration is trained with seeds 42, 123 and 456. Report the mean ± std, never the best seed.

### 8.2 Running a notebook

1. **Open the notebook** in Jupyter (`jupyter lab`) or VS Code, and select the `.venv` kernel.
2. **Paste the settings for the run you want** from §8.3 into the **settings cell** (cell 2), replacing its lines.

   > The settings cell keeps whatever the last person ran. The v3 notebook is committed with preset A (Model 3);
   > the v4 notebook with the commands-only KYPO2 run (preset D). Always check it before running.

3. **Kernel → Restart Kernel and Run All Cells.**
4. **Check the first output lines** (§8.4).

**Where the output goes:**
- **Per-seed results:** `RESULTS_DIR/<run tag>_seed<N>.json` (`runs/llm_v3/` for v3 runs).
- **Saved adapters:** `RESULTS_DIR/<run tag>_adapter_seed<N>/`.
- **The combined results file:** `RESULTS_NAME`, in the repository root.

**Resuming:** if a seed file already exists, that seed is **loaded instead of retrained** (`seed 42: already done,
loading ...`). A run interrupted part-way through a seed restarts that seed from scratch. To force a full retrain,
move the matching `runs/llm_v3/<run tag>_seed*.json` files away.

**The two cells at the bottom** of each notebook (a check line, and "Score the SAVED adapters on the real-human
sets") are only for §8.6. On a normal Run All, the last of them stops with
`AssertionError: set COMMANDS_ONLY = True ...`. That is harmless: everything above it has already run and been
saved.

### 8.3 Settings for each documented run

Every run uses `DATA_PATH = "data/v3/dataset4_llm_sequences.jsonl"`, `RESULTS_DIR = "runs/llm_v3"` and
`SEEDS = [42, 123, 456]`. Leave every setting not listed at its committed value.

**A. Model 3, the reported model** (`qlora_generative_patched v3.ipynb`):

```python
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PREFIX_TRAIN = False
VAL_MODE = "shift"
EVAL_PREFIXES = [1, 3, 5, None]
SCORE_BATCH = 8
ZERO_SHOT = False
COMMANDS_ONLY = False
EXTERNAL_TEST_PATH = None
RESULTS_NAME = "v3_model3_opt1.json"
```

- **Time:** about 25–40 min per epoch on an RTX 3070 Ti, 2–3 epochs per seed.
- **Expected:** full macro-F1 0.769 ± 0.069 (seeds: 0.810 / 0.826 / 0.672).

**B. Model 3, commands only, + KYPO** (same notebook): as A, but with:

```python
COMMANDS_ONLY = True
EXTERNAL_TEST_PATH = "external/kypo/kypo_sessions_ds4_commands_only.jsonl"
RESULTS_NAME = "v3_model3_cmds_kypo.json"
```

- **Time:** about 3–5 min per epoch.
- **Expected:** full macro-F1 0.763 ± 0.036; `external real humans labelled AI: 0.228 +/- 0.093`.

**C. v4** (`qlora_generative_patched v4.ipynb`, post hoc). Keep the committed v4 settings (`GAP_AUG = True`,
`PREFIX_TRAIN = True`, `PREFIX_W = 0.3`, `VAL_GAP_JITTER = True`, ...) and set:

```python
COMMANDS_ONLY = False
EXTERNAL_TEST_PATH = None
RESULTS_NAME = "v3_v4_frozen.json"
```

- **Time:** about 60–100 min per epoch.
- **Expected:** full macro-F1 0.896 ± 0.006.

**D. v4, commands only, + KYPO** (v4 notebook): as C, with `COMMANDS_ONLY = True`, `EXTERNAL_TEST_PATH` as in B,
and `RESULTS_NAME = "v3_v4_cmds_kypo.json"`.
- **Expected:** full 0.878 ± 0.006; real humans labelled AI 0.093 ± 0.012.

**E. Zero-shot Qwen2.5-7B** (v3 notebook; no training):

```python
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ZERO_SHOT = True
COMMANDS_ONLY = False
EXTERNAL_TEST_PATH = None
EVAL_PREFIXES = [3, None]
SCORE_BATCH = 2            # 1 if the GPU runs out of memory
RESULTS_NAME = "v3_zs_qwen7b.json"
```

- **What it needs:** about 15 GB of download, and the one-line memory fix in `build_model()`, which is already committed: `if USE_4BIT and not ZERO_SHOT:`.
- **Expected:** full macro-F1 0.322.

**F. Smoke test** (any notebook): `SMOKE = True` trains 1 seed for 1 epoch on a tiny subset, to check the pipeline
in minutes. **Its scores mean nothing.** Set it back to `False` afterwards.

### 8.4 Reading the output

| Line | What to check |
|---|---|
| `loaded 2818 \| dropped 0 duplicates, 0 test sessions identical to train` | 2818 = v3. If it says 2312, `DATA_PATH` points at v2b. |
| `shift validation: AI strategy '...' + scripted family '...' + N human participants held out` | fixed by `VAL_RULE_SEED = 2026`; must be the same on every run |
| `epoch N: val macro-F1 ...` / `early stop` | the epoch with the best validation score is kept |
| `test k=1/k=3/k=5/full: macro-F1 ...` | test scores after 1, 3, 5 commands and on the full session |
| `full-session confusion (rows=true, cols=pred)` | rows and columns are human, scripted, AI-driven |
| `AI recall by strategy` | the share of AI sessions caught, per evasion strategy |
| `latency per command update: p50 ... p95 ...` | time to re-score a session after a new command (about 170 ms on the 3070 Ti) |
| `external real humans labelled AI: x` | KYPO false-alarm rate (commands-only runs only) |
| `<k>: test macro-F1 mean +/- std` | the 3-seed summary to report |

### 8.5 Command-line alternative

`scripts/train_llm.py` is a plain-Python version of the notebook. For example, Model 3:

```powershell
python scripts/train_llm.py --data data/v3/dataset4_llm_sequences.jsonl --no-prefix --val-mode shift --out v3_model3_opt1.json
```

Run `python scripts/train_llm.py --help` for every option. It was generated from an earlier notebook version and
**was not used for the reported results**. The notebooks are the reference.

### 8.6 Scoring saved models on real humans (no retraining)

To score the commands-only adapters on KYPO and KYPO2 without retraining:
1. Use settings **B** (or **D** for v4), with a **new** `RESULTS_NAME`, e.g. `v3_model3_cmds_kypo2.json`.
2. **Run All.** The seeds load from `runs/llm_v3`.
3. **Run the cell "Score the SAVED adapters on the real-human sets"** at the bottom.
4. **Re-run the save cell** (`out = {"model": ...`).

**Checks in that cell's output:**
- `{'kypo': 1414, 'kypo2': 215}`;
- for each seed, `max difference from saved test probabilities` below about 0.035 (4-bit rounding);
- the KYPO rates must equal the original run (Model 3: 0.103 / 0.257 / 0.325; v4: 0.093 / 0.078 / 0.108).

The cell saves per-session predictions under `external_sets`, so trainee-level confidence intervals can be
computed.

### 8.7 The teammate's original recipe (reference)

`scripts/train_qlora_local.py` reads v2b and **overwrites `data/qlora_predictions.json`**. To run it on v3 without
changing it, use the wrapper, which changes only the paths and keeps all 3 seeds:

```powershell
copy Test-Results-Sabrina\run_teammate_v3.py .
python -u run_teammate_v3.py | Tee-Object -FilePath teammate_v3_log.txt
```

- **Time:** about 40 min of training per seed, plus prediction.
- **Expected:** `train: 1705, test: 1113`, then per seed macro-F1 of about 0.54–0.55, with about 400/1113 parse failures.
- **Output:** `Test-Results-Sabrina/v3_teammate_qlora_local.json`.

---

## 9. Comparing models

Comparing two macro-F1 numbers by eye is not enough. `scripts/compare_models.py` scores models on the **same test
sessions** and reports, for the LLM against each baseline:
- the macro-F1 difference, with a 95% paired bootstrap confidence interval;
- a Holm-corrected one-sided p-value;
- an exact McNemar test.

The LLM's predictions are averaged over its seeds (a seed ensemble).

**Decision rule:** "LLM BETTER" only if the whole 95% CI is above 0 **and** the Holm p < 0.05. A CI containing 0 means
"no significant difference".

```powershell
# Model 3 against the baselines, full sessions (needs runs/baselines_v3_predictions.json, §7.2)
python scripts/compare_models.py --data data/v3/dataset1_sessions.json --baselines runs/baselines_v3_predictions.json --llm Test-Results-Sabrina/v3_model3_opt1.json --llm-name "Model 3"

# the same after 3 commands
python scripts/compare_models.py --data data/v3/dataset1_sessions.json --baselines runs/baselines_v3_predictions.json --llm Test-Results-Sabrina/v3_model3_opt1.json --llm-name "Model 3" --prefix k=3

# only the held-out real agents (all human and scripted sessions kept)
python scripts/compare_models.py --data data/v3/dataset1_sessions.json --baselines runs/baselines_v3_predictions.json --llm Test-Results-Sabrina/v3_model3_opt1.json --llm-name "Model 3" --ai-families gemma-3-27b claude-haiku-5.5

# v4 against the equally augmented baselines
python scripts/compare_models.py --data data/v3/dataset1_sessions.json --baselines runs/baselines_v3_aug_predictions.json --llm Test-Results-Sabrina/v3_v4_frozen.json --llm-name "v4"

# add other LLM result files as extra contenders
python scripts/compare_models.py --data data/v3/dataset1_sessions.json --baselines runs/baselines_v3_predictions.json --llm Test-Results-Sabrina/v3_model3_opt1.json --llm-name "Model 3" --others qwen7b-zs=Test-Results-Sabrina/v3_zs_qwen7b.json
```

**Expected (first command):**
- Model 3 0.796 (ensemble);
- text −0.022, n.s.;
- text-cmds +0.019, n.s.;
- GRU +0.029, n.s.;
- GRU-cmds +0.035, better;
- XGBoost +0.148, better;
- timing +0.174, better.

The output is written to `runs/model3_comparison.json` (change it with `--out`).

**Limits:**
- **The teammate recipe's file** has labels but no probabilities, so `compare_models.py` cannot read it. Its comparison was done separately (test results §3.16).
- **The real-human (KYPO) comparisons** were also done separately (§3.13).

---

## 10. Part 8: evasion experiments

```powershell
python scripts/run_evasion_experiments.py | Tee-Object -FilePath evasion_results_v3.txt
```

**What it does:**
- reads v3 by default;
- trains XGBoost and the GRU inside the script, with one seed each;
- loads the final Model 3 predictions from `data/v3/qlora_predictions.json`.

**It runs three experiments:**
1. per-evasion-strategy results;
2. non-evasive against evasive AI recall;
3. a commands-only ablation for XGBoost and the GRU.

It runs in about 1.5 minutes on CPU.

**Expected (v3):**
- The first lines say `combined: 156 sessions`.
- Macro-F1: XGBoost about 0.645, GRU about 0.774, QLoRA 0.796.
- QLoRA AI recall: `combined` 0.526, `shorter_commands` 0.633, all evasive sessions 0.780.

**Reproducing the earlier v2b numbers** (the version in the first draft of the report):

```powershell
$env:HONEYAI_DATA_DIR = "data"
python scripts/run_evasion_experiments.py
Remove-Item Env:HONEYAI_DATA_DIR        # back to v3
```

The v2b run gives QLoRA 0.725 (the early single-seed run).

**How to read the output:**
- **Use the "AI-driven recall per evasion strategy" tables. Ignore "per-evasion macro-F1".** Every strategy except `none` contains only AI sessions, so a 3-class macro-F1 there is capped at about 0.33 by construction.
- **Read recall together with the false-alarm rate on humans.** A model that labels more sessions AI gets higher recall: XGBoost has the highest evasive recall, but the lowest macro-F1.
- **The script compares one-seed baselines with a 3-seed Model 3 ensemble,** so its ranking is not a test. Use §9 for claims.
- **Real agents need a separate analysis.** On the real LLM agents only, `combined` is caught 83–97% of the time, compared with 53–60% overall. The script cannot select that subset (test results §3.14).

---

## 11. Part 9: dashboard and Cowrie parser

### 11.1 Starting the dashboard

```powershell
streamlit run honeyai/dashboard.py        # or: python scripts/run_dashboard.py
```

- **It opens** at `http://localhost:8501`. If the browser doesn't open, type the address yourself.
- **The first load takes about 1 minute,** because it trains XGBoost and the GRU; later loads are cached.
- **To stop it,** press `Ctrl + C` in the terminal.
- **Check the caption under the title:** `Test data: data/v3/dataset1_sessions.json · QLoRA predictions: data/v3/qlora_predictions.json`. With `HONEYAI_DATA_DIR=data` it shows v2b instead.

### 11.2 The four tabs

| Tab | What it shows | How to use it |
|---|---|---|
| **Model Comparison** | Macro-F1 cards for XGBoost, GRU and QLoRA (about 0.644 / 0.774 / 0.796); per-class precision, recall and F1 tables; a confusion matrix per model; XGBoost feature importance | Overview. The per-class tables show where each model fails, e.g. XGBoost's scripted recall of about 0.20. |
| **Evasion Analysis** | AI recall per evasion strategy for each model (table and bar chart); the non-evasive against evasive recall drop | Which evasion strategy hurts which model |
| **Session Inspector** | Filters: actor label, evasion strategy, prediction (all / correct / misclassified). Pick a session to see each model's prediction (correct or wrong), then the full command trace: per step the gap, response category, typo flag and the response (expand "Response") | Case studies. E.g. filter to `combined` + `misclassified` to see the agents that fooled the models. |
| **Wild Data** | Upload a parsed sessions file (e.g. `data/wild_sessions.json`). It shows the number loaded and, for each session (first 20), the step count, mean gap and first 5 commands | **Inspection only: this tab does not classify.** Classification is done by the parser (§11.3). |

**Caveats to put in any figure caption:**
- **The dashboard's XGBoost and GRU are trained with one seed, while QLoRA is the 3-seed ensemble.** The cards are a demonstration; the paired comparison (§9) is the result.
- **The QLoRA card shows Model 3, not v4.**

**Screenshots:**
- **Part of the screen:** `Win + Shift + S`.
- **A whole long page** in Chrome or Edge: press `F12`, then `Ctrl + Shift + P`, type "full size screenshot", Enter.

### 11.3 Parsing and classifying a Cowrie log

```powershell
python scripts/parse_cowrie.py external/cowrie/cowrie.json              # parse only
python scripts/parse_cowrie.py external/cowrie/cowrie.json --classify   # parse + classify with XGBoost
```

**What the parser does:**
1. Groups events into sessions.
2. Keeps only successfully authenticated sessions with at least 2 commands.
3. Splits long sessions and converts them to the dataset1 format.
4. Writes `data/wild_sessions.json`.

**This overwrites a tracked file.** To undo: `git checkout -- data/wild_sessions.json`.

**Expected for the team log:**
- `13315 events loaded`, `1708 raw sessions`, `12 interactive sessions extracted`;
- 35 skipped (no successful login), 1663 skipped (fewer than 2 commands);
- labels: 9 scripted, 2 human, 1 AI-driven.

**Only XGBoost classifies,** trained on v2b. Cowrie logs contain no command output, so Model 3, which reads
responses, cannot classify them. Only a commands-only model could.

**How to read the result:** there is no ground truth. Manual inspection shows all 12 sessions are bots:
- 9 runs of an SSH-key implant bot;
- 3 honeypot-probing scripts.

The 3 "human" or "AI-driven" labels are bots that paused once, for 2–147 s, to wait for a download or a payload.
The clearest case is the same SSH-key bot: labelled scripted when it runs in under 1 s, and human when it pauses
8.4 s. Treat wild-data labels as a case study of the timing weakness, not as accuracy (test results §3.15).

---

## 12. Collecting new data

**Real LLM agents (needs an OpenRouter API key; costs money).** Follow `docs/DOCKER.md`:

```powershell
copy .env.example .env                      # then put the key in .env (never commit .env)
copy configs\agents.example.toml configs\agents.toml
docker compose build
docker compose run --rm honeyai python scripts/run_agent_matrix.py --dry-run
docker compose run --rm honeyai python scripts/run_agent_matrix.py
```

**Rules:**
- The agents only ever talk to the Python simulator; nothing runs on a real machine.
- Fix which families are held out before recording anything.
- Use no Qwen agents, because Qwen is the detector.

**Real humans.** Follow `docs/HUMAN_COLLECTION.md`:

```powershell
python scripts/collect_human.py --participant P101 --contributor SA
```

- **Consent:** check with your lecturer whether a consent form or ethics approval is needed first. The script asks for consent and records nothing without it.
- **IDs:** use anonymous IDs. Team members use the `P19x` range.

**Merging into a new dataset** (example: adjust the base file, the held-out families and the output folder):

```powershell
python scripts/merge_sessions.py contrib/agent_*.jsonl --base data/v3_sim/dataset1_sessions.json --test-families gemma-3-27b claude-haiku-5.5 --out data/v4_new
```

A new dataset needs its own data card and its own pre-declared evaluation plan. v3's test set must not be reused to
choose settings.

---

## 13. Reproducing the reported results

**Without a GPU,** every reported number can be recomputed from the saved files:

| Result (test results section) | Saved file | How to recompute |
|---|---|---|
| Model 3 on v3 (§3.10) | `Test-Results-Sabrina/v3_model3_opt1.json` | §7.2 then §9, first command |
| v4 on v3 (§3.11) | `Test-Results-Sabrina/v3_v4_frozen.json` | §7.2 (augmented) then §9, v4 command |
| Commands-only ablation (§3.12) | `v3_model3_cmds_kypo.json`, `v3_v4_cmds_kypo.json` | `compare_models.py` with `--others` (§9) |
| Real humans, KYPO / KYPO2 (§3.13) | `v3_model3_cmds_kypo2.json`, `v3_v4_cmds_kypo2.json` | the `external_sets` entries of those files |
| Part 8 on v3 (§3.14) | `evasion_results_v3.txt` | §10 |
| Wild Cowrie case study (§3.15) | `data/wild_sessions.json` | §11.3 |
| Teammate recipe on v3 (§3.16) | `v3_teammate_qlora_local.json`, `teammate_v3_log.txt` | — |
| Zero-shot Qwen-7B (§5.4) | `v3_zs_qwen7b.json` | §9, last command |
| Outside solutions, incl. the Palisade timing rule (§5) | the files above + `data/v3/dataset1_sessions.json` | test results §5.2 describes the rule (median gap < 1.7 s = AI) |
| Development runs on v2b (§3.1–§3.9) | `results_opt1_*`, `results_opt2_*`, `results_shift_*` | `compare_models.py` on `data/` (v2b) |

**With a GPU,** retrain with the settings in §8.3. Expect the same results within the seed spread. 4-bit and bf16
arithmetic is not bit-exact across GPUs and library versions.

---

## 14. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `NameError: name 'EXTERNAL_TEST_PATH' is not defined` (or any setting) | The kernel restarted and the settings cell did not run | Kernel → Restart and Run All. Check that the settings cell got a run number. |
| `AssertionError: set COMMANDS_ONLY = True ...` at the very end of Run All | The real-human scoring cell at the bottom ran with normal settings | Harmless: everything above it finished. Delete or skip that cell if you don't need it. |
| `seed 42: already done, loading ...` when you wanted to retrain | Finished seed files exist for this run tag | Move `runs/llm_v3/<tag>_seed*.json` away (§8.2) |
| `loaded 2312 ...` instead of 2818 | `DATA_PATH` points at v2b | Set `DATA_PATH = "data/v3/dataset4_llm_sequences.jsonl"` |
| `CUDA out of memory` | GPU too small for the batch | Lower `SCORE_BATCH` (2, then 1); for 1.5B+ models set `GRAD_CKPT = True`; for 7B zero-shot keep `ZERO_SHOT = True` |
| `torch.cuda.is_available()` is `False` | CPU-only PyTorch installed | Reinstall the CUDA build of PyTorch (pytorch.org) |
| Output stops while the progress bar finishes | Output through a pipe (`\| Tee-Object`) is buffered | Wait, or run with `python -u` next time |
| Dashboard or Part 8 shows 0.725 / v2b numbers | The `HONEYAI_DATA_DIR` override is set, or the code is older than `6d1859d` | `Remove-Item Env:HONEYAI_DATA_DIR`; `git pull` |
| `git status` lists `__pycache__/*.pyc` as modified | Compiled Python files were committed by mistake | `git restore honeyai/__pycache__ honeyai/experiments/__pycache__`; don't commit them |
| `data/wild_sessions.json` modified | The parser rewrote it | `git checkout -- data/wild_sessions.json` if unwanted |
| Results differ slightly between machines | GPU / CPU numerics, XGBoost threads | Expected: GRU and XGBoost within ±0.02; LLM within the seed spread |
| `Activate.ps1 cannot be loaded` | PowerShell execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Windows restarted during a long run | — | Re-run: finished seeds are kept. To avoid it: pause Windows Update, and `powercfg /change standby-timeout-ac 0` |

---

## 15. Limitations to keep in mind

- **Human sessions are simulated.** v2b and v3 humans come from a simulator with KYPO-calibrated timing. Real-human evidence is limited to KYPO and KYPO2: commands only, different tasks, and KYPO was used for calibration.
- **The full models are untested on real people.** They read server responses, and no real-human set has them.
- **Model 3 flags real people as AI more often than simple methods do.** On KYPO and KYPO2, the commands-only Model 3 flags 23–34% of sessions (seed-dependent), against 3–7% for timing and TF-IDF. v4 flags about 9–11%.
- **v4 is post hoc.** It was designed after seeing v2b test errors and is reported as a confirmed post-hoc finding.
- **The wild data is small and unlabelled:** 12 interactive sessions from about 3 hours of one honeypot.

The full list is in `docs/TEST_PLAN.md` §7 and test results §7.

---

## 16. Glossary

| Term | Meaning |
|---|---|
| **Macro-F1** | Average of the F1 scores of the three classes. The primary metric. |
| **AI recall (evasive)** | Share of AI sessions that use an evasion strategy and are labelled AI |
| **Human false attribution / false alarm** | Share of human sessions labelled AI (lower is better) |
| **k=1 / k=3 / k=5** | The score after only the first 1, 3 or 5 commands of each session (early attribution) |
| **Seed ensemble** | Predicted probabilities averaged over the 3 training seeds |
| **Shift validation** | A validation set that holds out an evasion strategy, a scripted family and whole participants, mimicking the test set |
| **Commands-only** | Input with all server responses removed. Needed for KYPO, KYPO2 and Cowrie. |
| **Gap augmentation** | v4's training trick: copies of scripted sessions with randomised timing, so the model can't treat "fast = script" as a shortcut |
| **Post hoc** | Designed after seeing test results, so reported with that disclosure |
| **Paired bootstrap / McNemar** | Statistical tests that compare two models on the same sessions |
| **Palisade rule** | Published LLM-agent detector: an agent replies in under 1.7 s |
| **QLoRA / LoRA** | Fine-tuning a 4-bit-quantised model by training small low-rank adapter matrices |
| **Run tag** | The name of a notebook run's files in `runs/`. Built from the settings, e.g. `ft_cmds_noprefix_shiftval_Qwen2.5-0.5B-Instruct`. |
