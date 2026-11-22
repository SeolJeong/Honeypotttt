# HoneyAI: test plan (UAT), benchmarking protocol and Model 3 validation

Version 2026-11-04, last updated 2026-11-18 (§4C, §4D, test statuses, §5, §6). Applies to `honeyai_code.zip` + `honeyai_datasets.zip` of the same date.

Every test case below was run in the development environment before this document was written, unless the **Status** column says otherwise. "Expected" values are what that run produced. Re-running on Colab should give **identical** results for deterministic steps, and results within the stated **tolerance** for seeded ones.

---

## 1. Scope and environments

| ID | Environment | Used for |
|---|---|---|
| E1 | Google Colab, CPU runtime | Data, KYPO, calibration, all non-LLM baselines, model comparison |
| E2 | Google Colab, GPU runtime (T4 or better) | Model 3 (QLoRA LLM) training and scoring |
| E3 | Docker (`docker compose`), or any machine with Python 3.10+ | Real LLM-agent collection via OpenRouter, human collection |

**Setup (E1/E2):** upload both zips and unzip into one folder:
```bash
unzip -q -o honeyai_code.zip -d honeyai && unzip -q -o honeyai_datasets.zip -d honeyai && cd honeyai
pip install -q xgboost            # E1; E2 also: pip install -q -r requirements-llm.txt
```

**Out of scope here:** collecting real participant data (dropped, see `docs/HUMAN_COLLECTION.md`) and an API. The dashboard and Cowrie parser (Part 9) are covered in `docs/USER_MANUAL.md` §11.

**Paths:** this plan was written while v2b lived in `data/v2b/`. In the repository, v2b is in `data/` and the final set in `data/v3/`; the commands below use the repository paths.

---

## 2. Test cases

Columns: **Status** = result in the development environment on 2026-11-04. Fill in **Colab result** and **Tester** when you run it.

### A. Data integrity

| ID | Test | Steps | Expected | Status | Colab result / tester |
|---|---|---|---|---|---|
| TC-A1 | Files present | `ls data data/v3 external/kypo external/kypo2` | `dataset1_sessions.json`, `dataset2_tabular_features.csv`, `dataset4_llm_sequences.jsonl` in `data/` (v2b) and `data/v3/`; `kypo_sessions.json` + `kypo_sessions_ds4_commands_only.jsonl` | Pass | |
| TC-A2 | **Reproducibility** | `python scripts/generate_sessions.py --human-timing configs/human_timing.json --out data/v2b_check` then `cmp data/v2b_check/dataset1_sessions.json data/dataset1_sessions.json` | Prints "dropped 6 within-split duplicates, 2 test sessions identical to a training session"; `cmp` prints nothing (identical) | Pass | |
| TC-A3 | Counts | `python -c "import json,collections as C; s=json.load(open('data/dataset1_sessions.json')); print(len(s), C.Counter((x['split'],x['actor_label']) for x in s))"` | 2312; train 480 human / 446 scripted / 556 AI; test 320 / 190 / 320 | Pass | |
| TC-A4 | **No duplicates / train-test leakage** | Count duplicate `interleaved_text` in `dataset4` and test texts that also appear in train | 0 and 0 | Pass | |
| TC-A5 | **Group-disjoint splits** | Compare train vs test `participant_id` (humans) and `generator_family` | No participant overlap; test-only families: `alternative_commands`, `human_like_delays`, `retry_on_failure`, `sim_agent_c` | Pass | |
| TC-A6 | Merge validation | `python scripts/merge_sessions.py --base data/dataset1_sessions.json --out /tmp/m` | "merged 2312 sessions (0 invalid, 0 incomplete skipped)"; no leakage warnings | Pass | |

### B. Real-human (KYPO) pipeline

| ID | Test | Steps | Expected | Status | Colab result / tester |
|---|---|---|---|---|---|
| TC-B1 | Download | `curl -sSL -o external/kypo/data.zip "https://zenodo.org/records/8136017/files/data.zip?download=1"` | about 0.4 MB zip | Pass | |
| TC-B2 | Convert | `python scripts/convert_kypo.py` | "trainees 267 -> 1414 sessions (357 on target hosts), 14658 commands"; gap p10 0.9 / p50 11.9 / p90 129.9 s | Pass | |
| TC-B3 | Calibrate | `python scripts/calibrate_human_timing.py` | best `median_lo 2, median_hi 10, sigma_lo 0.8, sigma_hi 2.2`; loss 0.542 → **0.009**; matches `configs/human_timing.json` | Pass | |
| TC-B4 | Real-human check | `python scripts/eval_kypo.py --train data/dataset1_sessions.json` | KYPO labelled AI: text-cmds 0.080 (target hosts 0.157), timing 0.050 (0.151) | Pass | |

### C. Baselines (Models 1, 2 and reference models)

| ID | Test | Steps | Expected (full-session macro-F1) | Tolerance | Status |
|---|---|---|---|---|---|
| TC-C1 | All baselines | `python scripts/run_baselines.py --data data/dataset1_sessions.json --out runs/baselines_v2b.json` | timing 0.667 · text 0.765 · text-cmds 0.737 · XGBoost 0.694 ± 0.001 · GRU 0.713 ± 0.014 · GRU-cmds 0.721 ± 0.012 | Deterministic models exact; GRU/XGBoost ± 0.02 (CPU threads differ) | Pass |
| TC-C2 | Prediction file | after TC-C1: `runs/baselines_v2b_predictions.json` exists | Per-session probabilities for every model, seed and prefix | — | Pass |

### D. Model 3 (LLM)

| ID | Test | Steps | Expected | Status |
|---|---|---|---|---|
| TC-D1 | **Smoke test** (E2) | In `notebooks/qlora_generative_patched.ipynb` set `SMOKE = True`, `DATA_PATH = "data/dataset4_llm_sequences.jsonl"`, run all | Finishes in minutes; prints "dropped 0 duplicates"; writes `qlora_patched_results.json`; no "unknown" from parse failures | Pass on CPU with a tiny random model (2026-11-04); superseded by the full GPU runs (TC-D2) |
| TC-D2 | **Full run** (E2) | `SMOKE = False`, same data | 3 seeds; per seed: epoch log with validation macro-F1, early stop, test at k=1/3/5/full, latency p50/p95; mean ± std summary | **Pass** (RTX 3070 Ti, not T4): v2b and v3, 3 seeds each (`docs/TEST_RESULTS_SABRINA.md` §3.9–§3.11) |
| TC-D3 | Commands-only + real humans (E2) | `COMMANDS_ONLY = True`, `EXTERNAL_TEST_PATH = "external/kypo/kypo_sessions_ds4_commands_only.jsonl"` | Same outputs, plus "external real humans labelled AI: x ± y" | **Pass** (RTX 3070 Ti): Model 3 and v4 on v3, 3 seeds, KYPO and KYPO2 (`docs/TEST_RESULTS_SABRINA.md` §3.12–§3.13) |
| TC-D4 | Truncation safety | Long v2b sessions (more than 1024 tokens) | Prompt ≤ 1024 tokens, newest step kept, answer never cut | Pass |
| TC-D6 | **Zero-shot open LLM** (E2) | `ZERO_SHOT = True`, `MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"`, `EVAL_PREFIXES = [3, None]`. Rename the results file afterwards, e.g. `zs_qwen7b.json` | 1 seed (no training); prints "label bias on training transcripts"; predictions spread over all 3 classes | Pass on CPU with **real Qwen2.5-0.5B weights** on 24 sessions: macro-F1 0.32 (chance level, as expected for an untrained 0.5B). **GPU run on v3 with Qwen2.5-7B (2026-11-18): macro-F1 0.322** (`docs/TEST_RESULTS_SABRINA.md` §5.4) |
| TC-D7 | **Zero-shot API LLM** (E1/E3) | `python scripts/zero_shot_llm.py --model openai/gpt-4o-mini --name gpt4o-mini-zs --dry-run`, then without `--dry-run`. Optionally `--few-shot 3 --name gpt4o-mini-fs3` | Cost estimate; `runs/zero_shot/<name>.jsonl`; parse-failure rate printed; re-running resumes | Pass against a fake server (429 retry, prose and bare-label replies parsed, unparseable replies counted). **Not run with real API models:** every candidate family overlaps with v3's agents (§4A rule 3) |
| TC-D5 | Latency | read `summary.latency_ms` | p50 / p95 ms per command update on T4. **Target: team to set** (suggestion: p95 ≤ 1000 ms) | **Measured** (RTX 3070 Ti, after 3 commands): Model 3 p50 168 / p95 183 ms; v4 166 / 173 ms. No target was agreed in advance, so no pass/fail is claimed |

### E. Data collection tools (E3)

| ID | Test | Steps | Expected | Status |
|---|---|---|---|---|
| TC-E1 | Model list | `python scripts/run_agent_matrix.py --list-models llama` | Real OpenRouter IDs with prices | **Pass:** real OpenRouter runs recorded v3's agent sessions (`data/v3/DATA_CARD.md`) |
| TC-E2 | Dry run | `python scripts/run_agent_matrix.py --dry-run` | Unknown model IDs are refused; plan + rough cost printed | **Pass** (real run, `configs/agents.toml`) |
| TC-E3 | Pilot | `--only <family>` | Sessions in `contrib/agent_<family>.jsonl`; refusals saved as `incomplete` | **Pass:** `contrib/agent_*.jsonl`; 4 incomplete Llama sessions recorded and skipped at merge |
| TC-E4 | Resume | Stop with Ctrl-C, run the same command again | Finished sessions skipped; incomplete ones retried | Pass (fake server) |
| TC-E5 | Human recorder | `python scripts/collect_human.py --participant P006` | Consent prompt; task ends automatically when solved; `contrib/human_P006.jsonl` | Pass (scripted input) |

---

## 3. Benchmarking protocol (all models)

**Fix these before looking at any test result:**

1. **Data and split:** v2b (`data/`) during development; v3 (`data/v3/`) for the final report. Same file and same split for every model.
2. **Validation:** each training script carves validation from `train`. Settings (epochs, LR, LoRA rank, model size) are chosen on **validation only**. The test set is scored once per final configuration.
3. **Equal effort:** every model gets the same number of tuning trials. Defaults for the baselines; if the LLM gets N trials, so does at least the best baseline.
4. **Seeds:** 3 (42, 123, 456) for every seeded model. Report mean ± std, never the best seed.
5. **Metrics:**

| Metric | Meaning | Where |
|---|---|---|
| Macro-F1, full session | **Primary metric** | all |
| Macro-F1 after 1/3/5 commands | Early attribution (RQ3) | all |
| AI recall on evasive sessions, and per strategy | Robustness to human-mimicry (RQ2) | all |
| Human false attribution (human → AI) | Cost of false alarms | all |
| ECE, abstention rate | Calibration of the LLM's confidence | LLM |
| p50 / p95 latency per command | Interactive use (RQ3) | LLM |
| KYPO real humans labelled AI | Real-human external check | commands-only models |

6. **Ablations:**
   - commands-only vs commands + responses (**the central one**)
   - round 0 (non-evasive training) vs round 2
   - 0.5B vs 1.5B
   - (with v3) training on real only vs simulated only vs simulated + real
7. **Uncertainty:** 95% bootstrap CI. A model "beats" another only by the test in §4.

---

## 4. Model 3 validation: is the LLM really better?

Comparing two macro-F1 numbers by eye is not enough. With 830 test sessions, a 0.02 gap can be noise. Use a **paired test**: both models are scored on the *same* sessions, so session difficulty cancels out.

**Steps (E1, after the LLM run on E2):**
1. Run the baselines (TC-C1). This writes `runs/baselines_v2b_predictions.json`.
2. Copy the LLM's `qlora_patched_results.json` (from the patched notebook) into the folder.
3. Run:
   ```bash
   python scripts/compare_models.py --llm qlora_patched_results.json --llm-name "QLoRA-0.5B"
   python scripts/compare_models.py --llm qlora_patched_results.json --llm-name "QLoRA-0.5B" --prefix k=3   # early attribution
   ```
   For `train_qlora.py` runs, pass the per-seed files instead: `--llm runs/qlora/<run>/seed*/test_predictions.jsonl`.
4. For the commands-only ablation, compare the commands-only LLM against the **commands-only** baselines (`text-cmds`, `gru-cmds`, `timing`).

**What the script does:**
- Averages each model's predicted probabilities over its seeds (a seed ensemble) and also shows per-seed mean ± std.
- For each baseline: the macro-F1 difference (LLM − baseline) with a **95% paired bootstrap CI** (5,000 resamples of sessions), a one-sided p-value with **Holm correction** across baselines, and an exact **McNemar test** on per-session correctness.
- The same paired difference for AI-evasive recall and human false attribution.

**Decision rule (agreed in advance):**

| Result vs a baseline | Verdict |
|---|---|
| CI of the macro-F1 difference entirely above 0 **and** Holm p < 0.05 | **LLM better** |
| CI contains 0 | No significant difference |
| CI entirely below 0 | LLM worse |

**Model 3 can be called "best" only if it is LLM BETTER against every baseline, and at minimum against the strongest one** (currently the text model, 0.765). If it isn't, report what it *is* better at, if anything (for example early attribution, evasive recall, or calibration), and cite spec §19.

The script was self-tested by treating the GRU as the "LLM" (results in §5).

---

## 4B. Model 3 configuration selection (declared 2026-11-05, before the shift-validation runs)

**Why this section exists.** Two Model 3 configurations have already been scored on the test set:
option 1 (full sessions only, seed 42: full-session macro-F1 0.724) and option 2 (partial-session
training, 3 seeds: 0.653 ± 0.011). Their validation set was a random 15% of `train`, which is
in-distribution (val 0.87–0.97 vs test 0.65–0.72) and so says nothing about the test shift.
Keeping whichever configuration scored higher on test would be selection on the test set.
From here on the rule below decides, and both earlier runs are reported as they are.

**Selection rule.**

1. **Validation mirrors the test shift** (`VAL_MODE = "shift"` in `notebooks/qlora_generative_patched.ipynb`
   and `scripts/train_llm.py`). Taken from `train` only:
   - one AI evasion strategy, held out entirely (unseen-strategy shift);
   - one scripted generator family, held out entirely (unseen-family shift);
   - 15% of human participants, held out as whole participants.

   The strategy and family are picked by a fixed rule, not by hand: `random.Random(2026)` (the dataset's
   generation seed) chooses from the sorted list of non-`none` AI strategies, then from the sorted list of
   scripted families. On v2b this gives **`hesitation_redundant_exploration`** and **`randomised_ordering`**:
   304 validation sessions (80 human, 112 scripted, 112 AI) and 1,178 training sessions.
   `VAL_RULE_SEED` must not be changed after seeing any result.
2. **Selection metric:** macro-F1 on **full** validation sessions, the same as the primary test metric.
   It is used both for early stopping and for choosing between configurations.
3. **Candidates, in order** (each one is seed 42 first, and is compared on validation only):
   1. option 1, full sessions only (`PREFIX_TRAIN = False`);
   2. option 2, partial-session training (`PREFIX_TRAIN = True`, `TRAIN_PREFIXES = [1, 3, 5, None]`);
   3. optional: the winner of 1 vs 2 with `Qwen2.5-1.5B-Instruct` instead of 0.5B.

   The configuration with the highest validation macro-F1 is the final Model 3 and is then run on
   seeds 123 and 456. A tie (within 0.01) goes to the cheaper configuration.
4. **Fixed hyperparameters:** LR 1e-4, LoRA r 16, MAX_LEN 768, MAX_EPOCHS 4, PATIENCE 1. They are not
   changed in response to any test result.
5. **Test set:** scored once per final configuration with 3 seeds and compared with `compare_models.py` (§4).
   The headline is the per-seed mean ± std. The seed-ensemble figure is secondary.
6. **Reporting:** every configuration tried is reported, with its validation and test scores: the two
   earlier random-validation runs, the shift-validation runs, and 1.5B if it was run.
   Note for §3.3 (equal effort): Model 3 gets 2–3 configuration trials, while the baselines use defaults.
   Say so in the report as a limitation.

**Known limitation of the shift validation.** `train` has only two AI families (`sim_agent_a`,
`sim_agent_b`), so the unseen-AI-family shift (test: `sim_agent_c`) cannot be mirrored without removing
half of the AI training data. Only the unseen-strategy and unseen-scripted-family shifts are mirrored.
Holding these sessions out also reduces training data by about 20%.

**Outcome (recorded 2026-11-08).**
- **Seed 42 validation:** option 1 scored 0.9597 and option 2 scored 0.9643, a tie within 0.01. The tie went to the cheaper option, so **option 1 is Model 3**.
- **Its three seeds (42, 123, 456)** are reported in `docs/TEST_RESULTS_SABRINA.md` §3.9.

**Post-hoc experiment: v4** (`notebooks/qlora_generative_patched v4.ipynb`, commit `8f433ab`).
- **What it adds to option 2:**
  - gap augmentation of scripted TRAIN sessions;
  - jittering of the scripted validation sessions;
  - prefix loss weight 0.3;
  - dataset-level loss-weight normalisation;
  - a gradient-accumulation loss fix.
- **It was designed after seeing test errors** (0/64 recall on the test family `human_like_delays`). Validation would not have selected it: its validation score was lower, and on a validation set it had itself changed.
- **Its v2b test score is therefore biased upward** by an unknown amount. It is reported only as an experiment, and compared only against baselines given the same augmentation (`run_baselines.py --gap-aug 0.5`).
- **Its configuration is frozen as of that commit.** It may be evaluated cleanly once, alongside Model 3, on v3 test sessions that did not exist when it was designed. v2b test sessions reused in v3 do not count as clean.

**Tuning budget actually spent on Model 3** (§3.3):
- the original notebook;
- option 1 and option 2 with random validation;
- option 1 and option 2 with shift validation;
- v4.

The baselines used their defaults, plus the gap augmentation in the equal-effort check.

---

## 4A. Comparing with existing LLMs (off the shelf)

This answers "why fine-tune instead of just asking an existing LLM?". The off-the-shelf models get **no training**, the same sessions and the same class definitions (the prompt is fixed in the code; don't tune it on test results).

| Contender | How | Notes |
|---|---|---|
| Qwen2.5-7B-Instruct, zero-shot | Patched notebook, `ZERO_SHOT = True` (TC-D6) | Open and free; fits a T4 in 4-bit. Scores the 3 labels with real probabilities, with domain-context calibration (Fei et al., 2023) on training transcripts, labels unused |
| Llama-3.1-8B-Instruct / Mistral-7B-Instruct, zero-shot | Same, change `MODEL_ID` | **Gated on Hugging Face**: accept the licence and log in first (`huggingface-cli login`) |
| GPT-4o-mini, Claude, Gemini (zero- or few-shot) | `scripts/zero_shot_llm.py` via OpenRouter (TC-D7) | Costs money; run `--dry-run` first. `--n 300` for a cheaper stratified subset |

Then put them all in **one** comparison against your Model 3:
```bash
python scripts/compare_models.py --llm qlora_patched_results.json --llm-name "QLoRA-0.5B (ours)" \
    --others qwen7b-zs=zs_qwen7b.json gpt4o-mini-zs=runs/zero_shot/gpt4o-mini-zs.jsonl gpt4o-mini-fs3=runs/zero_shot/gpt4o-mini-fs3.jsonl
```
If a contender was only run on a subset (`--n`), the comparison automatically uses the sessions every model has, and says so.

**Rules for a fair result:**
1. The same test sessions and prompt definitions for every LLM. Few-shot examples come from **train** only (the script does this).
2. Report each contender's parse-failure rate (API models).
3. **Family overlap:** once real agent data is in (v3), an LLM judging sessions produced by its own family (e.g. GPT-4o-mini judging gpt4o-mini agents) may recognise its own style. Use judges from families that generated no data, or report the overlap.
4. The claim to aim for: *"a fine-tuned 0.5B model beats zero-shot 7B-class and API models while being smaller and faster"*. That's only valid if §4's decision rule says LLM BETTER against them.

## 4C. Final evaluation on v3 (declared 2026-11-08, before any model has seen v3)

**Data:** `data/v3/` (see `data/v3/DATA_CARD.md`). v3 is frozen as built on 2026-11-08: 2,818 sessions,
every test session new. No real human sessions are in v3; the real-human checks are separate, test-only sets (§4D).

**Models.** Each is run exactly as already configured. No v3 result may change any setting.

| Model | Configuration | Seeds |
|---|---|---|
| **Model 3** | option 1 (§4B): `qlora_generative_patched v3.ipynb`, `PREFIX_TRAIN=False`, `VAL_MODE="shift"`, everything else default | 42, 123, 456 |
| v4 (post hoc, frozen) | `qlora_generative_patched v4.ipynb` at commit `8f433ab`, unchanged | 42, 123, 456 |
| Baselines | `run_baselines.py` defaults | 42, 123, 456 for the seeded ones |
| Baselines with augmentation | `run_baselines.py --gap-aug 0.5` (the equal-effort comparison for v4) | same |

The only change allowed is the data path, plus a separate results folder (`RESULTS_DIR = "runs/llm_v3"`) so that
v2b runs are not mistaken for finished v3 runs. The validation rule (`VAL_RULE_SEED = 2026`) is applied to v3's
training set unchanged.

**Primary result:** Model 3 against the un-augmented baselines, full sessions, all v3 test sessions. Decision rule as in §4.

**Reported subsets.** These are fixed now; `compare_models.py --ai-families ...` keeps all human and scripted test sessions:
1. **Real held-out agents:** AI = `gemma-3-27b`, `claude-haiku-5.5`. **The cleanest test**, because nobody designed anything against these families.
2. **Simulated AI:** AI = `sim_agent_a`, `sim_agent_b`, `sim_agent_c`. These are new sessions, but the same generator families as v2b's test set.
3. **Early attribution:** after 1, 3 and 5 commands, all sessions.

**v4** is compared only against the augmented baselines, with its post-hoc status stated. If v4 beats Model 3
on v3, that is reported as a post-hoc finding confirmed on fresh data. Model 3's label does not change.

**Real humans (test-only).** The team's own collection (`docs/HUMAN_COLLECTION.md`) was **dropped on 2026-11-10**:
the only available participant was a team member who built the system. No session was recorded.
Real-human evidence comes from the two public KYPO sets in §4D.

---

## 4D. Remaining runs (declared 2026-11-10, before any commands-only LLM result was seen)

**No further configuration changes** to Model 3 or v4. The only runs left are these.

**1. Commands-only ablation + KYPO real-human check (GPU).** Both models, trained on v3, 3 seeds.

| Run | Notebook | What changes from the §4C run | Results file |
|---|---|---|---|
| Model 3, commands only | v3 notebook (`PREFIX_TRAIN=False`, `VAL_MODE="shift"`) | `COMMANDS_ONLY=True`, `EXTERNAL_TEST_PATH`, `RESULTS_NAME` | `v3_model3_cmds_kypo.json` |
| v4, commands only | v4 notebook at `8f433ab`, unchanged | the same three settings | `v3_v4_cmds_kypo.json` |

`EXTERNAL_TEST_PATH = "external/kypo/kypo_sessions_ds4_commands_only.jsonl"`, `DATA_PATH` and `RESULTS_DIR = "runs/llm_v3"` as in §4C.

**2. Reporting, fixed now.**
- **Ablation:** for each model, v3 test macro-F1 with responses (§4C run) against commands-only, at full / k=1 / k=3 / k=5.
- **Commands-only comparison:** against the commands-only baselines only (`text-cmds`, `gru-cmds`, `timing`).
  Model 3 is compared with the plain baselines; v4 with the gap-augmented ones.
- **KYPO:** per seed and mean ± std, the share of real-human sessions labelled human / scripted / AI, for
  all sessions, target-host sessions and non-target-host sessions, next to the same baselines.
- **v4 on real agents:** AI recall on `gemma-3-27b` and `claude-haiku-5.5` alone is reported as well as the subset
  macro-F1, because that subset still contains the scripted families v4 was designed against.

**3. Second real-human set: KYPO2** (`external/kypo2/`, see `docs/DATA_CARDS.md`).
- **Source:** Ošlejšek, Macák, Dočkalová Burská, *Hands-on cybersecurity training behavior data for process mining*, Data in Brief 52 (2024), doi:10.1016/j.dib.2023.109956; Zenodo record 10298187, CC BY 4.0.
- **Only `data1` is used:** 52 trainees, November 2022, 215 sessions after conversion.
  - `data2` is not used. It is the 2022-05-04 training that is already in KYPO: all 4,507 records match.
- **Independence:** KYPO2 was not used to calibrate anything, and no model or setting has been chosen with it.
- **Scoring:** the commands-only models are scored on it once, from the adapters saved by the runs above, with no retraining.
- **Disclosure:** the commands-only baselines were scored on KYPO2 on 2026-11-10, before any LLM result:

  | Baseline | KYPO2 labelled AI |
  |---|---|
  | timing | 2.8% |
  | text-cmds | 6.5% (7.0% with augmentation) |
  | gru-cmds | 50.4% (46.4% with augmentation) |

**5. Off-the-shelf LLM (added 2026-11-17, before the run).** §4A's zero-shot comparison is run once.
- **Model:** `Qwen/Qwen2.5-7B-Instruct`, with `ZERO_SHOT = True`, on v3, with `EVAL_PREFIXES = [3, None]`.
- **One run only:** there is no training, so there are no seeds.
- **Prompt:** the zero-shot prompt already in the notebook, unchanged.
- **Why Qwen:** it is the only judge with no family overlap with v3's agents.
- **Comparison:** against Model 3 and the baselines, with `compare_models.py --others`.
- **Nothing else is tried:** no other zero-shot model, and no other prompt.
- **Code change, memory only:** `prepare_model_for_kbit_training` is skipped in zero-shot mode. It only prepares a model for training, and with it a 7B model would not fit an 8 GB GPU. Trained runs are unaffected.
- **Results file:** `v3_zs_qwen7b.json`.
- **Outcome (2026-11-18):** run once as declared. Macro-F1 0.322 on full sessions; Model 3 is better by +0.475 (`docs/TEST_RESULTS_SABRINA.md` §5.4).

**6. Teammate's original QLoRA script on v3 (added 2026-11-18, before the run).**
- **Script:** `scripts/train_qlora_local.py`, run once on v3 with 3 seeds (42, 123, 456).
- **Its recipe is unchanged:** LoRA r=8 on q/v, 15 epochs, LR 5e-5, MAX_SEQ_LEN 512, 480-token prompts, JSON generation.
- **How it is run:** through `run_teammate_v3.py`, which changes only three things:
  - the data path (v3);
  - the output path, so `data/qlora_predictions.json` is not overwritten;
  - it keeps all 3 seeds' predictions instead of only the best seed.
- **Role:** a reference for Model 3, not a candidate.
- **Reporting:** mean ± std over all 3 seeds, with the parse-failure rate.
- **Results file:** `Test-Results-Sabrina/v3_teammate_qlora_local.json`.
- **Outcome (2026-11-18):** run once as declared. Macro-F1 0.545 ± 0.004, with 37% unparseable outputs; Model 3 is better by +0.258 [+0.224, +0.294] (`docs/TEST_RESULTS_SABRINA.md` §3.16).

**7. Part 8 and Part 9 on v3 (code change, 2026-11-18).** `honeyai/experiments/evasion_analysis.py` and `honeyai/dashboard.py` now default to `data/v3`. Their QLoRA input is the final Model 3: `data/v3/qlora_predictions.json`, the 3-seed ensemble built from `Test-Results-Sabrina/v3_model3_opt1.json`, with macro-F1 0.796. Set `HONEYAI_DATA_DIR=data` to reproduce the earlier v2b numbers. No model was retrained for this.

**4. Caveats to state with these results.**
- **Calibration overlap:** the simulated humans' timing was calibrated on KYPO target-host sessions
  (`scripts/calibrate_human_timing.py`). KYPO is therefore not independent of the training data on timing.
  KYPO2 is.
- **Scope:** both sets are commands-only and come from security-training exercises with different tasks.
  The full models, which read server responses, are untested on real people.

---

## 5. Current results (development environment, 2026-11-04)

**Baselines on v2b test (830 sessions):**

| Model | Full | k=1 | k=3 | k=5 | AI recall (evasive) | Human → AI | KYPO → AI (all / target hosts) |
|---|---|---|---|---|---|---|---|
| Timing only | 0.667 | 0.377 | 0.668 | 0.707 | 0.691 | 0.147 | 0.047 / 0.118 |
| Text (TF-IDF) | **0.765** | 0.652 | 0.710 | 0.746 | 0.587 | 0.156 | n/a |
| Text, commands only | 0.737 | 0.660 | 0.724 | 0.723 | 0.681 | 0.166 | 0.080 / 0.157 |
| XGBoost | 0.694 ± 0.001 | 0.523 | **0.747** | 0.741 | **0.738** | **0.099** | n/a |
| GRU | 0.713 ± 0.014 | 0.611 | 0.655 | 0.716 | 0.606 | 0.153 | n/a |
| GRU, commands only | 0.721 ± 0.012 | 0.614 | 0.675 | 0.694 | 0.668 | 0.160 | 0.091 / 0.129 |

**Model 3 (LLM): measured on GPU after this section was written** (updated 2026-11-18). v2b rows below; the final v3 results, comparisons and real-human checks are in `docs/TEST_RESULTS_SABRINA.md` (§2, §3.10–§3.13, §5).

| Model | Full | k=1 | k=3 | k=5 | AI recall (evasive) | Human → AI | KYPO → AI |
|---|---|---|---|---|---|---|---|
| QLoRA 0.5B, Model 3 (TC-D2, v2b, 3 seeds) | 0.719 ± 0.036 | 0.391 | 0.670 | 0.713 | 0.705 (ensemble) | 0.137 | n/a |
| QLoRA 0.5B, commands only (TC-D3) | run on v3 instead: 0.763 ± 0.036 | 0.413 | 0.543 | 0.747 | 0.870 (ensemble) | 0.108 | 0.228 ± 0.093 (KYPO), 0.296 ± 0.165 (KYPO2) |

**Comparison script self-test** (GRU treated as the "LLM", full sessions, 830 sessions, 5,000 resamples):

| vs | Baseline F1 | Diff | 95% CI | Verdict |
|---|---|---|---|---|
| text | 0.765 | −0.048 | [−0.077, −0.020] | LLM WORSE |
| text-cmds | 0.737 | −0.020 | [−0.043, +0.004] | no significant difference |
| gru-cmds | 0.728 | −0.011 | [−0.033, +0.010] | no significant difference |
| xgboost | 0.695 | +0.022 | [−0.010, +0.054] | no significant difference |
| timing | 0.667 | +0.050 | [+0.015, +0.084] | LLM BETTER |

This is why the paired test matters: by raw numbers the GRU (0.717) "beats" XGBoost (0.695), but the CI includes 0. Baseline F1s here are seed ensembles, so they can differ slightly from the per-seed means in the table above.

---

## 6. Acceptance criteria ("ready for the report")

- [x] TC-A1–A6 and TC-B1–B4 pass (development environment; not re-run on Colab)
- [x] TC-C1 baselines reproduced within tolerance (v3 rebuilt on a second machine, 2026-11-17)
- [x] TC-D2 full LLM run: 3 seeds, settings chosen on validation, results file saved
- [x] TC-D3 commands-only + KYPO run (plus KYPO2)
- [x] §4 comparison run for full sessions and k=3; verdicts recorded
- [ ] Latency measured and compared with the agreed target: **measured, but no target was agreed** (TC-D5)
- [ ] Every number in the report names its data file, split, seeds and settings: **to check in the final report**
- [x] (When real data exists) everything repeated on `data/v3`

## 7. Known limitations (state these in the report)

- Human and AI sessions in v2b are **simulated policies**; only the scripts are real. Only human *timing* is calibrated to real data.
  - v3 adds 287 real-LLM-agent test sessions (5 model families), but its humans are still simulated.
- No real human sessions were collected by the team. KYPO and KYPO2 are the real-human evidence, but they use different tasks and have no outputs, and KYPO was used to calibrate simulated human timing.
- Which evasions succeed depends on simulator assumptions (calibration changed `timing_mimicry` recall from 0.75 to 0.91).
- XGBoost's cross-validation score (0.965) is inflated; use the test score.

## 8. Sign-off

| Role | Name | Date | Result |
|---|---|---|---|
| Tester | | | |
| Model 3 owner | | | |
| Reviewer | | | |
