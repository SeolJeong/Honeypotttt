# HoneyAI: test plan (UAT), benchmarking protocol and Model 3 validation

Version 2026-11-04. Applies to `honeyai_code.zip` + `honeyai_datasets.zip` of the same date.

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

**Out of scope here:** collecting real participant data (needs ethics approval first), the dashboard and API (not built yet).

---

## 2. Test cases

Columns: **Status** = result in the development environment on 2026-11-04. Fill in **Colab result** and **Tester** when you run it.

### A. Data integrity

| ID | Test | Steps | Expected | Status | Colab result / tester |
|---|---|---|---|---|---|
| TC-A1 | Files present | `ls data/v2b data/v2 external/kypo` | `dataset1_sessions.json`, `dataset2_tabular_features.csv`, `dataset4_llm_sequences.jsonl` in v2 and v2b; `kypo_sessions.json` + `kypo_sessions_ds4_commands_only.jsonl` | Pass | |
| TC-A2 | **Reproducibility** | `python scripts/generate_sessions.py --human-timing configs/human_timing.json --out data/v2b_check` then `cmp data/v2b_check/dataset1_sessions.json data/v2b/dataset1_sessions.json` | Prints "dropped 6 within-split duplicates, 2 test sessions identical to a training session"; `cmp` prints nothing (identical) | Pass | |
| TC-A3 | Counts | `python -c "import json,collections as C; s=json.load(open('data/v2b/dataset1_sessions.json')); print(len(s), C.Counter((x['split'],x['actor_label']) for x in s))"` | 2312; train 480 human / 446 scripted / 556 AI; test 320 / 190 / 320 | Pass | |
| TC-A4 | **No duplicates / train-test leakage** | Count duplicate `interleaved_text` in `dataset4` and test texts that also appear in train | 0 and 0 | Pass | |
| TC-A5 | **Group-disjoint splits** | Compare train vs test `participant_id` (humans) and `generator_family` | No participant overlap; test-only families: `alternative_commands`, `human_like_delays`, `retry_on_failure`, `sim_agent_c` | Pass | |
| TC-A6 | Merge validation | `python scripts/merge_sessions.py --base data/v2b/dataset1_sessions.json --out /tmp/m` | "merged 2312 sessions (0 invalid, 0 incomplete skipped)"; no leakage warnings | Pass | |

### B. Real-human (KYPO) pipeline

| ID | Test | Steps | Expected | Status | Colab result / tester |
|---|---|---|---|---|---|
| TC-B1 | Download | `curl -sSL -o external/kypo/data.zip "https://zenodo.org/records/8136017/files/data.zip?download=1"` | about 0.4 MB zip | Pass | |
| TC-B2 | Convert | `python scripts/convert_kypo.py` | "trainees 267 -> 1414 sessions (357 on target hosts), 14658 commands"; gap p10 0.9 / p50 11.9 / p90 129.9 s | Pass | |
| TC-B3 | Calibrate | `python scripts/calibrate_human_timing.py` | best `median_lo 2, median_hi 10, sigma_lo 0.8, sigma_hi 2.2`; loss 0.542 → **0.009**; matches `configs/human_timing.json` | Pass | |
| TC-B4 | Real-human check | `python scripts/eval_kypo.py --train data/v2b/dataset1_sessions.json` | KYPO labelled AI: text-cmds 0.080 (target hosts 0.157), timing 0.050 (0.151) | Pass | |

### C. Baselines (Models 1, 2 and reference models)

| ID | Test | Steps | Expected (full-session macro-F1) | Tolerance | Status |
|---|---|---|---|---|---|
| TC-C1 | All baselines | `python scripts/run_baselines.py --data data/v2b/dataset1_sessions.json --out runs/baselines_v2b.json` | timing 0.667 · text 0.765 · text-cmds 0.737 · XGBoost 0.694 ± 0.001 · GRU 0.713 ± 0.014 · GRU-cmds 0.721 ± 0.012 | Deterministic models exact; GRU/XGBoost ± 0.02 (CPU threads differ) | Pass |
| TC-C2 | Prediction file | after TC-C1: `runs/baselines_v2b_predictions.json` exists | Per-session probabilities for every model, seed and prefix | — | Pass |

### D. Model 3 (LLM)

| ID | Test | Steps | Expected | Status |
|---|---|---|---|---|
| TC-D1 | **Smoke test** (E2) | In `notebooks/qlora_generative_patched.ipynb` set `SMOKE = True`, `DATA_PATH = "data/v2b/dataset4_llm_sequences.jsonl"`, run all | Finishes in minutes; prints "dropped 0 duplicates"; writes `qlora_patched_results.json`; no "unknown" from parse failures | Pass on CPU with a tiny random model (real weights not tested here); **run on Colab GPU** |
| TC-D2 | **Full run** (E2) | `SMOKE = False`, same data | 3 seeds; per seed: epoch log with validation macro-F1, early stop, test at k=1/3/5/full, latency p50/p95; mean ± std summary | **Not run** (needs GPU) |
| TC-D3 | Commands-only + real humans (E2) | `COMMANDS_ONLY = True`, `EXTERNAL_TEST_PATH = "external/kypo/kypo_sessions_ds4_commands_only.jsonl"` | Same outputs, plus "external real humans labelled AI: x ± y" | Pass on CPU smoke test; **run on Colab GPU** |
| TC-D4 | Truncation safety | Long v2b sessions (more than 1024 tokens) | Prompt ≤ 1024 tokens, newest step kept, answer never cut | Pass |
| TC-D6 | **Zero-shot open LLM** (E2) | `ZERO_SHOT = True`, `MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"`, `EVAL_PREFIXES = [3, None]`. Rename the results file afterwards, e.g. `zs_qwen7b.json` | 1 seed (no training); prints "label bias on training transcripts"; predictions spread over all 3 classes | Pass on CPU with **real Qwen2.5-0.5B weights** on 24 sessions: macro-F1 0.32 (chance level, as expected for an untrained 0.5B) |
| TC-D7 | **Zero-shot API LLM** (E1/E3) | `python scripts/zero_shot_llm.py --model openai/gpt-4o-mini --name gpt4o-mini-zs --dry-run`, then without `--dry-run`. Optionally `--few-shot 3 --name gpt4o-mini-fs3` | Cost estimate; `runs/zero_shot/<name>.jsonl`; parse-failure rate printed; re-running resumes | Pass against a fake server (429 retry, prose and bare-label replies parsed, unparseable replies counted) |
| TC-D5 | Latency | read `summary.latency_ms` | p50 / p95 ms per command update on T4. **Target: team to set** (suggestion: p95 ≤ 1000 ms) | **Not run** on GPU |

### E. Data collection tools (E3)

| ID | Test | Steps | Expected | Status |
|---|---|---|---|---|
| TC-E1 | Model list | `python scripts/run_agent_matrix.py --list-models llama` | Real OpenRouter IDs with prices | Tested against a fake server only |
| TC-E2 | Dry run | `python scripts/run_agent_matrix.py --dry-run` | Unknown model IDs are refused; plan + rough cost printed | Fake server only |
| TC-E3 | Pilot | `--only <family>` | Sessions in `contrib/agent_<family>.jsonl`; refusals saved as `incomplete` | Fake server only |
| TC-E4 | Resume | Stop with Ctrl-C, run the same command again | Finished sessions skipped; incomplete ones retried | Pass (fake server) |
| TC-E5 | Human recorder | `python scripts/collect_human.py --participant P006` | Consent prompt; task ends automatically when solved; `contrib/human_P006.jsonl` | Pass (scripted input) |

---

## 3. Benchmarking protocol (all models)

**Fix these before looking at any test result:**

1. **Data and split:** `data/v2b` now; `data/v3` for the final report. Same file and same split for every model.
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

**Model 3 (LLM): not yet measured.** It needs a GPU (TC-D2 on Colab). A reduced CPU run was started here but stopped: at about 100 s per training step it would have taken over 4 hours for a 240-session sanity check. Fill in this row from `qlora_patched_results.json`, then run §4.

| Model | Full | k=1 | k=3 | k=5 | AI recall (evasive) | Human → AI | KYPO → AI |
|---|---|---|---|---|---|---|---|
| QLoRA 0.5B (TC-D2) | | | | | | | |
| QLoRA 0.5B, commands only (TC-D3) | | | | | | | |

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

- [ ] TC-A1–A6 and TC-B1–B4 pass on Colab
- [ ] TC-C1 baselines reproduced within tolerance
- [ ] TC-D2 full LLM run: 3 seeds, settings chosen on validation, results file saved
- [ ] TC-D3 commands-only + KYPO run
- [ ] §4 comparison run for full sessions and k=3; verdicts recorded
- [ ] Latency measured and compared with the agreed target
- [ ] Every number in the report names its data file, split, seeds and settings
- [ ] (When real data exists) everything repeated on `data/v3`

## 7. Known limitations (state these in the report)

- Human and AI sessions in v2b are **simulated policies**; only the scripts are real. Only human *timing* is calibrated to real data.
- Only one real human participant (case study); KYPO is the real-human evidence, but it uses different tasks and has no outputs.
- Which evasions succeed depends on simulator assumptions (calibration changed `timing_mimicry` recall from 0.75 to 0.91).
- XGBoost's cross-validation score (0.965) is inflated; use the test score.

## 8. Sign-off

| Role | Name | Date | Result |
|---|---|---|---|
| Tester | | | |
| Model 3 owner | | | |
| Reviewer | | | |
