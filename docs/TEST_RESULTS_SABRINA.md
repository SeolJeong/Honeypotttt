# Model 3 (QLoRA LLM): test results (Sabrina)

These are the results of every Model 3 run: development on v2b, then the final evaluation on v3 and on real
humans. The protocol, metrics and decision rules are in [`TEST_PLAN.md`](TEST_PLAN.md), §3, §4, §4B, §4C and §4D.

**Where to start:**
- the final results are in §2 (summary) and §3.10–§3.13;
- **§5 compares Model 3 with outside solutions** (published detectors and standard methods), not with Models 1 and 2.

**Environments:**
- LLM runs: local Windows PC, RTX 3070 Ti (8 GB), PowerShell venv and Jupyter. TEST_PLAN names a Colab T4; latency figures are from the 3070 Ti.
- Baselines: CPU. The v3 baselines were rebuilt on a second machine (Linux, CPU) on 2026-11-17 and match the recorded numbers (§3.12).

**Code:**
- `notebooks/qlora_generative_patched.ipynb` (v1), then `… v2.ipynb`, `… v3.ipynb` (Model 3);
- `… v4.ipynb` (post-hoc v4, frozen at commit `8f433ab`).

**Data:**
- **v2b (development):** `data/dataset4_llm_sequences.jsonl`, 1,482 train and 830 test sessions; 0 duplicates and 0 test sessions identical to a training session.
  - The v2b test set holds out an unseen AI family (`sim_agent_c`), unseen scripted families, unseen human participants, and two unseen evasion strategies (`avoid_systematic_enumeration` and `combined`).
  - It was used during development, so v2b test scores are biased upward.
- **v3 (final, fresh):** `data/v3/`, 1,705 train and 1,113 test sessions (320 human, 188 scripted, 605 AI, of which 287 are real LLM agents). See `data/v3/DATA_CARD.md`.
- **Real humans (test only, commands only):**
  - KYPO: 1,414 sessions, 357 on target hosts (`external/kypo/`);
  - KYPO2: 215 sessions, 12 on target hosts (`external/kypo2/`).

  Both are described in `docs/DATA_CARDS.md`.

---

## 1. Run log

| # | Date | Run | Settings | Seeds | Validation | Status |
|---|---|---|---|---|---|---|
| 0 | 2026-11-04 | Smoke test | `SMOKE=True` (tiny subset, 1 epoch) | 42 | random | Pipeline check only. Test F1 0.167 means a trivial classifier, as expected. **Not a result.** |
| 1 | 2026-11-05 | **Option 1:** full sessions only | `PREFIX_TRAIN=False`, MAX_LEN 768, LR 1e-4, LoRA r16, ≤3 epochs, patience 1 | 42 | random 15% of train | Done |
| 2 | 2026-11-05/06 | **Option 2:** partial-session training | `PREFIX_TRAIN=True`, `TRAIN_PREFIXES=[1,3,5,None]`, VAL_MAX 300, ≤4 epochs, patience 1 | 42, 123, 456 | random 15% of train (scored on partial sessions too) | Done |
| 3 | 2026-11-06 | Teammate's `scripts/train_qlora_local.py` (generative, JSON output) | MAX_SEQ_LEN 512 (train), 480 (test prompt), right truncation, LoRA r8, LR 5e-5, 15 fixed epochs | 42, 123, 456 | none | Done (reference) |
| 4 | 2026-11-06/07 | **Option 2, shift validation** (notebook v3) | as run 2, plus `VAL_MODE="shift"` (TEST_PLAN §4B): validation scored on full sessions, 1,178 training sessions | 42, 123, 456 | shift | Done |
| 5 | 2026-11-07 | **Option 1, shift validation** (Run A) | `PREFIX_TRAIN=False`, `VAL_MODE="shift"` | 42 | shift | Done. Validation 0.9597 against option 2's 0.9643: within 0.01, a **tie, so option 1 is selected (cheaper) under §4B** |
| 6 | 2026-11-07 | **v4 notebook**: option 2 + shift validation + gap augmentation (`GAP_AUG`, frac 0.5, p 0.7) + validation jitter + prefix weight 0.3 + loss-weighting and gradient-accumulation fixes | see §3.8 | 42, 123, 456 | shift, jittered | Done. **Designed after seeing test errors** (§3.8) |
| 7 | 2026-11-08 | **Option 1, shift validation: seeds 123 and 456** (completes Model 3) | identical to run 5 | 123, 456 | shift | Done. **Model 3 = runs 5 + 7** (§3.9) |
| 8 | 2026-11-09 | **Model 3 on v3** (TEST_PLAN §4C) | as runs 5 + 7; `DATA_PATH=data/v3/...`, `RESULTS_DIR=runs/llm_v3` | 42, 123, 456 | shift | Done (§3.10) |
| 9 | 2026-11-10 | **Frozen v4 on v3** (TEST_PLAN §4C) | v4 notebook unchanged; `DATA_PATH=data/v3/...`, `RESULTS_DIR=runs/llm_v3` | 42, 123, 456 | shift, jittered | Done (§3.11) |
| 10 | 2026-11-11 | **v4, commands only, + KYPO** (TEST_PLAN §4D) | as run 9, plus `COMMANDS_ONLY=True`, `EXTERNAL_TEST_PATH` (KYPO) | 42, 123, 456 | shift, jittered | Done (§3.12, §3.13). `Test-Results-Sabrina/v3_v4_cmds_kypo.json` |
| 11a | by 2026-11-12 | **Model 3, commands only** (TEST_PLAN §4D), first run | as run 8, plus `COMMANDS_ONLY=True` | 42, 123, 456 | shift | **Superseded**: KYPO was not loaded, so no real-human result. Test: full 0.760 ± 0.034, k=3 0.553 ± 0.038, k=5 0.745 ± 0.021 |
| 11b | 2026-11-13 | **Model 3, commands only, + KYPO**: same configuration, retrained | as 11a, plus `EXTERNAL_TEST_PATH` (KYPO) | 42, 123, 456 | shift | Done (§3.12, §3.13). `Test-Results-Sabrina/v3_model3_cmds_kypo.json`. **Reported**, because it is the only run with the KYPO result; the two runs agree (full 0.760 vs 0.763) |
| 12 | 2026-11-17 | **Model 3, commands only: KYPO2 scoring** (TEST_PLAN §4D item 3) | saved adapters of run 11b, **no retraining**. A reload check against the saved test probabilities gave max differences of 0.0093 / 0.0074 / 0.0114, and the KYPO shares were reproduced exactly | 42, 123, 456 | — | Done (§3.13). `Test-Results-Sabrina/v3_model3_cmds_kypo2.json` |
| 13 | 2026-11-17 | **v4, commands only: KYPO2 scoring** (TEST_PLAN §4D item 3) | saved adapters of run 10, **no retraining**. A reload check against the saved test probabilities gave max differences of 0.0253 / 0.0034 / 0.0335 (bf16 / 4-bit numerics with different batching). The KYPO shares were reproduced exactly (0.093 / 0.078 / 0.108) | 42, 123, 456 | — | Done (§3.13). `Test-Results-Sabrina/v3_v4_cmds_kypo2.json` |
| 14 | 2026-11-18 | **Zero-shot Qwen2.5-7B-Instruct** (TEST_PLAN §4A, declared in §4D item 5) | v3 notebook: `ZERO_SHOT=True`, `EVAL_PREFIXES=[3, None]`, `SCORE_BATCH=2`; `prepare_model_for_kbit_training` skipped in zero-shot mode (memory) | 1 run (no training) | shift (scored, not used) | Done (§5.4). `Test-Results-Sabrina/v3_zs_qwen7b.json` |
| 15 | 2026-11-18 | **Part 8 evasion experiments on v3**, the teammate's `scripts/run_evasion_experiments.py` | data path switched to `data/v3`; QLoRA input = final Model 3 ensemble (`data/v3/qlora_predictions.json`); XGBoost and the GRU are trained by the script (1 seed each) | — | — | Done (§3.14). Reproduced on two machines |
| 16 | 2026-11-18 | **Part 9: dashboard on v3, and the wild Cowrie case study** | `honeyai/dashboard.py` (same switch); `scripts/parse_cowrie.py external/cowrie/cowrie.json --classify` | — | — | Done (§3.15) |
| 17 | 2026-11-18 | **Teammate's `train_qlora_local.py` on v3** (TEST_PLAN §4D item 6) | recipe unchanged; run with `run_teammate_v3.py` (data path, output path, all seeds kept) | 42, 123, 456 | none | Done (§3.16). `Test-Results-Sabrina/v3_teammate_qlora_local.json` |

Model: `Qwen/Qwen2.5-0.5B-Instruct`, 4-bit NF4, LoRA on all linear layers, class-weighted loss. The model
predicts by scoring the likelihood of each of the 3 labels. Transcripts longer than MAX_LEN lose their
**oldest** commands.

---

## 2. Summary: test macro-F1

Seeded runs report the **per-seed mean ± std**. "Ensemble" means probabilities averaged over the 3 seeds.

### Final results: v3 (fresh test set, 1,113 sessions)

| Model | Full | k=1 | k=3 | k=5 | AI recall (evasive) | Human → AI | ECE (full) | KYPO real humans → AI |
|---|---|---|---|---|---|---|---|---|
| **Model 3 (pre-specified)** | **0.769 ± 0.069** (ensemble 0.796) | 0.386 ± 0.029 | 0.677 ± 0.031 | 0.712 ± 0.038 | 0.780 (ensemble) | 0.080 ± 0.025 | 0.121 ± 0.005 | n/a (reads responses) |
| Model 3, commands only | 0.763 ± 0.036 (ensemble 0.778) | 0.413 ± 0.004 | 0.543 ± 0.012 | 0.747 ± 0.012 | 0.870 (ensemble) | 0.108 ± 0.005 | 0.122 ± 0.007 | **0.228 ± 0.093** |
| *Best baseline* | *text 0.818* | *text 0.669* | *text 0.751* | *text 0.778* | — | — | — | *text-cmds 0.044* |
| **v4 (post hoc, frozen)** | **0.896 ± 0.006** (ensemble 0.899) | 0.704 ± 0.003 | 0.875 ± 0.009 | 0.904 ± 0.012 | 0.828 (ensemble) | 0.095 ± 0.015 | 0.062 ± 0.014 | n/a (reads responses) |
| v4, commands only | 0.878 ± 0.006 (ensemble 0.906) | 0.694 ± 0.028 | 0.844 ± 0.010 | 0.874 ± 0.007 | 0.875 (ensemble) | 0.115 ± 0.015 | 0.072 ± 0.017 | **0.093 ± 0.012** |
| *Best gap-augmented baseline* | *text-cmds 0.858* | *text 0.708* | *text-cmds 0.813* | *text-cmds 0.856* | — | — | — | *timing 0.057* |
| *Qwen2.5-7B-Instruct, zero-shot (outside, 1 run)* | *0.322* | — | *0.255* | — | *0.183* | *0.278* | *0.459* | — |
| *Teammate `train_qlora_local.py` (3 seeds; 37% unparseable)* | *0.545 ± 0.004* | — | — | — | — | *0.087 (vote)* | — (constant confidence) | — |

- On v3, **AI recall (evasive)** covers the 545 AI sessions whose `evasion_strategy` is not `none`.
- The KYPO column is the share of 1,414 real-human sessions labelled AI. Only commands-only models can be scored on KYPO, because it has no server responses (§3.13).
- **Latency** (p50 / p95 per command update, RTX 3070 Ti):

  | Model | p50 | p95 |
  |---|---|---|
  | Model 3 | 168 ms | 183 ms |
  | v4 | 166 ms | 173 ms |
  | v4, commands only | 170 ms | 182 ms |
  | Model 3, commands only | 160 ms | 510 ms |

  - The 510 ms is probably load on the PC (§3.7 saw the same).
  - It is measured after 3 commands; longer sessions take longer.
  - No latency target was set in advance (TEST_PLAN TC-D5), so none is applied here.

### Development: v2b (test set used during development; scores biased upward)

| Model | Full | k=1 | k=3 | k=5 | AI recall (evasive) | Human → AI | ECE (full) |
|---|---|---|---|---|---|---|---|
| QLoRA 0.5B, option 1 (seed 42 only) | **0.724** (95% CI 0.689–0.755) | 0.366 | 0.609 | 0.723 | **0.743** | 0.147 | not recorded |
| Teammate `train_qlora_local.py` (3 seeds) | 0.513 ± 0.011 | — | — | — | not computed | not computed | — (constant confidence) |
| **Model 3: QLoRA 0.5B, option 1, shift validation (3 seeds)** | **0.719 ± 0.036** | 0.391 ± 0.023 | 0.670 ± 0.009 | 0.713 ± 0.014 | 0.705 (ensemble) | 0.137 ± 0.022 | 0.206 ± 0.039 |
| QLoRA 0.5B, v4 (gap augmentation, 3 seeds) | **0.815 ± 0.021** | 0.690 ± 0.037 | 0.793 ± 0.019 | 0.822 ± 0.025 | 0.670 (ensemble) | 0.144 ± 0.066 | 0.113 ± 0.009 |
| *Best baseline with the same gap augmentation* | *text-cmds 0.827* | *text 0.718* | *text-cmds 0.781* | *text-cmds 0.826* | — | — | — |
| **QLoRA 0.5B, option 2, shift validation (3 seeds)** | 0.683 ± 0.003 | 0.655 ± 0.010 | **0.724 ± 0.020** | 0.700 ± 0.013 | 0.736 (ensemble) | 0.154 ± 0.054 | 0.214 ± 0.036 |
| QLoRA 0.5B, option 2 (3 seeds) | 0.653 ± 0.011 | 0.643 ± 0.056 | 0.678 ± 0.004 | 0.670 ± 0.005 | 0.564 ± 0.039 | 0.095 ± 0.028 | 0.227 ± 0.012 |
| *Best baseline at that point (TEST_PLAN §5)* | *text 0.765* | *text-cmds 0.660* | *XGBoost 0.747* | *text 0.746* | *XGBoost 0.738* | *XGBoost 0.099* | — |

- On v2b, **AI recall (evasive)** is the recall on AI sessions whose `evasion_strategy` is not `none` (288 sessions).
- **Human → AI** is the share of (simulated) human test sessions labelled AI.

### Option 2 per seed

| Seed | Best epoch (val F1) | Epochs run | Full | k=1 | k=3 | k=5 | Human → AI (full) |
|---|---|---|---|---|---|---|---|
| 42 | 2 (0.912) | 3 | 0.638 | 0.680 | 0.681 | 0.668 | 0.081 |
| 123 | 1 (0.865) | 2 | 0.662 | 0.564 | 0.680 | 0.677 | 0.134 |
| 456 | 1 (0.896) | 2 | 0.659 | 0.686 | 0.675 | 0.665 | 0.069 |

- Training took about 37–72 minutes per epoch.
- Inference latency was p50 161 ms and p95 173 ms per command update (CUDA).
- Option 1 seed 42 stopped after epoch 3; epoch 2 was best, with val 0.966.
  - Option 1's latency was p50 165 ms and p95 184 ms.

---

## 3. Paired comparison against the baselines (`scripts/compare_models.py`)

The method: a 95% paired bootstrap CI of the macro-F1 difference (LLM − baseline), Holm-corrected
one-sided p-values, and a McNemar test. A verdict of **LLM BETTER** needs the CI to be entirely above 0
and the Holm p-value to be below 0.05. The LLM's figure in these tables is the seed ensemble (probabilities averaged over the seeds).

### 3.1 Option 1, seed 42, full sessions

Ensemble of 1 seed: macro-F1 0.724, AI-evasive recall 0.743, human → AI 0.147.

| vs | Baseline F1 | Diff | 95% CI | p (Holm) | McNemar p | Verdict |
|---|---|---|---|---|---|---|
| text | 0.765 | −0.041 | [−0.076, −0.006] | 1.000 | 0.512 | LLM worse |
| text-cmds | 0.737 | −0.013 | [−0.040, +0.013] | 1.000 | 0.857 | no significant difference |
| gru-cmds | 0.728 | −0.004 | [−0.033, +0.024] | 1.000 | 0.488 | no significant difference |
| gru | 0.717 | +0.007 | [−0.022, +0.035] | 1.000 | 0.048 | no significant difference |
| xgboost | 0.695 | +0.029 | [+0.004, +0.054] | 0.048 | 0.488 | **LLM better** |
| timing | 0.667 | +0.057 | [+0.030, +0.083] | 0.000 | 0.001 | **LLM better** |

### 3.2 Option 2, 3 seeds, full sessions

Ensemble 0.634 (per-seed 0.653 ± 0.011), AI-evasive recall 0.542, human → AI 0.094.

| vs | Baseline F1 | Diff | 95% CI | p (Holm) | McNemar p | Verdict |
|---|---|---|---|---|---|---|
| text | 0.765 | −0.131 | [−0.163, −0.099] | 1.000 | 0.000 | LLM worse |
| text-cmds | 0.737 | −0.103 | [−0.134, −0.075] | 1.000 | 0.000 | LLM worse |
| gru-cmds | 0.728 | −0.094 | [−0.126, −0.064] | 1.000 | 0.000 | LLM worse |
| gru | 0.717 | −0.083 | [−0.115, −0.053] | 1.000 | 0.001 | LLM worse |
| xgboost | 0.695 | −0.061 | [−0.088, −0.034] | 1.000 | 0.000 | LLM worse |
| timing | 0.667 | −0.033 | [−0.063, −0.005] | 1.000 | 0.026 | LLM worse |

### 3.3 Option 2, 3 seeds, after 3 commands (k=3)

Ensemble 0.683 (per-seed 0.678 ± 0.004), AI-evasive recall 0.601, human → AI 0.119.

| vs | Baseline F1 | Diff | 95% CI | p (Holm) | McNemar p | Verdict |
|---|---|---|---|---|---|---|
| xgboost | 0.747 | −0.064 | [−0.093, −0.035] | 1.000 | 0.001 | LLM worse |
| text-cmds | 0.724 | −0.041 | [−0.067, −0.016] | 1.000 | 0.038 | LLM worse |
| text | 0.710 | −0.027 | [−0.054, −0.001] | 1.000 | 0.597 | LLM worse |
| gru-cmds | 0.687 | −0.004 | [−0.035, +0.027] | 1.000 | 0.880 | no significant difference |
| gru | 0.670 | +0.014 | [−0.023, +0.050] | 1.000 | 0.887 | no significant difference |
| timing | 0.668 | +0.015 | [−0.019, +0.049] | 1.000 | 0.314 | no significant difference |

### 3.4 Option 2, 3 seeds, after 1 command (k=1)

Ensemble 0.660 (per-seed 0.643 ± 0.056), AI-evasive recall 0.635, human → AI 0.237.

| vs | Baseline F1 | Diff | 95% CI | p (Holm) | McNemar p | Verdict |
|---|---|---|---|---|---|---|
| gru-cmds | 0.666 | −0.006 | [−0.043, +0.030] | 1.000 | 0.949 | no significant difference |
| text-cmds | 0.660 | +0.000 | [−0.026, +0.027] | 1.000 | 0.855 | no significant difference |
| text | 0.652 | +0.008 | [−0.019, +0.034] | 1.000 | 0.597 | no significant difference |
| gru | 0.647 | +0.013 | [−0.025, +0.052] | 1.000 | 0.502 | no significant difference |
| xgboost | 0.525 | +0.135 | [+0.099, +0.172] | 0.000 | 0.000 | **LLM better** |
| timing | 0.377 | +0.283 | [+0.246, +0.319] | 0.000 | 0.000 | **LLM better** |

### 3.5 Teammate's `train_qlora_local.py` (3 seeds, full sessions)

| Seed | Macro-F1 | Parse failures (out of 830) |
|---|---|---|
| 42 | 0.527 | 290 |
| 123 | 0.503 | 333 |
| 456 | 0.508 | 306 |
| **Mean ± std** | **0.513 ± 0.011** | 310 (37%) |

Seed 456 detail. The confusion matrix covers only the 524 parsed predictions; the 306 failures are
labelled `unknown` and scored as wrong.

| | precision | recall | F1 |
|---|---|---|---|
| human | 0.675 | 0.531 | 0.594 |
| scripted | 0.818 | 0.189 | 0.308 |
| AI-driven | 0.746 | 0.531 | 0.620 |

Confusion matrix (rows are the true class; columns are predicted human / scripted / AI), seed 456:

- human: [170, 0, 37]
- scripted: [28, 36, 21]
- AI-driven: [54, 8, 170]

**Why so many parse failures: input truncation, not the model.**
- At test time the prompt is truncated to 480 tokens **from the right** (the tokenizer's default `truncation_side`). That cuts off the end of the session and the `assistant` turn marker, so the model continues writing the transcript instead of answering in JSON.
- 364 of 830 test prompts are longer than 480 tokens: 129 human, 128 scripted and 107 AI.
- The parse failures per class follow the same pattern (seed 456: 113 human, 105 scripted, 88 AI). The few long sessions that still parse are rescued by the fallback that looks for a label name anywhere in the text.
- Training has the same problem at 512 tokens: the answer is cut off in 624 of 1,482 training sessions, so those sessions teach nothing.

**On the 524 parsed sessions alone, macro-F1 is 0.679 (seed 456).** That figure is **not** a fair score. It
drops the longest sessions, which are a different mix of classes, so use it only as a diagnostic.

**Other differences from the patched notebook**, so the gap is not attributable to one cause:
- no validation set and no early stopping (15 fixed epochs);
- LoRA r8 instead of r16, and LR 5e-5 instead of 1e-4;
- no class weighting;
- the stated confidence is always the 0.95 it was trained to print, so calibration and abstention cannot be measured;
- predictions come from free-text generation (which can fail) instead of scoring the 3 labels (which cannot).

### 3.6 Option 2 with shift validation (run 4, 3 seeds)

**Validation (shift, full sessions). This is the number used for selection:**

| Seed | Validation macro-F1 per epoch | Best | Epoch time |
|---|---|---|---|
| 42 | 0.964, 0.961 | **0.964** (epoch 1) | 50 min |
| 123 | 0.896, 0.931, 0.946, 0.937 | 0.946 (epoch 3) | 48 min for epoch 1, then 92–93 min |
| 456 | 0.954, 0.941 | 0.954 (epoch 1) | 52–53 min |

Seed 123's epochs slowed to about twice as long after epoch 1. That is a hardware slowdown (probably GPU memory
spilling into system RAM); the training work per epoch is the same.

**Test (per seed):**

| Seed | Full | k=1 | k=3 | k=5 | Human → AI (full) | ECE (full) |
|---|---|---|---|---|---|---|
| 42 | 0.685 | 0.649 | 0.735 | 0.699 | 0.228 | 0.228 |
| 123 | 0.680 | 0.655 | 0.697 | 0.686 | 0.138 | 0.251 |
| 456 | 0.685 | 0.669 | 0.741 | 0.717 | 0.097 | 0.163 |
| **Mean ± std** | **0.683 ± 0.003** | 0.655 ± 0.010 | **0.724 ± 0.020** | 0.700 ± 0.013 | 0.154 ± 0.054 | 0.214 ± 0.036 |

**Paired comparison against the baselines.** The LLM figure is the 3-seed ensemble.

| vs | Full | After 1 command | After 3 commands | After 5 commands |
|---|---|---|---|---|
| text | −0.081, worse | +0.003, n.s. | **+0.031 [+0.004, +0.058], better** | −0.041, worse |
| text-cmds | −0.053, worse | −0.004, n.s. | +0.017, n.s. | −0.018, n.s. |
| gru-cmds | −0.045, worse | −0.011, n.s. | **+0.054 [+0.026, +0.084], better** | −0.002, n.s. |
| gru | −0.033, worse | +0.008, n.s. | **+0.072 [+0.037, +0.106], better** | −0.014, n.s. |
| xgboost | −0.011, n.s. | **+0.130, better** | −0.006, n.s. | −0.037, worse |
| timing | +0.016, n.s. | **+0.278, better** | **+0.073 [+0.042, +0.104], better** | −0.002, n.s. |

"n.s." means no significant difference. Ensemble macro-F1: full 0.684, k=1 0.655, k=3 **0.741**, k=5 0.705.

**Against the earlier option 2 run (random validation), full sessions:** +0.050 [+0.031, +0.070], better.
The same configuration improved only by changing how the epoch is chosen: early stopping on the shift
validation set stopped two of the three seeds after epoch 1.

**Full-session confusion matrices** (rows are the true class; columns are predicted human / scripted / AI):

- seed 42: human [245, 2, 73], scripted [2, 84, 104], AI [33, 28, 259]
- seed 123: human [274, 2, 44], scripted [4, 90, 96], AI [73, 29, 218]
- seed 456: human [287, 2, 31], scripted [14, 84, 92], AI [70, 27, 223]

**AI recall by strategy, full sessions (seeds 42 / 123 / 456):**

- `combined`: 0.60 / 0.23 / 0.31
- `avoid_systematic_enumeration`: 0.96 / 0.91 / 0.95
- `timing_mimicry`: 0.97 / 0.94 / 0.91

### 3.7 Option 1 with shift validation (run 5, seed 42) and the §4B decision

- **Validation per epoch:** 0.941, **0.960**, 0.922. Early stopping chose epoch 2.
- **Training time:** 18–19 minutes per epoch.
- **Latency:** p50 297 ms and p95 496 ms. Option 2 measured about 161 ms, so this is probably load on the PC, not the model.

**Decision under TEST_PLAN §4B.**
- Seed 42 validation: option 1 0.9597, option 2 0.9643. The difference of 0.0046 is under 0.01, so it is a tie.
- A tie goes to the cheaper configuration, which is **option 1**. Option 1's validation score was not looked at before this was decided.

**Test (seed 42):**

| | Full | k=1 | k=3 | k=5 | AI recall (evasive) | Human → AI | ECE |
|---|---|---|---|---|---|---|---|
| Option 1, shift validation | 0.670 | 0.419 | 0.680 | 0.696 | 0.674 | 0.141 | 0.261 |
| Option 2, shift validation, seed 42 | 0.685 | 0.649 | 0.735 | 0.699 | — | 0.228 | 0.228 |

**Paired comparison against the baselines, full sessions:**
- Worse than text (−0.094), text-cmds (−0.067), gru-cmds (−0.058), gru (−0.047) and xgboost (−0.024, CI [−0.048, −0.001]).
- No significant difference from timing.

**Earlier in the session:**
- After 1 command, it is worse than every baseline except timing.
- After 3 commands, it is worse than xgboost and text-cmds and tied with the rest.

**Full-session confusion matrix** (rows are the true class; columns are predicted human / scripted / AI):

- human: [275, 0, 45]
- scripted: [17, 84, 89]
- AI: [75, 26, 219]

AI recall on `combined` is 0.20.

**Note:** on the test set, option 2 scored as well as or better than option 1 at every point. Validation could not tell the two apart,
and the tie-break chose option 1. Option 2 is reported in full as the partial-session-training experiment (§3.6).

---

### 3.8 v4: gap augmentation (run 6, 3 seeds), and an equal-effort check

**What v4 changes compared with v3** (notebook `qlora_generative_patched v4.ipynb`):
1. **`GAP_AUG`:** half of the scripted TRAIN sessions get an extra copy. In that copy, each `[GAP ...]` token is
   resampled with probability 0.7 from the gap distribution of the non-scripted TRAIN sessions.
2. **`VAL_GAP_JITTER`:** half of the held-out scripted validation sessions are jittered in place.
3. **`PREFIX_W = 0.3`:** partial-session examples get 0.3 of the loss weight of a full session.
4. **Loss weighting:** weights are normalised over the whole dataset, not per micro-batch.
5. **Gradient accumulation:** the loss is now divided by the accumulation steps correctly.

**Leakage review: none found.** The gap distribution is taken from TRAIN sessions after the validation split.
Augmentation touches only training (copies) and validation (in place). Test sessions are never modified.

**Why it must be disclosed: the change was motivated by test results.** The notebook's own note for fix #12 cites
test recall on `human_like_delays` (0/64). Validation could not have suggested it:
- the held-out validation family (`randomised_ordering`) does not fake timing;
- v4's validation score is lower (0.905–0.919, against 0.946–0.964 for v3);
- `VAL_GAP_JITTER` changes the validation set itself, so v4's and v3's validation scores are not comparable.

Under §4B, v4 could not have been selected. This is test-informed design (the "data snooping" pitfall in Arp et
al., USENIX Security 2022) and must be reported as such.

**Validation per seed:**

| Seed | Validation per epoch | Best | Epoch time |
|---|---|---|---|
| 42 | 0.919, 0.896 | 0.919 (epoch 1) | 77–78 min |
| 123 | 0.905, 0.892 | 0.905 (epoch 1) | 79–109 min |
| 456 | 0.910, 0.874 | 0.910 (epoch 1) | 82–88 min |

**Test macro-F1 per seed:**

| Seed | Full | k=1 | k=3 | k=5 |
|---|---|---|---|---|
| 42 | 0.828 | 0.705 | 0.798 | 0.838 |
| 123 | 0.832 | 0.726 | 0.813 | 0.842 |
| 456 | 0.785 | 0.640 | 0.766 | 0.787 |

**Where the gain comes from.** Full-session recall by family, mean of 3 seeds:

| Family | v3 (option 2, shift val) | v4 |
|---|---|---|
| scripted / `human_like_delays` | **0.00** | **0.93** |
| scripted / `retry_on_failure` | 0.35 | 1.00 |
| scripted / `alternative_commands` | 1.00 | 1.00 |
| human | 0.84 | 0.84 |
| AI `sim_agent_a` / `b` / `c` | 0.74 / 0.72 / 0.73 | **0.65 / 0.61 / 0.65** |

- Every training scripted session has only short gaps, so v3 learned "short gaps = scripted". The test family `human_like_delays` breaks that rule.
- **The gain is all in the scripted class. AI detection got worse:** evasive-AI recall fell from 0.736 to 0.670 (ensemble).
- ECE improved from 0.21 to 0.11.

**Equal-effort check.** The same augmentation was given to every baseline:
`python scripts/run_baselines.py --data data/v2b/dataset1_sessions.json --gap-aug 0.5 --out runs/baselines_v2b_gapaug.json`.
Raw gaps are resampled from the non-scripted training gaps, with the same fraction, probability and seed.

| Baseline | Full, no augmentation | Full, with augmentation |
|---|---|---|
| timing | 0.667 | 0.709 |
| text | 0.765 | 0.801 |
| text-cmds | 0.737 | **0.827** |
| XGBoost | 0.694 | 0.691 |
| GRU | 0.713 | 0.725 |
| GRU, commands only | 0.721 | 0.730 |

With augmentation, the text models also recall `human_like_delays` (0.92–0.94). The GRU does not (0.01).

**v4 against the augmented baselines** (paired, ensemble; Holm-corrected):

| vs | Full | After 1 command | After 3 commands | After 5 commands |
|---|---|---|---|---|
| text-cmds | +0.011 [−0.014, +0.035], n.s. | −0.001, n.s. | **+0.034 [+0.007, +0.060], better** | +0.022 [−0.004, +0.049], n.s. |
| text | **+0.037 [+0.013, +0.060], better** | −0.010, n.s. | **+0.039, better** | **+0.034, better** |
| gru-cmds | **+0.105, better** | **+0.037, better** | **+0.122, better** | **+0.131, better** |
| gru | **+0.109, better** | **+0.059, better** | **+0.115, better** | **+0.107, better** |
| timing | **+0.129, better** | **+0.327, better** | **+0.160, better** | **+0.131, better** |
| xgboost | **+0.146, better** | **+0.366, better** | **+0.256, better** | **+0.109, better** |

Against the un-augmented baselines, v4 is "better" than all six at every point. **That comparison is not
fair and should not be reported as the headline.**

**Fair reading:**
- With the same augmentation for everyone, v4 is tied with the best baseline (text-cmds) on full sessions.
- It is better than every baseline after 3 commands.
- It is better than all except the text models after 1 and 5 commands.

### 3.9 Model 3 (final, pre-specified): option 1 with shift validation, 3 seeds (runs 5 + 7)

This is the configuration the §4B rule selected on validation, before its test results were seen. **It is the
Model 3 result.** Runs 5 and 7 use identical settings; only the seeds differ.

| Seed | Best validation (epoch) | Epochs run | Full | k=1 | k=3 | k=5 | Human → AI (full) | ECE (full) |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.960 (2) | 3 | 0.670 | 0.419 | 0.680 | 0.696 | 0.141 | 0.261 |
| 123 | 0.921 (3) | 4 | 0.756 | 0.390 | 0.672 | 0.731 | 0.109 | 0.170 |
| 456 | 0.942 (2) | 3 | 0.731 | 0.364 | 0.658 | 0.712 | 0.163 | 0.187 |
| **Mean ± std** | | | **0.719 ± 0.036** | 0.391 ± 0.023 | 0.670 ± 0.009 | 0.713 ± 0.014 | 0.137 ± 0.022 | 0.206 ± 0.039 |

- The seeds vary a lot on full sessions (0.670–0.756).
- Seed 42, used for the §4B decision, happened to be the weakest.
- Epochs took 18–44 minutes.

**Paired comparison against the baselines (un-augmented, as pre-specified).** The LLM figure is the 3-seed ensemble.

| vs | Full (ensemble 0.722) | After 1 command (0.373) | After 3 commands (0.688) | After 5 commands (0.710) |
|---|---|---|---|---|
| text | −0.043 [−0.077, −0.009], worse | −0.280, worse | −0.023, n.s. | −0.037, worse |
| text-cmds | −0.015, n.s. | −0.287, worse | −0.037 [−0.071, −0.004], worse | −0.014, n.s. |
| gru-cmds | −0.007, n.s. | −0.294, worse | +0.000, n.s. | +0.002, n.s. |
| gru | +0.005, n.s. | −0.275, worse | +0.018, n.s. | −0.010, n.s. |
| xgboost | +0.027 [−0.002, +0.055], n.s. | −0.152, worse | −0.059, worse | −0.032, worse |
| timing | **+0.054 [+0.025, +0.082], better** | −0.004, n.s. | +0.019, n.s. | +0.003, n.s. |

**Full-session confusion matrices** (rows are the true class; columns are predicted human / scripted / AI):

| Seed | human | scripted | AI |
|---|---|---|---|
| 42 | [275, 0, 45] | [17, 84, 89] | [75, 26, 219] |
| 123 | [285, 0, 35] | [13, 124, 53] | [72, 24, 224] |
| 456 | [268, 0, 52] | [9, 109, 72] | [61, 21, 238] |

**Recall by family (full sessions, mean of 3 seeds):**

- human: 0.86
- scripted: `alternative_commands` 1.00, `retry_on_failure` 0.54, `human_like_delays` **0.10**
- AI: `sim_agent_a` 0.67, `sim_agent_b` 0.66, `sim_agent_c` (unseen) 0.72

**Reading:**
- **Full sessions:** Model 3 is significantly better only than the timing baseline. It is tied with text-cmds, both GRUs and XGBoost, and significantly worse than TF-IDF text.
- **Early in a session it is weak.** After 1 command it labels 63% of humans as AI, because it was never trained on partial sessions.
- **Its errors:**
  - scripted sessions with human-like timing are mistaken for AI;
  - the `combined` evasion strategy is missed (recall 0.20–0.33).
- **The unseen AI family is detected about as well as the seen ones** (0.72, against 0.66–0.67).

### 3.10 Final evaluation on v3: Model 3 (run 8, TEST_PLAN §4C)

Every v3 test session is new (1,113 sessions: 320 human, 188 scripted, 605 AI, of which 287 are real LLM agents).
The baselines were run on two machines, and their results match (XGBoost within 0.006).

**Validation and test per seed:**

| Seed | Validation per epoch | Best | Full | k=1 | k=3 | k=5 | Human → AI |
|---|---|---|---|---|---|---|---|
| 42 | 0.871, **0.927**, 0.730 | epoch 2 | 0.810 | 0.348 | 0.679 | 0.734 | 0.116 |
| 123 | 0.818, **0.912**, 0.873 | epoch 2 | 0.826 | 0.393 | 0.713 | 0.744 | 0.066 |
| 456 | **0.888**, 0.746 | epoch 1 | 0.672 | 0.417 | 0.638 | 0.658 | 0.059 |
| **Mean ± std** | | | **0.769 ± 0.069** | 0.386 ± 0.029 | 0.677 ± 0.031 | 0.712 ± 0.038 | 0.080 |

- Seed 456 stopped after epoch 1 and labels 182 of 605 AI sessions as human, which explains the large std.
- The 3-seed ensemble (0.796) is higher than the per-seed mean, because the seeds make different errors.
- Latency: p50 168 ms, p95 183 ms.

**Primary result: full sessions, all test sessions** (ensemble against ensemble baselines; Holm-corrected):

| vs | Baseline F1 | Diff | 95% CI | Verdict |
|---|---|---|---|---|
| text | 0.818 | −0.022 | [−0.049, +0.006] | no significant difference |
| text-cmds | 0.777 | +0.019 | [−0.009, +0.047] | no significant difference |
| GRU (Model 2) | 0.767 | +0.029 | [−0.003, +0.061] | no significant difference |
| GRU, commands only | 0.762 | +0.035 | [+0.005, +0.065] | **better** (Holm p 0.046, borderline) |
| XGBoost (Model 1) | 0.648 | +0.148 | [+0.115, +0.179] | **better** |
| timing | 0.622 | +0.174 | [+0.141, +0.206] | **better** |

**Subsets fixed in §4C** (full sessions; human and scripted test sessions kept in both):

| Subset | Model 3 ensemble (per seed) | Better than | Tied with | Worse than |
|---|---|---|---|---|
| Real held-out agents (Gemma, Claude), 708 sessions | 0.830 (0.813 ± 0.068). AI recall 0.913 | XGBoost, timing | text-cmds, GRU, GRU-cmds | **text** (−0.055) |
| Simulated AI, 826 sessions | 0.774 (0.753 ± 0.068). Evasive-AI recall 0.668 | **GRU, GRU-cmds**, XGBoost, timing | text, text-cmds | — |

**Early attribution** (all sessions):

| Point | Model 3 ensemble | Better than | Tied with | Worse than |
|---|---|---|---|---|
| After 1 command | 0.411 | — | timing | all others (labels 68% of humans as AI) |
| After 3 commands | 0.699 | — | XGBoost, GRUs, timing | text, text-cmds |
| After 5 commands | 0.723 | XGBoost, timing | text-cmds | text, GRU, GRU-cmds |

**Reading:**
- On fresh data, Model 3 is **statistically tied with the best baseline** (TF-IDF text) on full sessions.
- It is significantly better than Model 1 (XGBoost), the timing rule and the commands-only GRU.
- It is tied with Model 2 (GRU) overall, and better than both GRUs on the simulated evasive agents.
- It is not the best model:
  - TF-IDF text is better on the real-agent subset;
  - option 1 is weak early in a session.
- The seed variance (std 0.069) is large, so the per-seed mean is the conservative headline.

### 3.11 Final evaluation on v3: frozen v4 (run 9, TEST_PLAN §4C)

v4 was designed after seeing v2b test errors, so its v2b score was biased (§3.8). It was frozen at commit `8f433ab`
and run here once, unchanged, on v3. Only the data path and the results folder differ.

**Validation and test per seed:**

| Seed | Validation per epoch | Best | Full | k=1 | k=3 | k=5 | Human → AI | ECE |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.854, **0.923**, 0.896 | epoch 2 | 0.901 | 0.703 | 0.881 | 0.909 | 0.087 | 0.043 |
| 123 | 0.867, **0.901**, 0.897 | epoch 2 | 0.901 | 0.702 | 0.882 | 0.914 | 0.081 | 0.075 |
| 456 | 0.821, **0.894**, 0.853 | epoch 2 | 0.887 | 0.708 | 0.863 | 0.887 | 0.116 | 0.068 |
| **Mean ± std** | | | **0.896 ± 0.006** | 0.704 ± 0.003 | 0.875 ± 0.009 | 0.904 ± 0.012 | 0.095 | 0.062 |

- Seed variance is small (std 0.006, against Model 3's 0.069).
- Epochs took 59–106 minutes.
- Latency: p50 166 ms, p95 173 ms.

**v4 against the gap-augmented baselines and Model 3** (ensemble; Holm-corrected; diff with 95% CI):

| vs | Full, all sessions | Real held-out agents (708) | Simulated AI (826) | After 1 command | After 3 commands | After 5 commands |
|---|---|---|---|---|---|---|
| text-cmds | **+0.041 [+0.024, +0.058]** | **+0.032 [+0.014, +0.050]** | **+0.049** | +0.002, n.s. | **+0.065** | **+0.052** |
| text | **+0.045 [+0.028, +0.063]** | **+0.021 [+0.004, +0.038]** | **+0.061** | +0.001, n.s. | **+0.070** | **+0.068** |
| GRU | **+0.147** | **+0.161** | **+0.166** | **+0.087** | **+0.226** | **+0.187** |
| GRU, commands only | **+0.144** | **+0.161** | **+0.169** | **+0.046** | **+0.221** | **+0.181** |
| timing | **+0.152** | **+0.171** | **+0.139** | **+0.103** | **+0.208** | **+0.167** |
| XGBoost | **+0.187** | **+0.215** | **+0.208** | **+0.288** | **+0.227** | **+0.208** |
| Model 3 (option 1) | **+0.103 [+0.079, +0.127]** | **+0.116** | **+0.111** | **+0.297** | **+0.179** | **+0.185** |

Bold means "v4 better" under the §4 rule. v4's ensemble scores: full 0.899, real agents 0.946, simulated 0.885,
k=1 0.708, k=3 0.878, k=5 0.908.

**Recall by family (full sessions, mean of 3 seeds):**

- scripted: all three test families 1.00 (including `human_like_delays`)
- real agents, held out: `claude-haiku-5.5` 0.97, `gemma-3-27b` 0.94
- real agents, training families (held-out strategies): `deepseek-v3.2` 0.97, `mistral-small-3.2` 0.92, `llama3.1-8b` 0.83
- simulated agents: `sim_agent_c` 0.80, `sim_agent_b` 0.70, `sim_agent_a` **0.58**

**Reading:**
- **v4's improvement holds on fresh data.** With the same gap augmentation given to every baseline, v4 is significantly better than all six on full sessions, after 3 commands and after 5 commands. After 1 command it ties the two text models.
- **It also holds on the cleanest subset.** On the real agent families nobody designed against (Gemma, Claude), v4 is better than every baseline. Its smallest margin is over text: +0.021, CI [+0.004, +0.038].
- **v4 is better than Model 3 at every point** (+0.103 on full sessions).
- **Its remaining weakness is the simulated evasive agents:** recall is 0.58 on `sim_agent_a`, and evasive-AI recall is 0.828 overall.

**Limits on this result:**
- **The scripted test families are the ones v4 was designed around.** v3's scripted and simulated-AI test families are the same generators as v2b's (new sessions, same families), and `human_like_delays` is the family that motivated the augmentation. Both subsets above include those scripted sessions. Only the real-agent AI sessions come from families nothing was designed against.
- **Tuning effort is unequal.** v4 is the sixth Model 3 configuration. The baselines got their defaults plus the augmentation, and nothing else.
- **Under §4B and §4C, Model 3 remains option 1.** v4 is reported as a post-hoc configuration confirmed on fresh data.
- **All humans in v3 are simulated.** The real-human check (KYPO, KYPO2) is in §3.13. It covers the commands-only version of v4 only.

---

### 3.12 Commands-only ablation (runs 10 and 11b, TEST_PLAN §4D)

Both models were retrained on v3 with every server response removed (`COMMANDS_ONLY=True`); nothing else changed.

**Baselines rebuilt on 2026-11-17** (second machine, Linux CPU):
- `run_baselines.py --data data/v3/dataset1_sessions.json` and `--gap-aug 0.5`;
- the outputs are `runs/baselines_v3*.json` (not in git);
- the comparisons are in `runs/compare_v3/`.

They reproduce the recorded baselines (TC-C1):
- the deterministic baselines match exactly (text 0.818, text-cmds 0.777, timing 0.622; augmented text-cmds 0.858);
- XGBoost and the GRUs are within ±0.02;
- every verdict in §3.10 and §3.11 comes out the same.

**Ablation: with responses against commands only** (paired, ensemble; positive = responses help):

| Model | Full | After 1 command | After 3 commands | After 5 commands |
|---|---|---|---|---|
| Model 3 | +0.018 [−0.008, +0.045], n.s. | −0.004, n.s. | **+0.150 [+0.120, +0.181], responses help** | **−0.042 [−0.067, −0.018], commands-only better** |
| v4 | −0.007 [−0.020, +0.006], n.s. | −0.010, n.s. | +0.004, n.s. | +0.003, n.s. |

Per seed:
- Model 3: full 0.769 ± 0.069 with responses, against 0.763 ± 0.036 commands only;
- v4: full 0.896 ± 0.006 against 0.878 ± 0.006.

**Reading:**
- **Server responses do not improve full-session accuracy for either model.** For v4 they make no difference at any point.
- For Model 3, responses matter only after 3 commands, where commands-only Model 3 recognises almost no scripted sessions (2.7% recall, ensemble). After 5 commands, commands-only Model 3 is better.
- So the models' signal is mainly in the commands and their timing.
- Two limits:
  - the responses come from our simulator, so they may carry less information than real server output;
  - about 5–10% of full sessions exceed `MAX_LEN` and lose their oldest commands, while no commands-only session does.

**Commands-only LLMs against the commands-only baselines** (paired, ensemble, Holm-corrected; Model 3 against plain baselines, v4 against gap-augmented, as fixed in §4D):

| vs | Model 3 cmds, full (0.778) | Model 3 cmds, k=3 (0.549) | Model 3 cmds, real agents (0.782) | v4 cmds, full (0.906) | v4 cmds, k=3 (0.874) | v4 cmds, real agents (0.934) |
|---|---|---|---|---|---|---|
| text-cmds | +0.001, n.s. | −0.192, worse | −0.039 [−0.065, −0.015], worse | **+0.048 [+0.029, +0.067], better** | **+0.061, better** | +0.020 [−0.000, +0.042], n.s. |
| gru-cmds | +0.017, n.s. | −0.133, worse | −0.027, n.s. | **+0.151, better** | **+0.217, better** | **+0.150, better** |
| timing | **+0.156, better** | −0.122, worse | **+0.163, better** | **+0.159, better** | **+0.204, better** | **+0.159, better** |

"Real agents" is the §4C subset: Gemma and Claude, plus all human and scripted test sessions (708).

### 3.13 Real humans: KYPO and KYPO2 (TEST_PLAN §4D)

Every session in these sets is a real person, so any "AI" or "scripted" label is an error. Only commands-only
models can be scored, because neither set has server responses.

**KYPO (1,414 sessions; 357 on target hosts):**

| Model | Labelled AI, all | Target hosts | Other hosts | Labelled scripted | Labelled human |
|---|---|---|---|---|---|
| **Model 3, commands only** (3 seeds) | **0.228 ± 0.093** (0.103 / 0.257 / 0.325) | 0.213 | 0.233 | 0.000 | 0.772 |
| v4, commands only (3 seeds) | 0.093 ± 0.012 (0.093 / 0.078 / 0.108) | 0.116 | 0.085 | 0.001 | 0.906 |
| timing | 0.062 | 0.137 | 0.036 | 0.011 | 0.928 |
| text-cmds | **0.044** | 0.106 | 0.023 | 0.021 | 0.936 |
| gru-cmds (3 seeds) | 0.362 ± 0.098 | 0.317 | 0.377 | 0.000 | 0.638 |
| timing, gap-augmented | 0.057 | 0.120 | 0.035 | 0.049 | 0.895 |
| text-cmds, gap-augmented | 0.065 | 0.123 | 0.045 | 0.002 | 0.933 |
| gru-cmds, gap-augmented (3 seeds) | 0.358 ± 0.140 | 0.324 | 0.369 | 0.000 | 0.642 |

**KYPO2 (215 sessions; 12 on target hosts; never used for calibration or selection):**

| Model | Labelled AI, all | Labelled human |
|---|---|---|
| **Model 3, commands only** (3 seeds) | **0.296 ± 0.165** (0.065 / 0.386 / 0.437) | 0.704 |
| v4, commands only (3 seeds) | 0.109 ± 0.040 (0.070 / 0.093 / 0.163) | 0.887 |
| timing | **0.028** | 0.953 |
| text-cmds | 0.065 | 0.935 |
| gru-cmds | 0.504 ± 0.086 | 0.496 |
| timing / text-cmds / gru-cmds, gap-augmented | 0.028 / 0.070 / 0.464 ± 0.177 | 0.935 / 0.930 / 0.536 |

The KYPO2 baseline rates match those recorded in TEST_PLAN §4D before any LLM result was seen. With only 12
target-host sessions, the target-host split is not reported for KYPO2.

**Uncertainty and per-person view** (per-session predictions saved by runs 12 and 13). The CIs are 95%,
from bootstrapping whole trainees (5,000 resamples), because one person contributes many sessions.

| Model | KYPO: labelled AI | KYPO: trainees with ≥1 session labelled AI (265) | KYPO2: labelled AI | KYPO2: trainees with ≥1 session labelled AI (52) |
|---|---|---|---|---|
| **Model 3, commands only, 3-seed ensemble** | **0.249 [0.221, 0.278]** | **0.725** | **0.344 [0.253, 0.440]** | **0.635** |
| v4, commands only, 3-seed ensemble | 0.088 [0.068, 0.109] | 0.298 | 0.098 [0.045, 0.155] | 0.231 |
| timing | 0.062 [0.050, 0.074] | 0.291 | 0.028 [0.009, 0.050] | 0.115 |
| text-cmds | 0.044 [0.033, 0.054] | 0.192 | 0.065 [0.019, 0.124] | 0.135 |
| timing, gap-augmented | 0.057 [0.046, 0.068] | 0.268 | 0.028 [0.010, 0.050] | 0.115 |
| text-cmds, gap-augmented | 0.065 [0.047, 0.084] | 0.211 | 0.070 [0.022, 0.126] | 0.154 |

- **Averaging the seeds does not help Model 3:** its ensemble flags more real people than its per-seed mean (KYPO 0.249 vs 0.228; KYPO2 0.344 vs 0.296). For v4 the ensemble is slightly lower (0.088 and 0.098).
- **In practice, Model 3 would flag most real people at some point.** 73% of KYPO trainees and 64% of KYPO2 trainees have at least one session labelled AI. For v4 it is 30% and 23%; for timing and TF-IDF, 12–29%.
- The baselines' per-session predictions were recomputed on 2026-11-17 (the deterministic ones only; same models as above). The GRU is left out, because its per-session KYPO predictions were not saved.
- KYPO2 session IDs differ between the two machines' conversions only in the folder name, for example `Locust_3302_insider-2022-11-28-sb120-s0-w0` vs `Locust_3302-2022-11 da-sb120-s0-w0`.
  - Sessions were matched on the `sbNNN-sN-wN` part: all 215 match one-to-one.
  - The baseline KYPO2 rates are identical on both machines.

**Paired test on real humans** (3-seed ensemble against the same sessions; difference in the share labelled AI; 95% CI from bootstrapping whole trainees; positive = the LLM flags more real people). As in §4D, Model 3 is compared with the plain baselines and v4 with the gap-augmented ones.

| | KYPO vs timing | KYPO vs text-cmds | KYPO2 vs timing | KYPO2 vs text-cmds |
|---|---|---|---|---|
| **Model 3, commands only** | **+0.187 [+0.159, +0.217], more** | **+0.205 [+0.175, +0.237], more** | **+0.316 [+0.226, +0.418], more** | **+0.279 [+0.207, +0.353], more** |
| v4, commands only | **+0.031 [+0.009, +0.054], more** | **+0.023 [+0.006, +0.040], more** | **+0.070 [+0.018, +0.128], more** | +0.028 [−0.014, +0.072], no significant difference |

**Reading:**
- **Model 3 (commands only) flags 23% of real people as AI, against 4–6% for TF-IDF and timing.** Its three seeds range from 10% to 33%.
  - On the simulated v3 humans the same seeds all flag 10–12%, so the simulated test set gives no warning of this.
- **v4 (commands only) is stable across seeds** (KYPO 8–11%, KYPO2 7–16%), but it flags significantly more real people than timing on both sets, and more than TF-IDF on KYPO. On KYPO2 it is not significantly different from TF-IDF. The excess is small (2–7 points), against Model 3's 19–32 points.
- **KYPO2 is worse for Model 3 (30%) than KYPO (23%).** KYPO2 was never used for calibration or selection, so the calibration overlap was not flattering KYPO.
  - Its seeds disagree badly: seed 42 flags 6.5% of real people (the same as TF-IDF), while seeds 123 and 456 flag 39–44%.
- **Both LLMs are far better than the commands-only GRU** (36% on KYPO, 50% on KYPO2).
- **The calibration overlap does not flatter the LLMs.** Model 3's rate is similar on target hosts (the sessions used to calibrate the simulated humans' timing) and on other hosts; v4's is higher on target hosts.
- **The paired tests compare seed ensembles.** Per-seed rates are in the tables above.

### 3.14 Part 8 evasion experiments on v3 (run 15)

The teammate's Part 8 code (`honeyai/experiments/evasion_analysis.py`), unchanged except for the data path:
- the data is v3 (1,113 test sessions; 545 evasive AI sessions);
- the QLoRA input is the final Model 3, as the 3-seed ensemble;
- XGBoost and the GRU are trained inside the script with one seed each.

It gave the same numbers on two machines: XGBoost 0.645 vs 0.644, the GRU 0.774, and QLoRA 0.796 in both.

**Overall macro-F1 (script):** XGBoost 0.645, GRU 0.774, **QLoRA (Model 3) 0.796**.
- The script compares one-seed baselines with a 3-seed ensemble, so the ranking is not a test.
- The paired, ensemble-against-ensemble result is in §3.10: Model 3 is tied with the GRU (+0.029 [−0.003, +0.061]) and better than XGBoost (+0.148).

**AI-driven recall per evasion strategy** (Experiment 1). The left block is the script's output. The right block
covers the real LLM agents only (all 5 families), computed from the §3.10 models (3-seed ensembles,
`run_baselines.py`), because the script cannot select subsets.

| Strategy | n | XGBoost | GRU | **QLoRA** | Real agents: n | XGBoost | GRU | **Model 3** |
|---|---|---|---|---|---|---|---|---|
| none | 60 | 0.917 | 1.000 | 0.917 | 28 | 0.964 | 1.000 | 0.893 |
| timing_mimicry | 62 | 0.968 | 0.887 | 0.855 | 30 | 1.000 | 0.933 | 0.867 |
| typo_mimicry | 60 | 0.933 | 0.950 | 0.967 | 28 | 0.929 | 1.000 | 0.929 |
| shorter_commands | 60 | 0.850 | 0.750 | 0.633 | 28 | 0.893 | 1.000 | 0.964 |
| hesitation_redundant_exploration | 60 | 0.967 | 0.933 | 0.933 | 28 | 1.000 | 0.964 | 0.929 |
| avoid_systematic_enumeration | 147 | 0.966 | 0.946 | 0.939 | 69 | 0.957 | 1.000 | 0.957 |
| **combined** | 156 | 0.603 | 0.538 | 0.526 | 76 | **0.974** | **0.855** | **0.829** |

**Non-evasive vs evasive AI recall** (Experiment 2): XGBoost 0.917 → 0.846, GRU 1.000 → 0.800, QLoRA 0.917 → 0.780.

**Commands-only ablation** (Experiment 3; XGBoost and GRU only): XGBoost 0.645 → 0.641, GRU 0.774 → 0.768.
- Responses make no difference, as for both LLMs (§3.12).

**Reading:**
- **Model 3 is not the most robust model by recall.** XGBoost has the highest evasive-AI recall (0.846), but it gets it by labelling many sessions AI: it recalls only about 20% of scripted sessions, and its macro-F1 is the lowest. Recall has to be read together with the false-alarm rate on humans (§3.10: XGBoost 0.073, GRU 0.126, Model 3 0.081).
- **The weak strategies are weak on simulated agents only.**
  - `combined` drops every model to 0.53–0.60 overall, but on the real LLM agents the same strategy is caught 83–97% of the time.
  - `shorter_commands` hurts Model 3 overall (0.633), but it catches 96% of real agents using it.
- **This changes two of the findings in the earlier v2b report:**
  - "Combined evasion defeats all models (~29%), a ceiling for passive observation" is a property of the simulated combined agent, not of real agents.
  - "The GRU breaks on shorter commands (0.188)" is much weaker on v3 (0.750), and the GRU catches 100% of real agents using that strategy.
- **The script's "per-evasion macro-F1" tables are not used.** Every strategy except `none` contains only AI sessions, so a 3-class macro-F1 there is capped at about 0.33 by construction.

### 3.15 Part 9: dashboard on v3, and the wild Cowrie case study (run 16)

**Dashboard** (`streamlit run honeyai/dashboard.py`):
- It now shows v3 with the final Model 3: XGBoost 0.644, GRU 0.774, QLoRA 0.796.
- A caption names the data and predictions files.
- Checked by screenshot on 2026-11-18.
- As with Part 8, its XGBoost and GRU use one seed while QLoRA is the 3-seed ensemble, so the dashboard is a demo and §3.10 holds the comparison.

**Wild Cowrie sessions** (`python scripts/parse_cowrie.py external/cowrie/cowrie.json --classify`):
- **The log:** 13,315 events and 1,708 raw sessions from the team's live honeypot (about 5 hours, 3 Nov).
  - 35 sessions never authenticated.
  - 1,663 had fewer than 2 commands.
  - **12 interactive sessions remain.**
- **The classifier is XGBoost only.** Cowrie logs have no command output, so Model 3, which reads responses, cannot classify them.
- **XGBoost's labels:** 9 scripted, 2 human and 1 AI-driven. The same labels came out on two machines.
- **Manual inspection: all 12 are automated bots.** No ground truth exists, so this is a case study, not accuracy.

| Session | XGBoost | What it actually is |
|---|---|---|
| WILD-00001, -02, -07 to -12 (8 sessions) | scripted | A known SSH-key implant bot (`chattr -ia .ssh; lockr …; rm -rf .ssh; echo "ssh-rsa …"`): 3 commands within 1 s |
| WILD-00003 | scripted | Honeypot-detection probes (`printf 'check_…_small_math' "$((357*535+8))"`), about 0.02 s per command |
| WILD-00004 | **AI-driven** (confidence 0.45–0.56) | The same kind of probe script, with one 2–3 s wait for a result |
| WILD-00005 | **human** (1.00) | A shell-capability probe at about 0.02 s per command, then one **146.6 s** wait before `chmod +x /root/.16` (probably a payload upload) |
| WILD-00006 | **human** (0.98) | **The same SSH-key implant bot** as the 8 sessions above, identical commands, but with one 8.4 s pause |

**Reading:** XGBoost leans on timing, so a real bot that pauses once (for a download or a payload) is labelled
human or AI. The clearest example: the same SSH-key implant bot is labelled scripted 8 times when it runs in under 1 s,
and **human** once, when it pauses for 8.4 s. The same timing weakness the simulated data showed appears here on real traffic.

### 3.16 Teammate's original QLoRA script on v3 (run 17)

**Setup.**
- **Declared in TEST_PLAN §4D item 6,** before the run.
- **`scripts/train_qlora_local.py` with its recipe unchanged:**
  - LoRA r=8 on q/v, 15 fixed epochs, LR 5e-5;
  - 512-token training sequences and 480-token test prompts, both truncated from the right;
  - the answer generated as JSON.
- **Run with `run_teammate_v3.py`,** which changes only the data path, the output path and keeps all 3 seeds.
- **Hardware and time:** RTX 3070 Ti, about 37–43 minutes of training per seed.

| Seed | Macro-F1 | Parse failures (of 1,113) |
|---|---|---|
| 42 | 0.546 | 411 |
| 123 | 0.540 | 414 |
| 456 | 0.549 | 402 |
| **Mean ± std** | **0.545 ± 0.004** | **409 (37%)** |

**Against the other LLMs** (same 1,113 v3 test sessions).
- **How the teammate's model is summarised:** a majority vote over its 3 seeds (0.538), with "unknown" when the output could not be parsed or the seeds disagree. Its outputs are labels without probabilities, so `compare_models.py` cannot score it.
- **How the comparison was done:** paired bootstrap over sessions, 3,000 resamples, 95% CIs.

| | Full, all sessions | Real held-out agents (708) |
|---|---|---|
| Teammate recipe (majority vote) | 0.538 | 0.551 |
| **Model 3** − teammate | **+0.258 [+0.224, +0.294], better** | **+0.280 [+0.240, +0.321], better** |
| v4 − teammate | **+0.361 [+0.328, +0.396], better** | **+0.395 [+0.360, +0.434], better** |
| Qwen2.5-7B zero-shot − teammate | −0.216 [−0.263, −0.167], worse | −0.171 [−0.227, −0.115], worse |

Per-seed means: teammate 0.545 ± 0.004, Model 3 0.769 ± 0.069, v4 0.896 ± 0.006.

**Why it scores lower: truncation, as on v2b (§3.5).**
- **Long sessions fail most.** Parse failures rise from 17% for sessions of 5 or fewer commands to 70% for sessions of 10 or more. The right-truncated prompt loses the end of the session and the answer marker.
- **Scripted sessions fail most often (57%),** because they are the longest. That is also why the teammate model's scripted recall is only 18–20%.
- **On the sessions it can parse, its macro-F1 is 0.760** (677 sessions, majority vote). That figure is a diagnostic only: it drops the hardest, longest sessions, so it is not a fair score.
- **It flags simulated humans as AI at a similar rate:** 8.7%, against Model 3's 8.1% and v4's 9.4%.

**Reading:**
- **Same base model, same data, different recipe:** the patched notebook (Model 3) is better by +0.26 macro-F1 on fresh data, and by +0.28 on the clean real-agent subset.
- **This confirms the v2b result on fresh data:** there the gap was 0.513 against 0.719. Most of the gap comes from the right truncation and the unparseable outputs, not from the base model.
- **The teammate recipe still beats the zero-shot 7B model** (+0.216): fine-tuning helps even with the truncation problem.

## 4. Robustness to evasion on v2b: AI recall by strategy (full sessions)

For v3, see §3.14 (per strategy, including real agents), §3.10 and §3.11 (per family) and §5.2 (real agents faking human timing).

| Strategy | In training? | Option 1 (seed 42) | Option 2 (mean of 3 seeds) |
|---|---|---|---|
| none | yes | 0.94 | 0.63 |
| timing_mimicry | yes | 0.88 | 0.81 |
| typo_mimicry | yes | 0.91 | 0.67 |
| shorter_commands | yes | 0.88 | 0.44 |
| hesitation_redundant_exploration | yes | 0.91 | 0.62 |
| avoid_systematic_enumeration | **no (held out)** | 0.96 | 0.85 |
| combined | **no (held out)** | **0.29** | **0.19** |

Full-session confusion matrices (rows are the true class; columns are predicted human / scripted / AI):

- **Option 1, seed 42:**
  - human: [273, 0, 47]
  - scripted: [9, 97, 84]
  - AI: [65, 11, 244]
- **Option 2, seed 42:**
  - human: [292, 2, 26]
  - scripted: [51, 84, 55]
  - AI: [119, 25, 176]

---

## 5. Model 3 against outside solutions

This section answers: **why use Model 3 instead of an approach that already exists?** Models 1 and 2 are
part of this project and are compared in §3; they are not repeated here.

### 5.1 The outside solutions

| Outside solution | What it is | How it is tested here |
|---|---|---|
| **Palisade timing rule** (arXiv 2410.13919) | Published detector for LLM agents on SSH honeypots: an agent replies in **under 1.7 s**. The paper also sends prompt-injection probes. | Applied **literally** to v3: a session is flagged AI if its median gap between commands is under 1.7 s. Prompt injection cannot be tested in our simulator, so only the timing half is evaluated. |
| **Timing classifier** | The same timing signal, but fitted to our training data (depth-3 tree on mean / min / std gap). This is the strongest version of the Palisade timing idea. | `run_baselines.py` (`timing`) |
| **TF-IDF + logistic regression** | The standard off-the-shelf text classifier, run on exactly the same transcripts as the LLM | `run_baselines.py` (`text`, `text-cmds`) |
| **Feature-based gradient boosting** (Alt, RIT thesis 2026: macro-F1 0.809 on its own data) | Published classifier for automated / human / AI-assisted sessions | Its approach is what Model 1 (XGBoost) implements, so it is covered in §3. Its own numbers come from different data and classes, so they are **not** comparable with ours (DATA_PLAN §8). |
| **Off-the-shelf LLM, zero-shot** (TEST_PLAN §4A) | "Just ask an existing LLM": Qwen2.5-7B-Instruct, 14× larger than Model 3, no training | Run once, as declared in TEST_PLAN §4D item 5 (§5.4) |

### 5.2 Against the Palisade timing rule (published)

**Literal rule, v3 test.** The 1,074 sessions with at least 2 commands are scored; one-command sessions have no
gap to measure. Model figures are 3-seed ensembles on the same sessions. Differences are paired bootstrap 95% CIs
(5,000 resamples), with exact McNemar p-values.

| Sessions | Palisade rule: flagged AI | **Model 3** | Model 3 − rule | v4 (post hoc) |
|---|---|---|---|---|
| Real held-out agents, Gemma + Claude (178) | 0.640 | **0.899** | **+0.258 [+0.180, +0.337]**, p < 0.001 | 0.955 |
| ... of which `timing_mimicry` + `combined` (52) | **0.000** | **0.846** | **+0.846 [+0.731, +0.942]**, p < 0.001 | 0.962 |
| All AI (568) | 0.460 | **0.783** | **+0.324 [+0.276, +0.370]**, p < 0.001 | 0.838 |
| Humans, simulated (318): false alarms | 0.075 | 0.079 | +0.003 [−0.035, +0.041], p = 1.0 | 0.091 |
| Scripted (188): wrongly flagged AI | 0.660 | **0.266** | **−0.394 [−0.489, −0.293]** | 0.000 |

**Reading:**
- **Same false-alarm rate, many more agents caught.** At the same false-alarm rate on the simulated humans (7.5% vs 7.9%), Model 3 catches 90% of the real agents nobody designed against; the rule catches 64%.
- **Agents that fake human timing defeat the rule completely.** Real agents told to mimic human timing, or to combine tricks, slow down to 3–21 s per command (`data/v3/DATA_CARD.md`). The rule catches **none** of them; Model 3 catches 85%.
  - The Palisade authors themselves note that agents could add delays. This is that case, measured.
- **The rule cannot tell scripts from agents.** It flags 66% of automated scripts as AI, because scripts are fast too.

**Fitted timing classifier** (the rule's signal, fitted to our data; paired, Holm-corrected; §3.10, §3.11):

| | Full, all sessions | Real held-out agents (708) |
|---|---|---|
| **Model 3** vs timing | **+0.174 [+0.141, +0.206], better** | **+0.211 [+0.175, +0.247], better** |
| v4 vs timing (both gap-augmented) | **+0.152, better** | **+0.171, better** |

### 5.3 Against a standard text classifier (TF-IDF + logistic regression)

| | Full, all sessions | Real held-out agents (708) | After 3 commands |
|---|---|---|---|
| **Model 3** vs text | −0.022 [−0.049, +0.006], **no significant difference** | **−0.055 [−0.086, −0.024], worse** | **−0.052 [−0.085, −0.020], worse** |
| v4 vs text (both gap-augmented) | **+0.045 [+0.028, +0.063], better** | **+0.021 [+0.004, +0.038], better** | **+0.070, better** |

- **Model 3 does not beat a TF-IDF classifier.** It is tied overall and worse on the real held-out agents.
- **v4 does beat it**, including on the clean real-agent subset, when both get the same gap augmentation. v4 is post hoc (§3.8), so this is a confirmed post-hoc finding, not the pre-specified result.

### 5.4 Against an off-the-shelf LLM: zero-shot Qwen2.5-7B-Instruct (run 14)

**Setup.**
- **Declared first:** TEST_PLAN §4D item 5, on 2026-11-17, before the run.
- **Model:** `Qwen/Qwen2.5-7B-Instruct`, with no training, in the v3 notebook with `ZERO_SHOT = True`.
- **Prompt:** the fixed zero-shot prompt with the three class definitions (also in `scripts/zero_shot_llm.py`).
- **Calibration:** domain-context calibration on 60 training transcripts, labels unused.
- **Run once,** with no changes. Qwen is the only judge with no family overlap with v3's agents.
- **Scoring:** the same 1,113 v3 test sessions and the same label scoring as Model 3. `Test-Results-Sabrina/v3_zs_qwen7b.json`.

| | Full, all sessions | Real held-out agents (708) | After 3 commands |
|---|---|---|---|
| Qwen2.5-7B zero-shot, macro-F1 | 0.322 [0.292, 0.350] | 0.379 | 0.255 |
| **Model 3** − zero-shot | **+0.475 [+0.438, +0.511], better** | **+0.451 [+0.404, +0.496], better** | **+0.445 [+0.400, +0.490], better** |
| v4 − zero-shot | **+0.577 [+0.545, +0.610], better** | **+0.566 [+0.527, +0.607], better** | **+0.624 [+0.589, +0.657], better** |

The CIs are paired bootstraps over sessions (3,000 resamples); every McNemar p < 0.001. The zero-shot model is also
significantly worse than every baseline, including the timing rule (−0.301).

**Why it fails.**
- **It calls most sessions human (63%),** including 390 of 605 AI sessions, and only 20% of the held-out real agents are labelled AI (65% are labelled human).
- **It flags 28% of simulated humans as AI,** against Model 3's 8%.
- **Its confidence is poorly calibrated:** ECE 0.46.
- **It is slower:** latency p50 521 ms, p95 809 ms, against Model 3's 168 / 183 ms.

**Reading.** An existing, 14 times larger LLM asked with a definition-based prompt performs at chance level.
Fine-tuning a 0.5B model on labelled sessions is what makes the LLM approach work.

**Limits.**
- One model and one prompt were tried, as declared.
- A larger API model or few-shot prompting might do better. They were not run because of family overlap with v3's agents (TEST_PLAN §4A rule 3).

### 5.5 Real humans: false alarms against the same outside solutions

Commands-only models only (§3.13). This is the share of real people labelled AI; lower is better.

| | KYPO (1,414) | KYPO2 (215, fully independent) |
|---|---|---|
| Timing classifier | 0.062 | **0.028** |
| TF-IDF, commands only | **0.044** | 0.065 |
| **Model 3, commands only** | 0.228 ± 0.093 (seeds: 0.103, 0.257, 0.325) | 0.296 ± 0.165 (seeds: 0.065, 0.386, 0.437) |
| v4, commands only (vs gap-augmented: KYPO timing 0.057, text-cmds 0.065; KYPO2 0.028, 0.070) | 0.093 ± 0.012 | 0.109 ± 0.040 |

- **On real humans, Model 3 raises more false alarms than both simple outside methods** on both sets, and its rate varies widely between seeds. One seed matches TF-IDF; the other two are 4–6 times worse.
- v4 is close to them, but still higher. In paired tests on the seed ensemble it flags significantly more real people than both on KYPO, and than timing on KYPO2; it is tied with TF-IDF on KYPO2 (§3.13).
- On the simulated v3 humans, the same three Model 3 seeds all flag 10–12% as AI. So the simulated test set did not predict this.

### 5.6 What can be claimed

**Supported:**
1. **Model 3 is better than published timing-based detection** (Palisade), both as the literal rule and fitted to our data.
   - On fresh data, it catches far more real LLM agents at the same false-alarm rate on simulated humans.
   - It keeps working when agents fake human timing, which defeats the rule completely.
   - It separates scripts from agents, which the rule cannot.
2. **Model 3 is far better than asking an existing LLM:** +0.475 macro-F1 over zero-shot Qwen2.5-7B-Instruct (14× larger) on fresh data, and +0.451 on the held-out real agents. The zero-shot model performs at chance level.
3. **v4 (post hoc, disclosed)** is better than every outside solution tested **on the v3 test set**, including TF-IDF on the clean real-agent subset. On real humans it is not (point 5).

**Not supported:**
4. **Model 3 is not better than a standard TF-IDF text classifier** (tied overall; worse on real held-out agents).
5. **On real humans**, commands-only Model 3 flags far more people as AI than the timing and TF-IDF methods (+19 to +32 points, significant on both sets). v4 flags slightly more (+2 to +7 points): significant in 3 of 4 comparisons, and tied with TF-IDF on KYPO2.

**Suggested wording for the report:**
> "On fresh data, the pre-specified Model 3 is far better than an existing, 14× larger LLM asked zero-shot
> (macro-F1 0.796 vs 0.322), and detects far more real LLM agents than published timing-based detection
> (Palisade) at the same false-alarm rate, including agents that deliberately mimic human timing, which the timing
> rule misses entirely. It is statistically tied with a standard TF-IDF text classifier. A post-hoc variant (v4),
> which removes a timing shortcut found during development, is better than every outside method tested on that data. On real
> humans (KYPO, KYPO2), Model 3 raises far more false alarms than the simple methods; v4 raises slightly more."

---

## 6. Interpretation (development, v2b)

0. **The patches matter.** On the same data, model and seeds, the teammate's original script scores
   0.513 ± 0.011 and the patched notebook scores 0.653–0.724. Most of the gap comes from truncation
   (§3.5): 37% of the original script's outputs are unparseable.
1. **On full sessions (the primary metric), Model 3 is not the best model.**
   - Option 1 is significantly worse than the text (TF-IDF) baseline.
   - It is tied with text-cmds, gru-cmds and gru.
   - It is significantly better than XGBoost and timing-only.
   - Option 2 is significantly worse than all six baselines.
2. **Training on partial sessions moved accuracy from full sessions to the first command.**
   - At k=1, option 2 scores 0.643 against option 1's 0.366. That ties the best sequence baselines and beats XGBoost and timing by a wide margin.
   - On full sessions, option 2 scores 0.653 against option 1's 0.724.
   - Option 2's k=3 score (0.678) is higher than its full-session score (0.653). The prefix-trained model gets *worse* as it sees more of a session.
3. **The main error is AI sessions labelled human, and the `combined` strategy is the weak spot.**
   - Recall on `combined` is 0.29 for option 1 and 0.19 for option 2. This is the held-out strategy that stacks several human-mimicry tricks.
   - The other held-out strategy, `avoid_systematic_enumeration`, is detected well: 0.85–0.96.
   - Option 1 also confuses scripted with AI: 84 of 190 scripted sessions were labelled AI.
4. **The validation set is over-optimistic.**
   - The random 15% validation split is in-distribution: 0.87–0.97, against 0.65–0.72 on test.
   - It cannot tell which configuration will generalise to the test shift. Shift validation (TEST_PLAN §4B) is the fix.
5. **Calibration is poor (option 2).**
   - ECE on full sessions is 0.23, so the model is overconfident.
   - The 3-seed ensemble scores *below* the per-seed mean (0.634 against 0.653), which happens when overconfident seeds disagree.
   - Report the per-seed mean as the headline, not the ensemble.
6. **Seed stability.**
   - Option 2 is stable at k=3, k=5 and full (std ≤ 0.011).
   - It is unstable at k=1 (std 0.056): seed 123 labelled 127 of 320 AI sessions as scripted.

7. **Shift validation improved option 2 by +0.03 to +0.05 on full sessions.**
   - It also made its 3-command result the best of any model: better than text, gru, gru-cmds and timing, and tied with XGBoost and text-cmds.
   - The error moved:
     - fewer AI sessions are now labelled human;
     - recall on `combined` rose to 0.23–0.60;
     - but about half of the scripted sessions are now labelled AI (92–104 of 190).
   - The shift validation is still near its ceiling (0.95–0.96, against 0.68 on test), because it cannot hold out an AI family.

## 7. Caveats

**Development (v2b), as recorded at the time:**
- ~~Option 1 has only 1 seed.~~ **Resolved:** seeds 123 and 456 were run (§3.9).
- **Both early runs were scored on the test set before a selection rule existed.** From then on, option 1 vs option 2 was decided on shift validation (TEST_PLAN §4B), not by test numbers. Both runs are reported as they are.
- **The two options do not use comparable validation scores.** Option 2's validation included partial sessions, while option 1's used full sessions only.
- **The two options do not have the same epoch budget.** Option 1 had MAX_EPOCHS 3; option 2 had 4.
- **Early-attribution results are tested at 4 points (k = 1, 3, 5, full) without correction across those points.** Holm corrects across the baselines only. The 3-command "LLM better" results have CIs that clear 0 by a margin, except against text (lower bound +0.004), which is borderline.
- **The run-4 test numbers were seen before option 1's shift-validation run.** Under §4B, option 1 vs option 2 is still decided by validation only. Option 2's seed 42 validation score is 0.964.

**Final evaluation (v3 and real humans):**
- **v4 was designed after seeing v2b test errors** (§3.8). Its v3 results are a post-hoc finding confirmed on fresh data. It is compared only against the equally augmented baselines, and Model 3's label does not change (TEST_PLAN §4C).
- **v3's simulated test families were seen during development.** They are new sessions from the same generators as v2b's test set. Only the real-agent families `gemma-3-27b` and `claude-haiku-5.5` are fully clean.
- **Tuning effort is unequal.** Model 3 has had 6 configurations (TEST_PLAN §4B); the baselines use their defaults, plus the gap augmentation for the v4 comparison (Dodge et al., 2019).
- **All humans in v3 are simulated.** No real participant sessions were collected (`docs/HUMAN_COLLECTION.md`).
  - The real-human evidence is KYPO and KYPO2: commands only, different tasks, and no server responses.
  - So the full models, which read responses, are untested on real people.
- **KYPO is not independent of the training data on timing.** The simulated humans' think times were calibrated on KYPO target-host sessions (`scripts/calibrate_human_timing.py`). KYPO2 was not used for anything.
- **Real-human paired tests** use seed ensembles and the deterministic baselines only (timing, text-cmds). The GRU's per-session KYPO predictions were not saved.
- **The Model 3 commands-only configuration was trained twice** (runs 11a and 11b). Only 11b has the KYPO result; the two agree on the test set (full 0.760 vs 0.763).
- **Latency** was measured on an RTX 3070 Ti after 3 commands, not on the T4 named in TEST_PLAN, and no target was fixed in advance.

## 8. Next

Done:
1. Teammate's `train_qlora_local.py`: done (§3.5). Report it as the unpatched reference: 0.513 ± 0.011,
   with 37% parse failures caused by right truncation.
2. Shift-validation runs (TEST_PLAN §4B): done. Option 1 was selected, and its 3 seeds are complete (§3.9).
3. Model 3 and frozen v4 on v3 (TEST_PLAN §4C): done (§3.10, §3.11).
4. Commands-only ablation and KYPO real-human check (TC-D3, TEST_PLAN §4D): done (§3.12, §3.13).
5. Decided: Model 3 is option 1 with shift validation (§3.9), recorded in TEST_PLAN §4B. v4 is reported as a post-hoc experiment (§3.8, §3.11).

Still open:
6. **KYPO2 for the commands-only LLMs** (TEST_PLAN §4D item 3): done for both (runs 12 and 13, §3.13).
7. **Off-the-shelf LLM comparison (TEST_PLAN §4A):** done with zero-shot Qwen2.5-7B (run 14, §5.4). API models were not run (family overlap).

8. **Part 8 and Part 9 on v3** with the final Model 3: done (runs 15 and 16, §3.14 and §3.15).
9. **Teammate's `train_qlora_local.py` on v3:** done (run 17, §3.16): 0.545 ± 0.004, with 37% parse failures. Model 3 is better by +0.258.
10. **Optional:** label the 12 wild Cowrie sessions with the commands-only v4 (needs a GPU), as a case study next to XGBoost.

Declared in TEST_PLAN §3.6 but not run (time). Listed here so they are not silently missing:
- 0.5B vs 1.5B;
- round 0 (non-evasive training) vs round 2;
- training on real agents only vs simulated only vs both.
