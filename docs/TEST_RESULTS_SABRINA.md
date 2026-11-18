# Model 3 (QLoRA LLM): test results (Sabrina)

These are the results of Model 3 runs on the v2b dataset. The protocol, metrics and decision rule are in
[`TEST_PLAN.md`](TEST_PLAN.md), §3, §4 and §4B. The baseline numbers are in TEST_PLAN §5.

- **Environment:** local Windows PC, RTX 3070 Ti (8 GB), PowerShell venv and Jupyter.
- **Code:** `notebooks/qlora_generative_patched.ipynb`.
- **Data:** `data/dataset4_llm_sequences.jsonl` (v2b), 1,482 train and 830 test sessions.
  - Deduplication found 0 duplicates and 0 test sessions identical to a training session.
- **Test set:** 830 sessions: 320 human, 190 scripted and 320 AI.
  - The test set holds out: an unseen AI family (`sim_agent_c`), unseen scripted families, unseen human participants, and two unseen evasion strategies (`avoid_systematic_enumeration` and `combined`).

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
| 7 | 2026-11-08 | **Option 1, shift validation: seeds 123 and 456** (completes Model 3) | identical to run 5 | 123, 456 | shift | Done. **Model 3 = runs 5 + 7** (§3.9) |
| 9 | 2026-11-10 | **Frozen v4 on v3** (TEST_PLAN §4C) | v4 notebook unchanged; `DATA_PATH=data/v3/...`, `RESULTS_DIR=runs/llm_v3` | 42, 123, 456 | shift, jittered | Done (§3.11) |
| 8 | 2026-11-09 | **Model 3 on v3** (TEST_PLAN §4C) | as runs 5 + 7; `DATA_PATH=data/v3/...`, `RESULTS_DIR=runs/llm_v3` | 42, 123, 456 | shift | Done (§3.10) |
| 6 | 2026-11-07 | **v4 notebook**: option 2 + shift validation + gap augmentation (`GAP_AUG`, frac 0.5, p 0.7) + validation jitter + prefix weight 0.3 + loss-weighting and gradient-accumulation fixes | see §3.8 | 42, 123, 456 | shift, jittered | Done. **Designed after seeing test errors** (§3.8) |

Model: `Qwen/Qwen2.5-0.5B-Instruct`, 4-bit NF4, LoRA on all linear layers, class-weighted loss. The model
predicts by scoring the likelihood of each of the 3 labels. Transcripts longer than MAX_LEN lose their
**oldest** commands.

---

## 2. Summary: test macro-F1

Seeded runs report the **per-seed mean ± std**. Option 1 has only one seed so far.

| Model | Full | k=1 | k=3 | k=5 | AI recall (evasive) | Human → AI | ECE (full) |
|---|---|---|---|---|---|---|---|
| QLoRA 0.5B, option 1 (seed 42 only) | **0.724** (95% CI 0.689–0.755) | 0.366 | 0.609 | 0.723 | **0.743** | 0.147 | not recorded |
| Teammate `train_qlora_local.py` (3 seeds) | 0.513 ± 0.011 | — | — | — | not computed | not computed | — (constant confidence) |
| **v4 (post hoc, frozen) on v3** | **0.896 ± 0.006** | 0.704 ± 0.003 | 0.875 ± 0.009 | 0.904 ± 0.012 | 0.828 (ensemble) | 0.095 ± 0.015 | 0.062 ± 0.014 |
| *Best gap-augmented baseline on v3* | *text-cmds 0.858* | *text 0.708* | *text-cmds 0.813* | *text-cmds 0.856* | — | — | — |
| **Model 3 on v3 (fresh test, 1,113 sessions)** | **0.769 ± 0.069** (ensemble 0.796) | 0.386 ± 0.029 | 0.677 ± 0.031 | 0.712 ± 0.038 | 0.780 (ensemble) | 0.080 ± 0.025 | 0.121 ± 0.005 |
| *Best baseline on v3* | *text 0.818* | *text 0.669* | *text 0.751* | *text 0.778* | — | — | — |
| **Model 3: QLoRA 0.5B, option 1, shift validation (3 seeds)** | **0.719 ± 0.036** | 0.391 ± 0.023 | 0.670 ± 0.009 | 0.713 ± 0.014 | 0.705 (ensemble) | 0.137 ± 0.022 | 0.206 ± 0.039 |
| QLoRA 0.5B, v4 (gap augmentation, 3 seeds) | **0.815 ± 0.021** | 0.690 ± 0.037 | 0.793 ± 0.019 | 0.822 ± 0.025 | 0.670 (ensemble) | 0.144 ± 0.066 | 0.113 ± 0.009 |
| *Best baseline with the same gap augmentation* | *text-cmds 0.827* | *text 0.718* | *text-cmds 0.781* | *text-cmds 0.826* | — | — | — |
| **QLoRA 0.5B, option 2, shift validation (3 seeds)** | 0.683 ± 0.003 | 0.655 ± 0.010 | **0.724 ± 0.020** | 0.700 ± 0.013 | 0.736 (ensemble) | 0.154 ± 0.054 | 0.214 ± 0.036 |
| QLoRA 0.5B, option 2 (3 seeds) | 0.653 ± 0.011 | 0.643 ± 0.056 | 0.678 ± 0.004 | 0.670 ± 0.005 | 0.564 ± 0.039 | 0.095 ± 0.028 | 0.227 ± 0.012 |
| *Best baseline at that point (TEST_PLAN §5)* | *text 0.765* | *text-cmds 0.660* | *XGBoost 0.747* | *text 0.746* | *XGBoost 0.738* | *XGBoost 0.099* | — |

- **AI recall (evasive)** is the recall on AI sessions whose `evasion_strategy` is not `none` (288 sessions).
- **Human → AI** is the share of human sessions labelled AI.

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
- **All humans are simulated.** The real-human check (KYPO, collected sessions) has not been run for v4.

---

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

## 4. Robustness to evasion: AI recall by strategy (full sessions)

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

## 5. Interpretation

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

## 6. Caveats

- **Option 1 has only 1 seed.** Its 0.724 is not yet comparable to a 3-seed mean. Seeds 123 and 456 are still to run.
- **Both runs were scored on the test set before a selection rule existed.** From now on, option 1 vs option 2 is decided on shift validation (TEST_PLAN §4B), not by these test numbers. Both runs are reported here as they are.
- **The two options do not use comparable validation scores.** Option 2's validation included partial sessions, while option 1's used full sessions only.
- **The two options do not have the same epoch budget.** Option 1 had MAX_EPOCHS 3; option 2 had 4.
- **Model 3 has had more tuning than the baselines.** It has had 2 configuration trials so far; the baselines use defaults (TEST_PLAN §3.3).
- **Early-attribution results are tested at 4 points (k = 1, 3, 5, full) without correction across those points.** Holm corrects across the baselines only. The 3-command "LLM better" results have CIs that clear 0 by a margin, except against text (lower bound +0.004), which is borderline.
- **The run-4 test numbers were seen before option 1's shift-validation run.** Under §4B, option 1 vs option 2 is still decided by validation only. Option 2's seed 42 validation score is 0.964.
- **All data is simulated (v2b).** The real-human (KYPO) check for Model 3 (commands only, TC-D3) has not been run yet.

- **v4 was designed after seeing test errors** (§3.8). Its improvement must be reported with that disclosure, and against the equally augmented baselines.
- **The baselines were not tuned.** Only the gap augmentation was added to them; Model 3 has had 5 configurations (Dodge et al., 2019).

## 7. Next

1. Teammate's `train_qlora_local.py`: done (§3.5). Report it as the unpatched reference: 0.513 ± 0.011,
   with 37% parse failures caused by right truncation.
2. Shift-validation runs (TEST_PLAN §4B): done. Option 1 was selected, and its 3 seeds are complete (§3.9).
3. Optional: Qwen2.5-1.5B with the winning configuration, kept only if its validation score is higher.
4. Commands-only run with the KYPO real-human check (TC-D3).
5. Decided: Model 3 is option 1 with shift validation (§3.9), recorded in TEST_PLAN §4B. v4 is reported as a post-hoc experiment (§3.8), frozen at commit `8f433ab` for a clean test on new v3 test sessions.
