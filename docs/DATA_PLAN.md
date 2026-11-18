# HoneyAI: data plan (updated 2026-11-04, v2b + KYPO)

Every number and link in this file was re-checked against the actual files or sources on 2026-11-03/04. Items marked **NOT BUILT** don't exist yet. Nothing here has been committed.

---

## 0. One thing this plan cannot do

No data plan can *guarantee* the LLM beats the baselines. If we shape the data or the evaluation so that it does, for example by tuning on the test set, giving the LLM more tuning effort than the baselines, or generating data that favours it, the comparison stops meaning anything. A marker can spot that, and it's the one way to lose the "fair comparison" marks completely.

What this plan *does* do:
1. **Give the LLM its best legitimate chance.** It builds data where the LLM's claimed advantage, reading how each command reacts to the previous response, actually exists and matters.
2. **Make the result trustworthy whichever model wins.** Spec §19 gives you a strong claim for every outcome.

---

## 1. Status at a glance

| Item | Status | Where |
|---|---|---|
| v1 data (208 sessions) | Exists. **Produced by a script, not real people or LLMs** (one-letter inserted typos, `[OK]` placeholder responses, progress = step ÷ total). Don't report it as participant or agent data. | `data/` |
| v2 simulated data (2,311 sessions, deduplicated) | **Done**, regenerates identically | `data/v2/`, `data/v2/DATA_CARD.md` |
| Simulator + actors | **Done** | `honeyai/sim/` |
| Real LLM-agent recorder | **Done**: retries, refusal handling, served-model and provider recorded. Tested against a fake server, **not yet against a real model** | `scripts/run_llm_agent.py` |
| Multi-model runner (OpenRouter), resumable, cost estimate | **Done**, tested in Docker against a fake server | `scripts/run_agent_matrix.py`, `configs/agents.example.toml` |
| Docker setup | **Done**: image built and tested here; Docker Desktop not tested | `Dockerfile`, `docker-compose.yml`, `docs/DOCKER.md` |
| Real human recorder | **Done**, tested with scripted input | `scripts/collect_human.py` |
| Merge + validation + leakage warnings | **Done**, tested | `scripts/merge_sessions.py` |
| Teammate brief | **Done** | `contrib/TEAMMATE_PROMPT.md` |
| QLoRA LLM (Model 3) | **Code done**, CPU smoke test passed, **not yet trained on a GPU** | `honeyai/models/qlora_cls.py` (classification head), `notebooks/qlora_colab.ipynb` |
| Text floor baseline | **Done** | `scripts/text_baseline.py` |
| Duplicate / train-test leakage check | **Done**, built into generation and merging | `honeyai/sim/recording.py` (`dedupe`) |
| Team's generative QLoRA notebook (patched) | **Done**. CPU smoke test passed with Qwen's real tokenizer and a tiny random model (the weights CDN is blocked here). **Not yet run on a GPU with real weights** | `notebooks/qlora_generative_patched.ipynb` |
| Team's "large" dataset (824 sessions) | Same generator as v1 (canned responses, 1–5 steps); 19 duplicates plus 23 test sessions identical to training ones. Development data only | teammate |
| Timing-rule baseline (Palisade-inspired; promised in your proposal) | **NOT BUILT** | — |
| Zero-shot / off-the-shelf LLM baselines | **Done**: open models via `ZERO_SHOT = True` in the patched notebook (scoring tested with real Qwen 0.5B weights on CPU); API models via OpenRouter (tested against a fake server) | `notebooks/qlora_generative_patched.ipynb`, `scripts/zero_shot_llm.py`, `scripts/compare_models.py --others` |
| KYPO converter + real-human check | **Done** (1,414 real-human sessions; results in §4A) | `scripts/convert_kypo.py`, `scripts/eval_kypo.py` |
| Cowrie log parser (wild data) | **NOT BUILT** | — |
| Simulated-human timing recalibration | **Done** → `data/v2b` (§5A) | `scripts/calibrate_human_timing.py`, `configs/human_timing.json` |
| Colab notebook for KYPO + calibration | **Done**, replayed end to end from a fresh unzip | `notebooks/kypo_calibration_colab.ipynb` |
| Shortcut-audit script | **NOT BUILT** (run by hand once; results in §5) | — |
| Agent `--memory` option, prompt variants | **NOT BUILT** | — |
| Extra tasks (8 → 10–12) | **NOT BUILT** | — |
| XGBoost: participant-grouped CV on v2, missing proposal features | **NOT BUILT** | `honeyai/models/xgb_baseline.py` |

---

## 1A. Team alignment: one dataset, one split, one format

**Everyone trains and evaluates on the same files.** Results computed on different data can't go in the same table.

### Which data is which
| Data | Status | Use for |
|---|---|---|
| `data/` (v1, 208 sessions) | Script-generated, canned responses | **Retired.** Nothing new; keep only for history |
| Teammate's `dataset4_llm_sequences_large.jsonl` (824) | Same generator as v1; 19 duplicates + 23 test sessions identical to train | **Retired** for results; quick debugging only |
| `data/v2/` (2,311, deduplicated) | Simulator with the original human timing | Kept for reference (earlier numbers) |
| **`data/v2b/`** (2,312, deduplicated) | Same simulator, **human timing calibrated to real humans (KYPO)** | **The current shared development set.** All models run on this now |
| `data/v3/` | Built later by `merge_sessions.py` (v2 + real agents + real humans) | **The dataset for reported results** |
| `external/` (KYPO, 4-month, CyberLab, Cowrie) | Real but unlabelled / different environment | Calibration, real-human check, wild comparison. **Never merged into v2/v3** |

**Re-run needed:** the XGBoost 0.822 and GRU 0.855 in the teammate's summary must be re-computed on `data/v2b/` with its train/test split before they go next to any LLM number.

### Format contract (don't change without telling everyone)
- **Sessions:** `dataset1_sessions.json` (full traces). `dataset2_tabular_features.csv` (XGBoost) and `dataset4_llm_sequences.jsonl` (LLM) are **derived** from it. Never hand-edit them; regenerate.
- **Required fields per session:** `session_id`, `actor_label` (`human` / `scripted` / `AI-driven`), `task`, `split`, `evasion_strategy`, `generator_family`, `participant_id` (humans), `data_source`, `interactions`.
- **`dataset4` line:** `{"session_id", "split", "actor_label", "evasion_strategy", "generator_family", "participant_id", "task", "total_steps", "interleaved_text"}`. Text format: `[GAP X] CMD: …` / `RESP [category]: …`.
- **Splits are fixed by the merge script**, not chosen per model. Validation is carved from `train` inside each training script, the same way for every model.

### Names and IDs (fixed)
| Thing | Value |
|---|---|
| AI family labels (`--family`) | `llama3.1-8b`, `gpt4o-mini`, `mistral-7b` (+ any extra from `configs/agents.toml`). **Held-out test family: decide once, write it here: `________`** |
| Participant ID blocks (never names) | Currently **one participant: P006**. If more join later: P1: P006–P009 · P2: P010–P013 · P3: P014–P017 · P4: P018–P021 · P5: P022–P025 |
| Held-out test participants | **P006 (test only; see §4A)**. If more participants join, hold out at least a third of them |
| Held-out evasion strategies | `avoid_systematic_enumeration`, `combined` (merge default) |
| Contributed files | `contrib/agent_<family>.jsonl`, `contrib/human_<participant>.jsonl`. Never edited by hand |

### Folder layout
```
data/v2b/       shared dev set (python3 scripts/generate_sessions.py --human-timing configs/human_timing.json --out data/v2b)
data/v2/        previous version (original timing), reference only
data/v3/        merged real + simulated (python3 scripts/merge_sessions.py ...)
contrib/        raw contributed sessions (.jsonl): back these up daily
external/       downloaded public datasets (not in git)
  kypo/  ssh4month/  cyberlab/  cowrie/
runs/           model outputs (not in git)
```

### Before anyone reports a number
1. Run it on `data/v2b` (now) or `data/v3` (final), using the split in the file.
2. Use 3 seeds and report mean ± std, not the best seed.
3. Choose settings on validation, never on test.
4. Say which file, which commit or date, and which settings, in the results table.

---

## 2. Ground rules (these keep the results valid)

1. **Report test results on real data wherever it exists.** Simulated sessions can go into *training*. For AI and scripted, the reported test set is real agents and the real scripts. For humans, with only one real participant, report the simulated test personas **labelled as simulated**, plus the KYPO check and the one-person case study (§4A).
2. **One environment for labelled data.** Every labelled class runs against the same simulator. Never pool Cowrie, KYPO or other external data into training. Otherwise the model learns "which dataset is this from?".
3. **Split by group:** participants for humans, model families for AI, families for scripts. `merge_sessions.py` warns when this is violated.
4. **Choose settings on validation data only.** Pick learning rate, LoRA rank, 0.5B vs 1.5B and so on from validation scores. Run the test set once per final configuration.
5. **Give every model the same tuning budget.** If the LLM gets 10 tuning trials, XGBoost and the GRU get 10 too.
6. **Nothing hand-written.** Never write, edit or AI-generate a session file and label it as human or agent data.
7. **Ethics first.** Don't record any participant until the lecturer confirms what's needed (spec §6).

---

## 3. Data sources and their roles (all verified 2026-11-03)

| Source | Verified facts | Role | Not for |
|---|---|---|---|
| **Our simulator (v2)** | 2,311 sessions; train 1,482 / test 829; 1–16 steps (mean 6.2) | Training, development, ablations | Reported human or AI results |
| **Real LLM agents** (teammates) | Not collected yet | **Training and test** (held-out family) | — |
| **Real humans** (teammates) | Not collected yet | **Training and test** (held-out participants) | — |
| **Teammate's Cowrie log** | 18,775 events, 2,254 sessions, 45 source IPs, about 5 h (3 Nov 00:00–05:08 UTC). 2,124 sessions ran a single command (`uname -s -v -n -r -m`). Only 25 sessions ran more than 2 commands, all automated (median gap 0.021 s). **No outputs in the JSON log.** | Wild comparison, case studies, demo | Training, accuracy |
| **KYPO shell commands** (Švábenský et al., *Data in Brief* 2021; Zenodo 8136017, **CC BY 4.0**) | I count **21,108 records / 267 trainee files** (the record states 21,459 / 275, so recount before citing). Fields: command, timestamp, working dir, host, user. **No outputs.** Offensive training tasks (`nmap`, `ssh`, `fcrackzip`, `john`, 5,025 Metasploit commands). Median 65 commands per trainee. Raw median gap 12.8 s, **but that includes the run time of long commands** (scans, cracking). After quick commands on target hosts: p10/p50/p90 = **0.7 / 4.6 / 52.5 s**. Converted: 1,414 sessions (357 on target hosts reached over SSH) | (1) Calibrate simulated-human timing. (2) External **real-human** check of the commands-only models | Training; headline accuracy (different tasks) |
| **4-month SSH botnet dataset** (Boiko & Niiakyi, Zenodo 19815504, v2.1 Apr 2026, **CC BY 4.0**) | 145,425 events, 27 Jul–14 Nov 2025, Paramiko honeypot; fields `id, timestamp, session_id, ip, port, event_type, message, command, level`; 28 interactive shell sessions; no responses, no labels | Wild distribution comparison (the spec's §14 "4-month dataset") | Training, accuracy |
| **Palisade `LLM-Honeypot-Logs`** (Hugging Face) | Exists; **gated (manual approval)**; raw daily `cowrie.json` files from around Sept 2024; no labels visible | Optional case studies | Training, accuracy |
| **CyberLab honeynet** (Zenodo 3687527, v1.0, **CC BY 4.0**) | Cowrie on about 50 nodes, May 2019–Feb 2020; **293 daily `.json.gz` files, 10.02 GB** (verified via the Zenodo API); IPs pseudonymised | Optional pre-LLM-era wild comparison (download a week, not all of it) | Training, accuracy |
| `hotal/honeypot_logs` (Hugging Face) | From the *LLM Honeypot* paper (IEEE CNS 2024); **rows not inspected** (viewer blocked here); no licence listed | **Maybe:** realistic command/output pairs for the simulator, if that's what it contains | Labels (it has none) |
| **Alt RIT thesis** | Abstract verified (numbers in §8). **Data release unknown**: the PDF is behind a Cloudflare check | Related work, quoted precisely | A score to "beat" |

---

## 3A. External datasets: exact download links (all tested 2026-11-04)

| Dataset | Download | Size | Licence / cite |
|---|---|---|---|
| **KYPO** shell commands (real humans) | https://zenodo.org/records/8136017/files/data.zip?download=1 | 0.4 MB | CC BY 4.0. Cite Švábenský et al., *Data in Brief* 38 (2021), doi:10.1016/j.dib.2021.107398 |
| KYPO analysis tools (optional) | https://zenodo.org/records/8136017/files/toolset.zip?download=1 | 0.1 MB | same |
| **4-month SSH botnet** (CSV) | https://zenodo.org/records/19815504/files/honey_csv_v2.1.tar.gz?download=1 | 7.5 MB | CC BY 4.0. Cite Boiko & Niiakyi, doi:10.5281/zenodo.19815504 |
| 4-month SSH botnet (SQLite) | https://zenodo.org/records/19815504/files/honey_db_v2.1.tar.gz?download=1 | 7.9 MB | same |
| 4-month SSH botnet README | https://zenodo.org/records/19815504/files/README.md?download=1 | small | same |
| **CyberLab honeynet** (one day) | `https://zenodo.org/records/3687527/files/cyberlab_YYYY-MM-DD.json.gz?download=1` | 25–60 MB per day; 10 GB in total | CC BY 4.0. Cite the Zenodo record (doi:10.5281/zenodo.3687527) |

Record pages: [KYPO](https://zenodo.org/records/8136017) · [4-month](https://zenodo.org/records/19815504) · [CyberLab](https://zenodo.org/records/3687527)

```bash
mkdir -p external/kypo external/ssh4month external/cyberlab
curl -L -o external/kypo/data.zip "https://zenodo.org/records/8136017/files/data.zip?download=1"
curl -L -o external/ssh4month/honey_csv_v2.1.tar.gz "https://zenodo.org/records/19815504/files/honey_csv_v2.1.tar.gz?download=1"
curl -L -o external/ssh4month/README.md "https://zenodo.org/records/19815504/files/README.md?download=1"
for d in 2019-09-03 2019-09-04 2019-09-05 2019-09-06 2019-09-07 2019-09-08 2019-09-09; do
  curl -L -o external/cyberlab/cyberlab_$d.json.gz "https://zenodo.org/records/3687527/files/cyberlab_$d.json.gz?download=1"
done
```
Keep `external/` **out of git** and out of `data/`.

---

## 4. Targets (how much)

What drives the numbers is the precision of the test set. At 56 test sessions the macro-F1 confidence interval was about ±0.10, with 0.796 [0.687, 0.890]. To measure one condition's recall to roughly ±0.15 you need **at least 40 test sessions per condition**.

| Class | Groups | Sessions | Notes |
|---|---|---|---|
| Human (real) | **1 participant** (current reality; see §4A) | 8–24 | **Test only**, never training. 8 tasks, optionally a second and third round on fresh scenarios |
| AI (real) | **3 families**: `llama3.1-8b`, `gpt4o-mini`, `mistral-7b`; 1 held out | about 110–170 per family | 7 conditions × 8 tasks × 2–3 reps |
| Scripted | 7 families (done) | match the other classes | real scripts, `generate_sessions.py` |
| Simulated human/AI | — | v2 has 1,800 | training only |

**Current v2 test strata (verified):** AI `none`, `timing`, `typo`, `shorter` and `hesitation` have **32 each, which is below the 40 target**. `avoid_systematic_enumeration` and `combined` have 80 each. Humans: 8 test personas.

---

## 4A. Plan with only one real human participant

One person can't support "accuracy on unseen humans". Your spec (§6) covers this case: **say it's a controlled feasibility study and limit the human-class claims to match.** Here's how the human class is handled:

| Role | Data | What you can claim |
|---|---|---|
| Training humans | **Simulated personas** (v2), recalibrated to KYPO timing (step 7) | Nothing on their own: they're simulated |
| Main test, human class | Simulated test personas (SP13–SP20) | "On simulated humans…", always labelled simulated |
| **Real-human check #1** | **KYPO**: 267 real trainees, commands + timestamps, no outputs | Human false-attribution rate of the **commands-only** models (`--no-responses`) on real people. **This is the main real-human evidence** |
| **Real-human check #2** | Your 1 participant: 8 sessions (up to 24 with fresh scenarios) | A **case study**: report raw counts ("7 of 8 sessions labelled human"), not rates, and no confidence intervals |

Rules for the one participant:
- **Test only.** Never put them in training, otherwise nothing real is left to test on.
- **One ID, used once** (e.g. P006). Never record the same person under several IDs to look like more participants.
- If they're a teammate, they know the tasks, which biases the data. Say so in the report, or use an outsider if at all possible.
- Later rounds: they'll have learned the tasks, so record which round each session was (put it in `--contributor`, e.g. `P1-round2`) and expect them to look more efficient.

**Real-human check results (2026-11-04), commands-only models.** All KYPO sessions are real people, so "AI" is always wrong:

| Model | Trained on | Labelled AI: all KYPO (n=1,414) | Labelled AI: target hosts (n=357) | Labelled AI: simulated test humans |
|---|---|---|---|---|
| Text (TF-IDF, commands + timing) | v2 | 6.6% | 12.9% | 12.5% |
| Text | **v2b** | 8.0% | 15.7% | 16.6% |
| Timing (mean gap) | v2 | 5.2% | 15.4% | 9.7% |
| Timing | **v2b** | 5.0% | 15.1% | 14.7% |

With v2b, simulated and real humans get falsely flagged at similar rates on target hosts (about 15–17%). That's mild evidence that the calibrated personas resemble real people. Caveats to report: different tasks from ours; KYPO gaps include command run time, which pushes real people towards "human"; the AI share varies by KYPO training (0–13%). For the LLM, run the patched notebook with `COMMANDS_ONLY = True` and `EXTERNAL_TEST_PATH` set to the KYPO commands-only file.

Still worth trying: even **3–5 outside people** (classmates, about 20 minutes each) would allow a leave-one-person-out evaluation, which is a much stronger claim than one person. Ask, but don't let it block anything.

---

## 5. Audit results on v2 (re-run 2026-11-03 after the shell fixes)

| Check | Result | What it means |
|---|---|---|
| Text baseline (TF-IDF + LR), commands + responses | macro-F1 **0.765**; AI recall on `combined` **0.10** | Evasion clearly works; there's room for better models |
| Text baseline, commands only | 0.746 | Bag-of-words gains almost nothing from responses. That's expected, because it can't model "the next command used the previous output". The LLM should do better here if the hypothesis holds |
| Text baseline, round 0 (non-evasive training only) | 0.694 | Round 0 → round 2 effect exists |
| XGBoost (Model 1) | test **0.677** vs CV 0.945 | Large shift to unseen families and strategies. The CV is inflated because v2's `dataset2` has no `participant_id`, so humans fall back to grouping by session (**to fix**) |
| **Single feature: mean inter-command gap** | macro-F1 **0.701** (depth-3 tree) | **Timing alone beats full XGBoost.** The Palisade-style timing baseline will be strong; the LLM must beat it, not just XGBoost |
| Simulated timing vs KYPO | v2 simulated humans p10/p50/p90 **2.1 / 6.4 / 19.1 s**; KYPO real humans (quick commands, target hosts) **0.7 / 4.6 / 52.5 s** | **Correction:** an earlier version said simulated humans were too *fast* (6.8 vs 12.8 s). That compared against KYPO gaps that include scan run time. The real difference is **spread**: real people are far more variable. Fixed in v2b (§5A) |
| Simulated AI timing mimicry | median 5.90 s | Close to simulated humans, as intended |

---

## 5A. v2b: calibrated human timing (run 2026-11-04)

`scripts/calibrate_human_timing.py` grid-searched the persona think-time ranges against KYPO (quick commands, target hosts). Result: `median ~ U(2, 10)`, `sigma ~ U(0.8, 2.2)`; quantile error 0.542 → **0.009**. Fresh-seed check: p10/p25/p50/p75/p90 = 0.6 / 1.7 / 5.2 / 15.5 / 48.4 s vs KYPO 0.7 / 1.8 / 4.6 / 14.5 / 52.5 s. **Only timing is calibrated**; typos, hint-following and exploration are still the simulator's assumptions.

| Check on v2b | Result | vs v2 |
|---|---|---|
| Text baseline, commands + responses | **0.765**; AI recall on `combined` 0.17 | 0.765 |
| Text baseline, commands only | 0.737 | 0.746 |
| Text baseline, round 0 | 0.693 | 0.694 |
| XGBoost | **0.693** (CV 0.965, still inflated) | 0.677 |
| **Mean gap alone** | **0.622** | 0.701: timing is much less of a shortcut once human timing is realistic |
| AI recall under `timing_mimicry` (text baseline) | 0.91 | 0.75: the simulated agents' "human-like" timing is now *narrower* than real humans', so it stands out. **Which evasions work depends on the simulator's assumptions; say so in the report** |

## 6. Steps

### Phase A: this week (start the slow things)
1. **Ethics:** confirm with the lecturer whether consent forms or approval are needed. Nothing human-related starts before this.
2. **Send `contrib/TEAMMATE_PROMPT.md`** to teammates. Assign the families exactly: `llama3.1-8b`, `gpt4o-mini`, `mistral-7b`, or via OpenRouter in `configs/agents.toml` (see `docs/DOCKER.md`). Assign the participant-ID blocks from §1A.
3. **Pilot one real agent first** (you, on the 3070):
   ```bash
   ollama pull llama3.1:8b
   python3 scripts/run_llm_agent.py --base-url http://localhost:11434/v1 --api-key-env NONE \
       --model llama3.1:8b --family llama3.1-8b --evasion none --reps 1 --contributor <you>
   ```
   Read 3–4 sessions by hand. Check that the agent produces sensible commands and that the JSON is parsed correctly. **Do this before teammates spend time on full runs.**
4. **Cowrie:** ask your teammate to keep the sensor running for weeks and **enable TTY logs** (`var/lib/cowrie/tty/`) so outputs are kept. Anonymise IPs in anything you show.
5. **Download the external datasets** with the commands in §3A.
6. **Align the team on §1A:** everyone switches to `data/v2b/` now. The teammate re-runs XGBoost and the GRU on it, and runs `notebooks/qlora_generative_patched.ipynb` on `data/v2b/dataset4_llm_sequences.jsonl`, once normally and once with `COMMANDS_ONLY = True` plus the KYPO file as `EXTERNAL_TEST_PATH`.

### Phase B: improve the simulator (before scaling anything)
7. ~~Recalibrate simulated-human timing~~ **Done** (§5A); run `notebooks/kypo_calibration_colab.ipynb` to reproduce.
8. **Add 2–4 safe tasks** (spec allows 8–12), e.g. find which cron job runs a script, find what's filling the disk, find failed logins in a log. **NOT BUILT.**
9. **Raise each test condition to at least 40 sessions.** Increase the `sim_agent_c` reps in `generate_sessions.py`'s `plan()`.
10. **Agent variety:** add the `--memory N` and prompt-variant options to `run_llm_agent.py`, and write *new* evasion wordings that are used only in testing (round 2 must never test on the evasion examples used for retraining). **NOT BUILT.**
11. Regenerate after any simulator change: `python3 scripts/generate_sessions.py --human-timing configs/human_timing.json --out data/v2c`, then re-run the §5/§5A audit.

### Phase C: collect real data
12. **Agents:** each teammate runs their family with `--evasion all --reps 2` (about 112 sessions), more if time allows.
13. **Humans:** run `python3 scripts/collect_human.py --participant P0XX --contributor <initials>`, unassisted. Mix experience levels. No teammates as participants.

### Phase D: merge and audit
14. Merge:
    ```bash
    python3 scripts/merge_sessions.py contrib/*.jsonl --base data/v2b/dataset1_sessions.json \
        --test-participants <5-6 held-out IDs> --test-families mistral-7b --out data/v3
    ```
    The defaults also hold out `avoid_systematic_enumeration` and `combined` for testing. Fix every leakage warning.
15. **Shortcut audit:** for each single feature or metadata field (session length, first command, task, mean gap), train a depth-3 tree and check its test macro-F1. If one comes close to the full models, find out why before training. **Script NOT BUILT.**
16. Check every reported test condition has at least 40 sessions.

### Phase E: train and compare (the fair way)
17. **Baselines, all on the same splits and metrics:**
    - timing rule (**NOT BUILT**)
    - text baseline: `python3 scripts/text_baseline.py --data data/v3/dataset1_sessions.json`
    - XGBoost (fix the grouped CV first)
    - GRU
    - zero-shot Qwen2.5-7B-Instruct (**NOT BUILT**)
18. **LLM runs:**
    ```bash
    python3 scripts/train_qlora.py --data data/v3/dataset1_sessions.json               # main, 3 seeds
    python3 scripts/train_qlora.py --data data/v3/dataset1_sessions.json --no-responses  # central ablation
    python3 scripts/train_qlora.py --data data/v3/dataset1_sessions.json --round 0       # before retraining
    python3 scripts/train_qlora.py --data data/v3/dataset1_sessions.json \
        --model Qwen/Qwen2.5-1.5B --batch-size 4 --grad-accum 4                          # scale comparison
    ```
    The team's patched generative notebook (`notebooks/qlora_generative_patched.ipynb`) runs on the same `dataset4` file. Run both on the same split and keep whichever validates better; report both if both are kept.
19. **Training-data experiment** (for the generated-data bonus): train on real only, simulated only, and simulated + real; test all three on the same real test set.
20. Report mean ± std over 3 seeds and bootstrap CIs. Say a model "beats" another only when the difference is outside the CIs.

---

## 7. What legitimately helps the LLM (and what would invalidate it)

**Legitimate. Do these:**
- **Longer, response-rich sessions**, so there is adaptation to detect. Already the case in v2 (mean 6.2 steps vs about 3 in v1).
- **Training on partial sessions** (already the default), which matches the online scoring use case.
- **Diverse real agents and humans**, so the model learns behaviour instead of one generator's quirks.
- **Proper validation-based tuning** of LR, epochs, LoRA rank and model size (0.5B vs 1.5B).
- **The commands-only vs commands+responses ablation:** it's the evidence for *why* the LLM wins, if it does.

**Invalidating. Don't do these:**
- Tuning on, or repeatedly checking, the test set.
- Giving the LLM more tuning effort than the baselines.
- Designing simulator behaviour after seeing which model it favours.
- Dropping test conditions where the LLM does badly.
- Reporting simulated-data results as human or AI results.

If the LLM doesn't win, report it honestly. Spec §19: *"response-conditioned sequential information, not model scale, was the primary source of robustness"* is still a strong result.

---

## 8. Verified facts for the report

**Alt (RIT thesis, 2026), from the abstract:**
- 281,621 sessions over 2 months, collected with Cowrie and ELK. Classes: Hydra (automated), human, and AI-assisted (**PentestGPT/ChatGPT**, i.e. a human using AI tools).
- 24 features, 8 classifiers, 5-fold stratified CV **with SMOTE**; raw class imbalance about 6,231:1.
- Results: macro-F1 **0.809** (Gradient Boosting and Random Forest); AI-assisted F1 **above 0.888** in the top three models; human precision **1.000**; best human leave-one-out F1 **0.732** (KNN).
- Quote each number with its condition. Never treat them as a target to beat on our data.

**How we differ from Alt** (say this explicitly):
- autonomous agents imitating humans, vs humans assisted by AI
- the Hydra class is handled by our Stage-1 filter instead
- no SMOTE on sequences
- online partial-session scoring
- command–response modelling

**Palisade** (arXiv 2410.13919):
- Prompt-injection probes, then timing to confirm.
- The blog says LLM agents usually reply in **under ~1.5 s**; the paper's detection rule uses **under 1.7 s**. State which one you cite.
- The paper's own GPT-4o test: mean 1.20 s, median 1.06 s.
- The authors themselves note that agents could add delays to mimic humans. That supports your evasion-aware motivation, so cite it.
- *Verified via search summaries; arxiv.org isn't reachable from this session, so re-check against the PDF.*

**References to verify before submission:** I confirmed [1] (Alt) and the KYPO dataset. I didn't open the Aalborg thesis [2], VelLMes [5] or AdvancedShelLM [6]; check that their links resolve.
