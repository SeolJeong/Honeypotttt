# HoneyAI datasets: what's in `honeyai_datasets.zip`

Version 2026-11-04. This file is sent separately and is not inside the zip.

| Path in zip | Sessions | What it is | Use for |
|---|---|---|---|
| `data/v2b/` | 2,312 | **Current shared development set.** Simulator data with human timing calibrated to real humans (KYPO) | All model development and comparison now |
| `data/v2/` | 2,311 | Same simulator, original (uncalibrated) human timing | Reference only (earlier numbers) |
| `data/dataset1_sessions.json`, `dataset2_…`, `dataset4_…` (v1) | 208 | Original team data. **Script-generated:** placeholder responses, inserted typos | Retired. History only; never report as participant or LLM-agent data |
| `data/dataset3_auth_noise.csv` | 1,000 records | Stage-1 login-noise filter data. `auth_success` matches the label exactly | Stage-1 filter; don't report its score as a result |
| `external/kypo/kypo_sessions.json` | 1,414 | **Real humans** (KYPO), converted to HoneyAI sessions; commands + timing, **no outputs** | Real-human check of commands-only models; timing calibration |
| `external/kypo/kypo_sessions_ds4_commands_only.jsonl` | 1,414 | The same in LLM text format, commands only | `EXTERNAL_TEST_PATH` in the patched LLM notebook |
| `configs/human_timing.json` | — | Calibrated timing parameters plus their provenance | Reproducing v2b |

In each `data/vX/` folder, `dataset1_sessions.json` is the source. `dataset2_tabular_features.csv` (XGBoost) and `dataset4_llm_sequences.jsonl` (LLM) are derived from it, so **regenerate them, never hand-edit**.

---

## v2 / v2b: simulated controlled sessions

**How it's made:** `scripts/generate_sessions.py`. Every session runs on its own randomised fake Linux host (app name, paths, users, OS, ports, credentials), with 8 tasks from spec §6. Responses come from executing each command on that host, so they're consistent: a path revealed in one response exists when it's used later. Hints vary by task and session. Output is deterministic (seed 2026), and a duplicate check drops repeated sessions and any test session identical to a training one.

**Provenance (state this in the report):**

| `data_source` | Class | How real |
|---|---|---|
| `scripted_client_vs_simulator` | scripted | **Real scripts**: 7 families with genuine control flow |
| `simulated_policy` | human, AI-driven | **Not real people or real LLMs.** Parametric policies encoding the assumptions below |

**Simulated humans (personas SP01–SP20):**
- They follow a revealed hint with a persona-specific probability of 0.45–0.85; otherwise they repeat the failed command, explore, try `man`/`--help` or `sudo`, or search.
- Typos at 3–18% of commands, corrected on the next command about 85% of the time.
- Often use `cd` and relative paths.
- Think time is lognormal per persona: v2 median ~ U(3.5, 10) s, sigma ~ U(0.5, 0.9). **v2b: median ~ U(2, 10) s, sigma ~ U(0.8, 2.2)**, fitted to KYPO. In both, it grows with output length and after errors.

**Simulated AI agents (sim_agent_a/b/c):**
- They follow hints with probability 0.90–0.95; otherwise they run a targeted `find`/`grep`.
- Latency is lognormal around a family median (1.4–3.2 s), plus an output-length term.
- Absolute paths, polished commands; they stop when the task is done.
- Evasion strategies:
  - `timing_mimicry`: human-like gaps that ignore output length and errors (**not** calibrated to real humans)
  - `typo_mimicry`: typos that are always fixed on the next command
  - `shorter_commands`: plain forms
  - `hesitation_redundant_exploration`: 30% redundant commands
  - `avoid_systematic_enumeration`: no `find`/`grep -r`; hint-following drops to 0.75
  - `combined`: all of the above

**Splits (group-disjoint):**
- Humans: SP01–SP12 train, SP13–SP20 test.
- Scripts: `fixed_replay`, `state_machine`, `conditional_error_handling`, `randomised_ordering` train; `human_like_delays`, `alternative_commands`, `retry_on_failure` test only.
- AI: `sim_agent_a/b` with none/timing/typo/shorter/hesitation train. Test: `sim_agent_c` with all 7 conditions (unseen family), and `sim_agent_a/b` with `avoid_systematic_enumeration` and `combined` (unseen strategies).

| | v2 | v2b |
|---|---|---|
| Train (human / scripted / AI) | 480 / 446 / 556 | 480 / 446 / 556 |
| Test (human / scripted / AI) | 319 / 190 / 320 | 320 / 190 / 320 |
| Steps per session | 1–16 (mean 6.2) | 1–16 (mean 6.2) |
| Text baseline macro-F1 | 0.765 | 0.765 |
| XGBoost | 0.677 | 0.694 |
| Timing alone | 0.701 | 0.667 (learned 3-feature tree) / 0.622 (mean gap only) |

**Known gaps:** five AI test conditions have 32 sessions each (below the 40 target).

**v2b calibration target:** KYPO gaps after quick commands on target hosts, p10/p25/p50/p75/p90 = 0.7 / 1.8 / 4.6 / 14.5 / 52.5 s. Calibrated simulation: 0.6 / 1.7 / 5.2 / 15.5 / 48.4 s. **Only timing is calibrated.**

---

## KYPO (real humans, external)

**Source:** Švábenský, Vykopal, Seda, Čeleda. *Dataset of Shell Commands Used by Participants of Hands-on Cybersecurity Training.* Data in Brief 38 (2021). doi:10.1016/j.dib.2021.107398. Zenodo record 8136017 (v4), **CC BY 4.0**. Cite it whenever you use these files.

**Conversion** (`scripts/convert_kypo.py`):
- Bash commands only; Metasploit console commands are excluded by default.
- A new session starts on a host change or after 10 minutes idle.
- Sessions are cut into windows of at most 16 commands. The first command of each segment is dropped, because its gap is unknown.
- `has_typo` is a heuristic: a command followed by a near-identical one.
- IP addresses are dropped.

**Contents:** 267 trainees → 1,414 sessions (357 on target hosts: `server`, `web`, `client`, `laboratory`, reached over SSH; the rest on the trainee's own `attacker` machine), 14,658 commands.

**Limits:**
- No command outputs, so only commands-only models can be evaluated on it.
- Gaps include the run time of long commands.
- The tasks (exploitation training) differ from ours.
- It's an **external check**, never training data.

---

## Not included

- The teammate's 824-session "large" file: generated like v1, with 19 duplicates and 23 test sessions identical to training ones. Development only, if at all.
- Raw KYPO `data.zip`: download it from Zenodo (link in `docs/DATA_PLAN.md` §3A).
- Wild data (Cowrie, 4-month, CyberLab): unlabelled. Download links are in `docs/DATA_PLAN.md`.

## KYPO2: second real-human set (added 2026-11-10)

**Source:** Ošlejšek, Macák, Dočkalová Burská. *Hands-on cybersecurity training behavior data for process mining.*
Data in Brief 52 (2024). doi:10.1016/j.dib.2023.109956. Zenodo record 10298187, **CC BY 4.0**. Cite it whenever you use these files.

- **What is used:** `data1` only. It holds 52 trainees doing the "Locust 3302" insider-attack exercise in the KYPO Cyber Range in November 2022.
  - It has 3,653 command records (2,749 Bash, 904 Metasploit). Only the Bash commands are converted.
- **What is not used:** `data2` (48 trainees, May 2022). All 4,507 of its records are already in KYPO, as the 2022-05-04 training.
- **Overlap with KYPO:** none at record level (0 of 3,653). After conversion, 2 short sessions have the same text as a KYPO session.
- **Format:** the same as KYPO: command, timestamp, host, working directory. **No outputs.**
- **Converted with:**
  ```
  python scripts/convert_kypo.py --zip external/kypo2/data1_commands.zip --out external/kypo2/kypo2_sessions.json
  ```
  `data1_commands.zip` is the `command_histories` folder of `data1.zip`, re-zipped under a training/event path.
  - The result is 215 sessions, 12 of them on target hosts.
  - The gaps between commands are p10 1.0 s, p50 14.2 s, p90 130.4 s.
- **Use:** test-only, commands-only models, real-human false-alarm rate (TEST_PLAN §4D). It was not used for calibration.
- **Not for:** training, or headline accuracy.

