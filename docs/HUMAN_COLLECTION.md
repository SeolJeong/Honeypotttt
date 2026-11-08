# Collecting real human sessions for v3

Real people solve short tasks on the **simulated** Linux server, the same one the AI agents attack.
These sessions are the v3 real-human data. Nothing touches a real machine.

## Before you start
- **Ethics:** check with your lecturer whether a consent form or approval is needed (spec §6). The script
  shows a consent notice and records nothing unless the participant types `yes`.
- **Who:** ideally **6 or more people who are not on the team**. Team members know how the simulator works, so their
  sessions are not typical. If team members take part, give them IDs from the `P19x` range so they can be
  identified and excluded later.
- **IDs:** anonymous IDs only, never names. Use `P101`, `P102`, … in the order people take part, and
  keep a private note of which ID went to whom only if your ethics process requires it.
- **Time:** about 15–30 minutes per person for all 8 tasks.

## Setup (once, on the PC used for collection)
Use the repo with the updated `honeyai/sim/environment.py` (it fixes a crash on commands like `ls |`).
```powershell
cd C:\Side-Quest\Honeypotttt
.\.venv\Scripts\Activate.ps1
python scripts/collect_human.py --participant P000 --contributor SA --tasks identify_os_version
```
That last line is a test run: type `yes`, try a command or two, then `quit`. Afterwards, delete `contrib\human_P000.jsonl`.

## Running a participant
```powershell
python scripts/collect_human.py --participant P101 --contributor SA
```
Then hand over the keyboard. Tell them only this:
> "You're logged into a Linux server as user `deploy`. Each task says what to find. Use normal shell
> commands. Type `skip` to give up on a task, or `quit` to stop. There is no time limit, so work as you normally would."

**Do not** coach them, suggest commands, or let them search the web or ask an AI. Their own unaided behaviour
is what we're measuring. Each session is saved to `contrib\human_P101.jsonl`.

## What is recorded
- the commands they type;
- the simulated output;
- the time between the prompt appearing and Enter;
- task progress.

No names, keystrokes or IP addresses are recorded.

## After collection
Send the `contrib\human_P*.jsonl` files to whoever builds v3.

**Split rule (fixed now, before any data exists):**
- participants with odd IDs (P101, P103, …) go to **test**;
- participants with even IDs go to **train**;
- team members (`P19x`) go to **train** only.
