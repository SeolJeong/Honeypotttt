# HoneyAI

Evasion-aware online attribution of SSH attack sessions: **human vs scripted vs AI-driven**.

HoneyAI watches how an attacker behaves inside an SSH honeypot session and labels
it as a human, a script, or an AI agent, updating the prediction after every
command. The hard problem it targets is attribution when AI agents deliberately
imitate humans. See the project proposal for the full framing and literature review.

## Status: proof of concept

The end-to-end pipeline runs on **generated data**, so the whole team can work
without waiting on live infrastructure:

- **Scripted** sessions: generated directly (seven automation families).
- **Human** sessions: currently **synthetic placeholders** for pipeline validation.
  Replace with real consented-participant sessions for reported results.
- **AI-driven** sessions: currently **simulated**. Replace with real-LLM sessions
  (the agent loop is in `honeyai/generators/ai_driven.py`, run with your own keys).

All three emit the same schema, so swapping synthetic data for real data needs no
code change downstream.

## Quick start

```bash
pip install -r requirements.txt
python scripts/build_dataset.py        # writes data/sessions/poc_dataset.jsonl
python -m honeyai.models.xgb_baseline  # Model 1 (XGBoost), leakage-safe macro-F1
python -m honeyai.models.recurrent     # Model 2 (GRU sequence model)
python scripts/online_demo.py          # per-command scoring, terminal
python -m honeyai.dashboard.app        # per-command scoring, web dashboard
```

## Layout

```
honeyai/
  schema.py              # THE data contract: Session / Step (lock before changing)
  harness/tasks.py       # controlled fake-shell task environment (shared by all actors)
  generators/
    scripted.py          # 7 automation families
    synthetic_human.py   # simulated humans (PoC placeholder -> real participants)
    ai_driven.py         # simulated AI + live-API agent loop (-> real keys)
  features/extract.py    # ~25 tabular features for Model 1
  models/
    xgb_baseline.py      # Model 1: XGBoost, grouped leakage-resistant CV
    recurrent.py         # Model 2: GRU on interleaved sequence
    # llm_qlora.py       # Model 3: QLoRA fine-tuned LM              (TODO)
  scoring.py             # shared online-scoring + adaptive-threat score
  features/sequence.py   # per-step encoding for Model 2
  dashboard/app.py       # Gradio per-command scoring dashboard
scripts/
  build_dataset.py       # assemble the full dataset
  online_demo.py         # per-command scoring tracer bullet
data/sessions/           # generated sessions (JSONL)
```

## Evaluation discipline (do not weaken)

- Splits are **grouped** so no participant / script family / model family straddles
  train and test. This prevents the model from memorising an actor instead of
  learning behaviour. `honeyai/models/xgb_baseline.py` shows the pattern.
- Report **macro-F1** (classes are imbalanced), plus per-class precision/recall.
- The central experiment is **commands-only vs commands+responses**, and model
  behaviour **under evasion** (AI imitating humans). Design for these from the start.

## What's next

- Model 2 (recurrent sequence model) and Model 3 (QLoRA LM) on the interleaved
  command-response view (`Session.to_model_text`).
- Cowrie log parser -> schema, for unlabelled wild-traffic validation.
- Gradio dashboard driving `scripts/online_demo.py` logic.
- Replace synthetic human + simulated AI with real data.
