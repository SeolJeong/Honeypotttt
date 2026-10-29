"""Model 3: language model over the interleaved command-response transcript.

Two backends, selected by --backend:

  tfidf  (default, CPU, runs now): a word/char n-gram TF-IDF + logistic-regression
          classifier over Session.to_model_text(). This is a runnable STAND-IN so
          the three-model comparison produces numbers without a GPU. It is a
          language-based model (reads the text), but it is NOT the fine-tuned LLM.

  qlora  (GPU, run on Colab / CUDA box): the real deliverable. 4-bit QLoRA
          fine-tune of a 0.5-1.5B instruction model that reads the partial
          transcript and emits a structured class. Requires transformers, peft,
          bitsandbytes and a GPU, so it does not run in the CPU PoC container;
          run it where you have the GPU and drop its reported numbers into the
          comparison in place of the tfidf stand-in.

Both use the SAME leakage-resistant grouped CV as Models 1 and 2, so results are
directly comparable.

Run (CPU stand-in):  python -m honeyai.models.llm_model
Run (GPU, real):     python -m honeyai.models.llm_model --backend qlora
"""
from __future__ import annotations

import argparse
import os
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from honeyai.schema import read_sessions, Session, Step
from honeyai.schema import ActorLabel, ResponseCategory, ThinkTimeBin, EvasionStrategy
from honeyai.models.xgb_baseline import LABELS, LABEL_IDX, _groups

DATA = os.path.join("data", "sessions", "poc_dataset.jsonl")


def _to_text(session: dict) -> str:
    """Rebuild the interleaved command-response transcript from a dict session.
    Mirrors Session.to_model_text so both backends see the same view."""
    lines = []
    for s in session["steps"]:
        lines.append(f"[STEP {s['step']}] [GAP={s['think_time_bin'].upper()}]")
        lines.append(f"ACTION: {s['command']}")
        lines.append(f"RESULT: {s['system_response']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Backend A: TF-IDF + LogReg stand-in (CPU, runs now)
# ---------------------------------------------------------------------------

def run_tfidf(n_splits: int = 5, seed: int = 0):
    sessions = read_sessions(DATA)
    texts = np.array([_to_text(s) for s in sessions])
    y = np.array([LABEL_IDX[s["actor_label"]] for s in sessions])
    groups = np.array(_groups(sessions))

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    f1s, all_true, all_pred = [], [], []
    for tr, te in sgkf.split(texts, y, groups):
        vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                              token_pattern=r"[^\s]+", min_df=2, max_features=4000)
        Xtr = vec.fit_transform(texts[tr])
        Xte = vec.transform(texts[te])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=2.0)
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        f1s.append(f1_score(y[te], pred, average="macro"))
        all_true.extend(y[te]); all_pred.extend(pred)

    print("Model 3 (language model) - backend=tfidf  [CPU STAND-IN for QLoRA]")
    print(f"leakage-resistant {n_splits}-fold")
    print(f"macro-F1: {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}\n")
    print(classification_report(all_true, all_pred, target_names=LABELS, digits=3))
    print("confusion matrix (rows=true, cols=pred):")
    print("        " + "  ".join(f"{l[:6]:>7}" for l in LABELS))
    for i, row in enumerate(confusion_matrix(all_true, all_pred)):
        print(f"{LABELS[i][:6]:>6}  " + "  ".join(f"{v:>7}" for v in row))
    print("\nNOTE: this is the runnable stand-in. Replace with the QLoRA backend's")
    print("numbers (run on GPU) for the final three-model comparison.")
    return float(np.mean(f1s))


# ---------------------------------------------------------------------------
# Backend B: QLoRA fine-tune (GPU - run on Colab / CUDA box)
# ---------------------------------------------------------------------------

QLORA_INSTRUCTIONS = """
QLoRA backend (run on a GPU, e.g. Colab T4):

  pip install transformers peft bitsandbytes accelerate datasets

  Base model suggestion (0.5-1.5B, instruction-tuned, small enough for a T4):
    Qwen/Qwen2.5-0.5B-Instruct  or  TinyLlama/TinyLlama-1.1B-Chat-v1.0

Approach:
  1. Build examples: input = Session.to_model_text(up_to_step=k) for several k per
     session (this also trains early/online attribution); target = a JSON string
     {"class": "<human|scripted|ai-driven>"}.
  2. Group-aware split: hold out whole participants / model families / script
     families (reuse honeyai.models.xgb_baseline._groups), so no actor leaks.
  3. Load the base model in 4-bit (bitsandbytes nf4), attach a LoRA adapter
     (peft LoraConfig, r=16, alpha=32, target the attention/MLP projections),
     train with a causal-LM objective on the formatted prompt+completion.
  4. >= 3 seeds, save adapter weights, fixed preprocessing, class-weighted or
     balanced sampling for the minority classes.
  5. Inference: generate, parse the JSON class, compute macro-F1 on the SAME
     grouped, evasion-aware test set used by Models 1 and 2.

Keep output constrained to the JSON schema (never free prose) so parsing is
deterministic. Report numbers into the comparison table in place of the tfidf
stand-in. A runnable skeleton is in scripts/train_qlora.py.
"""


def run_qlora(*_args, **_kwargs):
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import peft  # noqa: F401
    except ImportError:
        print("QLoRA backend needs transformers + peft + bitsandbytes on a GPU.")
        print(QLORA_INSTRUCTIONS)
        return None
    print("Dependencies present. See scripts/train_qlora.py for the training entry "
          "point; run it on a GPU runtime.")
    print(QLORA_INSTRUCTIONS)
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["tfidf", "qlora"], default="tfidf")
    args = ap.parse_args()
    if args.backend == "tfidf":
        run_tfidf()
    else:
        run_qlora()
