"""QLoRA fine-tune skeleton for Model 3 (run on a GPU, e.g. Colab T4).

This is a runnable starting point, not a finished experiment. It does NOT run in
the CPU PoC container; open it on a GPU runtime, `pip install` the deps below, and
fill the two TODOs. Output numbers replace the tfidf stand-in in the comparison.

  pip install transformers peft bitsandbytes accelerate datasets torch

Design (per the build spec):
- input  = interleaved transcript truncated to k steps (also trains early attribution)
- target = {"class": "<human|scripted|ai-driven>"}  (constrained, never free prose)
- split  = group-aware (whole participants / model families / script families held out)
- >= 3 seeds, save adapter weights, class balancing for minority classes
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from honeyai.schema import read_sessions
from honeyai.models.xgb_baseline import _groups

DATA = os.path.join("data", "sessions", "poc_dataset.jsonl")
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # or TinyLlama/TinyLlama-1.1B-Chat-v1.0
LABELS = ["human", "scripted", "ai-driven"]


def _transcript(session: dict, k: int) -> str:
    lines = []
    for s in session["steps"][:k]:
        lines.append(f"[STEP {s['step']}] [GAP={s['think_time_bin'].upper()}]")
        lines.append(f"ACTION: {s['command']}")
        lines.append(f"RESULT: {s['system_response']}")
    return "\n".join(lines)


def build_examples(sessions, ks=(3, 5, 10, None)):
    """One example per (session, k). k=None means the full session."""
    examples = []
    for s in sessions:
        n = len(s["steps"])
        for k in ks:
            kk = n if k is None else min(k, n)
            if kk < 1:
                continue
            prompt = (
                "You are a security analyst. Classify the SSH session actor as "
                "exactly one of: human, scripted, ai-driven. Respond with JSON only.\n\n"
                f"SESSION:\n{_transcript(s, kk)}\n\nANSWER:"
            )
            target = json.dumps({"class": s["actor_label"]})
            examples.append({"prompt": prompt, "completion": " " + target,
                             "group": _groups([s])[0], "label": s["actor_label"]})
    return examples


def main():
    sessions = read_sessions(DATA)
    examples = build_examples(sessions)
    print(f"built {len(examples)} training examples from {len(sessions)} sessions")

    # --- group-aware train/test split -------------------------------------
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold
    y = np.array([LABELS.index(e["label"]) for e in examples])
    groups = np.array([e["group"] for e in examples])
    tr_idx, te_idx = next(StratifiedGroupKFold(5, shuffle=True, random_state=0)
                          .split(examples, y, groups))
    print(f"train {len(tr_idx)}  test {len(te_idx)}  (groups held out)")

    # --- TODO 1: load base model in 4-bit + attach LoRA -------------------
    # from transformers import (AutoModelForCausalLM, AutoTokenizer,
    #                           BitsAndBytesConfig, TrainingArguments)
    # from peft import LoraConfig, get_peft_model
    # bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    #                          bnb_4bit_compute_dtype="bfloat16")
    # tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    # model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb,
    #                                              device_map="auto")
    # lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    #                   task_type="CAUSAL_LM",
    #                   target_modules=["q_proj","k_proj","v_proj","o_proj"])
    # model = get_peft_model(model, lora)

    # --- TODO 2: tokenise prompt+completion, train (>=3 seeds), save adapter,
    #     then generate on the test set, parse {"class": ...}, compute macro-F1
    #     against the SAME grouped split used by Models 1 and 2.

    print("\nFill TODO 1 and TODO 2 on a GPU runtime to train and evaluate Model 3.")


if __name__ == "__main__":
    main()
