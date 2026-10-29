"""QLoRA fine-tune for Model 3 (run on a GPU, e.g. Google Colab T4).

Complete, runnable on a GPU runtime. It does NOT run in the CPU PoC container.
On Colab: set runtime to GPU, then:

    !pip install -q transformers peft bitsandbytes accelerate datasets
    !python scripts/train_qlora.py

Design (per the build spec):
- input  = interleaved transcript truncated to k steps (also trains early attribution)
- target = {"class": "<human|scripted|ai-driven>"}  (constrained JSON, never prose)
- split  = group-aware: whole participants / model families / script families held out
- 4-bit QLoRA LoRA adapter, saved to ./adapters/honeyai-qlora
- reports macro-F1 on the SAME grouped test split used by Models 1 and 2

Reproducibility: pass --seed and run 3+ times for mean/std, as the spec requires.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from honeyai.schema import read_sessions
from honeyai.models.xgb_baseline import _groups

DATA = os.path.join("data", "sessions", "poc_dataset.jsonl")
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # small enough for a T4; swap if you like
LABELS = ["human", "scripted", "ai-driven"]
ADAPTER_DIR = os.path.join("adapters", "honeyai-qlora")


def _transcript(session: dict, k: int) -> str:
    lines = []
    for s in session["steps"][:k]:
        lines.append(f"[STEP {s['step']}] [GAP={s['think_time_bin'].upper()}]")
        lines.append(f"ACTION: {s['command']}")
        lines.append(f"RESULT: {s['system_response']}")
    return "\n".join(lines)


PROMPT_TMPL = (
    "You are a security analyst. Classify the SSH session actor as exactly one of: "
    "human, scripted, ai-driven. Respond with JSON only, e.g. {{\"class\": \"human\"}}.\n\n"
    "SESSION:\n{transcript}\n\nANSWER:"
)


def build_examples(sessions, ks=(3, 5, 10, None)):
    rows = []
    for s in sessions:
        n = len(s["steps"])
        for k in ks:
            kk = n if k is None else min(k, n)
            if kk < 1:
                continue
            rows.append({
                "prompt": PROMPT_TMPL.format(transcript=_transcript(s, kk)),
                "answer": json.dumps({"class": s["actor_label"]}),
                "group": _groups([s])[0],
                "label": s["actor_label"],
            })
    return rows


def split_examples(rows, seed):
    from sklearn.model_selection import StratifiedGroupKFold
    y = np.array([LABELS.index(r["label"]) for r in rows])
    groups = np.array([r["group"] for r in rows])
    tr, te = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed)
                  .split(rows, y, groups))
    return [rows[i] for i in tr], [rows[i] for i in te]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--base-model", default=BASE_MODEL)
    args = ap.parse_args()

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, TrainingArguments, Trainer)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    sessions = read_sessions(DATA)
    rows = build_examples(sessions)
    train_rows, test_rows = split_examples(rows, args.seed)
    print(f"{len(sessions)} sessions -> {len(rows)} examples "
          f"(train {len(train_rows)}, test {len(test_rows)}, groups held out)")

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- build tokenised training set; mask the prompt so loss is on the answer only
    def encode(row, max_len=1024):
        prompt_ids = tok(row["prompt"], add_special_tokens=False)["input_ids"]
        answer_ids = tok(" " + row["answer"] + tok.eos_token,
                         add_special_tokens=False)["input_ids"]
        ids = (prompt_ids + answer_ids)[:max_len]
        labels = ([-100] * len(prompt_ids) + answer_ids)[:max_len]
        return {"input_ids": ids, "labels": labels,
                "attention_mask": [1] * len(ids)}

    train_enc = [encode(r) for r in train_rows]

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        def pad(key, fill):
            return torch.tensor([b[key] + [fill] * (maxlen - len(b[key])) for b in batch])
        return {"input_ids": pad("input_ids", tok.pad_token_id),
                "attention_mask": pad("attention_mask", 0),
                "labels": pad("labels", -100)}

    # ---- 4-bit base model + LoRA adapter (this is the QLoRA part)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir="./_qlora_out", per_device_train_batch_size=4,
        gradient_accumulation_steps=2, num_train_epochs=args.epochs,
        learning_rate=2e-4, bf16=True, logging_steps=20,
        save_strategy="no", report_to=[], seed=args.seed)
    trainer = Trainer(model=model, args=targs, train_dataset=train_enc,
                      data_collator=collate)
    trainer.train()

    os.makedirs(ADAPTER_DIR, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    tok.save_pretrained(ADAPTER_DIR)
    print(f"saved adapter -> {ADAPTER_DIR}")

    # ---- evaluate: generate on the test set, parse {"class": ...}, macro-F1
    from sklearn.metrics import f1_score, classification_report
    model.eval()
    preds, trues = [], []
    for r in test_rows:
        ids = tok(r["prompt"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=16, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        m = re.search(r'"class"\s*:\s*"(human|scripted|ai-driven)"', text)
        pred = m.group(1) if m else "scripted"   # default if unparseable
        preds.append(LABELS.index(pred))
        trues.append(LABELS.index(r["label"]))

    f1 = f1_score(trues, preds, average="macro")
    print(f"\nModel 3 (QLoRA) seed={args.seed}  macro-F1: {f1:.3f}")
    print(classification_report(trues, preds, target_names=LABELS, digits=3))
    print("Run 3+ seeds and report mean +/- std; compare against Models 1 and 2.")


if __name__ == "__main__":
    main()
