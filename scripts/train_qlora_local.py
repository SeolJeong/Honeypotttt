"""Part 7: QLoRA training — local GPU version.

Usage:
    python scripts/train_qlora_local.py
"""
import gc
import json
import os
import sys

import numpy as np
import torch
from collections import Counter

from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    Trainer, TrainingArguments, default_data_collator, set_seed,
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# --- config ---
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DS4 = os.path.join("data", "dataset4_llm_sequences.jsonl")
OUT = os.path.join("data", "qlora_predictions.json")
MAX_SEQ_LEN = 512
LABEL_NAMES = ["human", "scripted", "AI-driven"]
LABEL_MAP = {"human": 0, "scripted": 1, "AI-driven": 2}

SYSTEM_PROMPT = (
    "You are a security analyst classifying SSH honeypot sessions. "
    "Given an interleaved command-response transcript, classify the "
    "actor as human, scripted, or AI-driven. "
    'Respond with JSON only: {"class": "human|scripted|AI-driven", '
    '"confidence": 0.xx}'
)

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)


# --- helpers ---
def format_messages(text, label=None):
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    if label is not None:
        target = json.dumps({"class": label, "confidence": 0.95})
        msgs.append({"role": "assistant", "content": target})
    return msgs


def parse_prediction(text):
    text = text.strip()
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            cls = data.get("class", "")
            conf = float(data.get("confidence", 0.0))
            if cls in LABEL_NAMES:
                return cls, conf
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    for label in LABEL_NAMES:
        if label.lower() in text.lower():
            return label, 0.5
    return "unknown", 0.0


class SessionDataset(torch.utils.data.Dataset):
    def __init__(self, records, tokenizer, max_len=MAX_SEQ_LEN):
        self.records = records
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        prompt_msgs = format_messages(rec["interleaved_text"])
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        full_msgs = format_messages(rec["interleaved_text"], rec["actor_label"])
        full_text = self.tokenizer.apply_chat_template(
            full_msgs, tokenize=False
        )
        prompt_ids = self.tokenizer(
            prompt_text, add_special_tokens=False
        )["input_ids"]
        full_enc = self.tokenizer(
            full_text, truncation=True, max_length=self.max_len,
            padding="max_length", add_special_tokens=False,
        )
        input_ids = full_enc["input_ids"]
        attn_mask = full_enc["attention_mask"]
        labels = list(input_ids)
        prompt_len = min(len(prompt_ids), len(labels))
        for i in range(prompt_len):
            labels[i] = -100
        for i in range(len(labels)):
            if attn_mask[i] == 0:
                labels[i] = -100
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attn_mask),
            "labels": torch.tensor(labels),
        }


@torch.no_grad()
def predict_one(model, tokenizer, interleaved_text):
    msgs = format_messages(interleaved_text)
    prompt = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=480
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs, max_new_tokens=50, do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return parse_prediction(response)


def train_and_eval_seed(seed, train_recs, test_recs, tokenizer, n_epochs=15):
    print(f"\n{'='*50}")
    print(f"SEED {seed}")
    print(f"{'='*50}")
    set_seed(seed)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=BNB_CONFIG, device_map="auto",
    )
    model = get_peft_model(model, LORA_CONFIG)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    train_ds = SessionDataset(train_recs, tokenizer)

    args = TrainingArguments(
        output_dir=f"./qlora_seed{seed}",
        num_train_epochs=n_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        warmup_steps=10,
        fp16=True,
        logging_steps=20,
        save_strategy="no",
        report_to="none",
        seed=seed,
    )
    trainer = Trainer(
        model=model, train_dataset=train_ds,
        args=args, data_collator=default_data_collator,
    )
    trainer.train()

    model.eval()
    preds, confs, trues = [], [], []
    parse_fails = 0
    for i, rec in enumerate(test_recs):
        cls, conf = predict_one(model, tokenizer, rec["interleaved_text"])
        preds.append(cls)
        confs.append(conf)
        trues.append(rec["actor_label"])
        if cls == "unknown":
            parse_fails += 1
        if (i + 1) % 100 == 0:
            print(f"  predicted {i+1}/{len(test_recs)}...")

    f1 = f1_score(trues, preds, labels=LABEL_NAMES, average="macro", zero_division=0)
    print(f"\nmacro-F1: {f1:.3f}  (parse fails: {parse_fails}/{len(test_recs)})")
    print(classification_report(
        trues, preds, labels=LABEL_NAMES, target_names=LABEL_NAMES,
        digits=3, zero_division=0,
    ))
    print("confusion matrix (rows=true, cols=pred):")
    print(f"         {LABEL_NAMES}")
    cm = confusion_matrix(trues, preds, labels=LABEL_NAMES)
    for i, row in enumerate(cm):
        print(f"  {LABEL_NAMES[i]:>10s}  {row}")

    results = {
        "seed": seed, "macro_f1": f1, "parse_fails": parse_fails,
        "predictions": [
            {"session_id": rec["session_id"], "true_label": true,
             "predicted_label": pred, "confidence": conf}
            for rec, true, pred, conf in zip(test_recs, trues, preds, confs)
        ],
    }

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # load data
    records = []
    with open(DS4, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    train_recs = [r for r in records if r["split"] == "train"]
    test_recs = [r for r in records if r["split"] == "test"]
    print(f"train: {len(train_recs)}, test: {len(test_recs)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # run 3 seeds
    all_results = []
    for seed in [42, 123, 456]:
        res = train_and_eval_seed(seed, train_recs, test_recs, tokenizer)
        all_results.append(res)

    f1s = [r["macro_f1"] for r in all_results]
    print(f"\n{'='*50}")
    print(f"SUMMARY: test macro-F1 = {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}")
    for r in all_results:
        print(f"  seed {r['seed']}: {r['macro_f1']:.3f}  (parse fails: {r['parse_fails']})")

    best = max(all_results, key=lambda r: r["macro_f1"])
    out = {
        "model": MODEL_ID,
        "best_seed": best["seed"],
        "macro_f1_mean": round(float(np.mean(f1s)), 3),
        "macro_f1_std": round(float(np.std(f1s)), 3),
        "all_seeds": [{"seed": r["seed"], "macro_f1": r["macro_f1"]} for r in all_results],
        "predictions": best["predictions"],
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()