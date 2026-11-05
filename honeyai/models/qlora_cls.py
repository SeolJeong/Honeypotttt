"""Part 7: QLoRA fine-tuned language model detector (Model 3).

A small pretrained decoder LM (default Qwen2.5-0.5B) is loaded in 4-bit NF4,
LoRA adapters are trained on top, and a 3-way classification head reads the
final token of the interleaved command/response trace (honeyai.llm.format).

Why a classification head instead of generating JSON text:
  - it cannot produce malformed output, and the class probabilities are real
    softmax scores (so calibration/ECE means something), not numbers the
    model typed out;
  - class-weighted cross-entropy is a single line;
  - the required {"class", "confidence"} contract is produced by wrapping the
    probabilities (honeyai.llm.evaluate.structured).

Training follows the spec: 4-bit QLoRA, fixed preprocessing, class weights,
early stopping on a leakage-safe validation split, >=3 seeds, adapters and
the exact run config saved per seed.

Runs on a Colab T4 (16 GB). On CPU only `--smoke` works (tiny random model,
no 4-bit) and exists to check the pipeline, not to produce results.

Usage:
    python scripts/train_qlora.py                       # full run, 3 seeds
    python scripts/train_qlora.py --no-responses        # central ablation
    python scripts/train_qlora.py --round 0             # non-evasive training only
    python scripts/train_qlora.py --smoke               # CPU pipeline check
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from honeyai.llm.evaluate import print_report, report, structured, summarise_seeds
from honeyai.llm.format import (LABELS, FormatConfig, build_prompt, grouped_val_split,
                                load_sessions, make_examples, select_round)

EVAL_PREFIXES = [1, 3, 5, 10, None]  # None = full session (spec Experiment 4)


@dataclass
class QLoRAConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B"
    data_path: str = os.path.join("data", "dataset1_sessions.json")
    max_len: int = 1024
    load_4bit: bool = True

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    lr: float = 2e-4
    weight_decay: float = 0.0
    warmup_frac: float = 0.06
    epochs: int = 3
    batch_size: int = 8
    grad_accum: int = 2
    patience: int = 1           # epochs without val macro-F1 improvement
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456])

    train_prefixes: bool = True  # train on every prefix (online scoring)
    max_prefixes: int | None = 6  # per session; keeps 1-5 + full, thins the rest
    round: int = 2               # 0 = non-evasive training only, 2 = all train strategies
    abstain_below: float = 0.5
    fmt: FormatConfig = field(default_factory=FormatConfig)
    out_dir: str = os.path.join("runs", "qlora")
    smoke: bool = False

    def run_name(self) -> str:
        base = self.model_name.split("/")[-1]
        data = os.path.basename(os.path.dirname(self.data_path)) or "data"   # "data" for main's v2b
        return f"{base}_{data}_r{self.round}_{self.fmt.tag()}" + ("_smoke" if self.smoke else "")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fmt"] = self.fmt.to_dict()
        return d


# ── model construction ──────────────────────────────────────────────────

def _compute_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_model(cfg: QLoRAConfig, smoke_texts: list[str] | None = None):
    """Return (peft_model, tokenizer)."""
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if cfg.smoke:
        model, tok = _tiny_model_for_smoke(smoke_texts or [])
    else:
        tok = AutoTokenizer.from_pretrained(cfg.model_name)
        kwargs = {}
        if cfg.load_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=_compute_dtype())
            kwargs["device_map"] = {"": 0}
        else:
            kwargs["torch_dtype"] = _compute_dtype() if torch.cuda.is_available() else torch.float32
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name, num_labels=len(LABELS),
            id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)},
            **kwargs)
        if cfg.load_4bit:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    tok.truncation_side = "left"   # if anything is cut, keep the most recent steps
    model.config.pad_token_id = tok.pad_token_id

    lora = LoraConfig(task_type=TaskType.SEQ_CLS, r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
                      lora_dropout=cfg.lora_dropout, target_modules=cfg.target_modules)
    model = get_peft_model(model, lora)   # SEQ_CLS keeps the `score` head trainable
    if not cfg.load_4bit and torch.cuda.is_available():
        model = model.cuda()
    return model, tok


def _tiny_model_for_smoke(texts: list[str]):
    """Random 2-layer Qwen2 + a word-level tokenizer trained on the data.
    No downloads. Only for checking that the training/eval loop runs."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForSequenceClassification

    raw = Tokenizer(models.WordLevel(unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    raw.train_from_iterator(texts, trainers.WordLevelTrainer(special_tokens=["<pad>", "<unk>"]))
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>", unk_token="<unk>")
    conf = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                       num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                       max_position_embeddings=1024, num_labels=len(LABELS),
                       pad_token_id=tok.pad_token_id)
    return Qwen2ForSequenceClassification(conf), tok


# ── batching ────────────────────────────────────────────────────────────

def _batches(examples, tok, max_len, batch_size, shuffle, rng=None):
    idx = list(range(len(examples)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        chunk = [examples[j] for j in idx[i:i + batch_size]]
        enc = tok([e["text"] for e in chunk], padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt")
        enc["labels"] = torch.tensor([e["label"] for e in chunk])
        yield enc


def _device(model):
    return next(p for p in model.parameters()).device


@torch.no_grad()
def predict_proba(model, tok, examples, cfg: QLoRAConfig, batch_size: int = 16) -> np.ndarray:
    model.eval()
    dev = _device(model)
    out = []
    for enc in _batches(examples, tok, cfg.max_len, batch_size, shuffle=False):
        enc.pop("labels")
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.autocast(dev.type, dtype=_compute_dtype(), enabled=dev.type == "cuda"):
            logits = model(**enc).logits
        out.append(F.softmax(logits.float(), -1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(LABELS)))


def _class_weights(examples) -> torch.Tensor:
    counts = Counter(e["label"] for e in examples)
    total, k = len(examples), len(LABELS)
    return torch.tensor([total / (k * counts[c]) if counts[c] else 0.0 for c in range(k)],
                        dtype=torch.float32)


# ── training ────────────────────────────────────────────────────────────

def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_seed(cfg: QLoRAConfig, train_ex, val_ex, seed: int, save_dir: str):
    """Train with early stopping on val macro-F1 (over all prefixes).
    Returns the model reloaded at its best epoch, the tokenizer and a log."""
    from sklearn.metrics import f1_score

    _set_seed(seed)
    model, tok = build_model(cfg, smoke_texts=[e["text"] for e in train_ex + val_ex])
    dev = _device(model)
    weights = _class_weights(train_ex).to(dev)

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(train_ex) / cfg.batch_size / cfg.grad_accum)
    total = steps_per_epoch * cfg.epochs
    warm = max(1, int(total * cfg.warmup_frac))
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)))
    use_scaler = dev.type == "cuda" and _compute_dtype() == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    rng = random.Random(seed)
    y_val = np.array([e["label"] for e in val_ex])
    best, bad, log = -1.0, 0, []

    for epoch in range(cfg.epochs):
        model.train()
        t0, run_loss, n_b = time.time(), 0.0, 0
        optim.zero_grad()
        for b, enc in enumerate(_batches(train_ex, tok, cfg.max_len, cfg.batch_size, True, rng)):
            enc = {k: v.to(dev) for k, v in enc.items()}
            labels = enc.pop("labels")
            with torch.autocast(dev.type, dtype=_compute_dtype(), enabled=dev.type == "cuda"):
                logits = model(**enc).logits
            loss = F.cross_entropy(logits.float(), labels, weight=weights) / cfg.grad_accum
            scaler.scale(loss).backward()
            if (b + 1) % cfg.grad_accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()
                sched.step()
            run_loss += loss.item() * cfg.grad_accum
            n_b += 1
        if n_b % cfg.grad_accum:   # flush a trailing partial accumulation
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()
            sched.step()

        val_f1 = float(f1_score(y_val, predict_proba(model, tok, val_ex, cfg).argmax(1),
                                average="macro", labels=[0, 1, 2]))
        log.append({"epoch": epoch + 1, "train_loss": run_loss / max(1, n_b),
                    "val_macro_f1": val_f1, "secs": round(time.time() - t0, 1)})
        print(f"    epoch {epoch + 1}: loss {log[-1]['train_loss']:.4f}  "
              f"val macro-F1 {val_f1:.3f}  ({log[-1]['secs']}s)")

        if val_f1 > best:
            best, bad = val_f1, 0
            model.save_pretrained(save_dir)    # adapter + classification head only
            tok.save_pretrained(save_dir)
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"    early stop (best val macro-F1 {best:.3f})")
                break

    model = _reload_best(model, save_dir)
    return model, tok, log


def _reload_best(model, save_dir):
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    path = os.path.join(save_dir, "adapter_model.safetensors")
    set_peft_model_state_dict(model, load_file(path, device=str(_device(model))))
    return model


# ── latency (Experiment 7) ──────────────────────────────────────────────

@torch.no_grad()
def measure_latency(model, tok, examples, cfg: QLoRAConfig, n: int = 50) -> dict:
    """Per-command update latency: one partial trace scored at batch size 1."""
    model.eval()
    dev = _device(model)
    sample = examples[:n]
    for e in sample[:3]:   # warm-up
        predict_proba(model, tok, [e], cfg, batch_size=1)
    times = []
    for e in sample:
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        predict_proba(model, tok, [e], cfg, batch_size=1)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    mem = torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else None
    return {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95)),
            "n": len(times), "device": dev.type, "peak_gpu_mem_mb": mem}


# ── full experiment ─────────────────────────────────────────────────────

def run(cfg: QLoRAConfig) -> dict:
    sessions = load_sessions(cfg.data_path)
    train_s = select_round([s for s in sessions if s["split"] == "train"], cfg.round)
    test_s = [s for s in sessions if s["split"] == "test"]

    root = os.path.join(cfg.out_dir, cfg.run_name())
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    print(f"run: {cfg.run_name()}  ->  {root}")
    print(f"train pool: {len(train_s)} sessions "
          f"{dict(Counter(s['actor_label'] for s in train_s))}  | test: {len(test_s)}")

    all_seed_reports, latency = {}, None
    for seed in cfg.seeds:
        tr_s, va_s = grouped_val_split(train_s, seed=seed)
        train_ex = make_examples(tr_s, cfg.fmt, prefixes=cfg.train_prefixes, max_prefixes=cfg.max_prefixes)
        val_ex = make_examples(va_s, cfg.fmt, prefixes=True, max_prefixes=cfg.max_prefixes)
        print(f"\n--- seed {seed}: {len(tr_s)} train / {len(va_s)} val sessions "
              f"({len(train_ex)} / {len(val_ex)} examples) ---")

        seed_dir = os.path.join(root, f"seed{seed}")
        model, tok, log = train_one_seed(cfg, train_ex, val_ex, seed, seed_dir)

        seed_reports, preds_out = {}, []
        for k in EVAL_PREFIXES:
            test_ex = make_examples(test_s, cfg.fmt, k=k)
            probs = predict_proba(model, tok, test_ex, cfg)
            name = f"k={k}" if k else "full"
            seed_reports[name] = report(test_ex, probs, cfg.abstain_below)
            for e, p in zip(test_ex, probs):
                preds_out.append({"session_id": e["session_id"], "prefix": name,
                                  "steps_seen": e["steps_seen"], "true": e["actor_label"],
                                  "evasion_strategy": e["evasion_strategy"],
                                  "generator_family": e["generator_family"],
                                  **structured(p, cfg.abstain_below)})
        print_report(f"seed {seed} test, full session", seed_reports["full"])
        print("  early attribution macro-F1: " + "  ".join(
            f"{n}={r['macro_f1']:.3f}" for n, r in seed_reports.items()))

        if latency is None:
            latency = measure_latency(model, tok, make_examples(test_s, cfg.fmt, prefixes=True, max_prefixes=4), cfg)
            print(f"  latency p50 {latency['p50_ms']:.1f} ms  p95 {latency['p95_ms']:.1f} ms "
                  f"({latency['device']})")

        with open(os.path.join(seed_dir, "train_log.json"), "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)
        with open(os.path.join(seed_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(seed_reports, f, indent=2)
        with open(os.path.join(seed_dir, "test_predictions.jsonl"), "w", encoding="utf-8") as f:
            for row in preds_out:
                f.write(json.dumps(row) + "\n")
        all_seed_reports[seed] = seed_reports

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {name: summarise_seeds([all_seed_reports[s][name] for s in cfg.seeds])
               for name in next(iter(all_seed_reports.values()))}
    summary["latency"] = latency
    with open(os.path.join(root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== {cfg.run_name()}: mean +/- std over {len(cfg.seeds)} seeds ===")
    for name, m in summary.items():
        if name == "latency":
            continue
        line = f"  {name:>5s}  macro-F1 {m['macro_f1']['mean']:.3f} +/- {m['macro_f1']['std']:.3f}"
        if "ai_evasive_recall" in m:
            line += (f" | AI-evasive recall {m['ai_evasive_recall']['mean']:.3f}"
                     f" | human->AI {m['human_false_attribution']['mean']:.3f}")
        print(line)
    print(f"saved to {root}")
    return summary


# ── online scoring (for the API / dashboard, spec section 10) ───────────

class OnlineScorer:
    """Load a saved seed directory and score a growing session after every command.

        scorer = OnlineScorer("runs/qlora/Qwen2.5-0.5B_r2_full/seed42")
        scorer.score(steps_so_far)  # -> {"class": ..., "confidence": ..., "probs": {...}}
    """

    def __init__(self, seed_dir: str, load_4bit: bool | None = None):
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        run_cfg = json.load(open(os.path.join(os.path.dirname(seed_dir), "config.json"), encoding="utf-8"))
        fmt = FormatConfig(**run_cfg.pop("fmt"))
        self.cfg = QLoRAConfig(**{**run_cfg, "fmt": fmt})
        if load_4bit is not None:
            self.cfg.load_4bit = load_4bit
        if self.cfg.smoke:
            raise ValueError("smoke runs use a random model and cannot be reloaded for scoring")

        self.tok = AutoTokenizer.from_pretrained(seed_dir)
        kwargs = {}
        if self.cfg.load_4bit and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=_compute_dtype())
            kwargs["device_map"] = {"": 0}
        base = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.model_name, num_labels=len(LABELS), **kwargs)
        base.config.pad_token_id = self.tok.pad_token_id
        self.model = PeftModel.from_pretrained(base, seed_dir).eval()
        if not kwargs and torch.cuda.is_available():
            self.model = self.model.cuda()

    def score(self, steps: list[dict]) -> dict:
        ex = [{"text": build_prompt(steps, self.cfg.fmt), "label": 0}]
        probs = predict_proba(self.model, self.tok, ex, self.cfg, batch_size=1)[0]
        out = structured(probs, self.cfg.abstain_below)
        out["observations"] = len(steps)
        return out
