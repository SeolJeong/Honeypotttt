"""HoneyAI Model 3 (QLoRA LLM): plain-Python version of notebooks/qlora_generative_patched.ipynb.

Same code as the notebook (generated from it), with command-line options instead of a settings cell.
Run from the repo root, with the venv active:

    python scripts/train_llm.py --smoke                                   # pipeline check (minutes)
    python scripts/train_llm.py --out results_main.json                   # main run, 3 seeds
    python scripts/train_llm.py --commands-only --external external/kypo/kypo_sessions_ds4_commands_only.jsonl --out results_cmds.json
    python scripts/train_llm.py --zero-shot --model Qwen/Qwen2.5-7B-Instruct --out zs_qwen7b.json
    python scripts/train_llm.py --help                                    # all options

Then: python scripts/compare_models.py --llm results_main.json
"""

# ── settings (change here; choose values on VALIDATION results only) ──
import os
if os.path.basename(os.getcwd()) == "notebooks" and os.path.isdir(os.path.join("..", "data")):
    os.chdir("..")          # opened locally in Jupyter: work from the repo root
DATA_PATH = "data/dataset4_llm_sequences.jsonl"   # main's data/ (= v2b). In Colab without the repo: None -> upload
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
SEEDS = [42, 123, 456]
MAX_EPOCHS = 4              # upper bound; early stopping picks the epoch
PATIENCE = 1                # epochs without validation improvement before stopping
LR = 1e-4
LORA_R = 16
BATCH = 4
GRAD_ACCUM = 2
MAX_LEN = 768               # tokens; transcripts longer than this lose their OLDEST steps
VAL_FRAC = 0.15
VAL_MODE = "shift"          # "shift": validation mirrors the test shift (unseen AI strategy + unseen scripted family,
                            #          picked by a fixed rule, seed 2026) and is scored on FULL sessions (the selection metric)
                            # "random": old behaviour (~VAL_FRAC of every cell; in-distribution, over-optimistic)
VAL_RULE_SEED = 2026        # fixed before any results; do not change after seeing test numbers
PREFIX_TRAIN = True         # also train on partial sessions (online scoring)
TRAIN_PREFIXES = [1, 3, 5, None]   # partial lengths trained on (None = full session); matches the evaluation points
VAL_MAX = 300               # validation examples scored per epoch (random sample, fixed per seed)
SCORE_BATCH = 8             # sessions scored per forward pass
GRAD_CKPT = False           # gradient checkpointing: not needed for 0.5B on 8 GB; True saves memory for 1.5B+
RESULTS_DIR = "runs/llm"    # per-seed results + adapters; finished seeds are skipped on re-run
EVAL_PREFIXES = [1, 3, 5, None]   # None = full session
ABSTAIN_BELOW = 0.5         # confidence below this -> "unknown"
ZERO_SHOT = False           # True: NO training; score an off-the-shelf model (e.g. MODEL_ID = "Qwen/Qwen2.5-7B-Instruct")
COMMANDS_ONLY = False       # True: drop every RESP line (commands + timing only): the central ablation
EXTERNAL_TEST_PATH = None   # e.g. "external/kypo/kypo_sessions_ds4_commands_only.jsonl" ("upload" in Colab); needs COMMANDS_ONLY
SMOKE = False               # True: tiny subset, 1 seed, 1 epoch (pipeline check only)
RESULTS_NAME = "qlora_patched_results.json"   # output file (rename per run, e.g. results_main.json)


# ── command-line overrides (plain-Python version only) ──
import argparse
_ap = argparse.ArgumentParser(description="HoneyAI Model 3 (QLoRA). Same code as notebooks/qlora_generative_patched.ipynb")
_ap.add_argument("--data", default=DATA_PATH, help="dataset4 jsonl (default: main's data/)")
_ap.add_argument("--model", default=MODEL_ID)
_ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
_ap.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="max epochs (early stopping picks)")
_ap.add_argument("--patience", type=int, default=PATIENCE)
_ap.add_argument("--lr", type=float, default=LR)
_ap.add_argument("--lora-r", type=int, default=LORA_R)
_ap.add_argument("--batch", type=int, default=BATCH)
_ap.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
_ap.add_argument("--max-len", type=int, default=MAX_LEN)
_ap.add_argument("--smoke", action="store_true", help="tiny subset, 1 seed, 1 epoch")
_ap.add_argument("--zero-shot", action="store_true", help="no training: score an off-the-shelf model")
_ap.add_argument("--commands-only", action="store_true", help="drop response lines (central ablation)")
_ap.add_argument("--external", default=None, help="real-human test file (needs --commands-only)")
_ap.add_argument("--val-mode", default=VAL_MODE, choices=["shift", "random"])
_ap.add_argument("--no-prefix", action="store_true", help="train on full sessions only (option 1)")
_ap.add_argument("--out", default=RESULTS_NAME, help="results file name")
_a = _ap.parse_args()
DATA_PATH, MODEL_ID, SEEDS, MAX_EPOCHS, PATIENCE, LR = _a.data, _a.model, _a.seeds, _a.epochs, _a.patience, _a.lr
LORA_R, BATCH, GRAD_ACCUM, MAX_LEN = _a.lora_r, _a.batch, _a.grad_accum, _a.max_len
SMOKE, ZERO_SHOT, COMMANDS_ONLY, EXTERNAL_TEST_PATH, RESULTS_NAME = (
    _a.smoke, _a.zero_shot, _a.commands_only, _a.external, _a.out)
VAL_MODE = _a.val_mode
if _a.no_prefix:
    PREFIX_TRAIN = False


import copy, gc, json, math, os, random, re, time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          Trainer, TrainerCallback, TrainingArguments, set_seed)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

LABELS = ["human", "scripted", "AI-driven"]
CUDA = torch.cuda.is_available()
BF16 = CUDA and torch.cuda.is_bf16_supported()
USE_4BIT = CUDA                      # bitsandbytes 4-bit needs a CUDA GPU
DTYPE = torch.bfloat16 if BF16 else (torch.float16 if CUDA else torch.float32)
print("cuda:", CUDA, "| 4-bit:", USE_4BIT, "| dtype:", DTYPE)

SYSTEM_PROMPT = (
    "You are a security analyst classifying SSH honeypot sessions. Given an interleaved "
    "command-response transcript, classify the actor as human, scripted, or AI-driven. "
    'Respond with JSON only: {"class": "human|scripted|AI-driven"}'
)
if ZERO_SHOT:   # an untrained model needs the class definitions (same wording as scripts/zero_shot_llm.py)
    SYSTEM_PROMPT = (
        "You are a security analyst. You will see one SSH honeypot session: each step shows the time gap "
        "before the command ([GAP SHORT|MEDIUM|LONG|VERY_LONG]), the command, and the system's response.\n"
        "Classify who produced the commands:\n"
        "- human: a person typing; may pause, mistype, explore, repeat things.\n"
        "- scripted: a pre-written program; follows a fixed plan and does not adapt to what responses say.\n"
        "- AI-driven: an autonomous LLM agent; reasons about each response and adapts (for example follows a "
        "hint in the output), and may deliberately imitate a human.\n"
        'Respond with JSON only: {"class": "human|scripted|AI-driven"}'
    )
    SEEDS = SEEDS[:1]          # no training, so there is no seed variance


# ── load + deduplicate (fix #1) ──
if DATA_PATH is not None and not os.path.exists(DATA_PATH):
    print(f"{DATA_PATH} not found (cwd {os.getcwd()})")
    DATA_PATH = None
if DATA_PATH is None:
    from google.colab import files
    up = files.upload()
    DATA_PATH = list(up.keys())[0]
records = [json.loads(l) for l in open(DATA_PATH, encoding="utf-8") if l.strip()]

train_txt = {r["interleaved_text"] for r in records if r["split"] == "train"}
seen, clean = {"train": set(), "test": set()}, []
n_dup = n_leak = 0
for r in records:
    t, sp = r["interleaved_text"], r["split"]
    if sp == "test" and t in train_txt:
        n_leak += 1; continue
    if t in seen.setdefault(sp, set()):
        n_dup += 1; continue
    seen[sp].add(t); clean.append(r)
print(f"loaded {len(records)} | dropped {n_dup} duplicates, {n_leak} test sessions identical to train")
records = clean

train_all = [r for r in records if r["split"] == "train"]
test_recs = [r for r in records if r["split"] == "test"]
if SMOKE:
    rng = random.Random(0)
    train_all = [r for l in LABELS for r in rng.sample([x for x in train_all if x["actor_label"] == l], 6)]
    test_recs = [r for l in LABELS for r in rng.sample([x for x in test_recs if x["actor_label"] == l], 2)]
    SEEDS, MAX_EPOCHS = SEEDS[:1], 1
ext_recs = []
if EXTERNAL_TEST_PATH:
    if not COMMANDS_ONLY:
        print("EXTERNAL_TEST_PATH ignored: external data has no responses, set COMMANDS_ONLY = True")
    else:
        if EXTERNAL_TEST_PATH == "upload":
            from google.colab import files
            EXTERNAL_TEST_PATH = list(files.upload().keys())[0]
        ext_recs = [json.loads(l) for l in open(EXTERNAL_TEST_PATH, encoding="utf-8") if l.strip()]
        if SMOKE:
            ext_recs = ext_recs[:6]
        print(f"external test: {len(ext_recs)} sessions "
              f"({sum(bool(r.get('on_target_host')) for r in ext_recs)} on target hosts), all real humans")
print("train:", Counter(r["actor_label"] for r in train_all))
print("test: ", Counter(r["actor_label"] for r in test_recs))


# ── validation split from TRAIN only (fix #2) ──
# humans: whole participants held out if participant_id exists (v2+), otherwise sessions;
# scripted / AI: ~VAL_FRAC of each (family, evasion) cell. The test set is never touched.
def shift_holdouts(recs):
    # held-out AI evasion strategy + scripted family, picked by a fixed rule (VAL_RULE_SEED)
    rule = random.Random(VAL_RULE_SEED)
    strats = sorted({r["evasion_strategy"] for r in recs if r["actor_label"] == "AI-driven" and r["evasion_strategy"] != "none"})
    fams = sorted({r["generator_family"] for r in recs if r["actor_label"] == "scripted"})
    return rule.choice(strats), rule.choice(fams)

def val_split(recs, seed):
    rng = random.Random(seed)
    if VAL_MODE == "shift":
        strat, fam = shift_holdouts(recs)
        val_ids = {r["session_id"] for r in recs
                   if (r["actor_label"] == "AI-driven" and r["evasion_strategy"] == strat)
                   or (r["actor_label"] == "scripted" and r["generator_family"] == fam)}
        pids = sorted({r.get("participant_id") for r in recs if r["actor_label"] == "human" and r.get("participant_id")})
        held = set(rng.sample(pids, max(1, round(len(pids) * VAL_FRAC))))
        val_ids |= {r["session_id"] for r in recs if r["actor_label"] == "human" and r.get("participant_id") in held}
        print(f"shift validation: AI strategy '{strat}' + scripted family '{fam}' + {len(held)} human participants held out of training")
        return ([r for r in recs if r["session_id"] not in val_ids],
                [r for r in recs if r["session_id"] in val_ids])
    val_ids = set()
    pids = sorted({r.get("participant_id") for r in recs if r["actor_label"] == "human" and r.get("participant_id")})
    if len(pids) >= 3:
        k = max(1, round(len(pids) * VAL_FRAC))
        held = set(rng.sample(pids, k))
        val_ids |= {r["session_id"] for r in recs if r.get("participant_id") in held}
    cells = defaultdict(list)
    for r in recs:
        if r["actor_label"] != "human" or len(pids) < 3:
            cells[(r["actor_label"], r["generator_family"], r["evasion_strategy"])].append(r["session_id"])
    for ids in cells.values():
        if len(ids) < 3:                    # too small to split; keep in train
            continue
        rng.shuffle(ids)
        val_ids |= set(ids[:max(1, round(len(ids) * VAL_FRAC))])
    return ([r for r in recs if r["session_id"] not in val_ids],
            [r for r in recs if r["session_id"] in val_ids])


# ── transcript handling: prefixes + left truncation (fixes #7, #8) ──
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
_STEP = re.compile(r"^\[GAP ", re.M)

def steps_of(text):
    starts = [m.start() for m in _STEP.finditer(text)] or [0]
    chunks = [text[a:b].rstrip("\n") for a, b in zip(starts, starts[1:] + [len(text)])]
    if COMMANDS_ONLY:                      # keep "[GAP X] CMD: ..." and drop the response lines
        chunks = [c.split("\n", 1)[0] for c in chunks]
    return chunks

def prefix_lengths(n, cap):
    if cap is None or n <= cap:
        return list(range(1, n + 1))
    head = list(range(1, min(n, 5) + 1))
    rest = list(range(6, n))
    k = max(0, cap - len(head) - 1)
    picked = [rest[round(i * (len(rest) - 1) / max(1, k - 1))] for i in range(k)] if rest and k else []
    return sorted(set(head + picked + [n]))

def examples(recs, prefixes=False, k=None):
    out = []
    for r in recs:
        st = steps_of(r["interleaved_text"])
        n = len(st)
        lens = sorted({min(p, n) if p else n for p in TRAIN_PREFIXES}) if prefixes else [min(k, n) if k else n]
        for L in lens:
            out.append({**r, "text": "\n".join(st[:L]), "steps_seen": L})
    return out

def target_text(label):
    return json.dumps({"class": label})

def _chat_ids(msgs, **kw):
    out = tokenizer.apply_chat_template(msgs, tokenize=True, **kw)
    if hasattr(out, "keys"):                       # newer transformers return a dict
        out = out["input_ids"]
    return list(out)

_TEMPLATE_OVERHEAD = len(_chat_ids(
    [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ""},
     {"role": "assistant", "content": target_text("AI-driven")}]))

def prompt_ids(transcript):
    # tokenised chat prompt (system + user + assistant header); transcript cut from the LEFT
    ids = tokenizer(transcript, add_special_tokens=False)["input_ids"]
    budget = MAX_LEN - _TEMPLATE_OVERHEAD - 8
    if len(ids) > budget:
        transcript = "...[earlier steps truncated]\n" + tokenizer.decode(ids[-budget:])
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": transcript}]
    return _chat_ids(msgs, add_generation_prompt=True)

def answer_ids(label):
    # what follows the generation prompt for this label, incl. the end-of-turn token
    full = _chat_ids([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "x"},
                      {"role": "assistant", "content": target_text(label)}])
    pre = _chat_ids([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "x"}],
                    add_generation_prompt=True)
    assert full[:len(pre)] == pre
    return full[len(pre):]

ANSWER_IDS = {l: answer_ids(l) for l in LABELS}
print("answer tokens per label:", {l: len(v) for l, v in ANSWER_IDS.items()})


# ── dataset, collator, class-weighted trainer (fix #4) ──
class SessionDS(torch.utils.data.Dataset):
    def __init__(self, exs, weights):
        self.items = []
        for e in exs:
            p, a = prompt_ids(e["text"]), ANSWER_IDS[e["actor_label"]]
            self.items.append({"input_ids": p + a, "labels": [-100] * len(p) + a,
                               "weight": weights[e["actor_label"]]})
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    pad = tokenizer.pad_token_id
    return {
        "input_ids": torch.tensor([b["input_ids"] + [pad] * (n - len(b["input_ids"])) for b in batch]),
        "attention_mask": torch.tensor([[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]),
        "labels": torch.tensor([b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]),
        "weight": torch.tensor([b["weight"] for b in batch], dtype=torch.float32),
    }

def class_weights(exs):
    c = Counter(e["actor_label"] for e in exs)
    return {l: len(exs) / (len(LABELS) * c[l]) if c[l] else 0.0 for l in LABELS}

def _lm(model):
    # the causal LM inside the PEFT wrapper (LoRA layers stay active)
    return model.get_base_model() if hasattr(model, "get_base_model") else model

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # speed: run the 151k-word output layer ONLY at the ~10 answer positions, not at every token
        w = inputs.pop("weight")
        labels = inputs.pop("labels")
        lm = _lm(model)
        h = lm.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).last_hidden_state
        tgt = labels[:, 1:]
        mask = tgt != -100
        logits = lm.lm_head(h[:, :-1][mask]).float()
        tok = F.cross_entropy(logits, tgt[mask], reduction="none")
        rows = torch.arange(tgt.size(0), device=tgt.device).unsqueeze(1).expand_as(tgt)[mask]
        per_ex = torch.zeros(tgt.size(0), device=tok.device, dtype=tok.dtype).index_add_(0, rows, tok)
        per_ex = per_ex / mask.sum(1).clamp(min=1).to(tok.dtype)
        loss = (per_ex * w.to(tok.device)).sum() / w.sum().clamp(min=1e-8)
        return (loss, None) if return_outputs else loss


# ── prediction by scoring all 3 labels (fix #3): real probabilities, no parsing ──
@torch.no_grad()
def predict_proba(model, exs):
    # SCORE_BATCH sessions x 3 labels per forward pass; output layer only at answer positions
    model.eval()
    lm = _lm(model)
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(exs), SCORE_BATCH):
        chunk = exs[i:i + SCORE_BATCH]
        seqs, rows, cols, toks = [], [], [], []
        for e in chunk:
            p = prompt_ids(e["text"])
            for l in LABELS:
                a = ANSWER_IDS[l]
                r = len(seqs)
                seqs.append(p + a)
                for j, t in enumerate(a):
                    rows.append(r); cols.append(len(p) - 1 + j); toks.append(t)
        n = max(map(len, seqs))
        ids = torch.tensor([s + [tokenizer.pad_token_id] * (n - len(s)) for s in seqs], device=dev)
        att = torch.tensor([[1] * len(s) + [0] * (n - len(s)) for s in seqs], device=dev)
        rows_t, cols_t, toks_t = (torch.tensor(x, device=dev) for x in (rows, cols, toks))
        with torch.autocast(dev.type, dtype=DTYPE, enabled=CUDA):
            h = lm.model(input_ids=ids, attention_mask=att).last_hidden_state
            logits = lm.lm_head(h[rows_t, cols_t]).float()
        lp = F.log_softmax(logits, -1)[torch.arange(len(toks), device=dev), toks_t]
        seq = torch.zeros(len(seqs), device=dev).index_add_(0, rows_t, lp)
        out.append(F.softmax(seq.view(len(chunk), len(LABELS)), 1).cpu().numpy())
    probs = np.concatenate(out) if out else np.zeros((0, len(LABELS)))
    if ZS_CF is not None and len(probs):
        # zero-shot only: divide out the model's average label bias (estimated on
        # training transcripts, see run_seed), then renormalise
        probs = probs / ZS_CF
        probs = probs / probs.sum(1, keepdims=True)
    return probs

ZS_CF = None    # set in zero-shot mode only

def macro_f1(exs, probs):
    y = [LABELS.index(e["actor_label"]) for e in exs]
    return f1_score(y, probs.argmax(1), average="macro", labels=[0, 1, 2], zero_division=0)

def ece(probs, y, bins=15):
    conf, ok = probs.max(1), (probs.argmax(1) == y).astype(float)
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(e)


# ── early stopping on validation (fix #2) ──
class EarlyStopOnVal(TrainerCallback):
    def __init__(self, model, val_ex):
        self.model, self.val_ex = model, val_ex
        self.best, self.bad, self.best_state, self.log = -1.0, 0, None, []
    def on_epoch_begin(self, args, state, control, **kw):
        self.t0 = time.time()
    def on_epoch_end(self, args, state, control, **kw):
        secs = time.time() - getattr(self, "t0", time.time())
        f1 = macro_f1(self.val_ex, predict_proba(self.model, self.val_ex))
        self.log.append({"epoch": round(state.epoch), "val_macro_f1": round(float(f1), 4), "train_secs": round(secs)})
        print(f"  epoch {round(state.epoch)}: val macro-F1 {f1:.3f}  (training {secs / 60:.1f} min)")
        if f1 > self.best:
            self.best, self.bad = f1, 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.named_parameters() if v.requires_grad}
        else:
            self.bad += 1
            if self.bad >= PATIENCE:
                print(f"  early stop (best val macro-F1 {self.best:.3f})")
                control.should_training_stop = True
        self.model.train()

def build_model():
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    kw = {"dtype" if (major, minor) >= (4, 56) else "torch_dtype": DTYPE}
    if USE_4BIT:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=DTYPE)
        kw["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kw)
    if USE_4BIT:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=GRAD_CKPT,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    return get_peft_model(model, lora)

def run_seed(seed):
    global ZS_CF
    ZS_CF = None                       # calibration only ever applies to the current zero-shot run
    print(f"\n{'=' * 50}\nSEED {seed}\n{'=' * 50}")
    set_seed(seed)
    tr, va = val_split(train_all, seed)
    tr_ex = examples(tr, prefixes=PREFIX_TRAIN)
    va_ex = examples(va, prefixes=PREFIX_TRAIN and VAL_MODE != "shift")   # shift: full sessions = the selection metric
    if len(va_ex) > VAL_MAX and VAL_MODE != "shift":
        va_ex = random.Random(seed).sample(va_ex, VAL_MAX)
    print(f"train {len(tr)} sessions -> {len(tr_ex)} examples | val {len(va)} sessions -> {len(va_ex)} examples scored")
    model = build_model()
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {t:,}")
    cb = EarlyStopOnVal(model, va_ex)
    train_ds = SessionDS(tr_ex, class_weights(tr_ex))
    total_steps = math.ceil(len(train_ds) / (BATCH * GRAD_ACCUM)) * MAX_EPOCHS
    args = TrainingArguments(
        output_dir=f"./qlora_seed{seed}", num_train_epochs=MAX_EPOCHS,     # fix #6: actually used
        per_device_train_batch_size=BATCH, gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR, warmup_steps=max(1, int(0.05 * total_steps)), lr_scheduler_type="linear", weight_decay=0.0,
        bf16=BF16, fp16=CUDA and not BF16, logging_steps=20, save_strategy="no",
        report_to="none", seed=seed, remove_unused_columns=False, use_cpu=not CUDA)
    if ZERO_SHOT:                                          # no training: LoRA starts as a no-op (B = 0)
        # domain-context calibration (Fei et al. 2023): the model's average prediction on a
        # sample of TRAINING transcripts (text only, labels unused) estimates its label bias
        cal = random.Random(seed).sample(tr_ex, min(60, len(tr_ex)))
        ZS_CF = predict_proba(model, cal).mean(0)
        print("  label bias on training transcripts:", {l: round(float(x), 3) for l, x in zip(LABELS, ZS_CF)})
        cb.best = macro_f1(va_ex, predict_proba(model, va_ex))
        print(f"  zero-shot validation macro-F1 {cb.best:.3f}")
    else:
        WeightedTrainer(model=model, args=args, train_dataset=train_ds,
                        data_collator=collate, callbacks=[cb]).train()
        with torch.no_grad():                              # restore the best validation epoch
            for k, v in model.named_parameters():
                if k in cb.best_state:
                    v.copy_(cb.best_state[k].to(v.device))

    res = {"seed": seed, "best_val_macro_f1": round(float(cb.best), 4), "epochs": cb.log, "by_prefix": {}, "predictions": []}
    for k in EVAL_PREFIXES:
        ex = examples(test_recs, k=k)
        probs = predict_proba(model, ex)
        y = np.array([LABELS.index(e["actor_label"]) for e in ex])
        pred = probs.argmax(1)
        name = f"k={k}" if k else "full"
        ai = y == 2
        ev = np.array([e["evasion_strategy"] for e in ex])
        res["by_prefix"][name] = {
            "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0)),
            "ece": ece(probs, y),
            "abstain_rate": float((probs.max(1) < ABSTAIN_BELOW).mean()),
            "human_false_attribution": float((pred[y == 0] == 2).mean()) if (y == 0).any() else None,
            "ai_recall_by_strategy": {str(s): float((pred[ai & (ev == s)] == 2).mean()) for s in sorted(set(ev[ai]))},
            "confusion": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        }
        for e, p in zip(ex, probs):
            i = int(p.argmax())
            res["predictions"].append({
                "session_id": e["session_id"], "prefix": name, "steps_seen": e["steps_seen"],
                "true_label": e["actor_label"], "evasion_strategy": e["evasion_strategy"],
                "generator_family": e["generator_family"],
                "predicted_label": LABELS[i] if p[i] >= ABSTAIN_BELOW else "unknown",
                "confidence": round(float(p[i]), 4), "probs": {l: round(float(x), 4) for l, x in zip(LABELS, p)}})
        print(f"  test {name}: macro-F1 {res['by_prefix'][name]['macro_f1']:.3f}")
    full = res["by_prefix"]["full"]
    print("  full-session confusion (rows=true, cols=pred)", LABELS, full["confusion"])
    print("  AI recall by strategy:", {s: round(v, 2) for s, v in full["ai_recall_by_strategy"].items()})
    # per-command update latency (RQ3): score one partial session (3 commands) at a time
    lat = []
    for e in examples(test_recs[:50], k=3):
        t0 = time.perf_counter()
        predict_proba(model, [e])
        if CUDA:
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
    res["latency_ms"] = {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)),
                         "n": len(lat), "device": "cuda" if CUDA else "cpu"}
    print(f"  latency per command update: p50 {res['latency_ms']['p50']:.0f} ms, p95 {res['latency_ms']['p95']:.0f} ms")
    if ext_recs:                           # every external session is a real human
        ex = examples(ext_recs)
        pred = predict_proba(model, ex).argmax(1)
        def shares(mask):
            p = pred[mask]
            return {l: float((p == i).mean()) if len(p) else None for i, l in enumerate(LABELS)}
        tgt = np.array([bool(e.get("on_target_host")) for e in ex])
        res["external"] = {"n": len(ex), "all": shares(np.ones(len(ex), bool)), "target_hosts": shares(tgt)}
        print(f"  external real humans labelled AI (false attribution): {res['external']['all']['AI-driven']:.3f}"
              f" | on target hosts: {res['external']['target_hosts']['AI-driven']}")
    model.save_pretrained(os.path.join(RESULTS_DIR, f"{RUN_TAG}_adapter_seed{seed}"))
    del model; gc.collect()
    if CUDA:
        torch.cuda.empty_cache()
    return res


# ── run all seeds; report MEAN +/- STD (never the best test seed) ──
RUN_TAG = (f"{'zs' if ZERO_SHOT else 'ft'}_{'cmds' if COMMANDS_ONLY else 'full'}"
           f"{'' if PREFIX_TRAIN else '_noprefix'}{'_shiftval' if VAL_MODE == 'shift' else ''}_{MODEL_ID.split('/')[-1]}{'_smoke' if SMOKE else ''}")
os.makedirs(RESULTS_DIR, exist_ok=True)
results = []
for s in SEEDS:
    seed_file = os.path.join(RESULTS_DIR, f"{RUN_TAG}_seed{s}.json")
    if os.path.exists(seed_file):          # resume: this seed already finished in an earlier run
        print(f"seed {s}: already done, loading {seed_file}")
        results.append(json.load(open(seed_file, encoding="utf-8")))
        continue
    r = run_seed(s)
    with open(seed_file, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=1)
    print(f"seed {s}: saved {seed_file}")
    results.append(r)
summary = {}
for name in results[0]["by_prefix"]:
    vals = [r["by_prefix"][name]["macro_f1"] for r in results]
    summary[name] = {"macro_f1_mean": float(np.mean(vals)), "macro_f1_std": float(np.std(vals))}
    print(f"{name:>5s}: test macro-F1 {np.mean(vals):.3f} +/- {np.std(vals):.3f}")
# bootstrap CI for seed 1's full-session result (resampling sessions)
r0 = [p for p in results[0]["predictions"] if p["prefix"] == "full"]
y = np.array([LABELS.index(p["true_label"]) for p in r0])
yp = np.array([LABELS.index(max(p["probs"], key=p["probs"].get)) for p in r0])
rng = np.random.default_rng(0)
bs = [f1_score(y[i], yp[i], average="macro", labels=[0, 1, 2], zero_division=0) for i in (rng.integers(0, len(y), len(y)) for _ in range(2000))]
summary["full_ci95_seed0"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
print("95% bootstrap CI (first seed, full session):", [round(x, 3) for x in summary["full_ci95_seed0"]])
summary["latency_ms"] = results[0]["latency_ms"]
if results[0].get("external"):
    v = [r["external"]["all"]["AI-driven"] for r in results]
    summary["external_human_false_attribution"] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
    print(f"external real humans labelled AI: {np.mean(v):.3f} +/- {np.std(v):.3f}")


out = {"model": MODEL_ID, "data": os.path.basename(DATA_PATH), "dedup": {"duplicates": n_dup, "test_equals_train": n_leak},
       "config": {k: globals()[k] for k in ["SEEDS", "MAX_EPOCHS", "PATIENCE", "LR", "LORA_R", "BATCH", "GRAD_ACCUM",
                                           "MAX_LEN", "VAL_FRAC", "VAL_MODE", "VAL_RULE_SEED", "PREFIX_TRAIN", "TRAIN_PREFIXES", "VAL_MAX", "SCORE_BATCH",
                                           "GRAD_CKPT", "RESULTS_DIR", "ABSTAIN_BELOW", "SMOKE",
                                           "COMMANDS_ONLY", "EXTERNAL_TEST_PATH", "ZERO_SHOT", "RESULTS_NAME"]},
       "summary": summary, "seeds": results}
with open(RESULTS_NAME, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print(f"saved {RESULTS_NAME} (per-seed files and adapters in {RESULTS_DIR}/)")
try:
    from google.colab import files
    os.system(f"zip -qr qlora_patched_adapters.zip {RESULTS_DIR}")
    files.download(RESULTS_NAME); files.download("qlora_patched_adapters.zip")
except ImportError:
    pass


