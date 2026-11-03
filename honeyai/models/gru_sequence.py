"""Part 6: GRU sequence model (Model 2).

Tests whether sequential command-response ordering improves attribution
over flat aggregate features (Model 1 XGBoost). Each session step is
encoded as:
  - command token embeddings (averaged from a learned vocabulary)
  - response token embeddings (averaged, same vocabulary)
  - timing bin (one-hot: SHORT / MEDIUM / LONG / VERY_LONG)
  - response category (one-hot: 6 categories)
  - scalars: has_typo, task_progress, normalised cmd length, normalised gap

The step sequence feeds a single-layer GRU; the final hidden state goes
through dropout into a 3-class linear head.

Usage:
    python scripts/train_gru.py
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score

DATA_DIR = "data"
DS1 = os.path.join(DATA_DIR, "dataset1_sessions.json")

LABEL_MAP = {"human": 0, "scripted": 1, "AI-driven": 2}
LABEL_NAMES = ["human", "scripted", "AI-driven"]
TIMING_BINS = ["short", "medium", "long", "very_long"]
RESP_CATS = [
    "success", "not_found_with_hint", "not_found_no_hint",
    "permission_denied", "command_not_found", "partial_result",
]

MAX_CMD_TOK = 10    # tokens kept per command
MAX_RESP_TOK = 16   # tokens kept per response
VOCAB_MAX = 150     # vocabulary size (top tokens + PAD + UNK)
EMBED_DIM = 24
HIDDEN_DIM = 48
N_ONEHOT = len(TIMING_BINS) + len(RESP_CATS)  # 10
N_SCALAR = 4        # has_typo, progress, cmd_len_norm, gap_norm


# ── tokeniser + vocabulary ──────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric (keep paths and flags)."""
    return [t for t in re.split(r"[^a-z0-9._/-]+", text.lower()) if t]


class Vocab:
    """Word vocabulary built from training data only (no test leakage)."""

    PAD, UNK = 0, 1

    def __init__(self, max_size: int = VOCAB_MAX):
        self.max_size = max_size
        self.t2i: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}

    def build(self, token_lists: list[list[str]]):
        counts = Counter(t for toks in token_lists for t in toks)
        for tok, _ in counts.most_common(self.max_size - 2):
            self.t2i[tok] = len(self.t2i)

    def encode(self, tokens: list[str], max_len: int) -> list[int]:
        ids = [self.t2i.get(t, self.UNK) for t in tokens[:max_len]]
        return ids + [self.PAD] * (max_len - len(ids))

    def __len__(self):
        return len(self.t2i)


# ── data loading + encoding ─────────────────────────────────────────────

def load_sessions():
    """Load dataset1, split into train/test lists."""
    with open(DS1, encoding="utf-8") as f:
        sessions = json.load(f)
    train = [s for s in sessions if s["split"] == "train"]
    test = [s for s in sessions if s["split"] == "test"]
    return train, test


def build_vocab(sessions: list[dict]) -> Vocab:
    """Build vocabulary from training sessions only."""
    all_tok: list[list[str]] = []
    for s in sessions:
        for st in s["interactions"]:
            all_tok.append(tokenize(st["command"]))
            all_tok.append(tokenize(st["system_response"]))
    v = Vocab()
    v.build(all_tok)
    return v


def encode_sessions(sessions: list[dict], vocab: Vocab) -> dict:
    """Convert sessions into padded tensors for the GRU."""
    max_steps = max(len(s["interactions"]) for s in sessions)
    n = len(sessions)

    cmd = np.zeros((n, max_steps, MAX_CMD_TOK), dtype=np.int64)
    resp = np.zeros((n, max_steps, MAX_RESP_TOK), dtype=np.int64)
    cats = np.zeros((n, max_steps, N_ONEHOT), dtype=np.float32)
    scal = np.zeros((n, max_steps, N_SCALAR), dtype=np.float32)
    lens = np.zeros(n, dtype=np.int64)
    labs = np.zeros(n, dtype=np.int64)

    for i, sess in enumerate(sessions):
        steps = sess["interactions"]
        lens[i] = len(steps)
        labs[i] = LABEL_MAP[sess["actor_label"]]

        for j, st in enumerate(steps):
            cmd[i, j] = vocab.encode(tokenize(st["command"]), MAX_CMD_TOK)
            resp[i, j] = vocab.encode(tokenize(st["system_response"]), MAX_RESP_TOK)

            # timing bin one-hot
            tb = st.get("think_time_bin", "medium").lower()
            if tb in TIMING_BINS:
                cats[i, j, TIMING_BINS.index(tb)] = 1.0

            # response category one-hot
            rc = st.get("response_category", "success")
            if rc in RESP_CATS:
                cats[i, j, len(TIMING_BINS) + RESP_CATS.index(rc)] = 1.0

            # scalars (normalised to roughly 0-1)
            scal[i, j, 0] = 1.0 if st.get("has_typo") in (True, "True", "true") else 0.0
            scal[i, j, 1] = float(st.get("task_progress", 0.0))
            scal[i, j, 2] = min(len(st["command"]) / 50.0, 1.0)
            scal[i, j, 3] = min(float(st.get("inter_command_gap_s", 0)) / 30.0, 1.0)

    return {
        k: torch.from_numpy(v)
        for k, v in [
            ("cmd", cmd), ("resp", resp), ("cats", cats),
            ("scal", scal), ("lens", lens), ("labs", labs),
        ]
    }


# ── model ────────────────────────────────────────────────────────────────

class GRUClassifier(nn.Module):
    """Single-layer GRU with token-averaged embeddings per step."""

    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                 n_classes=3, dropout=0.4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        input_dim = embed_dim * 2 + N_ONEHOT + N_SCALAR  # cmd + resp + cats + scal
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, n_classes)

    def _avg(self, ids):
        """Embed token ids and mean-pool (ignoring PAD=0)."""
        emb = self.embed(ids)                               # (B, S, T, E)
        mask = (ids != 0).unsqueeze(-1).float()              # (B, S, T, 1)
        return (emb * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)  # (B, S, E)

    def forward(self, cmd, resp, cats, scal, lens):
        x = torch.cat([self._avg(cmd), self._avg(resp), cats, scal], dim=-1)
        packed = pack_padded_sequence(
            x, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)                              # h: (1, B, H)
        return self.head(self.drop(h.squeeze(0)))            # (B, 3)


# ── training helpers ─────────────────────────────────────────────────────

def _subset(data: dict, idx) -> dict:
    """Slice all tensors in a data dict by the given indices."""
    idx_t = torch.tensor(idx)
    return {k: v[idx_t] for k, v in data.items()}


def class_weights(labels: torch.Tensor) -> torch.Tensor:
    """Inverse-frequency weights for imbalanced classes."""
    counts = Counter(labels.numpy().tolist())
    total = len(labels)
    n_cls = len(counts)
    w = torch.zeros(max(counts.keys()) + 1)
    for c, cnt in counts.items():
        w[c] = total / (n_cls * cnt)
    return w


def build_groups(sessions: list[dict], n_splits: int) -> np.ndarray:
    """Assign group ids for StratifiedGroupKFold.

    Human: group by participant_id (disjoint people per fold).
    Scripted / AI: group by generator_family if enough families exist
    for the requested fold count; otherwise per-session fallback.
    """
    fam_sets: dict[str, set] = {}
    for s in sessions:
        fam_sets.setdefault(s["actor_label"], set()).add(
            s.get("generator_family") or s["session_id"]
        )
    use_fam = {lb: len(fs) >= n_splits for lb, fs in fam_sets.items()}

    gmap: dict[str, int] = {}
    gid = 0
    groups = []
    for s in sessions:
        lb = s["actor_label"]
        if lb == "human":
            key = s.get("participant_id") or s["session_id"]
        elif use_fam.get(lb, False):
            key = s["generator_family"]
        else:
            key = s["session_id"]
        if key not in gmap:
            gmap[key] = gid
            gid += 1
        groups.append(gmap[key])
    return np.array(groups)


def _train_model(data, vocab_size, seed, n_epochs=60, lr=5e-3):
    """Train a GRU on all data in `data`, return the trained model."""
    torch.manual_seed(seed)
    model = GRUClassifier(vocab_size)
    cw = class_weights(data["labs"])
    criterion = nn.CrossEntropyLoss(weight=cw)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for _ in range(n_epochs):
        model.train()
        logits = model(data["cmd"], data["resp"], data["cats"],
                       data["scal"], data["lens"])
        loss = criterion(logits, data["labs"])
        optim.zero_grad()
        loss.backward()
        optim.step()

    return model


@torch.no_grad()
def _predict(model, data):
    """Return predicted class indices."""
    model.eval()
    logits = model(data["cmd"], data["resp"], data["cats"],
                   data["scal"], data["lens"])
    return logits.argmax(dim=1).numpy()


# ── main routines ────────────────────────────────────────────────────────

def cross_validate(train_sess, enc_train, vocab_size, n_splits=3, seed=42):
    """Grouped stratified CV on the training split."""
    labs = enc_train["labs"].numpy()
    groups = build_groups(train_sess, n_splits)
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )

    fold_f1s = []
    all_t, all_p = [], []

    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(
            np.arange(len(labs)), labs, groups)):
        model = _train_model(_subset(enc_train, tr_idx), vocab_size,
                             seed=seed + fold)
        preds = _predict(model, _subset(enc_train, va_idx))
        f1 = f1_score(labs[va_idx], preds, average="macro")
        fold_f1s.append(f1)
        all_t.extend(labs[va_idx])
        all_p.extend(preds)
        print(f"  fold {fold + 1}: macro-F1 = {f1:.3f}")

    print(f"\n  CV macro-F1: {np.mean(fold_f1s):.3f} +/- {np.std(fold_f1s):.3f}")
    print("\n  CV classification report (all folds pooled):")
    print(classification_report(all_t, all_p,
                                target_names=LABEL_NAMES, digits=3))
    return fold_f1s


def train_and_evaluate(seeds=(42, 123, 456)):
    """Full pipeline: CV on train, then held-out test over multiple seeds."""
    train_sess, test_sess = load_sessions()
    print(f"train: {len(train_sess)}, test: {len(test_sess)}")

    vocab = build_vocab(train_sess)
    enc_train = encode_sessions(train_sess, vocab)
    enc_test = encode_sessions(test_sess, vocab)
    vs = len(vocab)
    print(f"vocab: {vs} tokens\n")

    # --- grouped CV on training split ---
    print("--- cross-validation (training split) ---")
    cross_validate(train_sess, enc_train, vs, n_splits=3, seed=42)

    # --- held-out test (3 seeds, report mean +/- std) ---
    print("\n--- held-out test evaluation ---")
    seed_f1s = []
    best_f1, best_preds = -1.0, None

    for seed in seeds:
        model = _train_model(enc_train, vs, seed=seed)
        preds = _predict(model, enc_test)
        true = enc_test["labs"].numpy()
        f1 = f1_score(true, preds, average="macro")
        seed_f1s.append(f1)
        print(f"  seed {seed}: macro-F1 = {f1:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            best_preds = preds

    print(f"\n  test macro-F1: {np.mean(seed_f1s):.3f} +/- {np.std(seed_f1s):.3f}")

    true = enc_test["labs"].numpy()
    print(f"\n  best seed report (F1={best_f1:.3f}):")
    print(classification_report(true, best_preds,
                                target_names=LABEL_NAMES, digits=3))
    print("  confusion matrix (rows=true, cols=pred):")
    print(f"           {LABEL_NAMES}")
    cm = confusion_matrix(true, best_preds)
    for i, row in enumerate(cm):
        print(f"    {LABEL_NAMES[i]:>10s}  {row}")


def main():
    train_and_evaluate()


if __name__ == "__main__":
    main()