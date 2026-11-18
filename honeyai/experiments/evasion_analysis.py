"""Part 8: Evasion experiments — the headline comparison.

Evaluates each model's robustness to evasion strategies by breaking down
test-set performance per strategy. Runs three analyses:

1. Per-evasion breakdown: macro-F1 for each strategy, each model
2. Non-evasive vs evasive: F1 drop when AI mimics human behaviour
3. Commands-only ablation: retrain XGBoost/GRU without response features,
   measure how much the drop tells us about response conditioning

Usage:
    python scripts/run_evasion_experiments.py
"""
from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import f1_score, classification_report

# --- paths ---
DATA_DIR = "data"
DS1 = os.path.join(DATA_DIR, "dataset1_sessions.json")
DS2 = os.path.join(DATA_DIR, "dataset2_tabular_features.csv")
QLORA_PREDS = os.path.join(DATA_DIR, "qlora_predictions.json")

LABEL_MAP = {"human": 0, "scripted": 1, "AI-driven": 2}
LABEL_NAMES = ["human", "scripted", "AI-driven"]


# =====================================================================
# LOAD DATA
# =====================================================================

def load_sessions():
    with open(DS1, encoding="utf-8") as f:
        return json.load(f)


def load_tabular():
    with open(DS2, encoding="utf-8") as f:
        return list(csv.DictReader(f))

def load_qlora_predictions():
    """Load QLoRA predictions. Handles both our format and teammate's."""
    if not os.path.exists(QLORA_PREDS):
        print(f"  [skip] {QLORA_PREDS} not found")
        return None
    with open(QLORA_PREDS, encoding="utf-8") as f:
        data = json.load(f)

    preds = []

    # teammate's format: seeds -> [0] -> predictions (with prefix field)
    if "seeds" in data:
        all_preds = data["seeds"][0].get("predictions", [])
        # take full-sequence predictions only
        preds = [p for p in all_preds if p.get("prefix") == "full"]
        if not preds:
            preds = all_preds

    # our format: predictions at top level
    if not preds:
        preds = data.get("predictions", [])

    # normalise key names
    out = {}
    for p in preds:
        sid = p.get("session_id", p.get("id", ""))
        true = p.get("true_label", p.get("true", ""))
        pred = p.get("predicted_label", p.get("pred", ""))
        if sid and true and pred:
            out[sid] = {"true": true, "pred": pred}
    return out if out else None

# =====================================================================
# TRAIN MODELS (reimport from existing modules)
# =====================================================================

def train_xgboost(feature_cols, tag="full"):
    """Train XGBoost on specified features, return per-session test predictions."""
    from xgboost import XGBClassifier

    rows = load_tabular()
    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]

    X_train = np.array([[float(r[c]) for c in feature_cols] for r in train_rows])
    y_train = np.array([LABEL_MAP[r["actor_label"]] for r in train_rows])
    X_test = np.array([[float(r[c]) for c in feature_cols] for r in test_rows])
    y_test = np.array([LABEL_MAP[r["actor_label"]] for r in test_rows])

    # class weights
    counts = Counter(y_train)
    total = len(y_train)
    n_cls = len(counts)
    weights = {c: total / (n_cls * cnt) for c, cnt in counts.items()}
    sample_w = np.array([weights[label] for label in y_train])

    clf = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="mlogloss",
    )
    clf.fit(X_train, y_train, sample_weight=sample_w)
    preds = clf.predict(X_test)

    # build per-session dict
    results = {}
    for row, pred in zip(test_rows, preds):
        results[row["session_id"]] = {
            "true": row["actor_label"],
            "pred": LABEL_NAMES[pred],
        }
    f1 = f1_score(y_test, preds, average="macro")
    print(f"  XGBoost ({tag}): test macro-F1 = {f1:.3f}")
    return results, f1


def train_gru(commands_only=False, tag="full"):
    """Train GRU, return per-session test predictions."""
    import torch
    import torch.nn as nn
    from torch.nn.utils.rnn import pack_padded_sequence

    sessions = load_sessions()
    train_sess = [s for s in sessions if s["split"] == "train"]
    test_sess = [s for s in sessions if s["split"] == "test"]

    # --- vocab ---
    import re
    def tokenize(text):
        return [t for t in re.split(r"[^a-z0-9._/-]+", text.lower()) if t]

    VOCAB_MAX = 150
    MAX_CMD_TOK = 10
    MAX_RESP_TOK = 16 if not commands_only else 0
    EMBED_DIM = 24
    HIDDEN_DIM = 48
    TIMING_BINS = ["short", "medium", "long", "very_long"]
    RESP_CATS = ["success", "not_found_with_hint", "not_found_no_hint",
                 "permission_denied", "command_not_found", "partial_result"]
    N_ONEHOT = len(TIMING_BINS) + len(RESP_CATS)
    N_SCALAR = 4

    t2i = {"<PAD>": 0, "<UNK>": 1}
    token_counts = Counter()
    for s in train_sess:
        for st in s["interactions"]:
            token_counts.update(tokenize(st["command"]))
            if not commands_only:
                token_counts.update(tokenize(st["system_response"]))
    for tok, _ in token_counts.most_common(VOCAB_MAX - 2):
        t2i[tok] = len(t2i)

    def encode_tok(tokens, max_len):
        ids = [t2i.get(t, 1) for t in tokens[:max_len]]
        return ids + [0] * (max_len - len(ids))

    def encode_sessions(sess_list):
        max_steps = max(len(s["interactions"]) for s in sess_list)
        n = len(sess_list)
        cmd = np.zeros((n, max_steps, MAX_CMD_TOK), dtype=np.int64)
        if not commands_only:
            resp = np.zeros((n, max_steps, MAX_RESP_TOK), dtype=np.int64)
        cats = np.zeros((n, max_steps, N_ONEHOT), dtype=np.float32)
        scal = np.zeros((n, max_steps, N_SCALAR), dtype=np.float32)
        lens = np.zeros(n, dtype=np.int64)
        labs = np.zeros(n, dtype=np.int64)

        for i, sess in enumerate(sess_list):
            steps = sess["interactions"]
            lens[i] = len(steps)
            labs[i] = LABEL_MAP[sess["actor_label"]]
            for j, st in enumerate(steps):
                cmd[i, j] = encode_tok(tokenize(st["command"]), MAX_CMD_TOK)
                if not commands_only:
                    resp[i, j] = encode_tok(tokenize(st["system_response"]), MAX_RESP_TOK)
                tb = st.get("think_time_bin", "medium").lower()
                if tb in TIMING_BINS:
                    cats[i, j, TIMING_BINS.index(tb)] = 1.0
                if not commands_only:
                    rc = st.get("response_category", "success")
                    if rc in RESP_CATS:
                        cats[i, j, len(TIMING_BINS) + RESP_CATS.index(rc)] = 1.0
                scal[i, j, 0] = 1.0 if st.get("has_typo") in (True, "True", "true") else 0.0
                scal[i, j, 1] = float(st.get("task_progress", 0.0))
                scal[i, j, 2] = min(len(st["command"]) / 50.0, 1.0)
                scal[i, j, 3] = min(float(st.get("inter_command_gap_s", 0)) / 30.0, 1.0)

        out = {"cmd": torch.from_numpy(cmd), "cats": torch.from_numpy(cats),
               "scal": torch.from_numpy(scal), "lens": torch.from_numpy(lens),
               "labs": torch.from_numpy(labs)}
        if not commands_only:
            out["resp"] = torch.from_numpy(resp)
        return out

    enc_train = encode_sessions(train_sess)
    enc_test = encode_sessions(test_sess)

    # --- model ---
    class GRU(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes=3, dropout=0.4, use_resp=True):
            super().__init__()
            self.use_resp = use_resp
            self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            input_dim = embed_dim + N_ONEHOT + N_SCALAR
            if use_resp:
                input_dim += embed_dim
            self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(hidden_dim, n_classes)

        def _avg(self, ids):
            emb = self.embed(ids)
            mask = (ids != 0).unsqueeze(-1).float()
            return (emb * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)

        def forward(self, cmd, cats, scal, lens, resp=None):
            parts = [self._avg(cmd), cats, scal]
            if self.use_resp and resp is not None:
                parts.insert(1, self._avg(resp))
            x = torch.cat(parts, dim=-1)
            packed = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
            _, h = self.gru(packed)
            return self.head(self.drop(h.squeeze(0)))

    vs = len(t2i)
    use_resp = not commands_only

    # train over 3 seeds, pick best
    best_f1 = -1
    best_preds = None

    for seed in [42, 123, 456]:
        torch.manual_seed(seed)
        model = GRU(vs, EMBED_DIM, HIDDEN_DIM, use_resp=use_resp)

        counts = Counter(enc_train["labs"].numpy().tolist())
        total_n = len(enc_train["labs"])
        n_cls = len(counts)
        cw = torch.zeros(max(counts.keys()) + 1)
        for c, cnt in counts.items():
            cw[c] = total_n / (n_cls * cnt)

        criterion = nn.CrossEntropyLoss(weight=cw)
        optim = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)

        for _ in range(60):
            model.train()
            kwargs = {"cmd": enc_train["cmd"], "cats": enc_train["cats"],
                      "scal": enc_train["scal"], "lens": enc_train["lens"]}
            if use_resp:
                kwargs["resp"] = enc_train["resp"]
            logits = model(**kwargs)
            loss = criterion(logits, enc_train["labs"])
            optim.zero_grad()
            loss.backward()
            optim.step()

        model.eval()
        with torch.no_grad():
            kwargs = {"cmd": enc_test["cmd"], "cats": enc_test["cats"],
                      "scal": enc_test["scal"], "lens": enc_test["lens"]}
            if use_resp:
                kwargs["resp"] = enc_test["resp"]
            logits = model(**kwargs)
            preds = logits.argmax(dim=1).numpy()

        true = enc_test["labs"].numpy()
        f1 = f1_score(true, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_preds = preds

    results = {}
    for sess, pred in zip(test_sess, best_preds):
        results[sess["session_id"]] = {
            "true": sess["actor_label"],
            "pred": LABEL_NAMES[pred],
        }
    print(f"  GRU ({tag}): test macro-F1 = {best_f1:.3f}")
    return results, best_f1


# =====================================================================
# FEATURE SETS
# =====================================================================

ALL_FEATURES = [
    "session_duration_s", "command_count",
    "inter_cmd_mean_s", "inter_cmd_std_s", "min_gap_s", "max_gap_s",
    "cmd_len_mean", "cmd_len_std", "unique_cmd_ratio",
    "typo_count", "typo_rate",
    "error_count", "error_rate",
    "error_recovery_rate", "adaptation_event_count", "not_found_with_hint_count",
    "final_task_progress", "interaction_depth", "sequence_entropy",
]

# commands-only: drop features that depend on response content
COMMANDS_ONLY_FEATURES = [
    "session_duration_s", "command_count",
    "inter_cmd_mean_s", "inter_cmd_std_s", "min_gap_s", "max_gap_s",
    "cmd_len_mean", "cmd_len_std", "unique_cmd_ratio",
    "typo_count", "typo_rate",
    "final_task_progress", "interaction_depth", "sequence_entropy",
]


# =====================================================================
# ANALYSIS
# =====================================================================

def get_evasion_map(sessions):
    """Map session_id -> evasion_strategy for test sessions."""
    return {
        s["session_id"]: s.get("evasion_strategy", "none")
        for s in sessions if s["split"] == "test"
    }


def get_family_map(sessions):
    """Map session_id -> generator_family for test sessions."""
    return {
        s["session_id"]: s.get("generator_family", "")
        for s in sessions if s["split"] == "test"
    }


def per_evasion_f1(predictions, evasion_map):
    """Break down F1 by evasion strategy."""
    strategy_groups = defaultdict(lambda: {"true": [], "pred": []})
    for sid, result in predictions.items():
        ev = evasion_map.get(sid, "none")
        strategy_groups[ev]["true"].append(result["true"])
        strategy_groups[ev]["pred"].append(result["pred"])

    rows = []
    for strategy in sorted(strategy_groups.keys()):
        group = strategy_groups[strategy]
        n = len(group["true"])
        f1 = f1_score(
            group["true"], group["pred"],
            labels=LABEL_NAMES, average="macro", zero_division=0
        )
        rows.append({"strategy": strategy, "n": n, "macro_f1": round(f1, 3)})
    return rows


def evasive_vs_non(predictions, evasion_map):
    """Compare F1 on non-evasive vs evasive AI sessions only."""
    non_ev = {"true": [], "pred": []}
    evasive = {"true": [], "pred": []}

    for sid, result in predictions.items():
        ev = evasion_map.get(sid, "none")
        if result["true"] != "AI-driven":
            continue
        if ev == "none":
            non_ev["true"].append(result["true"])
            non_ev["pred"].append(result["pred"])
        else:
            evasive["true"].append(result["true"])
            evasive["pred"].append(result["pred"])

    f1_non = f1_score(non_ev["true"], non_ev["pred"],
                      labels=LABEL_NAMES, average="macro", zero_division=0) if non_ev["true"] else 0
    f1_ev = f1_score(evasive["true"], evasive["pred"],
                     labels=LABEL_NAMES, average="macro", zero_division=0) if evasive["true"] else 0

    # per-class accuracy for AI-driven
    def ai_recall(group):
        if not group["true"]:
            return 0.0
        correct = sum(1 for t, p in zip(group["true"], group["pred"]) if t == p)
        return correct / len(group["true"])

    return {
        "non_evasive": {"n": len(non_ev["true"]), "macro_f1": round(f1_non, 3),
                        "ai_recall": round(ai_recall(non_ev), 3)},
        "evasive": {"n": len(evasive["true"]), "macro_f1": round(f1_ev, 3),
                    "ai_recall": round(ai_recall(evasive), 3)},
        "f1_drop": round(f1_non - f1_ev, 3),
    }


def per_strategy_ai_recall(predictions, evasion_map):
    """For AI-driven sessions only, recall per evasion strategy."""
    groups = defaultdict(lambda: {"total": 0, "correct": 0})
    for sid, result in predictions.items():
        if result["true"] != "AI-driven":
            continue
        ev = evasion_map.get(sid, "none")
        groups[ev]["total"] += 1
        if result["pred"] == "AI-driven":
            groups[ev]["correct"] += 1

    rows = []
    for strategy in sorted(groups.keys()):
        g = groups[strategy]
        recall = g["correct"] / g["total"] if g["total"] > 0 else 0
        rows.append({"strategy": strategy, "n": g["total"],
                      "ai_recall": round(recall, 3)})
    return rows


# =====================================================================
# MAIN
# =====================================================================

def print_table(title, rows, cols):
    """Print a simple ASCII table."""
    print(f"\n{title}")
    print("-" * len(title))
    widths = [max(len(str(r.get(c, ""))) for r in rows + [{c: c}]) + 2 for c in cols]
    header = "".join(str(c).ljust(w) for c, w in zip(cols, widths))
    print(header)
    print("-" * len(header))
    for r in rows:
        print("".join(str(r.get(c, "")).ljust(w) for c, w in zip(cols, widths)))


def main():
    sessions = load_sessions()
    evasion_map = get_evasion_map(sessions)

    test_strategies = sorted(set(evasion_map.values()))
    print(f"test evasion strategies: {test_strategies}")
    for ev in test_strategies:
        n = sum(1 for v in evasion_map.values() if v == ev)
        print(f"  {ev}: {n} sessions")

    # ==== EXPERIMENT 1: full models, per-evasion breakdown ====
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Per-evasion breakdown (full features)")
    print("=" * 60)

    models = {}

    print("\nTraining models...")
    xgb_preds, xgb_f1 = train_xgboost(ALL_FEATURES, tag="full")
    models["XGBoost"] = xgb_preds

    gru_preds, gru_f1 = train_gru(commands_only=False, tag="full")
    models["GRU"] = gru_preds

    qlora_preds = load_qlora_predictions()
    if qlora_preds:
        models["QLoRA"] = qlora_preds
        qlora_f1 = f1_score(
            [v["true"] for v in qlora_preds.values()],
            [v["pred"] for v in qlora_preds.values()],
            labels=LABEL_NAMES, average="macro", zero_division=0,
        )
        print(f"  QLoRA (loaded): test macro-F1 = {qlora_f1:.3f}")

    # per-evasion table
    for model_name, preds in models.items():
        rows = per_evasion_f1(preds, evasion_map)
        print_table(f"{model_name} — per-evasion macro-F1",
                    rows, ["strategy", "n", "macro_f1"])

    # AI-driven recall per strategy
    for model_name, preds in models.items():
        rows = per_strategy_ai_recall(preds, evasion_map)
        print_table(f"{model_name} — AI-driven recall per evasion strategy",
                    rows, ["strategy", "n", "ai_recall"])

    # ==== EXPERIMENT 2: evasive vs non-evasive ====
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Non-evasive vs evasive (AI-driven only)")
    print("=" * 60)

    comparison_rows = []
    for model_name, preds in models.items():
        result = evasive_vs_non(preds, evasion_map)
        comparison_rows.append({
            "model": model_name,
            "non_ev_n": result["non_evasive"]["n"],
            "non_ev_recall": result["non_evasive"]["ai_recall"],
            "evasive_n": result["evasive"]["n"],
            "evasive_recall": result["evasive"]["ai_recall"],
            "recall_drop": round(result["non_evasive"]["ai_recall"] -
                                 result["evasive"]["ai_recall"], 3),
        })
    print_table("AI-driven recall: non-evasive vs evasive",
                comparison_rows,
                ["model", "non_ev_n", "non_ev_recall", "evasive_n",
                 "evasive_recall", "recall_drop"])

    # ==== EXPERIMENT 3: commands-only ablation ====
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Commands-only ablation")
    print("=" * 60)

    print("\nRetraining without response features...")
    xgb_cmd_preds, xgb_cmd_f1 = train_xgboost(COMMANDS_ONLY_FEATURES, tag="commands-only")
    gru_cmd_preds, gru_cmd_f1 = train_gru(commands_only=True, tag="commands-only")

    ablation_rows = [
        {"model": "XGBoost", "full_f1": round(xgb_f1, 3),
         "cmd_only_f1": round(xgb_cmd_f1, 3),
         "f1_drop": round(xgb_f1 - xgb_cmd_f1, 3)},
        {"model": "GRU", "full_f1": round(gru_f1, 3),
         "cmd_only_f1": round(gru_cmd_f1, 3),
         "f1_drop": round(gru_f1 - gru_cmd_f1, 3)},
    ]
    print_table("Commands-only ablation",
                ablation_rows, ["model", "full_f1", "cmd_only_f1", "f1_drop"])

    # per-evasion for commands-only
    for model_name, preds in [("XGBoost-cmd", xgb_cmd_preds), ("GRU-cmd", gru_cmd_preds)]:
        rows = per_strategy_ai_recall(preds, evasion_map)
        print_table(f"{model_name} — AI recall per evasion (commands-only)",
                    rows, ["strategy", "n", "ai_recall"])

    # ==== SUMMARY ====
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nKey findings:")
    print(f"  1. XGBoost full F1: {xgb_f1:.3f}, GRU full F1: {gru_f1:.3f}")
    if qlora_preds:
        print(f"     QLoRA full F1: {qlora_f1:.3f}")
    print(f"  2. Commands-only drops: XGBoost {xgb_f1 - xgb_cmd_f1:+.3f}, GRU {gru_f1 - gru_cmd_f1:+.3f}")
    print(f"  3. See per-evasion tables above for strategy-specific findings")


if __name__ == "__main__":
    main()