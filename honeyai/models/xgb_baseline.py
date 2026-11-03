"""Part 5: XGBoost three-class attribution (Model 1).

Trains on the 19 numeric features from dataset2_tabular_features.csv.
Uses StratifiedGroupKFold for leakage-resistant CV (group by participant
for humans, by generator_family for scripted/AI, with per-session fallback
when a class has fewer groups than folds).

Usage:
    python scripts/train_xgboost.py
"""
from __future__ import annotations

import csv
import os
import json
import numpy as np
from collections import Counter

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from xgboost import XGBClassifier

DATA_DIR = "data"
DS2 = os.path.join(DATA_DIR, "dataset2_tabular_features.csv")

# the 19 numeric features (order must match dataset2 columns)
FEATURE_COLS = [
    "session_duration_s", "command_count",
    "inter_cmd_mean_s", "inter_cmd_std_s", "min_gap_s", "max_gap_s",
    "cmd_len_mean", "cmd_len_std", "unique_cmd_ratio",
    "typo_count", "typo_rate",
    "error_count", "error_rate",
    "error_recovery_rate", "adaptation_event_count", "not_found_with_hint_count",
    "final_task_progress", "interaction_depth", "sequence_entropy",
]

LABEL_MAP = {"human": 0, "scripted": 1, "AI-driven": 2}
LABEL_NAMES = ["human", "scripted", "AI-driven"]


def load_dataset() -> dict:
    """Load dataset2, return dict with train/test arrays and metadata."""
    with open(DS2, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]

    def to_arrays(subset):
        X = np.array([[float(r[c]) for c in FEATURE_COLS] for r in subset])
        y = np.array([LABEL_MAP[r["actor_label"]] for r in subset])
        return X, y

    X_train, y_train = to_arrays(train_rows)
    X_test, y_test = to_arrays(test_rows)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "train_rows": train_rows, "test_rows": test_rows,
    }


def build_groups(rows: list[dict], n_splits: int) -> np.ndarray:
    """Assign a group id to each row for StratifiedGroupKFold.

    - human: group by participant_id (P001, P002, P003)
    - scripted / AI-driven: group by generator_family if there are
      enough families for the requested folds; otherwise fall back
      to per-session grouping (each session = its own group).
    """
    # count distinct families per class in this subset
    class_families: dict[str, set] = {}
    for r in rows:
        label = r["actor_label"]
        fam = r.get("generator_family", r["session_id"])
        class_families.setdefault(label, set()).add(fam)

    # decide grouping strategy per class
    use_family: dict[str, bool] = {}
    for label, fams in class_families.items():
        use_family[label] = len(fams) >= n_splits

    # assign group ids: same family/participant -> same integer
    group_map: dict[str, int] = {}
    next_id = 0
    groups = []

    for r in rows:
        label = r["actor_label"]
        if label == "human":
            key = r.get("participant_id", r["session_id"])
        elif use_family.get(label, False):
            key = r["generator_family"]
        else:
            key = r["session_id"]  # per-session fallback

        if key not in group_map:
            group_map[key] = next_id
            next_id += 1
        groups.append(group_map[key])

    return np.array(groups)


def class_weights(y: np.ndarray) -> dict[int, float]:
    """Inverse-frequency class weights for imbalanced labels."""
    counts = Counter(y)
    total = len(y)
    n_classes = len(counts)
    return {c: total / (n_classes * cnt) for c, cnt in counts.items()}


def cross_validate(X, y, groups, n_splits=3, seed=42):
    """Grouped stratified CV. Returns per-fold metrics."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    weights = class_weights(y)
    sample_w = np.array([weights[label] for label in y])

    fold_f1s = []
    all_true, all_pred = [], []

    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(X, y, groups)):
        clf = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            eval_metric="mlogloss",
            use_label_encoder=False,
        )
        clf.fit(X[tr_idx], y[tr_idx], sample_weight=sample_w[tr_idx])
        preds = clf.predict(X[va_idx])

        f1 = f1_score(y[va_idx], preds, average="macro")
        fold_f1s.append(f1)
        all_true.extend(y[va_idx])
        all_pred.extend(preds)

        print(f"  fold {fold+1}: macro-F1 = {f1:.3f}")

    print(f"\n  CV macro-F1: {np.mean(fold_f1s):.3f} +/- {np.std(fold_f1s):.3f}")
    print("\n  CV classification report (all folds pooled):")
    print(classification_report(all_true, all_pred,
                                target_names=LABEL_NAMES, digits=3))
    return fold_f1s


def train_and_evaluate(seed=42):
    """Train on full training split, evaluate on held-out test split."""
    data = load_dataset()
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    print(f"train: {len(X_train)} sessions, test: {len(X_test)} sessions")
    print(f"train label dist: {dict(Counter(y_train))}")
    print(f"test  label dist: {dict(Counter(y_test))}")

    # --- grouped CV on training split ---
    print("\n--- cross-validation (training split) ---")
    groups = build_groups(data["train_rows"], n_splits=3)
    cross_validate(X_train, y_train, groups, n_splits=3, seed=seed)

    # --- final model on full training split ---
    print("\n--- held-out test evaluation ---")
    weights = class_weights(y_train)
    sample_w = np.array([weights[label] for label in y_train])

    clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        eval_metric="mlogloss",
        use_label_encoder=False,
    )
    clf.fit(X_train, y_train, sample_weight=sample_w)
    preds = clf.predict(X_test)

    test_f1 = f1_score(y_test, preds, average="macro")
    print(f"\ntest macro-F1: {test_f1:.3f}")
    print("\nclassification report:")
    print(classification_report(y_test, preds,
                                target_names=LABEL_NAMES, digits=3))
    print("confusion matrix (rows=true, cols=pred):")
    print(f"         {LABEL_NAMES}")
    cm = confusion_matrix(y_test, preds)
    for i, row in enumerate(cm):
        print(f"  {LABEL_NAMES[i]:>10s}  {row}")

    # --- feature importance ---
    print("\nfeature importance (gain):")
    importances = clf.feature_importances_
    ranked = sorted(zip(FEATURE_COLS, importances), key=lambda x: -x[1])
    for name, imp in ranked:
        bar = "#" * int(imp * 50)
        print(f"  {name:30s} {imp:.4f}  {bar}")

    return clf, test_f1


def main():
    train_and_evaluate(seed=42)


if __name__ == "__main__":
    main()