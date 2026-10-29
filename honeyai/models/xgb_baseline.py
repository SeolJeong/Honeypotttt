"""Model 1: XGBoost tabular baseline.

Trains on the aggregate features and reports macro-F1 with a GROUP-AWARE,
LEAKAGE-RESISTANT split: no participant appears in both train and test (humans),
and the evaluation is stratified by class. This is the fair-comparison discipline
the spec requires; do not switch to a naive random split for the real results.

Run:  python -m honeyai.models.xgb_baseline
"""
from __future__ import annotations

import os
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

from honeyai.schema import read_sessions
from honeyai.features.extract import extract_matrix

DATA = os.path.join("data", "sessions", "poc_dataset.jsonl")
LABELS = ["human", "scripted", "ai-driven"]
LABEL_IDX = {l: i for i, l in enumerate(LABELS)}


def _groups(sessions: list[dict]) -> list[str]:
    """Group key to prevent leakage: humans grouped by participant, scripted by
    script family, AI by model family. Sessions sharing a group never straddle the
    train/test boundary.

    Adaptive fallback: if a non-human class has fewer than 2 distinct families, a
    family-level grouping would hold the ENTIRE class out of training in some folds
    (the model then never learns it, scoring 0). In that case we group that class
    by session instead - still leakage-safe (a session never spans the split), but
    the class appears in both train and test. When a 2nd family is added later,
    grouping reverts to family-disjoint automatically.
    """
    from collections import defaultdict
    fams_by_label: dict[str, set] = defaultdict(set)
    for s in sessions:
        if s["actor_label"] != "human":
            fams_by_label[s["actor_label"]].add(s.get("generator_family") or "x")

    groups = []
    for s in sessions:
        lab = s["actor_label"]
        if lab == "human":
            groups.append("hum_" + (s.get("participant_id") or s["session_id"]))
        elif len(fams_by_label[lab]) >= 2:
            groups.append(lab + "_" + (s.get("generator_family") or "x"))
        else:
            # single-family class: group per session so it isn't fully held out
            groups.append(lab + "_" + s["session_id"])
    return groups


def run(n_splits: int = 5, seed: int = 0):
    sessions = read_sessions(DATA)
    X, y_str, feat_names = extract_matrix(sessions)
    X = np.array(X, dtype=float)
    y = np.array([LABEL_IDX[l] for l in y_str])
    groups = np.array(_groups(sessions))

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    f1s, all_true, all_pred = [], [], []
    for tr, te in sgkf.split(X, y, groups):
        clf = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, eval_metric="mlogloss",
            random_state=seed,
        )
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        f1s.append(f1_score(y[te], pred, average="macro"))
        all_true.extend(y[te]); all_pred.extend(pred)

    print(f"Model 1 (XGBoost) - leakage-resistant {n_splits}-fold")
    print(f"macro-F1: {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}\n")
    print(classification_report(all_true, all_pred, target_names=LABELS, digits=3))
    print("confusion matrix (rows=true, cols=pred):")
    print("        " + "  ".join(f"{l[:6]:>7}" for l in LABELS))
    cm = confusion_matrix(all_true, all_pred)
    for i, row in enumerate(cm):
        print(f"{LABELS[i][:6]:>6}  " + "  ".join(f"{v:>7}" for v in row))

    # feature importance (top 8), to show what drives the decision
    clf_full = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                             eval_metric="mlogloss", random_state=seed)
    clf_full.fit(X, y)
    imp = sorted(zip(feat_names, clf_full.feature_importances_),
                 key=lambda t: -t[1])[:8]
    print("\ntop features:")
    for name, score in imp:
        print(f"  {name:<24} {score:.3f}")
    return float(np.mean(f1s))


if __name__ == "__main__":
    run()
