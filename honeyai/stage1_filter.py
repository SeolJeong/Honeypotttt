"""Stage-1 auth-noise filter.

Separates brute-force login spam from sessions that actually succeed and
enter the interactive shell (Stage 2, where the 3-class attribution model
runs).

The dataset has 1,000 auth records:
  - 967 brute_force (failed, hundreds of attempts, fast gaps)
  -  33 credential_stuffing (single successful login, slower gap)

We train a simple logistic-regression classifier on three numeric features
and report accuracy + per-class metrics with stratified cross-validation.
In production this runs as a gate on live Cowrie traffic.

Usage:
    python -m honeyai.stage1_filter          # train + evaluate
    python -m honeyai.stage1_filter --predict data/dataset3_auth_noise.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

DATA = os.path.join("data", "dataset3_auth_noise.csv")

# the three numeric columns that separate noise from interactive sessions
FEATURE_COLS = ["auth_success", "attempts_in_session", "inter_attempt_gap_ms"]


def load(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load dataset3 → (X, y, record_ids)."""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    X = np.array([
        [
            1.0 if r["auth_success"] == "True" else 0.0,
            float(r["attempts_in_session"]),
            float(r["inter_attempt_gap_ms"]),
        ]
        for r in rows
    ], dtype=float)

    # binary label: 1 = enters stage 2 (interactive), 0 = noise
    y = np.array([1 if r["enter_stage2"] == "True" else 0 for r in rows])
    ids = [r["record_id"] for r in rows]
    return X, y, ids


def evaluate(n_splits: int = 5, seed: int = 0):
    """Stratified k-fold evaluation. Prints metrics."""
    X, y, _ = load(DATA)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scaler = StandardScaler()

    f1s = []
    all_true, all_pred = [], []

    for tr, te in skf.split(X, y):
        X_tr = scaler.fit_transform(X[tr])
        X_te = scaler.transform(X[te])
        clf = LogisticRegression(random_state=seed, max_iter=500)
        clf.fit(X_tr, y[tr])
        pred = clf.predict(X_te)
        f1s.append(f1_score(y[te], pred, average="macro"))
        all_true.extend(y[te])
        all_pred.extend(pred)

    labels = ["noise (brute_force)", "interactive (stage2)"]
    print(f"Stage-1 filter — stratified {n_splits}-fold CV")
    print(f"macro-F1: {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}\n")
    print(classification_report(all_true, all_pred, target_names=labels, digits=3))


def predict(path: str):
    """Train on full dataset3, predict on given file, print results."""
    X_train, y_train, _ = load(DATA)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    clf = LogisticRegression(random_state=0, max_iter=500)
    clf.fit(X_train_s, y_train)

    X_new, _, ids = load(path)
    X_new_s = scaler.transform(X_new)
    preds = clf.predict(X_new_s)

    for rid, p in zip(ids, preds):
        label = "interactive" if p == 1 else "noise"
        print(f"{rid}: {label}")

    n_int = int(preds.sum())
    print(f"\n{n_int} interactive / {len(preds) - n_int} noise out of {len(preds)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", help="CSV to classify (same columns as dataset3)")
    args = ap.parse_args()

    if args.predict:
        predict(args.predict)
    else:
        evaluate()


if __name__ == "__main__":
    main()
