"""TF-IDF + logistic regression on the exact same text the LM sees.

Not one of the three spec models: it is a cheap floor. If the QLoRA LM does
not beat this, "pretrained LM semantics help" is not supported, whatever the
GRU does. Uses the same formatter, splits, prefixes and metrics as Model 3.

Usage:
    python scripts/text_baseline.py                  # full input
    python scripts/text_baseline.py --no-responses   # commands-only ablation
    python scripts/text_baseline.py --round 0
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from honeyai.llm.evaluate import print_report, report
from honeyai.llm.format import FormatConfig, load_sessions, make_examples, select_round


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("data", "dataset1_sessions.json"))
    ap.add_argument("--round", type=int, choices=[0, 2], default=2)
    ap.add_argument("--no-responses", action="store_true")
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--no-category", action="store_true")
    ap.add_argument("--shuffle-steps", action="store_true")
    a = ap.parse_args()
    fmt = FormatConfig(include_responses=not a.no_responses, include_timing=not a.no_timing,
                       include_category=not a.no_category, shuffle_steps=a.shuffle_steps)

    sessions = load_sessions(a.data)
    train = make_examples(select_round([s for s in sessions if s["split"] == "train"], a.round),
                          fmt, prefixes=True, max_prefixes=8)
    test_s = [s for s in sessions if s["split"] == "test"]

    vec = TfidfVectorizer(token_pattern=r"[^\s]+", ngram_range=(1, 2), sublinear_tf=True)
    X = vec.fit_transform([e["text"] for e in train])
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=10)
    clf.fit(X, [e["label"] for e in train])

    print(f"text baseline | {a.data} | round {a.round} | input {fmt.tag()} | {len(train)} train examples")
    for k in [1, 3, 5, 10, None]:
        ex = make_examples(test_s, fmt, k=k)
        probs = clf.predict_proba(vec.transform([e["text"] for e in ex]))
        probs = probs[:, np.argsort(clf.classes_)]   # columns in LABELS order
        r = report(ex, probs)
        if k is None:
            print_report("test, full session", r)
        else:
            print(f"  k={k}: macro-F1 {r['macro_f1']:.3f}  AI-evasive recall "
                  f"{r['ai_evasive_recall']:.3f}")


if __name__ == "__main__":
    main()
