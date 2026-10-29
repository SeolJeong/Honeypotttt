"""Per-command (online) scoring demo.

Proves the real-time concept the brief wants: instead of scoring a whole finished
session, we replay one session command-by-command and re-score after every step,
showing how the attacker-type prediction firms up as evidence accumulates. This is
the logic the dashboard will drive; here it prints to the terminal.

Run:  python scripts/online_demo.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from honeyai.schema import read_sessions
from honeyai.features.extract import extract, FEATURE_NAMES
from honeyai.models.xgb_baseline import LABELS, LABEL_IDX, _groups

DATA = os.path.join("data", "sessions", "poc_dataset.jsonl")


def partial_session(session: dict, k: int) -> dict:
    """A copy of the session truncated to its first k steps."""
    return {**session, "steps": session["steps"][:k]}


def main():
    sessions = read_sessions(DATA)
    # simple holdout just for the demo (the real eval uses grouped CV)
    idx = np.arange(len(sessions))
    tr, te = train_test_split(idx, test_size=0.2, random_state=0,
                              stratify=[s["actor_label"] for s in sessions])
    Xtr = np.array([[extract(sessions[i])[k] for k in FEATURE_NAMES] for i in tr])
    ytr = np.array([LABEL_IDX[sessions[i]["actor_label"]] for i in tr])
    clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                        eval_metric="mlogloss", random_state=0)
    clf.fit(Xtr, ytr)

    # pick one interesting test session per class to replay
    shown = set()
    for i in te:
        s = sessions[i]
        lab = s["actor_label"]
        if lab in shown or len(s["steps"]) < 3:
            continue
        shown.add(lab)
        print("=" * 64)
        print(f"Replaying {s['session_id']}  true label: {lab}  "
              f"(family={s.get('generator_family')}, evasion={s.get('evasion_strategy')})")
        print("-" * 64)
        for k in range(1, len(s["steps"]) + 1):
            feats = [extract(partial_session(s, k))[f] for f in FEATURE_NAMES]
            proba = clf.predict_proba(np.array([feats]))[0]
            cmd = s["steps"][k - 1]["command"]
            probs = "  ".join(f"{LABELS[j][:3]}:{proba[j]:.2f}" for j in range(len(LABELS)))
            pred = LABELS[int(np.argmax(proba))]
            print(f"  step {k:>2} | {cmd[:34]:<34} | {probs} | -> {pred}")
        if len(shown) == len(LABELS):
            break
    print("=" * 64)
    print("Each row is the prediction AFTER that command - this is what the "
          "dashboard streams live.")


if __name__ == "__main__":
    main()
