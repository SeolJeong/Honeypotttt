"""Shared online-scoring logic, used by both the CLI demo and the dashboard.

Wraps a trained classifier and exposes per-command scoring over a partial session,
plus an adaptive-threat score. Keeping this in one place means the dashboard and any
batch evaluation score identically.
"""
from __future__ import annotations

import numpy as np

from honeyai.features.extract import extract, FEATURE_NAMES
from honeyai.models.xgb_baseline import LABELS, LABEL_IDX


def partial(session: dict, k: int) -> dict:
    return {**session, "steps": session["steps"][:k]}


def threat_score(session_prefix: dict, proba: np.ndarray) -> float:
    """Adaptive-threat score in [0, 1]. Combines P(reasoning actor) with observed
    adaptation depth, so a capable human can outrank a weak agent (per the spec:
    attribution and threat are separate signals).
    """
    p_ai = float(proba[LABEL_IDX["ai-driven"]])
    p_human = float(proba[LABEL_IDX["human"]])
    feats = extract(session_prefix)
    adaptation = 0.5 * feats["hint_follow_rate"] + 0.5 * feats["progress_after_failure"]
    depth = min(feats["n_steps"] / 10.0, 1.0)
    # reasoning actors (AI or human) with high adaptation and depth score highest
    reasoning = max(p_ai, 0.6 * p_human)
    return round(float(min(1.0, 0.6 * reasoning + 0.3 * adaptation + 0.1 * depth)), 3)


class OnlineScorer:
    def __init__(self, clf):
        self.clf = clf

    def score_step(self, session: dict, k: int) -> dict:
        """Score after the k-th command. Returns probs, prediction, threat."""
        pref = partial(session, k)
        x = np.array([[extract(pref)[f] for f in FEATURE_NAMES]], dtype=float)
        proba = self.clf.predict_proba(x)[0]
        return {
            "step": k,
            "command": session["steps"][k - 1]["command"],
            "response_category": session["steps"][k - 1]["response_category"],
            "probs": {LABELS[i]: float(proba[i]) for i in range(len(LABELS))},
            "prediction": LABELS[int(np.argmax(proba))],
            "threat": threat_score(pref, proba),
        }

    def score_session(self, session: dict) -> list[dict]:
        return [self.score_step(session, k) for k in range(1, len(session["steps"]) + 1)]


def train_default_scorer(sessions: list[dict], seed: int = 0) -> OnlineScorer:
    """Train Model 1 on all provided sessions (dashboard convenience)."""
    from xgboost import XGBClassifier
    X = np.array([[extract(s)[f] for f in FEATURE_NAMES] for s in sessions], dtype=float)
    y = np.array([LABEL_IDX[s["actor_label"]] for s in sessions])
    clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                        eval_metric="mlogloss", random_state=seed)
    clf.fit(X, y)
    return OnlineScorer(clf)
