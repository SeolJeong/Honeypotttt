"""Shared evaluation for the text models (QLoRA LM and the TF-IDF floor).

Every model returns a probability matrix (n_examples x 3, LABELS order); this
module turns it into the spec section 13 metrics so all text models are
scored identically:

  macro-F1, per-class P/R/F1, confusion matrix
  AI-evasive recall           (AI sessions with evasion_strategy != none)
  per-evasion-strategy AI recall
  human false-attribution     (human sessions predicted AI-driven)
  expected calibration error  (15 equal-width bins on max prob)
  abstention rate             (max prob < threshold -> "unknown")
  PR-AUC per class (one-vs-rest)
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, f1_score)

from honeyai.llm.format import LABELS

AI = LABELS.index("AI-driven")
HUMAN = LABELS.index("human")


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def structured(probs_row: np.ndarray, abstain_below: float = 0.0) -> dict:
    """The spec section 8 output contract: {"class": ..., "confidence": ...}."""
    i = int(np.argmax(probs_row))
    conf = float(probs_row[i])
    cls = LABELS[i] if conf >= abstain_below else "unknown"
    return {"class": cls, "confidence": round(conf, 4),
            "probs": {l: round(float(p), 4) for l, p in zip(LABELS, probs_row)}}


def report(examples: list[dict], probs: np.ndarray, abstain_below: float = 0.5) -> dict:
    y = np.array([e["label"] for e in examples])
    pred = probs.argmax(1)
    evasion = np.array([e["evasion_strategy"] for e in examples])

    out = {
        "n": int(len(y)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1, 2])),
        "accuracy": float((pred == y).mean()),
        "ece": expected_calibration_error(probs, y),
        "abstention_rate": float((probs.max(1) < abstain_below).mean()),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        "per_class": classification_report(y, pred, labels=[0, 1, 2], target_names=LABELS,
                                           output_dict=True, zero_division=0),
    }

    pr_auc = {}
    for c, name in enumerate(LABELS):
        if (y == c).any():
            pr_auc[name] = float(average_precision_score(y == c, probs[:, c]))
    out["pr_auc"] = pr_auc

    ai = y == AI
    ai_ev = ai & (evasion != "none")
    out["ai_evasive_recall"] = float((pred[ai_ev] == AI).mean()) if ai_ev.any() else None
    out["ai_naive_recall"] = float((pred[ai & ~ai_ev] == AI).mean()) if (ai & ~ai_ev).any() else None
    out["ai_recall_by_strategy"] = {
        s: float((pred[ai & (evasion == s)] == AI).mean())
        for s in sorted(set(evasion[ai]))
    }
    hu = y == HUMAN
    out["human_false_attribution"] = float((pred[hu] == AI).mean()) if hu.any() else None
    return out


def print_report(name: str, r: dict) -> None:
    print(f"\n== {name}  (n={r['n']}) ==")
    print(f"macro-F1 {r['macro_f1']:.3f} | acc {r['accuracy']:.3f} | "
          f"ECE {r['ece']:.3f} | abstain {r['abstention_rate']:.2f}")
    if r["ai_evasive_recall"] is not None:
        print(f"AI recall: naive {r['ai_naive_recall']:.3f}  evasive {r['ai_evasive_recall']:.3f}"
              f"  | human->AI false attribution {r['human_false_attribution']:.3f}")
    print("AI recall by strategy: " +
          ", ".join(f"{k}={v:.2f}" for k, v in r["ai_recall_by_strategy"].items()))
    print("confusion (rows=true, cols=pred) " + str(LABELS))
    for name_, row in zip(LABELS, r["confusion_matrix"]):
        print(f"  {name_:>10s} {row}")


def summarise_seeds(reports: list[dict], keys=("macro_f1", "ai_evasive_recall",
                                                "human_false_attribution", "ece")) -> dict:
    out = {}
    for k in keys:
        vals = [r[k] for r in reports if r.get(k) is not None]
        if vals:
            out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n_seeds": len(vals)}
    return out
