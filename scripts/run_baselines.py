"""Every non-LLM model on one dataset, one split, the same metrics.

Models (each keeps its original training regime, no tuning on test):
  timing      depth-3 tree on timing features only (mean / min / std gap):
              the data-driven version of the Palisade-style timing signal
  text        TF-IDF + logistic regression on the LLM's exact input format
  xgboost     Model 1 (19 session features, class weights)
  gru         Model 2 (honeyai.models.gru_sequence), full-batch, 60 epochs
plus commands-only variants (text-cmds, gru-cmds; timing is commands-only by
construction) for the central ablation and for the KYPO real-human check.

Metrics (honeyai.llm.evaluate.report, the same as the LLM notebook):
macro-F1 after 1/3/5 commands and on full sessions, AI recall per evasion
strategy, human->AI false attribution, ECE; seeded models: mean +/- std over seeds.
Early-k evaluation truncates TEST sessions to k commands; models are trained on
full sessions except text (prefix-trained, as in text_baseline.py).

    python scripts/run_baselines.py --data data/v2b/dataset1_sessions.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from honeyai.features import extract_one
from honeyai.llm.evaluate import report
from honeyai.llm.format import LABEL2ID, LABELS, FormatConfig, load_sessions, make_examples
from honeyai.models import gru_sequence as G
from honeyai.models.xgb_baseline import FEATURE_COLS, class_weights

KS = [1, 3, 5, None]
SEEDS = [42, 123, 456]


def truncate(sessions, k):
    if k is None:
        return sessions
    out = []
    for s in sessions:
        t = dict(s)
        t["interactions"] = s["interactions"][:k]
        t["total_steps"] = len(t["interactions"])
        out.append(t)
    return out


def gap_augment(train, frac, p, seed):
    """Equal-effort counterpart of the LLM notebook's GAP_AUG (v4): `frac` of the scripted TRAIN sessions get an
    extra copy in which each inter-command gap is, with probability `p`, resampled from the gaps of the
    NON-scripted TRAIN sessions. Applied to training data only; test and KYPO sessions are never modified."""
    rng = random.Random(seed)
    pool = [st["inter_command_gap_s"] for s in train if s["actor_label"] != "scripted"
            for st in s["interactions"] if st.get("inter_command_gap_s") is not None]
    scripted = sorted(s["session_id"] for s in train if s["actor_label"] == "scripted")
    picked = set(rng.sample(scripted, round(len(scripted) * frac)))
    out = list(train)
    for s in train:
        if s["session_id"] in picked:
            t = copy.deepcopy(s)
            t["session_id"] = s["session_id"] + "#gapaug"
            for st in t["interactions"]:
                if st.get("inter_command_gap_s") is not None and rng.random() < p:
                    st["inter_command_gap_s"] = rng.choice(pool)
            out.append(t)
    return out


def as_examples(sessions):
    return [{"label": LABEL2ID[s["actor_label"]], "evasion_strategy": s.get("evasion_strategy", "none")}
            for s in sessions]


def onehot_probs(labels_idx):
    p = np.zeros((len(labels_idx), 3))
    p[np.arange(len(labels_idx)), labels_idx] = 1.0
    return p


# ── timing ──
def timing_feats(s):
    g = [st["inter_command_gap_s"] for st in s["interactions"][1:]] or [s["interactions"][0]["inter_command_gap_s"]]
    return [float(np.mean(g)), float(np.min(g)), float(np.std(g))]


def fit_timing(train):
    m = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0)
    m.fit([timing_feats(s) for s in train], [LABEL2ID[s["actor_label"]] for s in train])
    return lambda ss: m.predict_proba([timing_feats(s) for s in ss])


# ── text ──
def fit_text(train, fmt):
    ex = make_examples(train, fmt, prefixes=True, max_prefixes=8)
    vec = TfidfVectorizer(token_pattern=r"[^\s]+", ngram_range=(1, 2), sublinear_tf=True)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=10)
    clf.fit(vec.fit_transform([e["text"] for e in ex]), [e["label"] for e in ex])
    return lambda ss: clf.predict_proba(vec.transform([e["text"] for e in make_examples(ss, fmt)]))


# ── xgboost ──
def fit_xgb(train, seed):
    X = np.array([[float(extract_one(s)[c]) for c in FEATURE_COLS] for s in train])
    y = np.array([LABEL2ID[s["actor_label"]] for s in train])
    w = class_weights(y)
    clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                        random_state=seed, eval_metric="mlogloss")
    clf.fit(X, y, sample_weight=np.array([w[v] for v in y]))
    return lambda ss: clf.predict_proba(np.array([[float(extract_one(s)[c]) for c in FEATURE_COLS] for s in ss]))


# ── gru ──
def gru_ready(sessions, commands_only):
    out = []
    for s in sessions:
        t = copy.deepcopy(s)
        for st in t["interactions"]:
            if commands_only or st.get("system_response") is None:
                st["system_response"] = ""
                st["response_category"] = None
            if st.get("task_progress") is None:
                st["task_progress"] = 0.0
            if commands_only:
                st["task_progress"] = 0.0       # progress is derived from responses
        out.append(t)
    return out


def fit_gru(train, seed, commands_only):
    tr = gru_ready(train, commands_only)
    vocab = G.build_vocab(tr)
    model = G._train_model(G.encode_sessions(tr, vocab), len(vocab), seed=seed)

    @torch.no_grad()
    def predict(ss):
        ss = gru_ready(ss, commands_only)
        enc = G.encode_sessions(ss, vocab)
        model.eval()
        logits = model(enc["cmd"], enc["resp"], enc["cats"], enc["scal"], enc["lens"])
        return torch.softmax(logits, 1).numpy()
    return predict


def evaluate(predict, test, kypo):
    by_k, preds = {}, {}
    for k in KS:
        ts = truncate(test, k)
        name = "full" if k is None else f"k={k}"
        probs = np.asarray(predict(ts))
        by_k[name] = report(as_examples(ts), probs)
        preds[name] = {s["session_id"]: [round(float(x), 5) for x in p] for s, p in zip(ts, probs)}
    ext = None
    if kypo is not None:
        p = np.asarray(predict(kypo)).argmax(1)
        tgt = np.array([bool(s.get("on_target_host")) for s in kypo])
        ext = {"all_AI": float((p == 2).mean()), "target_hosts_AI": float((p[tgt] == 2).mean())}
    return by_k, ext, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("data", "dataset1_sessions.json"))
    ap.add_argument("--kypo", default=os.path.join("external", "kypo", "kypo_sessions.json"))
    ap.add_argument("--out", default=os.path.join("runs", "baselines.json"))
    ap.add_argument("--gru-epochs", type=int, default=60)
    ap.add_argument("--gap-aug", type=float, default=0.0, help="fraction of scripted train sessions given a gap-jittered copy (v4: 0.5)")
    ap.add_argument("--gap-aug-p", type=float, default=0.7, help="per-gap resampling probability (v4: 0.7)")
    ap.add_argument("--gap-aug-seed", type=int, default=2026)
    a = ap.parse_args()

    sessions = load_sessions(a.data)
    train = [s for s in sessions if s["split"] == "train"]
    test = [s for s in sessions if s["split"] == "test"]
    if a.gap_aug > 0:
        n0 = len(train)
        train = gap_augment(train, a.gap_aug, a.gap_aug_p, a.gap_aug_seed)
        print(f"gap augmentation: +{len(train) - n0} jittered scripted train copies (frac {a.gap_aug}, p {a.gap_aug_p})")
    kypo = json.load(open(a.kypo, encoding="utf-8")) if os.path.exists(a.kypo) else None
    print(f"data {a.data}: train {len(train)}, test {len(test)}; KYPO {'%d sessions' % len(kypo) if kypo else 'not found'}")

    full, cmds = FormatConfig(), FormatConfig(include_responses=False)
    specs = [
        ("timing", False, lambda s: fit_timing(train), [None], True),
        ("text", False, lambda s: fit_text(train, full), [None], False),
        ("text-cmds", True, lambda s: fit_text(train, cmds), [None], True),
        ("xgboost", False, lambda s: fit_xgb(train, s), SEEDS, False),
        ("gru", False, lambda s: fit_gru(train, s, False), SEEDS, False),
        ("gru-cmds", True, lambda s: fit_gru(train, s, True), SEEDS, True),
    ]
    results, all_preds = {}, {}
    for name, cmds_only, fit, seeds, on_kypo in specs:
        t0 = time.time()
        runs = []
        for s in seeds:
            if s is not None:
                torch.manual_seed(s)
                np.random.seed(s)
            pred = fit(s)
            runs.append(evaluate(pred, test, kypo if on_kypo else None))
        agg = {}
        for kname in runs[0][0]:
            rs = [r[0][kname] for r in runs]
            agg[kname] = {
                "macro_f1": [float(np.mean([r["macro_f1"] for r in rs])), float(np.std([r["macro_f1"] for r in rs]))],
                "ai_evasive_recall": float(np.mean([r["ai_evasive_recall"] for r in rs])),
                "human_false_attribution": float(np.mean([r["human_false_attribution"] for r in rs])),
                "ai_recall_by_strategy": {st: float(np.mean([r["ai_recall_by_strategy"][st] for r in rs]))
                                          for st in rs[0]["ai_recall_by_strategy"]},
                "per_class_f1": {l: float(np.mean([r["per_class"][l]["f1-score"] for r in rs])) for l in LABELS},
            }
        ext = [r[1] for r in runs if r[1]]
        all_preds[name] = {str(s): r[2] for s, r in zip(seeds, runs)}
        results[name] = {"commands_only": cmds_only, "n_seeds": len(seeds), "by_k": agg,
                         "kypo": {k: float(np.mean([e[k] for e in ext])) for k in ext[0]} if ext else None,
                         "secs": round(time.time() - t0, 1)}
        f = agg["full"]
        print(f"{name:<10} full macro-F1 {f['macro_f1'][0]:.3f} ± {f['macro_f1'][1]:.3f} | "
              f"k=1 {agg['k=1']['macro_f1'][0]:.3f} k=3 {agg['k=3']['macro_f1'][0]:.3f} k=5 {agg['k=5']['macro_f1'][0]:.3f} | "
              f"AI-evasive rec {f['ai_evasive_recall']:.3f} | human->AI {f['human_false_attribution']:.3f}"
              + (f" | KYPO->AI {results[name]['kypo']['all_AI']:.3f} (target {results[name]['kypo']['target_hosts_AI']:.3f})"
                 if results[name]["kypo"] else "") + f"  [{results[name]['secs']}s]", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"data": a.data, "seeds": SEEDS, "gap_aug": [a.gap_aug, a.gap_aug_p, a.gap_aug_seed], "results": results}, fh, indent=1)
    pred_out = os.path.splitext(a.out)[0] + "_predictions.json"
    with open(pred_out, "w", encoding="utf-8") as fh:
        json.dump({"data": a.data, "labels": LABELS, "models": all_preds}, fh)
    print(f"wrote {a.out} and {pred_out} (per-session probabilities for scripts/compare_models.py)")


if __name__ == "__main__":
    main()
