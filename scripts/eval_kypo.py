"""Real-human check: how often do commands-only models call real KYPO humans AI?

Every KYPO session is a real person, so the only possible errors are
"human -> scripted" and "human -> AI-driven". The headline number is the
HUMAN FALSE-ATTRIBUTION RATE (share of real human sessions labelled AI-driven).

KYPO has no command outputs, so only models that ignore responses are valid:
  - text-cmds: TF-IDF + logistic regression on commands + timing tokens
               (same formatter as the LLM with include_responses=False)
  - timing:    depth-3 tree on the mean inter-command gap (Palisade-style signal)
For the LLM, run the patched notebook with COMMANDS_ONLY = True and
EXTERNAL_TEST_PATH pointing at external/kypo/kypo_sessions_ds4_commands_only.jsonl.

Caveats to report with the number:
  - KYPO tasks (exploitation training) differ from our 8 tasks, so this
    measures real humans AND a task/domain shift at the same time
  - KYPO gaps include runtime of long commands (scans, cracking); the
    simulator's commands are instant, which pushes KYPO towards "human"
  - target-host sessions (commands typed on a machine reached over SSH) are the
    closest match to a honeypot and are reported separately

    python scripts/eval_kypo.py --train data/v2b/dataset1_sessions.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from honeyai.llm.format import LABELS, FormatConfig, load_sessions, make_examples

CMDS_ONLY = FormatConfig(include_responses=False)


def mean_gap(sess: dict, skip_first: bool) -> float:
    st = sess["interactions"][1:] if skip_first and len(sess["interactions"]) > 1 else sess["interactions"]
    return float(np.mean([s["inter_command_gap_s"] for s in st])) if st else 0.0


def shares(preds: list[str]) -> dict:
    c = Counter(preds)
    return {l: round(c[l] / max(1, len(preds)), 3) for l in LABELS}


def main():
    ap = argparse.ArgumentParser()
    default_train = os.path.join("data", "dataset1_sessions.json")   # main's data/ is v2b
    ap.add_argument("--train", default=default_train)
    ap.add_argument("--kypo", default=os.path.join("external", "kypo", "kypo_sessions.json"))
    ap.add_argument("--out", default=os.path.join("runs", "kypo_eval.json"))
    a = ap.parse_args()

    sims = load_sessions(a.train)
    train = [s for s in sims if s["split"] == "train"]
    sim_test_h = [s for s in sims if s["split"] == "test" and s["actor_label"] == "human"]
    kypo = json.load(open(a.kypo, encoding="utf-8"))
    tgt = [s for s in kypo if s.get("on_target_host")]
    print(f"train: {a.train} ({len(train)} sessions) | KYPO: {len(kypo)} sessions ({len(tgt)} on target hosts)")

    # text, commands only
    tr_ex = make_examples(train, CMDS_ONLY, prefixes=True, max_prefixes=8)
    vec = TfidfVectorizer(token_pattern=r"[^\s]+", ngram_range=(1, 2), sublinear_tf=True)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=10)
    clf.fit(vec.fit_transform([e["text"] for e in tr_ex]), [e["actor_label"] for e in tr_ex])

    def text_pred(sessions):
        ex = make_examples(sessions, CMDS_ONLY)
        return [str(p) for p in clf.predict(vec.transform([e["text"] for e in ex]))]   # fitted on label strings

    # timing only (simulated step 0 is time-since-login, KYPO has none: skip it on the sim side)
    tree = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=0)
    tree.fit([[mean_gap(s, True)] for s in train], [s["actor_label"] for s in train])

    def time_pred(sessions, skip_first):
        return list(tree.predict([[mean_gap(s, skip_first)] for s in sessions]))

    rows = {
        "text-cmds": {"kypo_all": shares(text_pred(kypo)), "kypo_target_hosts": shares(text_pred(tgt)),
                      "sim_test_humans": shares(text_pred(sim_test_h))},
        "timing": {"kypo_all": shares(time_pred(kypo, False)), "kypo_target_hosts": shares(time_pred(tgt, False)),
                   "sim_test_humans": shares(time_pred(sim_test_h, True))},
    }
    by_training = {}
    for t in sorted({s["task"] for s in kypo}):
        sub = [s for s in kypo if s["task"] == t]
        by_training[t] = {"n": len(sub), "text-cmds_AI": shares(text_pred(sub))["AI-driven"],
                          "timing_AI": shares(time_pred(sub, False))["AI-driven"]}

    print("\nshare of sessions labelled as each class (all KYPO sessions are real humans)")
    print(f"{'model':<10} {'subset':<20} {'human':>7} {'scripted':>9} {'AI':>7}   <- AI = human false-attribution")
    for m, r in rows.items():
        for sub, sh in r.items():
            print(f"{m:<10} {sub:<20} {sh['human']:>7.3f} {sh['scripted']:>9.3f} {sh['AI-driven']:>7.3f}")
    print("\nby KYPO training (share labelled AI):")
    for t, v in by_training.items():
        print(f"  {t:<34} n={v['n']:<4} text-cmds {v['text-cmds_AI']:.3f}  timing {v['timing_AI']:.3f}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"train_file": a.train, "kypo_file": a.kypo, "n_kypo": len(kypo), "n_target_hosts": len(tgt),
                   "results": rows, "by_training": by_training}, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
