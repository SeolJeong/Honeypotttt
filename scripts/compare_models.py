"""Is Model 3 (the LLM) actually better than the other models? Paired test.

Every model is scored on the SAME test sessions. For each baseline B:
  - macro-F1(LLM) - macro-F1(B), with a 95% paired bootstrap CI (resampling
    sessions; both models are scored on each resample)
  - one-sided bootstrap p-value P(diff <= 0), Holm-corrected across baselines
  - exact McNemar test on per-session correctness
  - the same paired difference for AI-evasive recall and for the human
    false-attribution rate (lower is better)
Decision rule (fix it BEFORE looking at test results):
  "LLM better than B" only if the 95% CI of the macro-F1 difference is
  entirely above 0 AND the Holm-corrected p < 0.05. CI containing 0 means
  "no significant difference". Seeds: predicted probabilities are averaged over
  the seeds a model has (a seed ensemble); per-seed mean +/- std is also shown.

Inputs
  --baselines runs/baselines_v2b_predictions.json     (scripts/run_baselines.py)
  --llm FILE [FILE ...]   any of:
        qlora_patched_results.json                    (patched notebook)
        runs/qlora/<run>/seed*/test_predictions.jsonl (scripts/train_qlora.py)
  --data data/v2b/dataset1_sessions.json               (true labels, strategies)

    python scripts/compare_models.py --llm qlora_patched_results.json
    python scripts/compare_models.py --llm runs/qlora/*/seed*/test_predictions.jsonl --prefix k=3
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import f1_score

from honeyai.llm.format import LABELS, load_sessions


def load_llm(paths: list[str], prefix: str) -> dict[str, dict[str, list[float]]]:
    """-> {seed_name: {session_id: probs}}"""
    seeds = {}
    for p in paths:
        if p.endswith(".jsonl"):                                   # train_qlora.py
            rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            name = os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p)
            seeds[name] = {r["session_id"]: [r["probs"][l] for l in LABELS] for r in rows
                           if r.get("prefix", "full") == prefix and r.get("probs")}
        else:                                                      # patched notebook
            d = json.load(open(p, encoding="utf-8"))
            for sd in d["seeds"]:
                seeds[f"seed{sd['seed']}"] = {r["session_id"]: [r["probs"][l] for l in LABELS]
                                              for r in sd["predictions"] if r["prefix"] == prefix}
    return {k: v for k, v in seeds.items() if v}


def metrics(y, pred, evasive_ai, human):
    return {"macro_f1": float(f1_score(y, pred, average="macro", labels=[0, 1, 2])),
            "ai_evasive_recall": float((pred[evasive_ai] == 2).mean()) if evasive_ai.any() else float("nan"),
            "human_false_attribution": float((pred[human] == 2).mean()) if human.any() else float("nan")}


def mcnemar_exact(a_ok, b_ok):
    b = int((a_ok & ~b_ok).sum())      # A right, B wrong
    c = int((~a_ok & b_ok).sum())      # A wrong, B right
    n = b + c
    if n == 0:
        return 1.0, b, c
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p), b, c


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (len(items) - i) * p))
        out[k] = running
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", default=os.path.join("runs", "baselines_v2b_predictions.json"))
    ap.add_argument("--llm", nargs="*", default=[], help="LLM prediction file(s); see docstring")
    ap.add_argument("--llm-name", default="LLM")
    ap.add_argument("--llm-from-baselines", default=None, help="treat this baseline as the 'LLM' (to test the script)")
    ap.add_argument("--others", nargs="*", default=[], metavar="NAME=FILE",
                    help="extra models to compare against, e.g. zero-shot LLMs: qwen7b-zs=zs_qwen7b.json "
                         "gpt4o-mini-zs=runs/zero_shot/gpt4o-mini.jsonl (same file formats as --llm)")
    ap.add_argument("--data", default=os.path.join("data", "dataset1_sessions.json"))
    ap.add_argument("--prefix", default="full", help="full | k=1 | k=3 | k=5")
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--out", default=os.path.join("runs", "model3_comparison.json"))
    a = ap.parse_args()

    base = json.load(open(a.baselines, encoding="utf-8"))["models"]
    models = {m: {s: v[a.prefix] for s, v in seeds.items()} for m, seeds in base.items()}
    if a.llm_from_baselines:
        llm_name, llm = a.llm_from_baselines, models.pop(a.llm_from_baselines)
    else:
        if not a.llm:
            sys.exit("give --llm prediction file(s) (or --llm-from-baselines NAME to test the script)")
        llm_name, llm = a.llm_name, load_llm(a.llm, a.prefix)
        if not llm:
            sys.exit(f"no '{a.prefix}' predictions found in {a.llm}")

    for spec in a.others:
        name, _, path = spec.partition("=")
        if not path:
            sys.exit(f"--others needs NAME=FILE, got {spec!r}")
        got = load_llm([path], a.prefix)
        if not got:
            sys.exit(f"no '{a.prefix}' predictions in {path}")
        models[name] = got

    truth = {s["session_id"]: s for s in load_sessions(a.data) if s["split"] == "test"}
    ids = set(truth)
    for seeds in [llm, *models.values()]:
        for v in seeds.values():
            ids &= set(v)
    ids = sorted(ids)
    if len(ids) < len(truth):
        print(f"NOTE: comparing on the {len(ids)} test sessions every model has (of {len(truth)})")
    y = np.array([LABELS.index(truth[i]["actor_label"]) for i in ids])
    evasive_ai = np.array([truth[i]["actor_label"] == "AI-driven" and truth[i]["evasion_strategy"] != "none" for i in ids])
    human = y == 0

    def ensemble(seeds):
        return np.mean([[v[i] for i in ids] for v in seeds.values()], axis=0).argmax(1)

    def per_seed(seeds):
        f = [metrics(y, np.array([v[i] for i in ids]).argmax(1), evasive_ai, human)["macro_f1"] for v in seeds.values()]
        return float(np.mean(f)), float(np.std(f)), len(f)

    llm_pred = ensemble(llm)
    llm_m = metrics(y, llm_pred, evasive_ai, human)
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(ids), len(ids)) for _ in range(a.boot)]

    rows, pvals = {}, {}
    for name, seeds in models.items():
        bp = ensemble(seeds)
        bm = metrics(y, bp, evasive_ai, human)
        d_f1, d_ev, d_hf = [], [], []
        for b in boots:
            lm, mm = metrics(y[b], llm_pred[b], evasive_ai[b], human[b]), metrics(y[b], bp[b], evasive_ai[b], human[b])
            d_f1.append(lm["macro_f1"] - mm["macro_f1"])
            d_ev.append(lm["ai_evasive_recall"] - mm["ai_evasive_recall"])
            d_hf.append(lm["human_false_attribution"] - mm["human_false_attribution"])
        d_f1 = np.array(d_f1)
        pvals[name] = float((d_f1 <= 0).mean())
        p_mc, b_only, c_only = mcnemar_exact(llm_pred == y, bp == y)
        rows[name] = {
            "baseline": bm, "baseline_per_seed_f1": per_seed(seeds),
            "diff_macro_f1": float(llm_m["macro_f1"] - bm["macro_f1"]),
            "diff_macro_f1_ci95": [float(np.percentile(d_f1, 2.5)), float(np.percentile(d_f1, 97.5))],
            "diff_ai_evasive_recall_ci95": [float(np.nanpercentile(d_ev, 2.5)), float(np.nanpercentile(d_ev, 97.5))],
            "diff_human_false_attr_ci95": [float(np.nanpercentile(d_hf, 2.5)), float(np.nanpercentile(d_hf, 97.5))],
            "p_one_sided": pvals[name], "mcnemar_p": p_mc, "llm_only_correct": b_only, "baseline_only_correct": c_only,
        }
    for name, p in holm(pvals).items():
        r = rows[name]
        r["p_holm"] = p
        lo, hi = r["diff_macro_f1_ci95"]
        r["verdict"] = ("LLM BETTER" if lo > 0 and p < 0.05 else "LLM WORSE" if hi < 0 else "no significant difference")

    m, s, n = per_seed(llm)
    print(f"\n{llm_name} on {len(ids)} test sessions ({a.prefix}): macro-F1 {llm_m['macro_f1']:.3f} "
          f"(seed ensemble of {n}; per-seed {m:.3f} ± {s:.3f}) | AI-evasive recall {llm_m['ai_evasive_recall']:.3f} "
          f"| human->AI {llm_m['human_false_attribution']:.3f}\n")
    print(f"{'vs baseline':<11} {'base F1':>7} {'diff':>7} {'95% CI':>17} {'p(Holm)':>8} {'McNemar p':>9}  verdict")
    for name, r in sorted(rows.items(), key=lambda kv: -kv[1]["baseline"]["macro_f1"]):
        lo, hi = r["diff_macro_f1_ci95"]
        print(f"{name:<11} {r['baseline']['macro_f1']:>7.3f} {r['diff_macro_f1']:>+7.3f} [{lo:+.3f}, {hi:+.3f}] "
              f"{r['p_holm']:>8.3f} {r['mcnemar_p']:>9.3f}  {r['verdict']}")
    best = max(rows, key=lambda k: rows[k]["baseline"]["macro_f1"])
    print(f"\nbest baseline: {best}. Model 3 is 'best' only if it is LLM BETTER against every baseline above "
          f"(at least against {best}).")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"llm": llm_name, "prefix": a.prefix, "n_sessions": len(ids), "llm_metrics": llm_m,
                   "llm_per_seed_f1": per_seed(llm), "comparisons": rows}, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
