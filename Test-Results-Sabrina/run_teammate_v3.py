"""Run the teammate's scripts/train_qlora_local.py on v3, with its training recipe UNCHANGED.

Only two paths and the saving of predictions differ from running the script directly:
  - reads  data/v3/dataset4_llm_sequences.jsonl          (instead of data/dataset4_llm_sequences.jsonl, v2b)
  - writes Test-Results-Sabrina/v3_teammate_qlora_local.json   (NOT data/qlora_predictions.json, which it would overwrite)
  - keeps the predictions of ALL 3 seeds (the script itself keeps only the best seed, chosen on the test set)
Everything else (LoRA r=8 on q/v, 15 epochs, LR 5e-5, MAX_SEQ_LEN 512, 480-token prompts, JSON generation) is the
script's own code. The per-seed file is written after every seed, so a crash keeps the seeds already finished.

Run from the repo root:   python run_teammate_v3.py
"""
import json
import os
import sys

sys.path.insert(0, "scripts")
import train_qlora_local as t  # noqa: E402  (the teammate's script, imported, not modified)

t.DS4 = os.path.join("data", "v3", "dataset4_llm_sequences.jsonl")
t.OUT = os.path.join("Test-Results-Sabrina", "v3_teammate_qlora_local_best_seed.json")
PER_SEED = os.path.join("Test-Results-Sabrina", "v3_teammate_qlora_local.json")
assert os.path.exists(t.DS4), f"missing {t.DS4}"

_orig = t.train_and_eval_seed
_done = []


def _keep_every_seed(*args, **kwargs):
    res = _orig(*args, **kwargs)
    _done.append(res)
    with open(PER_SEED, "w", encoding="utf-8") as f:
        json.dump({"model": t.MODEL_ID, "script": "scripts/train_qlora_local.py (unchanged recipe)",
                   "data": t.DS4, "seeds": _done}, f, indent=1)
    print(f"saved {len(_done)} seed(s) to {PER_SEED}", flush=True)
    return res


t.train_and_eval_seed = _keep_every_seed
t.main()
