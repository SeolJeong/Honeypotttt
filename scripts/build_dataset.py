"""Build the full PoC dataset: scripted + synthetic-human + simulated-AI sessions.

Writes one JSON Lines file with all sessions in the shared schema. Replace the
synthetic-human and simulated-AI parts with real participant / real-API sessions
for final results; the output format is identical.

Run:  python scripts/build_dataset.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from honeyai.generators import scripted, synthetic_human, ai_driven
from honeyai.schema import write_sessions

OUT = os.path.join("data", "sessions", "poc_dataset.jsonl")


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    scr = scripted.generate(n_per_family_per_task=3, seed=0)
    hum = synthetic_human.generate(n_per_task=18, seed=1)
    ai = ai_driven.generate(n_per_task=18, seed=2, evasion_mix=True)
    alls = scr + hum + ai
    write_sessions(alls, OUT)
    print(f"scripted : {len(scr)}")
    print(f"human    : {len(hum)}  (SYNTHETIC - replace with real participants)")
    print(f"ai-driven: {len(ai)}  (SIMULATED - replace with real-API sessions)")
    print(f"total    : {len(alls)} -> {OUT}")


if __name__ == "__main__":
    main()
