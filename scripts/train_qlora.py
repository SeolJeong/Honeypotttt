"""Entry point for Part 7 QLoRA training (Model 3).

Examples:
    python scripts/train_qlora.py                         # round 2, full input, 3 seeds
    python scripts/train_qlora.py --round 0               # non-evasive training only
    python scripts/train_qlora.py --no-responses          # commands-only ablation
    python scripts/train_qlora.py --model Qwen/Qwen2.5-1.5B --batch-size 4 --grad-accum 4
    python scripts/train_qlora.py --smoke                 # CPU pipeline check, no download
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from honeyai.llm.format import FormatConfig
from honeyai.models.qlora_cls import QLoRAConfig, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=QLoRAConfig.model_name)
    ap.add_argument("--data", default=QLoRAConfig.data_path, help="dataset1-style sessions JSON")
    ap.add_argument("--max-prefixes", type=int, default=6)
    ap.add_argument("--round", type=int, choices=[0, 2], default=2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--no-4bit", action="store_true", help="full-precision LoRA (no QLoRA)")
    ap.add_argument("--no-prefix-training", action="store_true",
                    help="train on full sessions only (default: every prefix)")
    ap.add_argument("--no-responses", action="store_true", help="ablation: commands only")
    ap.add_argument("--no-timing", action="store_true", help="ablation: drop GAP tokens")
    ap.add_argument("--no-category", action="store_true", help="ablation: drop response category")
    ap.add_argument("--shuffle-steps", action="store_true", help="ablation: destroy ordering")
    ap.add_argument("--out-dir", default=os.path.join("runs", "qlora"))
    ap.add_argument("--smoke", action="store_true",
                    help="tiny random model on CPU, 1 seed, 2 epochs; checks the pipeline only")
    a = ap.parse_args()

    cfg = QLoRAConfig(
        model_name=a.model, data_path=a.data, max_prefixes=a.max_prefixes, round=a.round, seeds=a.seeds, epochs=a.epochs, lr=a.lr,
        batch_size=a.batch_size, grad_accum=a.grad_accum, max_len=a.max_len, lora_r=a.lora_r,
        lora_alpha=2 * a.lora_r, load_4bit=not a.no_4bit,
        train_prefixes=not a.no_prefix_training, out_dir=a.out_dir,
        fmt=FormatConfig(include_responses=not a.no_responses, include_timing=not a.no_timing,
                         include_category=not a.no_category, shuffle_steps=a.shuffle_steps),
    )
    if a.smoke:
        cfg.smoke, cfg.load_4bit, cfg.seeds, cfg.epochs = True, False, a.seeds[:1], 2
        cfg.lr = 1e-3
    run(cfg)


if __name__ == "__main__":
    main()
