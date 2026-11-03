"""Entry point for Part 5 XGBoost training.

Usage:
    python scripts/train_xgboost.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from honeyai.models.xgb_baseline import main

if __name__ == "__main__":
    main()