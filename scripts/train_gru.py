"""Entry point for Part 6 GRU training.

Usage:
    python scripts/train_gru.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from honeyai.models.gru_sequence import main

if __name__ == "__main__":
    main()