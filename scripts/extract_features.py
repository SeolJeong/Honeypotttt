"""Entry point for Part 4 feature extraction.

Usage:
    python scripts/extract_features.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from honeyai.features import main

if __name__ == "__main__":
    main()
