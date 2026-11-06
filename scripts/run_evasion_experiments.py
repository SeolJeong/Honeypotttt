"""Entry point for Part 8 evasion experiments.

Usage:
    python scripts/run_evasion_experiments.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from honeyai.experiments.evasion_analysis import main

if __name__ == "__main__":
    main()