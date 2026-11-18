"""Entry point for Cowrie log parsing.

Usage:
    python scripts/parse_cowrie.py path/to/cowrie.json
    python scripts/parse_cowrie.py path/to/cowrie.json --classify
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from honeyai.cowrie_parser import main

if __name__ == "__main__":
    main()