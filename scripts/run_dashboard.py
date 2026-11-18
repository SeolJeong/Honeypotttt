"""Entry point for Part 9 dashboard.

Usage:
    streamlit run honeyai/dashboard.py
    or:
    python scripts/run_dashboard.py
"""
import os
import subprocess
import sys


def main():
    dashboard = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "honeyai", "dashboard.py"
    )
    subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard])


if __name__ == "__main__":
    main()