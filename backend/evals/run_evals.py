"""Run the full UC-01 eval suite directly.

Example:
    python evals/run_evals.py
"""

import sys
from pathlib import Path

# Make src and evals importable
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root))

from evals.runner import main

if __name__ == "__main__":
    main()
