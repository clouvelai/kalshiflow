#!/usr/bin/env python3
"""
CLI entry point for the pair index builder.

Usage:
    cd backend
    uv run python scripts/build_pair_index.py build [--dry-run] [--no-llm] [-v]
    uv run python scripts/build_pair_index.py status
"""

import sys
import os

# Ensure backend/src is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kalshiflow_rl.pair_index.cli import main

if __name__ == "__main__":
    main()
