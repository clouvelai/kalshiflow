"""
Auto-curator for Captain memory files.

Pure Python, zero LLM calls. Runs at the start of each Captain cycle
to keep memory files within size limits. Keeps header lines + most recent
entries, truncating from the middle.

journal.jsonl is never truncated (audit trail).
"""

import logging
import os
from typing import Any, Dict

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.memory.auto_curator")

# Curation rules: file -> {max_lines, header_lines}
# header_lines are always preserved at the top of the file
CURATION_RULES: Dict[str, Dict[str, int]] = {
    "AGENTS.md": {"max_lines": 60, "header_lines": 6},
    "SIGNALS.md": {"max_lines": 30, "header_lines": 4},
    "PLAYBOOK.md": {"max_lines": 40, "header_lines": 4},
}


def auto_curate(data_dir: str) -> Dict[str, Any]:
    """Run auto-curation on all managed memory files.

    For each file: if lines > max, keep header + most recent entries, truncate.

    Args:
        data_dir: Path to the memory data directory

    Returns:
        Dict of actions taken, e.g. {"AGENTS.md": "truncated 85 -> 60"}
    """
    actions = {}

    for filename, rules in CURATION_RULES.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            continue

        max_lines = rules["max_lines"]
        header_lines = rules["header_lines"]

        try:
            with open(filepath, "r") as f:
                lines = f.readlines()

            if len(lines) <= max_lines:
                continue

            original_count = len(lines)

            # Keep header + most recent entries
            header = lines[:header_lines]
            # Keep the tail to fill up to max_lines
            tail_count = max_lines - header_lines - 1  # -1 for truncation marker
            tail = lines[-tail_count:] if tail_count > 0 else []

            truncated = header + [f"\n... (auto-curated: {original_count - len(header) - len(tail)} lines removed) ...\n\n"] + tail

            with open(filepath, "w") as f:
                f.writelines(truncated)

            msg = f"truncated {original_count} -> {len(truncated)}"
            actions[filename] = msg
            logger.info(f"[SINGLE_ARB:CURATE] {filename}: {msg}")

        except Exception as e:
            actions[filename] = f"error: {e}"
            logger.warning(f"[SINGLE_ARB:CURATE] {filename} failed: {e}")

    return actions
