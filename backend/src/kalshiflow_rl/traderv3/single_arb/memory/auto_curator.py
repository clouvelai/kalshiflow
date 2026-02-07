"""
Auto-curator for Captain memory files.

Pure Python, zero LLM calls. Runs at the start of each Captain cycle
to keep memory files within size limits.

AGENTS.md has protected sections (Red Lines, Validated Lessons) that are
NEVER truncated. Only "Current Cycle Notes" and overflow from
"Hypotheses Under Test" are truncated when over limit.

journal.jsonl is never truncated (audit trail).
"""

import logging
import os
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.memory.auto_curator")

# Curation rules: file -> {max_lines, header_lines}
# header_lines are always preserved at the top of the file
CURATION_RULES: Dict[str, Dict[str, int]] = {
    "AGENTS.md": {"max_lines": 60, "header_lines": 2},
    "SIGNALS.md": {"max_lines": 30, "header_lines": 4},
    "PLAYBOOK.md": {"max_lines": 40, "header_lines": 4},
}

# Sections in AGENTS.md that are NEVER truncated
PROTECTED_SECTIONS = {"Red Lines", "Validated Lessons", "RULES", "RULES (IF condition THEN action | confidence | source)"}

# Sections that CAN be truncated (in truncation priority order: first = truncated first)
TRUNCATABLE_SECTIONS = ["Current Cycle Notes", "Hypotheses Under Test", "LESSONS (evidence-backed | [ticker] [date] [outcome] [takeaway])"]


def _parse_sections(lines: List[str]) -> List[Tuple[str, List[str], bool]]:
    """Parse a markdown file into sections.

    Returns list of (section_name, section_lines, is_protected).
    Lines before the first section header are grouped as "__header__".
    """
    sections: List[Tuple[str, List[str], bool]] = []
    current_name = "__header__"
    current_lines: List[str] = []

    for line in lines:
        # Match ## Section Name headers
        match = re.match(r"^##\s+(.+)$", line.strip())
        if match:
            # Save previous section
            sections.append((
                current_name,
                current_lines,
                current_name in PROTECTED_SECTIONS,
            ))
            current_name = match.group(1).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    # Save final section
    sections.append((
        current_name,
        current_lines,
        current_name in PROTECTED_SECTIONS,
    ))

    return sections


def _curate_agents_md(lines: List[str], max_lines: int) -> List[str]:
    """Smart curation for AGENTS.md that respects protected sections.

    Protected sections (Red Lines, Validated Lessons) are NEVER truncated.
    Truncation happens in this order:
    1. Current Cycle Notes (truncated first, keep last 5 lines)
    2. Hypotheses Under Test (truncated second, keep last 8 lines)
    3. Unrecognized sections (truncated from tail)
    """
    if len(lines) <= max_lines:
        return lines

    sections = _parse_sections(lines)

    # Count protected lines
    protected_line_count = sum(
        len(sec_lines) for _, sec_lines, is_prot in sections if is_prot
    )
    header_lines = sum(
        len(sec_lines) for name, sec_lines, _ in sections if name == "__header__"
    )

    # Budget for truncatable sections
    budget = max_lines - protected_line_count - header_lines - 1  # -1 for truncation marker

    if budget < 5:
        # Protected sections alone exceed limit - just keep protected + header
        result = []
        for name, sec_lines, is_prot in sections:
            if name == "__header__" or is_prot:
                result.extend(sec_lines)
        return result

    # Distribute budget across truncatable sections
    truncatable = [(name, sec_lines) for name, sec_lines, is_prot in sections
                   if not is_prot and name != "__header__"]

    # Truncation targets (lines to keep per section when over budget)
    keep_limits = {
        "Current Cycle Notes": 5,
        "Hypotheses Under Test": 8,
    }

    # First pass: calculate what we'd keep at minimum
    total_truncatable = sum(len(sl) for _, sl in truncatable)

    if total_truncatable <= budget:
        # Everything fits, no truncation needed beyond what's already removed
        result = []
        for name, sec_lines, _ in sections:
            result.extend(sec_lines)
        return result

    # Need to truncate. Start with the highest-priority truncation targets
    overflow = total_truncatable - budget
    truncated_sections: Dict[str, List[str]] = {}

    for sec_name, sec_lines in truncatable:
        keep = keep_limits.get(sec_name, len(sec_lines))

        if overflow > 0 and len(sec_lines) > keep:
            can_trim = len(sec_lines) - keep
            trim = min(can_trim, overflow)
            # Keep header line (## ...) + last N lines
            if sec_lines:
                header_line = [sec_lines[0]] if sec_lines[0].strip().startswith("##") else []
                remaining = sec_lines[len(header_line):]
                kept = remaining[-(keep - len(header_line)):] if remaining else []
                removed = len(sec_lines) - len(header_line) - len(kept)
                if removed > 0:
                    truncated_sections[sec_name] = (
                        header_line +
                        [f"... ({removed} lines truncated) ...\n"] +
                        kept
                    )
                else:
                    truncated_sections[sec_name] = sec_lines
                overflow -= trim
            else:
                truncated_sections[sec_name] = sec_lines
        else:
            truncated_sections[sec_name] = sec_lines

    # Reassemble
    result = []
    for name, sec_lines, is_prot in sections:
        if name == "__header__" or is_prot:
            result.extend(sec_lines)
        elif name in truncated_sections:
            result.extend(truncated_sections[name])
        else:
            result.extend(sec_lines)

    return result


def auto_curate(data_dir: str) -> Dict[str, Any]:
    """Run auto-curation on all managed memory files.

    For AGENTS.md: uses section-aware truncation that protects Red Lines
    and Validated Lessons sections.
    For other files: keeps header + most recent entries, truncates from middle.

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

            if filename == "AGENTS.md":
                # Section-aware truncation for AGENTS.md
                curated = _curate_agents_md(lines, max_lines)
            else:
                # Simple header + tail truncation for other files
                header = lines[:header_lines]
                tail_count = max_lines - header_lines - 1
                tail = lines[-tail_count:] if tail_count > 0 else []
                curated = header + [f"\n... (auto-curated: {original_count - len(header) - len(tail)} lines removed) ...\n\n"] + tail

            with open(filepath, "w") as f:
                f.writelines(curated)

            msg = f"truncated {original_count} -> {len(curated)}"
            actions[filename] = msg
            logger.info(f"[SINGLE_ARB:CURATE] {filename}: {msg}")

        except Exception as e:
            actions[filename] = f"error: {e}"
            logger.warning(f"[SINGLE_ARB:CURATE] {filename} failed: {e}")

    return actions
