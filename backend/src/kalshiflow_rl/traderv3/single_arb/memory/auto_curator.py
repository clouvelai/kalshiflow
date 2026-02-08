"""
Auto-curator for Captain memory files.

Pure Python, zero LLM calls. Runs at the start of each Captain cycle
to keep memory files within size limits.

AGENTS.md has protected sections (RULES) that are NEVER truncated.
Truncatable sections (LESSONS, HYPOTHESES, etc.) are trimmed to keep
the most recent entries when over limit. Deduplication collapses
near-identical lines.

journal.jsonl is never truncated (audit trail).
"""

import json
import logging
import os
import re
import time
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.memory.auto_curator")

# Curation rules: file -> {max_lines, header_lines}
# header_lines are always preserved at the top of the file
CURATION_RULES: Dict[str, Dict[str, int]] = {
    "AGENTS.md": {"max_lines": 60, "header_lines": 2},
    "SIGNALS.md": {"max_lines": 30, "header_lines": 4},
    "PLAYBOOK.md": {"max_lines": 40, "header_lines": 4},
    "THESES.md": {"max_lines": 30, "header_lines": 2},
}

# Section name keywords that mark a section as PROTECTED (never truncated).
# Uses substring matching so "## RULES (IF ...)" still matches "RULES".
PROTECTED_KEYWORDS = {"RULES", "DATA_LESSONS"}

# Section name keywords that mark a section as TRUNCATABLE, in priority order.
# First keyword = truncated first. Uses substring matching.
# Each maps to how many content lines to keep (excluding the ## header line).
TRUNCATABLE_KEEP = {
    "Current Cycle Notes": 3,
    "Hypotheses": 6,
    "HYPOTHESES": 6,
    "LESSONS": 10,
}

# Similarity threshold for deduplication (0.0-1.0, higher = stricter)
DEDUP_THRESHOLD = 0.82


def _is_protected(section_name: str) -> bool:
    """Check if a section name matches any protected keyword."""
    for kw in PROTECTED_KEYWORDS:
        if kw in section_name:
            return True
    return False


def _get_keep_limit(section_name: str) -> int:
    """Get the keep-limit for a section, or -1 if not explicitly truncatable."""
    for kw, limit in TRUNCATABLE_KEEP.items():
        if kw in section_name:
            return limit
    return -1  # -1 means "no explicit limit, truncate as last resort"


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
                _is_protected(current_name),
            ))
            current_name = match.group(1).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    # Save final section
    sections.append((
        current_name,
        current_lines,
        _is_protected(current_name),
    ))

    return sections


def _dedup_lines(lines: List[str], threshold: float = DEDUP_THRESHOLD) -> List[str]:
    """Remove near-duplicate lines from a list.

    Keeps the LAST occurrence of similar lines (most recent).
    Only deduplicates lines that start with "- " (list items).
    """
    if len(lines) <= 1:
        return lines

    # Work backwards so we keep the latest entry
    kept: List[str] = []
    kept_stripped: List[str] = []

    for line in reversed(lines):
        stripped = line.strip()

        # Only dedup list items (lines starting with -)
        if not stripped.startswith("-"):
            kept.append(line)
            continue

        # Check if this line is similar to any already-kept line
        is_dupe = False
        for k in kept_stripped:
            ratio = SequenceMatcher(None, stripped, k).ratio()
            if ratio >= threshold:
                is_dupe = True
                break

        if not is_dupe:
            kept.append(line)
            kept_stripped.append(stripped)

    kept.reverse()
    return kept


def _archive_entry(data_dir: str, content: str, section: str) -> None:
    """Append a truncated entry to journal.jsonl as archived_lesson.

    Preserves lessons that would otherwise be lost during auto-curation,
    making them searchable via recall_context (vector search).
    """
    entry = {
        "content": content,
        "type": "archived_lesson",
        "metadata": {"section": section, "reason": "auto_curated"},
        "timestamp": time.time(),
    }
    journal_path = os.path.join(data_dir, "journal.jsonl")
    try:
        with open(journal_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning(f"[CURATE] Failed to archive entry: {e}")


def _curate_agents_md(lines: List[str], max_lines: int, data_dir: str = "") -> List[str]:
    """Smart curation for AGENTS.md that respects protected sections.

    1. Parse into sections
    2. Deduplicate within each truncatable section
    3. Truncate sections in priority order if still over limit
    4. Protected sections (RULES) are NEVER truncated
    """
    sections = _parse_sections(lines)

    # Step 1: Deduplicate within truncatable sections
    deduped_sections = []
    dedup_removed = 0
    for name, sec_lines, is_prot in sections:
        if is_prot or name == "__header__":
            deduped_sections.append((name, sec_lines, is_prot))
        else:
            # Separate header line from content
            if sec_lines and sec_lines[0].strip().startswith("##"):
                header_line = [sec_lines[0]]
                content = sec_lines[1:]
            else:
                header_line = []
                content = sec_lines

            deduped_content = _dedup_lines(content)
            removed = len(content) - len(deduped_content)
            dedup_removed += removed

            deduped_sections.append((name, header_line + deduped_content, is_prot))

    if dedup_removed > 0:
        logger.info(f"[CURATE] Deduplication removed {dedup_removed} near-duplicate lines")

    sections = deduped_sections

    # Check if we're within limit after dedup
    total_lines = sum(len(sl) for _, sl, _ in sections)
    if total_lines <= max_lines:
        result = []
        for _, sec_lines, _ in sections:
            result.extend(sec_lines)
        return result

    # Step 2: Truncate sections in priority order
    # Count fixed lines (protected + header)
    protected_line_count = sum(
        len(sec_lines) for _, sec_lines, is_prot in sections
        if is_prot or _ == "__header__"
    )

    # Budget for truncatable sections
    budget = max_lines - protected_line_count

    if budget < 5:
        # Protected sections alone exceed limit - just keep protected + header
        result = []
        for name, sec_lines, is_prot in sections:
            if name == "__header__" or is_prot:
                result.extend(sec_lines)
        return result

    # Collect truncatable sections with their keep limits
    truncatable = []
    for name, sec_lines, is_prot in sections:
        if is_prot or name == "__header__":
            continue
        keep_limit = _get_keep_limit(name)
        truncatable.append((name, sec_lines, keep_limit))

    total_truncatable = sum(len(sl) for _, sl, _ in truncatable)

    if total_truncatable <= budget:
        # Everything fits after dedup
        result = []
        for _, sec_lines, _ in sections:
            result.extend(sec_lines)
        return result

    # Need to truncate. Apply keep limits.
    overflow = total_truncatable - budget
    truncated_sections: Dict[str, List[str]] = {}

    # Sort by priority: sections with explicit keep limits first (lowest limit = truncated most aggressively)
    priority_order = sorted(
        truncatable,
        key=lambda x: x[2] if x[2] >= 0 else 9999,
    )

    for sec_name, sec_lines, keep_limit in priority_order:
        if overflow <= 0:
            truncated_sections[sec_name] = sec_lines
            continue

        # Default keep limit for sections without explicit limits
        if keep_limit < 0:
            keep_limit = max(3, len(sec_lines) // 2)

        # +1 for the ## header line
        keep = keep_limit + 1 if sec_lines and sec_lines[0].strip().startswith("##") else keep_limit

        if len(sec_lines) > keep:
            can_trim = len(sec_lines) - keep
            trim = min(can_trim, overflow)
            actual_keep = len(sec_lines) - trim

            # Keep header line + last N content lines
            header_line = [sec_lines[0]] if sec_lines[0].strip().startswith("##") else []
            remaining = sec_lines[len(header_line):]
            content_keep = actual_keep - len(header_line)
            kept = remaining[-content_keep:] if content_keep > 0 else []
            removed_lines = remaining[:len(remaining) - len(kept)] if content_keep > 0 else remaining
            removed = len(removed_lines)

            # Archive removed lines to journal.jsonl before discarding
            if removed > 0 and data_dir:
                for line in removed_lines:
                    stripped = line.strip()
                    if stripped and stripped.startswith("-"):
                        _archive_entry(data_dir, stripped, sec_name)
                logger.info(f"[CURATE] Archived {removed} entries from '{sec_name}' to journal.jsonl")

            if removed > 0:
                truncated_sections[sec_name] = (
                    header_line +
                    [f"... ({removed} older entries curated) ...\n"] +
                    kept
                )
            else:
                truncated_sections[sec_name] = sec_lines
            overflow -= trim
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


def _curate_playbook_md(lines: List[str], max_lines: int, data_dir: str = "") -> List[str]:
    """Section-aware curation for PLAYBOOK.md.

    Preserves structural sections (EXIT_RULES, BANKROLL, ACTIVE_STRATEGIES,
    CUSTOM_STRATEGIES) and truncates narrative sections (POSITIONS).
    """
    sections = _parse_sections(lines)

    # Structural sections to always keep
    structural_keywords = {"EXIT_RULES", "BANKROLL", "ACTIVE_STRATEGIES", "CUSTOM_STRATEGIES"}

    total = sum(len(sl) for _, sl, _ in sections)
    if total <= max_lines:
        result = []
        for _, sl, _ in sections:
            result.extend(sl)
        return result

    # Dedup all non-structural sections
    deduped = []
    for name, sec_lines, is_prot in sections:
        is_structural = any(kw in name for kw in structural_keywords)
        if name == "__header__" or is_structural:
            deduped.append((name, sec_lines, True))
        else:
            if sec_lines and sec_lines[0].strip().startswith("##"):
                header = [sec_lines[0]]
                content = sec_lines[1:]
            else:
                header = []
                content = sec_lines
            content = _dedup_lines(content)
            deduped.append((name, header + content, False))

    # Count structural lines
    structural_lines = sum(
        len(sl) for _, sl, is_struct in deduped if is_struct
    )

    budget = max_lines - structural_lines
    if budget < 5:
        result = []
        for _, sl, is_struct in deduped:
            if is_struct:
                result.extend(sl)
        return result

    # Truncate non-structural sections to fit budget
    non_structural = [(n, sl) for n, sl, is_struct in deduped if not is_struct]
    total_ns = sum(len(sl) for _, sl in non_structural)

    if total_ns <= budget:
        result = []
        for _, sl, _ in deduped:
            result.extend(sl)
        return result

    # Need to truncate: keep last N lines from each non-structural section
    overflow = total_ns - budget
    truncated_map: Dict[str, List[str]] = {}

    for sec_name, sec_lines in non_structural:
        if overflow <= 0:
            truncated_map[sec_name] = sec_lines
            continue

        keep = max(3, len(sec_lines) // 3)
        if len(sec_lines) > keep:
            trim = min(len(sec_lines) - keep, overflow)
            header_line = [sec_lines[0]] if sec_lines[0].strip().startswith("##") else []
            remaining = sec_lines[len(header_line):]
            content_keep = len(sec_lines) - trim - len(header_line)
            kept = remaining[-content_keep:] if content_keep > 0 else []
            removed_lines = remaining[:len(remaining) - len(kept)] if content_keep > 0 else remaining
            removed = len(removed_lines)

            # Archive removed lines to journal.jsonl
            if removed > 0 and data_dir:
                for line in removed_lines:
                    stripped = line.strip()
                    if stripped and stripped.startswith("-"):
                        _archive_entry(data_dir, stripped, f"PLAYBOOK:{sec_name}")

            if removed > 0:
                truncated_map[sec_name] = (
                    header_line +
                    [f"... ({removed} older entries curated) ...\n"] +
                    kept
                )
            else:
                truncated_map[sec_name] = sec_lines
            overflow -= trim
        else:
            truncated_map[sec_name] = sec_lines

    # Reassemble preserving original order
    result = []
    for name, sec_lines, is_struct in deduped:
        if is_struct:
            result.extend(sec_lines)
        elif name in truncated_map:
            result.extend(truncated_map[name])
        else:
            result.extend(sec_lines)

    return result


def auto_curate(data_dir: str) -> Dict[str, Any]:
    """Run auto-curation on all managed memory files.

    For AGENTS.md: section-aware truncation + deduplication.
    For PLAYBOOK.md: section-aware truncation preserving structural sections.
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
                # Still run dedup even if under limit (for AGENTS.md)
                if filename == "AGENTS.md":
                    curated = _curate_agents_md(lines, max_lines, data_dir=data_dir)
                    if len(curated) < len(lines):
                        with open(filepath, "w") as f:
                            f.writelines(curated)
                        actions[filename] = f"deduped {len(lines)} -> {len(curated)}"
                        logger.info(f"[SINGLE_ARB:CURATE] {filename}: deduped {len(lines)} -> {len(curated)}")
                continue

            original_count = len(lines)

            if filename == "AGENTS.md":
                curated = _curate_agents_md(lines, max_lines, data_dir=data_dir)
            elif filename == "PLAYBOOK.md":
                curated = _curate_playbook_md(lines, max_lines, data_dir=data_dir)
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


# ---------------------------------------------------------------------------
# Journal distillation
# ---------------------------------------------------------------------------

# State file for tracking last distillation timestamp
_DISTILL_STATE_FILE = "distill_state.json"

# Distillation interval in seconds (4 hours)
DISTILL_INTERVAL_SECONDS = 4 * 60 * 60


def _read_distill_state(data_dir: str) -> Dict[str, Any]:
    """Read the distillation state file."""
    path = os.path.join(data_dir, _DISTILL_STATE_FILE)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.loads(f.read())
    except Exception:
        return {}


def _write_distill_state(data_dir: str, state: Dict[str, Any]) -> None:
    """Write the distillation state file."""
    path = os.path.join(data_dir, _DISTILL_STATE_FILE)
    try:
        with open(path, "w") as f:
            f.write(json.dumps(state))
    except Exception as e:
        logger.warning(f"[DISTILL] Failed to write state: {e}")


def _read_journal_entries(data_dir: str, since_ts: float = 0) -> List[Dict]:
    """Read journal entries, optionally filtering by timestamp."""
    journal_path = os.path.join(data_dir, "journal.jsonl")
    if not os.path.exists(journal_path):
        return []

    entries = []
    try:
        with open(journal_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("timestamp", 0) >= since_ts:
                        entries.append(entry)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.warning(f"[DISTILL] Failed to read journal: {e}")

    return entries


def _extract_event_ticker(entry: Dict) -> str:
    """Extract event ticker from a journal entry's metadata or content."""
    meta = entry.get("metadata", {})

    # Direct event_ticker in metadata
    if meta.get("event_ticker"):
        return meta["event_ticker"]

    # Extract from market ticker (e.g. KXNFLMENTION-SB26-YES → KXNFLMENTION-SB26)
    ticker = meta.get("ticker", "")
    if ticker:
        # Remove trailing -YES/-NO and last segment (market suffix)
        parts = ticker.split("-")
        if len(parts) >= 2:
            # Event ticker is usually the first 2 segments
            return "-".join(parts[:2])

    # Try parsing from content
    content = entry.get("content", "")
    # Pattern: TRADE: buy 5 yes TICKER-WITH-DASHES @42c
    # The ticker is usually the 4th word after TRADE:
    if content.startswith("TRADE:") or content.startswith("FAILED ORDER:"):
        parts = content.split()
        for p in parts:
            if p.startswith("KX") or p.startswith("kx"):
                dash_parts = p.split("-")
                if len(dash_parts) >= 2:
                    return "-".join(dash_parts[:2])

    return "unknown"


def _extract_strategy_tag(entry: Dict) -> str:
    """Infer a strategy tag from journal entry content/metadata."""
    content = entry.get("content", "").lower()
    meta = entry.get("metadata", {})

    # Check for explicit strategy tags
    if meta.get("strategy"):
        return meta["strategy"]

    # Infer from content patterns
    if "arb" in content or "execute_arb" in content:
        return "arb"
    if "mention" in content or "simulation" in content:
        return "mentions"
    if "spread" in content:
        return "spread"
    if "exit" in content or "take profit" in content or "cut loss" in content:
        return "exit"

    return "other"


def distill_journal(data_dir: str) -> Dict[str, Any]:
    """Distill journal.jsonl trade entries into data-grounded lessons.

    Reads journal entries, groups by strategy and event, computes stats,
    and writes a protected DATA_LESSONS section in AGENTS.md.

    Pure Python, zero LLM cost. Designed to run every 4 hours.

    Args:
        data_dir: Path to the memory data directory

    Returns:
        Dict with distillation stats
    """
    state = _read_distill_state(data_dir)
    now = time.time()

    # Read all journal entries (we rebuild DATA_LESSONS from full history)
    entries = _read_journal_entries(data_dir, since_ts=0)

    if not entries:
        logger.info("[DISTILL] No journal entries to distill")
        return {"status": "empty", "entries": 0}

    # Filter to trade-related entries
    trade_types = {"trade", "arb_execution", "order_cancelled", "order_expired"}
    trade_entries = [
        e for e in entries
        if e.get("type") in trade_types
        or e.get("content", "").startswith("TRADE:")
        or e.get("content", "").startswith("FAILED ORDER:")
        or e.get("content", "").startswith("ARB:")
    ]

    # Also collect archived lessons for context
    archived = [e for e in entries if e.get("type") == "archived_lesson"]

    if not trade_entries and not archived:
        logger.info("[DISTILL] No trade entries or archived lessons to distill")
        return {"status": "no_trades", "entries": len(entries)}

    # --- Group trades by strategy ---
    strategy_stats: Dict[str, Dict[str, Any]] = {}
    for entry in trade_entries:
        tag = _extract_strategy_tag(entry)
        if tag not in strategy_stats:
            strategy_stats[tag] = {
                "total": 0, "filled": 0, "failed": 0,
                "cancelled": 0, "expired": 0,
                "total_cost_cents": 0, "total_pnl_cents": 0,
            }
        stats = strategy_stats[tag]
        stats["total"] += 1

        meta = entry.get("metadata", {})
        status = meta.get("status", "")
        content = entry.get("content", "")

        if status == "failed" or "FAILED" in content:
            stats["failed"] += 1
        elif status in ("canceled", "cancelled"):
            stats["cancelled"] += 1
        elif status == "expired":
            stats["expired"] += 1
        else:
            stats["filled"] += 1
            # Accumulate cost
            contracts = meta.get("contracts", 0) or 0
            price = meta.get("price_cents", 0) or 0
            stats["total_cost_cents"] += contracts * price

    # --- Group trades by event ---
    event_stats: Dict[str, Dict[str, Any]] = {}
    for entry in trade_entries:
        event = _extract_event_ticker(entry)
        if event not in event_stats:
            event_stats[event] = {
                "total": 0, "filled": 0, "failed": 0,
                "strategies": set(),
            }
        ev = event_stats[event]
        ev["total"] += 1
        ev["strategies"].add(_extract_strategy_tag(entry))

        meta = entry.get("metadata", {})
        status = meta.get("status", "")
        content = entry.get("content", "")
        if status == "failed" or "FAILED" in content:
            ev["failed"] += 1
        else:
            ev["filled"] += 1

    # --- Generate DATA_LESSONS lines ---
    today = time.strftime("%Y-%m-%d", time.gmtime())
    lessons: List[str] = []

    # Strategy-level lessons
    for tag, stats in sorted(strategy_stats.items(), key=lambda x: -x[1]["total"]):
        total = stats["total"]
        filled = stats["filled"]
        failed = stats["failed"]
        cancelled = stats["cancelled"]
        expired = stats["expired"]

        if total == 0:
            continue

        fill_rate = filled / total * 100 if total > 0 else 0
        parts = [f"{tag}: {filled}/{total} filled ({fill_rate:.0f}%)"]

        if failed > 0:
            parts.append(f"{failed} failed")
        if cancelled > 0:
            parts.append(f"{cancelled} cancelled")
        if expired > 0:
            parts.append(f"{expired} expired")

        lessons.append(f"- {', '.join(parts)} | data:journal | {today}\n")

    # Event-level lessons (top 5 by volume)
    sorted_events = sorted(event_stats.items(), key=lambda x: -x[1]["total"])
    for event, ev in sorted_events[:5]:
        if event == "unknown":
            continue
        strategies = ", ".join(sorted(ev["strategies"]))
        fail_note = f", {ev['failed']} failed" if ev["failed"] > 0 else ""
        lessons.append(
            f"- {event}: {ev['total']} trades ({strategies}){fail_note} | data:journal | {today}\n"
        )

    # Archived lesson count
    if archived:
        lessons.append(
            f"- {len(archived)} archived lessons preserved in journal.jsonl (searchable via recall_context) | data:journal | {today}\n"
        )

    if not lessons:
        logger.info("[DISTILL] No lessons generated from journal data")
        return {"status": "no_lessons", "trade_entries": len(trade_entries)}

    # --- Write DATA_LESSONS section to AGENTS.md ---
    agents_path = os.path.join(data_dir, "AGENTS.md")
    if not os.path.exists(agents_path):
        logger.warning("[DISTILL] AGENTS.md not found, skipping write")
        return {"status": "no_agents_md"}

    try:
        with open(agents_path, "r") as f:
            content = f.read()

        # Build the new DATA_LESSONS section
        section_content = "## DATA_LESSONS (auto-generated from journal, do not edit)\n"
        for lesson in lessons:
            section_content += lesson
        section_content += "\n"

        # Replace existing DATA_LESSONS section or append
        data_lessons_pattern = re.compile(
            r"## DATA_LESSONS[^\n]*\n(?:.*?\n)*?(?=## |\Z)",
            re.DOTALL,
        )

        if data_lessons_pattern.search(content):
            # Replace existing section
            new_content = data_lessons_pattern.sub(section_content, content)
        else:
            # Append at the end
            new_content = content.rstrip("\n") + "\n\n" + section_content

        with open(agents_path, "w") as f:
            f.write(new_content)

        logger.info(
            f"[DISTILL] Wrote {len(lessons)} DATA_LESSONS to AGENTS.md "
            f"(from {len(trade_entries)} trades, {len(archived)} archived)"
        )

    except Exception as e:
        logger.error(f"[DISTILL] Failed to write AGENTS.md: {e}")
        return {"status": "write_error", "error": str(e)}

    # Update state
    state["last_distill_ts"] = now
    state["last_distill_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    state["entries_processed"] = len(trade_entries)
    state["lessons_generated"] = len(lessons)
    _write_distill_state(data_dir, state)

    return {
        "status": "ok",
        "trade_entries": len(trade_entries),
        "archived_entries": len(archived),
        "lessons_generated": len(lessons),
        "strategies": list(strategy_stats.keys()),
        "events": list(event_stats.keys()),
    }
