"""
Memory tools for the arb deep agent.

Uses DualMemoryStore (file + pgvector) for storage and search.
Includes curator tools for maintenance (dedup, consolidation, pruning).
"""

import logging
import time
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

logger = logging.getLogger("kalshiflow_rl.traderv3.deep_agent.tools.memory")

_memory_store = None  # DualMemoryStore


def set_memory_store(store: Any) -> None:
    """Set the DualMemoryStore instance."""
    global _memory_store
    _memory_store = store


# ── Agent tools (used by Captain, EventAnalyst, MemoryCurator) ──────────────


@tool
async def memory_store(
    content: str,
    memory_type: str = "learning",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Store a learning or insight in dual memory (file + pgvector).

    Use this to record patterns, mistakes, strategy adjustments, or market insights
    that should persist across sessions. File store always succeeds.

    Args:
        content: The learning/insight text to store
        memory_type: Category (learning, mistake, strategy, market_insight, pair_analysis, validation)
        metadata: Optional metadata dict (pair_id, ticker, confidence, etc.)

    Returns:
        Dict with status and storage details
    """
    if not _memory_store:
        logger.warning("memory_store called but DualMemoryStore not initialized")
        return {"error": "Memory store not available"}
    try:
        return await _memory_store.store(
            content=content,
            memory_type=memory_type,
            metadata=metadata,
        )
    except Exception as e:
        return {"error": str(e)}


@tool
async def memory_search(
    query: str,
    memory_type: Optional[str] = None,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Search memory for relevant past learnings and insights.

    Uses pgvector semantic search (primary) with file keyword fallback.
    Always call this before making decisions to recall past patterns.

    Args:
        query: Semantic search query
        memory_type: Optional filter by type (learning, mistake, strategy, etc.)
        limit: Maximum results to return

    Returns:
        List of matching memory entries with content, type, similarity score
    """
    if not _memory_store:
        logger.warning("memory_search called but DualMemoryStore not initialized")
        return [{"error": "Memory store not available"}]
    try:
        return await _memory_store.search(
            query=query,
            memory_type=memory_type,
            limit=limit,
        )
    except Exception as e:
        return [{"error": str(e)}]


# ── EventAnalyst tools ──────────────────────────────────────────────────────


@tool
async def save_validation(
    event_ticker: str,
    status: str,
    reasoning: str,
    confidence: float = 0.8,
    risk_factors: Optional[List[str]] = None,
    recommended_side: Optional[str] = None,
    recommended_max_price: Optional[int] = None,
) -> Dict[str, Any]:
    """Save an event validation result to the file cache.

    Called by EventAnalyst after analyzing ALL markets in an event.
    Validation is event-level: the entire event is approved or rejected.
    Captain checks this cache before any trade via get_validation_status.

    Args:
        event_ticker: Kalshi event ticker (e.g. 'KXFEDCHAIRNOM-29')
        status: 'approved' or 'rejected'
        reasoning: Detailed explanation referencing candle trends across markets
        confidence: Confidence score 0-1
        risk_factors: List of identified risks
        recommended_side: 'yes' or 'no' - which Kalshi side to trade across the event
        recommended_max_price: Suggested max entry price in cents

    Returns:
        Dict with status confirmation
    """
    if not _memory_store:
        return {"error": "Memory store not available"}

    data = {
        "status": status,
        "reasoning": reasoning,
        "confidence": confidence,
        "risk_factors": risk_factors or [],
    }
    if recommended_side:
        data["spread_assessment"] = {
            "recommended_side": recommended_side,
            "recommended_max_price": recommended_max_price,
        }

    _memory_store.save_validation(event_ticker, data)
    return {"event_ticker": event_ticker, "status": status, "saved": True}


# ── MemoryCurator tools ─────────────────────────────────────────────────────


@tool
async def get_memory_stats() -> Dict[str, Any]:
    """Get memory store statistics.

    Returns entry counts by type, staleness info, and storage details.
    Used by MemoryCurator to assess memory health.

    Returns:
        Dict with journal_entries, type_counts, validations_cached, etc.
    """
    if not _memory_store:
        return {"error": "Memory store not available"}
    return _memory_store.get_stats()


@tool
async def dedup_memories(similarity_threshold: float = 0.88) -> Dict[str, Any]:
    """Find and mark duplicate memories based on content similarity.

    Scans recent journal entries for near-duplicates (keyword overlap).
    Returns pairs of duplicates found for review.

    Args:
        similarity_threshold: Minimum overlap ratio to flag as duplicate (0-1)

    Returns:
        Dict with duplicate pairs found and dedup count
    """
    if not _memory_store:
        return {"error": "Memory store not available"}

    file_store = _memory_store.file_store
    entries = file_store.get_journal(limit=200)

    # Simple keyword-based dedup (cosine would need embeddings)
    duplicates = []
    seen_content = {}

    for entry in entries:
        content = entry.get("content", "").lower().strip()
        if not content:
            continue

        # Check for exact or near-exact duplicates
        words = set(content.split())
        found_dup = False
        for existing_content, existing_entry in seen_content.items():
            existing_words = set(existing_content.split())
            if not words or not existing_words:
                continue
            overlap = len(words & existing_words) / max(len(words | existing_words), 1)
            if overlap >= similarity_threshold:
                duplicates.append({
                    "original": existing_entry.get("content", "")[:100],
                    "duplicate": content[:100],
                    "overlap": round(overlap, 3),
                })
                found_dup = True
                break

        if not found_dup:
            seen_content[content] = entry

    return {
        "duplicates_found": len(duplicates),
        "entries_scanned": len(entries),
        "pairs": duplicates[:20],
    }


@tool
async def consolidate_memories(memory_type: str = "learning") -> Dict[str, Any]:
    """Merge related memories of a given type into a summary.

    Reads all entries of a type, groups by topic similarity, and identifies
    candidates for consolidation. Does NOT auto-merge (returns recommendations).

    Args:
        memory_type: Type of memories to consolidate (learning, mistake, etc.)

    Returns:
        Dict with consolidation recommendations
    """
    if not _memory_store:
        return {"error": "Memory store not available"}

    file_store = _memory_store.file_store
    entries = file_store.get_journal(limit=100, memory_type=memory_type)

    # Group by metadata keys (pair_id, ticker, etc.)
    groups: Dict[str, List[Dict]] = {}
    ungrouped = []

    for entry in entries:
        meta = entry.get("metadata", {})
        pair_id = meta.get("pair_id")
        ticker = meta.get("ticker") or meta.get("kalshi_ticker")

        key = pair_id or ticker
        if key:
            groups.setdefault(key, []).append(entry)
        else:
            ungrouped.append(entry)

    recommendations = []
    for key, group in groups.items():
        if len(group) >= 3:
            recommendations.append({
                "group_key": key,
                "count": len(group),
                "newest": group[0].get("content", "")[:100],
                "oldest": group[-1].get("content", "")[:100],
                "recommendation": "consolidate",
            })

    return {
        "type": memory_type,
        "total_entries": len(entries),
        "groups": len(groups),
        "ungrouped": len(ungrouped),
        "consolidation_candidates": len(recommendations),
        "recommendations": recommendations[:10],
    }


@tool
async def prune_stale_memories(max_age_hours: float = 168.0) -> Dict[str, Any]:
    """Identify stale memories older than max_age_hours.

    Reports stale entries but does NOT delete them (read-only analysis).
    Returns count and preview for curator to decide.

    Args:
        max_age_hours: Maximum age in hours before considering stale (default 168 = 7 days)

    Returns:
        Dict with stale entry count and previews
    """
    if not _memory_store:
        return {"error": "Memory store not available"}

    file_store = _memory_store.file_store
    entries = file_store.get_journal(limit=500)

    cutoff = time.time() - (max_age_hours * 3600)
    stale = []

    for entry in entries:
        ts = entry.get("timestamp", 0)
        if ts and ts < cutoff:
            stale.append({
                "content": entry.get("content", "")[:100],
                "type": entry.get("type"),
                "age_hours": round((time.time() - ts) / 3600, 1),
            })

    return {
        "max_age_hours": max_age_hours,
        "total_scanned": len(entries),
        "stale_count": len(stale),
        "stale_preview": stale[:15],
    }
