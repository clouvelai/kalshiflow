"""Module-level @tool functions for the MentionsSpecialist subagent.

For Phase 1, these tools delegate to the existing mentions system
(mentions_simulator, mentions_context, mentions_semantic) rather than
duplicating that complex pipeline. The context provides clean access
to the index and config needed by those modules.

The existing mentions_tools module-level globals (_index, _file_store, etc.)
are still set by the coordinator for the old code path. These new tools
call the same underlying functions.
"""

from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from .context import MentionsToolContext

# Single module-level context
_ctx: Optional[MentionsToolContext] = None


def set_context(ctx: MentionsToolContext) -> None:
    """Set the shared MentionsToolContext. Called by coordinator."""
    global _ctx
    _ctx = ctx


# ---------------------------------------------------------------------------
# For Phase 1, re-export the existing mentions tools directly.
# The existing tools use their own module-level globals which are set
# by the coordinator in the old path. Since both old and new paths
# call set_mentions_dependencies(), the same tools work in both modes.
#
# In a future Phase 2, these would be rewritten to use _ctx directly.
# ---------------------------------------------------------------------------

from ..single_arb.mentions_tools import (
    get_mentions_status,
    simulate_probability,
    trigger_simulation,
    compute_edge,
    get_event_context,
    get_mention_context,
    query_wordnet,
    get_mentions_rules,
    get_mentions_summary,
)

__all__ = [
    "set_context",
    "get_mentions_status",
    "simulate_probability",
    "trigger_simulation",
    "compute_edge",
    "get_event_context",
    "get_mention_context",
    "query_wordnet",
    "get_mentions_rules",
    "get_mentions_summary",
]
