"""Agent tools module - clean tool layer for Captain/Commando/Mentions.

Provides module-level @tool functions backed by testable context classes.
Each tool file has a set_context() function called by the coordinator.

Usage:
    from ..agent_tools import captain_tools, commando_tools, mentions_tools
    from ..agent_tools.context import CaptainToolContext, CommandoToolContext
    from ..agent_tools.session import TradingSession

    session = TradingSession(order_group_id="...", order_ttl=60)
    captain_ctx = CaptainToolContext(gateway, index, file_store, session)
    commando_ctx = CommandoToolContext(gateway, index, file_store, session)

    captain_tools.set_context(captain_ctx)
    commando_tools.set_context(commando_ctx)
"""

from . import captain_tools, commando_tools, mentions_tools
from .context import CaptainToolContext, CommandoToolContext, MentionsToolContext
from .session import TradingSession

__all__ = [
    "captain_tools",
    "commando_tools",
    "mentions_tools",
    "CaptainToolContext",
    "CommandoToolContext",
    "MentionsToolContext",
    "TradingSession",
]
