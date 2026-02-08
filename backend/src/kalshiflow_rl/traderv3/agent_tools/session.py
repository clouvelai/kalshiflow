"""TradingSession - Shared mutable state between Captain and Commando tools.

Owned by the coordinator, passed to both tool context classes.
"""

import time
from dataclasses import dataclass, field
from typing import Set


@dataclass
class TradingSession:
    """Shared state for a single trading session.

    The coordinator creates one at startup. Both CaptainToolContext and
    CommandoToolContext reference the same instance, ensuring order ID
    tracking and session metadata are consistent.
    """

    order_group_id: str = ""
    order_ttl: int = 60  # seconds
    captain_order_ids: Set[str] = field(default_factory=set)
    sniper_order_ids: Set[str] = field(default_factory=set)
    started_at: float = field(default_factory=time.time)

    def reset(self) -> None:
        """Clear session state (e.g. on order group reset)."""
        self.captain_order_ids.clear()
        self.sniper_order_ids.clear()
        self.started_at = time.time()
