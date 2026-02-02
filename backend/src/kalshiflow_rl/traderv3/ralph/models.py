"""Data models for the RALPH self-healing agent."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class IssueSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IssueCategory(Enum):
    PRICING_BUG = "pricing_bug"
    POSITION_TRACKING = "position_tracking"
    API_ERROR = "api_error"
    WEBSOCKET_FAILURE = "websocket_failure"
    AGENT_CRASH = "agent_crash"
    STATE_CORRUPTION = "state_corruption"
    PROCESS_DEATH = "process_death"
    MEMORY_CORRUPTION = "memory_corruption"


class RALPHPhase(Enum):
    MONITORING = "monitoring"
    DIAGNOSING = "diagnosing"
    FIXING = "fixing"
    VALIDATING = "validating"
    RESTARTING = "restarting"
    COMPLETE = "complete"
    ESCALATED = "escalated"


class FixResultStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    ESCALATE = "escalate"
    TIMEOUT = "timeout"
    ROLLBACK = "rollback"


@dataclass
class DetectedIssue:
    """An issue detected by the RALPH monitor."""
    issue_id: str
    severity: IssueSeverity
    category: IssueCategory
    summary: str
    evidence: List[Dict[str, Any]]
    source: str  # "health", "websocket", "memory", "process"
    detected_at: float
    occurrence_count: int = 1
    attempt_count: int = 0
    status: str = "new"  # new, handling, fixed, escalated
    last_attempt_at: Optional[float] = None

    @property
    def should_escalate(self) -> bool:
        return self.attempt_count >= 3

    @property
    def severity_rank(self) -> int:
        return {
            IssueSeverity.CRITICAL: 0,
            IssueSeverity.HIGH: 1,
            IssueSeverity.MEDIUM: 2,
            IssueSeverity.LOW: 3,
        }[self.severity]


@dataclass
class FixAttempt:
    """A record of a fix attempt by RALPH."""
    issue_id: str
    attempt_number: int
    claude_prompt: str
    claude_output: str
    result: FixResultStatus
    git_diff: str
    git_commit: Optional[str] = None
    validation_passed: bool = False
    trader_restarted: bool = False
    timestamp: float = 0.0
    duration_seconds: float = 0.0
    files_changed: List[str] = field(default_factory=list)


@dataclass
class FixResult:
    """Result from a Claude Code fix session."""
    status: FixResultStatus
    output: str
    files_changed: List[str] = field(default_factory=list)
    error: Optional[str] = None
    duration_seconds: float = 0.0


# Mapping from issue category to likely affected files (starting points for Claude)
CATEGORY_FILE_MAP: Dict[IssueCategory, List[str]] = {
    IssueCategory.PRICING_BUG: [
        "deep_agent/tools.py",
        "core/state_container.py",
        "deep_agent/trade_executor.py",
    ],
    IssueCategory.POSITION_TRACKING: [
        "core/state_container.py",
        "clients/position_listener.py",
        "deep_agent/tools.py",
    ],
    IssueCategory.API_ERROR: [
        "clients/demo_client.py",
        "clients/lifecycle_client.py",
        "clients/auth_utils.py",
    ],
    IssueCategory.WEBSOCKET_FAILURE: [
        "core/websocket_manager.py",
        "clients/orderbook_integration.py",
        "clients/trades_integration.py",
    ],
    IssueCategory.AGENT_CRASH: [
        "deep_agent/agent.py",
        "deep_agent/tools.py",
        "strategies/plugins/deep_agent.py",
    ],
    IssueCategory.STATE_CORRUPTION: [
        "core/state_container.py",
        "core/coordinator.py",
        "deep_agent/tools.py",
    ],
    IssueCategory.PROCESS_DEATH: [
        "app.py",
        "core/coordinator.py",
    ],
    IssueCategory.MEMORY_CORRUPTION: [
        "deep_agent/reflection.py",
        "deep_agent/tools.py",
        "deep_agent/memory/",
    ],
}
