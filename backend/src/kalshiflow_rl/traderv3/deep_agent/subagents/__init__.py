"""Deep Agent Subagents - Specialized agents that receive delegated tasks.

The main deep agent delegates to subagents via the task() tool.
Each subagent has its own system prompt, tools, and runs an isolated
LLM conversation. The parent agent receives only the final summary,
never the intermediate tool calls -- solving context bloat.
"""

from .base import SubAgent, SubAgentResult
from .issue_reporter import IssueReportingAgent
from .registry import SUBAGENT_REGISTRY

__all__ = [
    "SubAgent",
    "SubAgentResult",
    "IssueReportingAgent",
    "SUBAGENT_REGISTRY",
]
