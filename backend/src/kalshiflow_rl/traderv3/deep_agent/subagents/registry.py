"""Subagent registry - maps names to subagent classes.

The task() tool uses this registry to look up and instantiate subagents.
New subagents are registered by adding them here.
"""

from typing import Dict, Type

from .base import SubAgent
from .issue_reporter import IssueReportingAgent

# Registry: name -> subagent class
SUBAGENT_REGISTRY: Dict[str, Type[SubAgent]] = {
    "issue_reporter": IssueReportingAgent,
}
