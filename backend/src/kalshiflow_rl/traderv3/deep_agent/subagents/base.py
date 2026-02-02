"""Base subagent class for the deep agent delegation framework.

Subagents run isolated LLM conversations with their own system prompts
and tool sets. The parent agent delegates via task() and receives only
the final summary -- intermediate tool calls stay contained.

Pattern:
    Parent: task(agent="issue_reporter", input="prices showing 372c...")
    SubAgent: runs its own LLM loop with specialized tools
    Parent: receives {"summary": "Filed issue da-12345 [critical/pricing_bug]..."}
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from anthropic import AsyncAnthropic

logger = logging.getLogger("deep_agent.subagents")


@dataclass
class SubAgentResult:
    """Result returned from a subagent to the parent agent."""
    success: bool
    summary: str  # Concise text summary for the parent (under 300 words)
    data: Dict[str, Any] = field(default_factory=dict)  # Structured data
    error: Optional[str] = None


class SubAgent(ABC):
    """Base class for deep agent subagents.

    Each subagent defines:
    - name: unique identifier used by the parent's task() tool
    - description: helps the parent decide when to delegate
    - system_prompt: detailed instructions for the subagent's LLM
    - tools: list of tool schemas the subagent can use
    - tool handlers: execute the subagent's tool calls
    """

    # Subclasses must set these
    name: str = ""
    description: str = ""
    max_turns: int = 5
    model: str = "claude-haiku-4-20250414"  # Cheap model for subagent work

    def __init__(
        self,
        client: AsyncAnthropic,
        memory_dir: "Path",
        ws_manager: Optional[Any] = None,
    ):
        self._client = client
        self._memory_dir = memory_dir
        self._ws_manager = ws_manager

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the subagent's system prompt."""
        ...

    @abstractmethod
    def get_tools(self) -> List[Dict[str, Any]]:
        """Return the subagent's tool schemas."""
        ...

    @abstractmethod
    async def handle_tool_call(self, tool_name: str, tool_input: Dict) -> Any:
        """Execute a tool call and return the result."""
        ...

    async def run(self, task_input: str) -> SubAgentResult:
        """Run the subagent's LLM loop to completion.

        The subagent gets its own conversation, executes tool calls,
        and returns a final SubAgentResult to the parent.
        """
        start = time.time()
        messages = [{"role": "user", "content": task_input}]
        tools = self.get_tools()
        system_prompt = self.get_system_prompt()

        logger.info("[subagent.%s] Starting (input: %d chars)", self.name, len(task_input))

        for turn in range(self.max_turns):
            try:
                response = await asyncio.wait_for(
                    self._client.messages.create(
                        model=self.model,
                        max_tokens=1024,
                        system=system_prompt,
                        messages=messages,
                        tools=tools if tools else None,
                    ),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                return SubAgentResult(
                    success=False,
                    summary="Subagent LLM call timed out",
                    error="timeout",
                )
            except Exception as e:
                return SubAgentResult(
                    success=False,
                    summary=f"Subagent LLM error: {str(e)[:200]}",
                    error=str(e),
                )

            # Check for end_turn (no tool use, just text response)
            if response.stop_reason == "end_turn":
                text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        text += block.text
                duration = time.time() - start
                logger.info(
                    "[subagent.%s] Completed in %.1fs (%d turns)",
                    self.name, duration, turn + 1,
                )
                return SubAgentResult(success=True, summary=text)

            # Process tool calls
            if response.stop_reason == "tool_use":
                # Add assistant response to conversation
                messages.append({"role": "assistant", "content": response.content})

                # Execute each tool call
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        try:
                            result = await self.handle_tool_call(block.name, block.input)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps(result, default=str) if not isinstance(result, str) else result,
                            })
                        except Exception as e:
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": json.dumps({"error": str(e)}),
                                "is_error": True,
                            })

                messages.append({"role": "user", "content": tool_results})

        # Exhausted turns
        return SubAgentResult(
            success=False,
            summary="Subagent exhausted max turns without completing",
            error="max_turns_exceeded",
        )
