"""
WebSocket handler for /ws/agent endpoint.

Provides real-time agent streaming to the frontend:
- Agent thinking/reasoning
- Tool calls and results
- Subagent delegation
- User message handling
"""

import json
import logging
from typing import Any

from starlette.websockets import WebSocket, WebSocketDisconnect

logger = logging.getLogger("kalshiflow_rl.traderv3.ws_agent_handler")


class AgentWebSocketHandler:
    """
    Handles WebSocket connections for agent streaming.

    Protocol:
    - Client -> Server: {"type": "user_message", "message": "..."}
    - Client -> Server: {"type": "get_state"}
    - Server -> Client: {"type": "agent_message", "data": {...}}
    - Server -> Client: {"type": "system_state", "data": {...}}
    """

    def __init__(self, arb_strategy: Any = None):
        self._arb_strategy = arb_strategy

    def set_arb_strategy(self, strategy: Any) -> None:
        """Set the arb strategy reference."""
        self._arb_strategy = strategy

    async def handle_websocket(self, websocket: WebSocket) -> None:
        """Handle a WebSocket connection for agent interaction."""
        await websocket.accept()
        logger.info("Agent WebSocket connected")

        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "error": "Invalid JSON"})
                    continue

                msg_type = msg.get("type")

                if msg_type == "user_message":
                    message = msg.get("message", "")
                    if message and self._arb_strategy:
                        response = await self._arb_strategy.handle_user_message(message)
                        await websocket.send_json({
                            "type": "agent_response",
                            "data": {"response": response},
                        })
                    else:
                        await websocket.send_json({
                            "type": "agent_response",
                            "data": {"response": "No strategy running"},
                        })

                elif msg_type == "get_state":
                    if self._arb_strategy:
                        state = self._arb_strategy.get_status()
                        await websocket.send_json({
                            "type": "system_state",
                            "data": state,
                        })
                    else:
                        await websocket.send_json({
                            "type": "system_state",
                            "data": {"running": False},
                        })

                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})

        except WebSocketDisconnect:
            logger.info("Agent WebSocket disconnected")
        except Exception as e:
            logger.error(f"Agent WebSocket error: {e}")
