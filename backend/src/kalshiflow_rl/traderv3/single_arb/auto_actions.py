"""AutoActionManager - Deterministic automated trading actions.

Extends the Sniper pattern to more scenarios. Each auto-action is a callback
registered on the AttentionRouter. Actions fire deterministically when conditions
are met, but Captain retains override authority via configure_automation tool.

Auto-actions:
  1. Hard P&L Stops: Exit positions that cross loss threshold (-12c/ct default)
  2. Time-Pressure Exits: Exit positions in events closing < 30min
  3. Regime Gating: Disable sniper for events entering toxic regime

Items still appear in Captain's attention feed marked with auto_handled,
so the Captain knows what happened and can override if needed.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, TYPE_CHECKING

from .models import AttentionItem

if TYPE_CHECKING:
    from .index import EventArbIndex
    from .sniper import Sniper
    from ..gateway.client import KalshiGateway

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.auto_actions")


@dataclass
class AutoActionResult:
    """Result of an auto-action execution."""
    acted: bool = False
    summary: str = ""
    order_id: str = ""
    error: str = ""


@dataclass
class AutoActionConfig:
    """Per-action configuration with Captain override support."""
    enabled: bool = True
    overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Per-ticker/event overrides: {"TICKER-XYZ": {"threshold": -15}, "EVENT-ABC": {"hold_through": True}}


class AutoActionManager:
    """Manages automated trading actions triggered by AttentionRouter items.

    Each action checks conditions and executes via the trading gateway.
    Captain can override behavior per action+ticker via configure_automation tool.
    """

    def __init__(
        self,
        gateway: "KalshiGateway",
        index: "EventArbIndex",
        sniper: Optional["Sniper"] = None,
        config=None,
        broadcast_callback: Optional[Callable] = None,
    ):
        self._gateway = gateway
        self._index = index
        self._sniper = sniper
        self._config = config
        self._broadcast = broadcast_callback

        # Per-action configs
        self._stop_loss = AutoActionConfig()
        self._time_exit = AutoActionConfig()
        self._regime_gate = AutoActionConfig()

        # Default thresholds
        self._stop_loss_threshold = -12  # cents per contract
        self._time_exit_minutes = 30     # minutes before close
        self._regime_gate_cooldown = 300.0  # seconds

        # Cooldown tracking
        self._regime_gate_last: Dict[str, float] = {}  # event_ticker -> last gate time
        self._action_log: List[Dict] = []  # recent actions for stats
        self._pending_resume_tasks: List[asyncio.Task] = []  # for cleanup on stop()

    async def on_attention_item(self, item: AttentionItem) -> None:
        """Called by AttentionRouter for each item before Captain sees it.

        If an auto-action fires, marks item.data["auto_handled"] with summary.
        """
        try:
            if item.category == "position_risk" and self._stop_loss.enabled:
                result = await self._handle_stop_loss(item)
                if result.acted:
                    item.data["auto_handled"] = result.summary
                    self._log_action("stop_loss", item, result)

            elif item.category == "settlement_approaching" and self._time_exit.enabled:
                result = await self._handle_time_exit(item)
                if result.acted:
                    item.data["auto_handled"] = result.summary
                    self._log_action("time_exit", item, result)

            elif item.category == "regime_change" and self._regime_gate.enabled:
                result = await self._handle_regime_gate(item)
                if result.acted:
                    item.data["auto_handled"] = result.summary
                    self._log_action("regime_gate", item, result)

        except Exception as e:
            logger.error(f"[AUTO_ACTION] Error processing {item.category}: {e}")

    # ------------------------------------------------------------------
    # Shared exit helpers
    # ------------------------------------------------------------------

    def _find_market(self, event_ticker: str, ticker: str):
        """Look up MarketMeta via public index API."""
        event = self._index.events.get(event_ticker)
        if event:
            return event.markets.get(ticker)
        return None

    async def _execute_exit(
        self, ticker: str, event_ticker: str, side: str, quantity: int,
        label: str, detail: str,
    ) -> AutoActionResult:
        """Execute a sell order to exit a position. Shared by stop_loss and time_exit."""
        try:
            market_meta = self._find_market(event_ticker, ticker)
            if not market_meta:
                return AutoActionResult(error=f"Market {ticker} not found")

            exit_price = market_meta.yes_bid if side == "yes" else (100 - market_meta.yes_ask if market_meta.yes_ask else None)
            if exit_price is None or exit_price <= 0:
                return AutoActionResult(error=f"No valid exit price for {ticker}")

            response = await self._gateway.place_order(
                ticker=ticker,
                side=side,
                action="sell",
                count=quantity,
                type="limit",
                yes_price=exit_price if side == "yes" else None,
                no_price=exit_price if side == "no" else None,
            )

            order_id = response.get("order", {}).get("order_id", "")
            if not order_id:
                return AutoActionResult(error=f"Exit order failed: no order_id in response for {ticker}")
            return AutoActionResult(
                acted=True,
                summary=f"{label}: sold {quantity}ct {ticker} @ {exit_price}c ({detail})",
                order_id=order_id,
            )

        except Exception as e:
            logger.error(f"[AUTO_ACTION:{label.upper()}] Failed to exit {ticker}: {e}")
            return AutoActionResult(error=str(e))

    # ------------------------------------------------------------------
    # Action 1: Hard P&L Stops
    # ------------------------------------------------------------------

    async def _handle_stop_loss(self, item: AttentionItem) -> AutoActionResult:
        """Exit position if P&L crosses loss threshold."""
        pnl_per_ct = item.data.get("pnl_per_contract", 0)
        ticker = item.data.get("ticker", "")
        side = item.data.get("side", "")
        quantity = item.data.get("quantity", 0)

        if not ticker or not side or quantity <= 0:
            return AutoActionResult()

        # Check for per-ticker override
        override = self._stop_loss.overrides.get(ticker, {})
        threshold = override.get("threshold", self._stop_loss_threshold)

        if pnl_per_ct > threshold:
            return AutoActionResult()  # Not at threshold yet

        return await self._execute_exit(
            ticker, item.event_ticker, side, quantity,
            "stop_loss", f"pnl={pnl_per_ct:+d}c/ct",
        )

    # ------------------------------------------------------------------
    # Action 2: Time-Pressure Exits
    # ------------------------------------------------------------------

    async def _handle_time_exit(self, item: AttentionItem) -> AutoActionResult:
        """Exit positions in events approaching settlement."""
        event_ticker = item.event_ticker
        ttc_hours = item.data.get("ttc_hours", float("inf"))
        ticker = item.data.get("ticker", "")
        side = item.data.get("side", "")
        quantity = item.data.get("quantity", 0)

        if not ticker or not side or quantity <= 0:
            return AutoActionResult()

        # Check for per-event override (Captain may say "hold through settlement")
        override = self._time_exit.overrides.get(event_ticker, {})
        if override.get("hold_through", False):
            return AutoActionResult()

        threshold_hours = override.get("threshold_minutes", self._time_exit_minutes) / 60.0
        if ttc_hours > threshold_hours:
            return AutoActionResult()

        return await self._execute_exit(
            ticker, event_ticker, side, quantity,
            "time_exit", f"ttc={ttc_hours:.1f}h",
        )

    # ------------------------------------------------------------------
    # Action 3: Regime Gating
    # ------------------------------------------------------------------

    async def _handle_regime_gate(self, item: AttentionItem) -> AutoActionResult:
        """Disable sniper for events entering toxic regime."""
        if not self._sniper:
            return AutoActionResult()

        event_ticker = item.event_ticker
        regime = item.data.get("regime", "")

        if regime != "toxic":
            return AutoActionResult()

        # Cooldown check
        last_gate = self._regime_gate_last.get(event_ticker, 0)
        if time.time() - last_gate < self._regime_gate_cooldown:
            return AutoActionResult()

        # Check for per-event override
        override = self._regime_gate.overrides.get(event_ticker, {})
        if override.get("ignore_regime", False):
            return AutoActionResult()

        # Actually pause the sniper and schedule resume after cooldown
        self._sniper.pause(f"toxic regime on {event_ticker}")
        self._regime_gate_last[event_ticker] = time.time()
        task = asyncio.create_task(self._resume_sniper_after(event_ticker))
        self._pending_resume_tasks.append(task)
        task.add_done_callback(lambda t: self._pending_resume_tasks.remove(t) if t in self._pending_resume_tasks else None)

        return AutoActionResult(
            acted=True,
            summary=f"regime_gate: toxic regime on {event_ticker}, sniper paused {self._regime_gate_cooldown}s",
        )

    async def _resume_sniper_after(self, event_ticker: str) -> None:
        """Resume sniper after regime gate cooldown expires."""
        await asyncio.sleep(self._regime_gate_cooldown)
        if self._sniper:
            self._sniper.resume()
            logger.info(f"[AUTO_ACTION:REGIME_GATE] Resumed sniper after {self._regime_gate_cooldown}s (event={event_ticker})")

    # ------------------------------------------------------------------
    # Configuration (called by Captain via configure_automation tool)
    # ------------------------------------------------------------------

    def configure(self, action_name: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Configure an auto-action. Returns the updated configuration.

        Args:
            action_name: "stop_loss", "time_exit", or "regime_gate"
            settings: Action-specific settings. Common keys:
                - "enabled": bool — enable/disable action globally
                - "ticker" or "event": str — key for per-ticker/event override
                - Other keys are action-specific (threshold, hold_through, etc.)
        """
        action_map = {
            "stop_loss": self._stop_loss,
            "time_exit": self._time_exit,
            "regime_gate": self._regime_gate,
        }

        action_cfg = action_map.get(action_name)
        if not action_cfg:
            return {"error": f"Unknown action: {action_name}. Valid: {list(action_map.keys())}"}

        # Global enable/disable
        if "enabled" in settings:
            action_cfg.enabled = bool(settings["enabled"])

        # Per-ticker/event override
        override_key = settings.get("ticker") or settings.get("event")
        if override_key:
            override_settings = {k: v for k, v in settings.items()
                                 if k not in ("enabled", "ticker", "event")}
            if override_settings:
                action_cfg.overrides[override_key] = {
                    **action_cfg.overrides.get(override_key, {}),
                    **override_settings,
                }

        # Action-specific global settings
        if action_name == "stop_loss" and "threshold" in settings and not override_key:
            self._stop_loss_threshold = int(settings["threshold"])
        if action_name == "time_exit" and "threshold_minutes" in settings and not override_key:
            self._time_exit_minutes = int(settings["threshold_minutes"])
        if action_name == "regime_gate" and "cooldown" in settings and not override_key:
            self._regime_gate_cooldown = float(settings["cooldown"])

        return self.get_config(action_name)

    def get_config(self, action_name: Optional[str] = None) -> Dict[str, Any]:
        """Get current configuration for one or all actions."""
        configs = {
            "stop_loss": {
                "enabled": self._stop_loss.enabled,
                "threshold": self._stop_loss_threshold,
                "overrides": dict(self._stop_loss.overrides),
            },
            "time_exit": {
                "enabled": self._time_exit.enabled,
                "threshold_minutes": self._time_exit_minutes,
                "overrides": dict(self._time_exit.overrides),
            },
            "regime_gate": {
                "enabled": self._regime_gate.enabled,
                "cooldown": self._regime_gate_cooldown,
                "overrides": dict(self._regime_gate.overrides),
            },
        }
        if action_name:
            return configs.get(action_name, {"error": f"Unknown: {action_name}"})
        return configs

    def stop(self) -> None:
        """Cancel pending resume tasks on shutdown."""
        for task in self._pending_resume_tasks:
            task.cancel()
        self._pending_resume_tasks.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get auto-action stats."""
        return {
            "config": self.get_config(),
            "recent_actions": self._action_log[-10:],
            "regime_gate_active": {
                k: round(time.time() - v, 1)
                for k, v in self._regime_gate_last.items()
                if time.time() - v < self._regime_gate_cooldown
            },
        }

    def _log_action(self, action_name: str, item: AttentionItem, result: AutoActionResult) -> None:
        """Log an action for stats/debugging and broadcast to frontend."""
        entry = {
            "action": action_name,
            "event_ticker": item.event_ticker,
            "market_ticker": item.market_ticker,
            "summary": result.summary,
            "order_id": result.order_id,
            "timestamp": time.time(),
        }
        self._action_log.append(entry)
        # Keep only last 50
        if len(self._action_log) > 50:
            self._action_log = self._action_log[-50:]
        logger.info(f"[AUTO_ACTION:{action_name.upper()}] {result.summary}")

        # Broadcast to frontend
        if self._broadcast:
            try:
                asyncio.create_task(self._broadcast({
                    "type": "auto_action_fired",
                    "data": entry,
                }))
            except Exception:
                pass  # Best-effort broadcast
