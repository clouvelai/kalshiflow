"""
Tests for TraderStateMachine.

Covers valid/invalid transitions, transition tracking, error state handling,
callbacks, timeouts, and property checks.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from kalshiflow_rl.traderv3.core.state_machine import (
    TraderStateMachine,
    TraderState,
    StateTransition,
    StateMetrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _advance(sm: TraderStateMachine, *states: TraderState) -> None:
    """Walk the state machine through a sequence of states."""
    for s in states:
        result = await sm.transition_to(s, context="test")
        assert result, f"Expected transition to {s} to succeed"


# ---------------------------------------------------------------------------
# TestValidTransitions
# ---------------------------------------------------------------------------

class TestValidTransitions:

    @pytest.mark.asyncio
    async def test_startup_to_initializing(self):
        sm = TraderStateMachine(event_bus=None)
        result = await sm.transition_to(TraderState.INITIALIZING, context="boot")
        assert result is True
        assert sm.current_state == TraderState.INITIALIZING

    @pytest.mark.asyncio
    async def test_initializing_to_orderbook(self):
        sm = TraderStateMachine(event_bus=None)
        await _advance(sm, TraderState.INITIALIZING)
        result = await sm.transition_to(TraderState.ORDERBOOK_CONNECT, context="connect ob")
        assert result is True
        assert sm.current_state == TraderState.ORDERBOOK_CONNECT

    @pytest.mark.asyncio
    async def test_orderbook_to_ready(self):
        sm = TraderStateMachine(event_bus=None)
        await _advance(sm, TraderState.INITIALIZING, TraderState.ORDERBOOK_CONNECT)
        result = await sm.transition_to(TraderState.READY, context="skip trading")
        assert result is True
        assert sm.current_state == TraderState.READY

    @pytest.mark.asyncio
    async def test_ready_to_shutdown(self):
        sm = TraderStateMachine(event_bus=None)
        await _advance(
            sm,
            TraderState.INITIALIZING,
            TraderState.ORDERBOOK_CONNECT,
            TraderState.READY,
        )
        result = await sm.transition_to(TraderState.SHUTDOWN, context="bye")
        assert result is True
        assert sm.current_state == TraderState.SHUTDOWN

    @pytest.mark.asyncio
    async def test_invalid_startup_to_ready(self):
        sm = TraderStateMachine(event_bus=None)
        result = await sm.transition_to(TraderState.READY, context="shortcut")
        assert result is False
        assert sm.current_state == TraderState.STARTUP

    @pytest.mark.asyncio
    async def test_shutdown_is_terminal(self):
        sm = TraderStateMachine(event_bus=None)
        await _advance(
            sm,
            TraderState.INITIALIZING,
            TraderState.ORDERBOOK_CONNECT,
            TraderState.READY,
            TraderState.SHUTDOWN,
        )
        for target in TraderState:
            if target == TraderState.SHUTDOWN:
                continue
            result = await sm.transition_to(target, context="escape")
            assert result is False or target == TraderState.SHUTDOWN
        assert sm.current_state == TraderState.SHUTDOWN

    @pytest.mark.asyncio
    async def test_error_to_startup_recovery(self):
        sm = TraderStateMachine(event_bus=None)
        await _advance(sm, TraderState.INITIALIZING, TraderState.ORDERBOOK_CONNECT, TraderState.READY)
        await sm.transition_to(TraderState.ERROR, context="oops")
        result = await sm.transition_to(TraderState.STARTUP, context="recover")
        assert result is True
        assert sm.current_state == TraderState.STARTUP

    @pytest.mark.asyncio
    async def test_same_state_noop(self):
        sm = TraderStateMachine(event_bus=None)
        transitions_before = sm._total_transitions
        result = await sm.transition_to(TraderState.STARTUP, context="noop")
        assert result is True
        assert sm._total_transitions == transitions_before


# ---------------------------------------------------------------------------
# TestTransitionTracking
# ---------------------------------------------------------------------------

class TestTransitionTracking:

    @pytest.mark.asyncio
    async def test_counter_increment(self):
        sm = TraderStateMachine(event_bus=None)
        await _advance(
            sm,
            TraderState.INITIALIZING,
            TraderState.ORDERBOOK_CONNECT,
            TraderState.READY,
        )
        assert sm._total_transitions == 3

    @pytest.mark.asyncio
    async def test_history_recording(self):
        sm = TraderStateMachine(event_bus=None)
        await sm.transition_to(TraderState.INITIALIZING, context="boot")
        assert len(sm._transition_history) == 1
        t = sm._transition_history[0]
        assert t.from_state == TraderState.STARTUP
        assert t.to_state == TraderState.INITIALIZING
        assert t.context == "boot"

    @pytest.mark.asyncio
    async def test_history_50_cap(self):
        sm = TraderStateMachine(event_bus=None)
        # Bounce between READY and ACTING to generate many transitions
        await _advance(
            sm,
            TraderState.INITIALIZING,
            TraderState.ORDERBOOK_CONNECT,
            TraderState.READY,
        )
        # 3 transitions so far; need 57 more to reach 60
        for _ in range(57):
            await sm.transition_to(TraderState.ACTING, context="act")
            await sm.transition_to(TraderState.READY, context="back")
        # 3 + 57*2 = 117 total transitions, but history capped at 50
        assert sm._total_transitions > 50
        assert len(sm._transition_history) <= 50


# ---------------------------------------------------------------------------
# TestErrorState
# ---------------------------------------------------------------------------

class TestErrorState:

    @pytest.mark.asyncio
    async def test_enter_error_state(self):
        sm = TraderStateMachine(event_bus=None)
        await _advance(sm, TraderState.INITIALIZING, TraderState.ORDERBOOK_CONNECT, TraderState.READY)
        await sm.enter_error_state("test failure", error=RuntimeError("boom"))
        assert sm.current_state == TraderState.ERROR
        assert sm._error_count >= 1
        assert "boom" in sm._last_error

    @pytest.mark.asyncio
    async def test_error_count_increment(self):
        sm = TraderStateMachine(event_bus=None)
        await _advance(sm, TraderState.INITIALIZING, TraderState.ORDERBOOK_CONNECT, TraderState.READY)
        await sm.enter_error_state("err1", error=RuntimeError("a"))
        count_after_first = sm._error_count

        # Recover then error again
        await sm.transition_to(TraderState.STARTUP, context="recover")
        await _advance(sm, TraderState.INITIALIZING, TraderState.ORDERBOOK_CONNECT, TraderState.READY)
        await sm.enter_error_state("err2", error=RuntimeError("b"))
        assert sm._error_count == count_after_first + 1


# ---------------------------------------------------------------------------
# TestCallbacks
# ---------------------------------------------------------------------------

class TestCallbacks:

    @pytest.mark.asyncio
    async def test_on_enter_callback(self):
        sm = TraderStateMachine(event_bus=None)
        called_with = {}

        async def on_ready(state, metrics):
            called_with["state"] = state
            called_with["metrics"] = metrics

        sm.register_state_callback(TraderState.READY, on_ready, on_enter=True)
        await _advance(
            sm,
            TraderState.INITIALIZING,
            TraderState.ORDERBOOK_CONNECT,
            TraderState.READY,
        )
        assert called_with["state"] == TraderState.READY
        assert isinstance(called_with["metrics"], StateMetrics)

    @pytest.mark.asyncio
    async def test_callback_exception_isolation(self):
        sm = TraderStateMachine(event_bus=None)

        async def bad_callback(state, metrics):
            raise RuntimeError("callback exploded")

        sm.register_state_callback(TraderState.INITIALIZING, bad_callback, on_enter=True)
        # Transition should still succeed despite callback error
        result = await sm.transition_to(TraderState.INITIALIZING, context="boot")
        assert result is True
        assert sm.current_state == TraderState.INITIALIZING


# ---------------------------------------------------------------------------
# TestTimeout
# ---------------------------------------------------------------------------

class TestTimeout:

    @pytest.mark.asyncio
    async def test_ready_never_expires(self):
        sm = TraderStateMachine(event_bus=None)
        await _advance(
            sm,
            TraderState.INITIALIZING,
            TraderState.ORDERBOOK_CONNECT,
            TraderState.READY,
        )
        # Even with a huge elapsed time, READY should not time out
        sm._state_entered_at = 0  # pretend we entered long ago
        with patch("kalshiflow_rl.traderv3.core.state_machine.time") as mock_time:
            mock_time.time.return_value = 999_999_999.0
            assert sm.check_state_timeout() is False

    @pytest.mark.asyncio
    async def test_startup_timeout(self):
        sm = TraderStateMachine(event_bus=None)
        # STARTUP timeout is 30s
        sm._state_entered_at = 100.0
        with patch("kalshiflow_rl.traderv3.core.state_machine.time") as mock_time:
            mock_time.time.return_value = 131.0  # 31s elapsed, exceeds 30s
            assert sm.check_state_timeout() is True


# ---------------------------------------------------------------------------
# TestProperties
# ---------------------------------------------------------------------------

class TestProperties:

    @pytest.mark.asyncio
    async def test_is_terminal_shutdown(self):
        sm = TraderStateMachine(event_bus=None)
        await _advance(
            sm,
            TraderState.INITIALIZING,
            TraderState.ORDERBOOK_CONNECT,
            TraderState.READY,
            TraderState.SHUTDOWN,
        )
        assert sm.is_terminal_state is True

    @pytest.mark.asyncio
    async def test_is_operational_ready(self):
        sm = TraderStateMachine(event_bus=None)
        await _advance(
            sm,
            TraderState.INITIALIZING,
            TraderState.ORDERBOOK_CONNECT,
            TraderState.READY,
        )
        assert sm.is_operational_state is True

    @pytest.mark.asyncio
    async def test_is_healthy_ready(self):
        sm = TraderStateMachine(event_bus=None)
        await _advance(
            sm,
            TraderState.INITIALIZING,
            TraderState.ORDERBOOK_CONNECT,
            TraderState.READY,
        )
        assert sm.is_healthy() is True

    @pytest.mark.asyncio
    async def test_not_healthy_error(self):
        sm = TraderStateMachine(event_bus=None)
        await _advance(
            sm,
            TraderState.INITIALIZING,
            TraderState.ORDERBOOK_CONNECT,
            TraderState.READY,
        )
        await sm.enter_error_state("fail", error=RuntimeError("x"))
        assert sm.is_healthy() is False
