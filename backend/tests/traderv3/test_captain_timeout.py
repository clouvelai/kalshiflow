"""Unit tests for Captain _run_with_timeout() — CancelledError propagation and timeout events.

Verifies:
- External CancelledError propagates (not caught as timeout)
- TimeoutError emits captain_cycle_complete event with status=timeout
- Normal completion does not emit timeout event
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest


def _make_captain(event_callback=None):
    """Create a minimal ArbCaptain with mocked dependencies (bypasses heavy __init__)."""
    from kalshiflow_rl.traderv3.single_arb.captain import ArbCaptain

    with patch.object(ArbCaptain, "__init__", lambda self, **kw: None):
        captain = ArbCaptain()

    captain._event_callback = event_callback
    captain._cycle_count = 0
    captain._errors = []
    captain._running = True
    captain._consecutive_timeouts = 0

    return captain


class TestCancelledErrorPropagation:
    """CancelledError from external cancellation must propagate, not be swallowed."""

    @pytest.mark.asyncio
    async def test_cancelled_error_raises(self):
        callback = AsyncMock()
        captain = _make_captain(event_callback=callback)

        async def cancelled_coro():
            raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await captain._run_with_timeout(cancelled_coro(), "reactive")

    @pytest.mark.asyncio
    async def test_cancelled_error_no_timeout_event(self):
        callback = AsyncMock()
        captain = _make_captain(event_callback=callback)

        async def cancelled_coro():
            raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await captain._run_with_timeout(cancelled_coro(), "strategic")

        # No events should have been emitted
        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancelled_error_no_recorded_error(self):
        captain = _make_captain(event_callback=AsyncMock())

        async def cancelled_coro():
            raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await captain._run_with_timeout(cancelled_coro(), "deep_scan")

        assert len(captain._errors) == 0


class TestTimeoutEmitsCycleComplete:
    """TimeoutError should record error and emit captain_cycle_complete with status=timeout."""

    @pytest.mark.asyncio
    async def test_timeout_emits_event(self):
        """Timeout emits captain_cycle_complete with status=timeout."""
        callback = AsyncMock()
        captain = _make_captain(event_callback=callback)

        async def dummy_coro():
            pass

        # Mock wait_for to raise TimeoutError (close coro to avoid RuntimeWarning)
        async def _timeout_wait_for(coro, *, timeout):
            coro.close()
            raise asyncio.TimeoutError()

        with patch("asyncio.wait_for", side_effect=_timeout_wait_for):
            await captain._run_with_timeout(dummy_coro(), "reactive")

        # Verify captain_cycle_complete with status=timeout was emitted
        assert callback.call_count == 1
        event = callback.call_args[0][0]
        assert event["type"] == "captain_cycle_complete"
        assert event["data"]["status"] == "timeout"
        assert event["data"]["mode"] == "reactive"
        assert event["data"]["cycle_num"] == 1
        assert event["data"]["duration_s"] == 45.0  # reactive timeout

    @pytest.mark.asyncio
    async def test_timeout_records_error(self):
        callback = AsyncMock()
        captain = _make_captain(event_callback=callback)
        captain._cycle_count = 7

        async def dummy_coro():
            pass

        async def _timeout_wait_for(coro, *, timeout):
            coro.close()
            raise asyncio.TimeoutError()

        with patch("asyncio.wait_for", side_effect=_timeout_wait_for):
            await captain._run_with_timeout(dummy_coro(), "deep_scan")

        assert "timeout_deep_scan_8" in captain._errors

    @pytest.mark.asyncio
    async def test_timeout_uses_mode_specific_duration(self):
        """Each mode should use its own timeout value in the emitted event."""
        for mode, expected_timeout in [("reactive", 45.0), ("strategic", 120.0), ("deep_scan", 180.0)]:
            callback = AsyncMock()
            captain = _make_captain(event_callback=callback)

            async def dummy_coro():
                pass

            async def _timeout_wait_for(coro, *, timeout):
                coro.close()
                raise asyncio.TimeoutError()

            with patch("asyncio.wait_for", side_effect=_timeout_wait_for):
                await captain._run_with_timeout(dummy_coro(), mode)

            event = callback.call_args[0][0]
            assert event["data"]["duration_s"] == expected_timeout, \
                f"Mode {mode} should use {expected_timeout}s timeout"

    @pytest.mark.asyncio
    async def test_successful_completion_no_timeout_event(self):
        callback = AsyncMock()
        captain = _make_captain(event_callback=callback)

        async def fast_coro():
            return "done"

        await captain._run_with_timeout(fast_coro(), "strategic")

        # _emit_event should not have been called (no timeout)
        callback.assert_not_called()
        assert len(captain._errors) == 0
