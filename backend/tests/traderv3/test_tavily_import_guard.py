"""Unit tests for TavilyService import guard flags.

Verifies that _async_unavailable and _sync_unavailable flags prevent
repeated ImportError retries after first failure.
"""

from unittest.mock import MagicMock, patch

from kalshiflow_rl.traderv3.single_arb.tavily_service import TavilySearchService


def make_service():
    budget = MagicMock()
    budget.should_fallback.return_value = False
    return TavilySearchService(
        api_key="test-key",
        budget_manager=budget,
    )


class TestAsyncImportGuard:
    def test_flag_set_on_import_error(self):
        """_async_unavailable should be set True after ImportError."""
        svc = make_service()
        assert svc._async_unavailable is False

        with patch.dict("sys.modules", {"tavily": None}):
            result = svc._get_async_client()

        assert result is None
        assert svc._async_unavailable is True

    def test_short_circuits_after_flag_set(self):
        """After flag is set, _get_async_client should return None immediately."""
        svc = make_service()
        svc._async_unavailable = True

        # Should NOT attempt import at all
        result = svc._get_async_client()
        assert result is None


class TestSyncImportGuard:
    def test_flag_set_on_import_error(self):
        """_sync_unavailable should be set True after ImportError."""
        svc = make_service()
        assert svc._sync_unavailable is False

        with patch.dict("sys.modules", {"tavily": None}):
            result = svc._get_sync_client()

        assert result is None
        assert svc._sync_unavailable is True

    def test_short_circuits_after_flag_set(self):
        """After flag is set, _get_sync_client should return None immediately."""
        svc = make_service()
        svc._sync_unavailable = True

        result = svc._get_sync_client()
        assert result is None


class TestInitHasFlags:
    def test_init_flags_default_false(self):
        """Both flags should start as False."""
        svc = make_service()
        assert svc._async_unavailable is False
        assert svc._sync_unavailable is False
