"""Tests for AccountHealthService multi-group protection (Captain + MM order groups)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kalshiflow_rl.traderv3.single_arb.account_health import AccountHealthService


def _make_service(order_group_id="group-captain"):
    """Create an AccountHealthService with mocked dependencies."""
    svc = AccountHealthService(
        gateway=MagicMock(),
        index=MagicMock(),
        session=MagicMock(),
        order_group_id=order_group_id,
    )
    return svc


class TestRegisterOrderGroup:
    def test_register_adds_to_protected(self):
        svc = _make_service("group-captain")
        svc.register_order_group("group-mm")
        assert "group-captain" in svc._protected_order_groups
        assert "group-mm" in svc._protected_order_groups

    def test_initial_group_in_protected(self):
        svc = _make_service("group-captain")
        assert "group-captain" in svc._protected_order_groups

    def test_no_initial_group(self):
        svc = _make_service(order_group_id=None)
        assert len(svc._protected_order_groups) == 0

    def test_register_multiple_groups(self):
        svc = _make_service("group-captain")
        svc.register_order_group("group-mm-1")
        svc.register_order_group("group-mm-2")
        assert len(svc._protected_order_groups) == 3


class TestStaleOrderProtection:
    @pytest.mark.asyncio
    async def test_skips_captain_group_orders(self):
        svc = _make_service("group-captain")
        order = MagicMock()
        order.order_id = "order-1"
        order.order_group_id = "group-captain"
        order.created_time = "2020-01-01T00:00:00Z"
        order.expiration_time = None

        svc._gateway.get_orders = AsyncMock(return_value=[order])
        svc._gateway.cancel_order = AsyncMock()

        result = await svc._check_stale_orders()
        svc._gateway.cancel_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_mm_group_orders(self):
        svc = _make_service("group-captain")
        svc.register_order_group("group-mm")

        order = MagicMock()
        order.order_id = "order-2"
        order.order_group_id = "group-mm"
        order.created_time = "2020-01-01T00:00:00Z"
        order.expiration_time = None

        svc._gateway.get_orders = AsyncMock(return_value=[order])
        svc._gateway.cancel_order = AsyncMock()

        result = await svc._check_stale_orders()
        svc._gateway.cancel_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancels_unprotected_stale_orders(self):
        svc = _make_service("group-captain")
        svc.register_order_group("group-mm")

        order = MagicMock()
        order.order_id = "order-stale"
        order.order_group_id = "group-other"
        order.created_time = "2020-01-01T00:00:00Z"  # Very old
        order.expiration_time = None

        svc._gateway.get_orders = AsyncMock(return_value=[order])
        svc._gateway.cancel_order = AsyncMock()

        result = await svc._check_stale_orders()
        svc._gateway.cancel_order.assert_called_once_with("order-stale")


class TestOrderGroupHygiene:
    @pytest.mark.asyncio
    async def test_skips_all_protected_groups(self):
        svc = _make_service("group-captain")
        svc.register_order_group("group-mm")

        groups = [
            {"order_group_id": "group-captain"},
            {"order_group_id": "group-mm"},
            {"order_group_id": "group-orphan"},
        ]
        svc._gateway.list_order_groups = AsyncMock(return_value=groups)
        svc._gateway.delete_order_group = AsyncMock()

        result = await svc._check_order_groups()
        # Only the orphan group should be considered for cleanup
        # (actual deletion depends on further logic in _check_order_groups)
