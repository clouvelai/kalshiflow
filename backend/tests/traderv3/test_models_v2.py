"""Unit tests for Captain V2 Pydantic models.

Pure T1 tests — no async, no mocking, no network.
Validates serialization, defaults, and model construction.
"""

from kalshiflow_rl.traderv3.single_arb.models import (
    ArbLegResult,
    ArbResult,
    CycleDiff,
    EventSemantics,
    EventSnapshot,
    MarketSnapshot,
    MarketState,
    MemoryEntry,
    NewsArticle,
    NewsSearchResult,
    OrderResult,
    PortfolioState,
    Position,
    RecallResult,
    RestingOrder,
    SniperStatus,
)


class TestMarketSnapshot:
    def test_defaults(self):
        m = MarketSnapshot(ticker="TEST-A", title="Test Market A")
        assert m.ticker == "TEST-A"
        assert m.yes_bid is None
        assert m.yes_ask is None
        assert m.vpin == 0.0
        assert m.volume_5m == 0

    def test_full_construction(self):
        m = MarketSnapshot(
            ticker="MKT-1", title="Market 1",
            yes_bid=40, yes_ask=45, spread=5,
            microprice=42.5, vpin=0.35,
            book_imbalance=0.2, volume_5m=150,
        )
        assert m.spread == 5
        assert m.microprice == 42.5

    def test_json_roundtrip(self):
        m = MarketSnapshot(ticker="MKT-1", title="Market 1", yes_bid=40, yes_ask=45)
        data = m.model_dump()
        m2 = MarketSnapshot(**data)
        assert m2.ticker == "MKT-1"
        assert m2.yes_bid == 40


class TestEventSnapshot:
    def test_defaults(self):
        e = EventSnapshot(event_ticker="EVT-1", title="Test Event")
        assert e.mutually_exclusive is True
        assert e.regime == "normal"
        assert e.markets == {}
        assert e.semantics is None

    def test_with_markets(self):
        m = MarketSnapshot(ticker="MKT-A", title="A")
        e = EventSnapshot(
            event_ticker="EVT-1", title="Test",
            markets={"MKT-A": m}, market_count=1,
        )
        assert e.market_count == 1
        assert "MKT-A" in e.markets

    def test_with_semantics(self):
        sem = EventSemantics(what="NFL game", who=["Eagles", "Chiefs"], domain="sports")
        e = EventSnapshot(event_ticker="EVT-1", title="Test", semantics=sem)
        assert e.semantics.domain == "sports"
        assert len(e.semantics.who) == 2


class TestEventSemantics:
    def test_defaults(self):
        s = EventSemantics()
        assert s.what == ""
        assert s.who == []
        assert s.search_terms == []

    def test_full_construction(self):
        s = EventSemantics(
            what="Presidential debate", who=["Candidate A", "Candidate B"],
            when="Tonight 8PM EST", domain="politics",
            search_terms=["debate", "polls"],
        )
        assert s.domain == "politics"
        assert len(s.search_terms) == 2


class TestMarketState:
    def test_empty(self):
        ms = MarketState()
        assert ms.events == []
        assert ms.total_events == 0

    def test_with_events(self):
        e = EventSnapshot(event_ticker="EVT-1", title="Test")
        ms = MarketState(events=[e], total_events=1, total_markets=3)
        assert ms.total_events == 1
        assert ms.total_markets == 3


class TestPosition:
    def test_construction(self):
        p = Position(
            ticker="MKT-A", event_ticker="EVT-1",
            side="yes", quantity=10, cost_cents=400,
            exit_price=45, current_value_cents=450,
            unrealized_pnl_cents=50,
        )
        assert p.side == "yes"
        assert p.unrealized_pnl_cents == 50


class TestPortfolioState:
    def test_empty(self):
        ps = PortfolioState()
        assert ps.balance_cents == 0
        assert ps.positions == []
        assert ps.total_positions == 0

    def test_with_positions(self):
        p = Position(ticker="MKT-A", side="yes", quantity=5, unrealized_pnl_cents=25)
        ps = PortfolioState(
            balance_cents=50000, balance_dollars=500.0,
            positions=[p], total_positions=1,
            total_unrealized_pnl_cents=25, total_cost_cents=200,
        )
        assert ps.balance_dollars == 500.0
        assert ps.total_positions == 1


class TestOrderResult:
    def test_success(self):
        r = OrderResult(
            order_id="ord-001", status="resting",
            ticker="MKT-A", side="yes", action="buy",
            contracts=5, price_cents=40, ttl_seconds=30,
            new_balance_cents=49800, new_balance_dollars=498.0,
        )
        assert r.new_balance_cents == 49800
        assert r.error is None

    def test_failure(self):
        r = OrderResult(error="Insufficient balance")
        assert r.error == "Insufficient balance"
        assert r.order_id == ""


class TestArbResult:
    def test_preview(self):
        leg = ArbLegResult(ticker="MKT-A", side="yes", contracts=5, price_cents=40)
        r = ArbResult(
            status="preview", event_ticker="EVT-1",
            direction="long", legs=[leg],
            legs_total=3, legs_executed=0,
        )
        assert r.status == "preview"
        assert len(r.legs) == 1

    def test_completed(self):
        r = ArbResult(
            status="completed", event_ticker="EVT-1",
            direction="long", legs_executed=3, legs_total=3,
            total_cost_cents=300,
            new_balance_cents=49700, new_balance_dollars=497.0,
        )
        assert r.legs_executed == r.legs_total


class TestRestingOrder:
    def test_construction(self):
        o = RestingOrder(
            order_id="ord-001", ticker="MKT-A",
            side="yes", action="buy",
            price_cents=40, remaining_count=5,
            queue_position=2,
        )
        assert o.queue_position == 2


class TestNewsSearchResult:
    def test_empty(self):
        r = NewsSearchResult(query="test")
        assert r.articles == []
        assert r.count == 0

    def test_with_articles(self):
        a = NewsArticle(title="Breaking News", url="https://example.com", score=0.95)
        r = NewsSearchResult(query="test", articles=[a], count=1, stored_in_memory=True)
        assert r.stored_in_memory is True
        assert r.articles[0].score == 0.95


class TestRecallResult:
    def test_empty(self):
        r = RecallResult(query="test")
        assert r.results == []
        assert r.count == 0

    def test_with_entries(self):
        e = MemoryEntry(content="learned something", memory_type="learning", similarity=0.85)
        r = RecallResult(query="test", results=[e], count=1)
        assert r.results[0].similarity == 0.85


class TestSniperStatus:
    def test_disabled(self):
        s = SniperStatus(enabled=False)
        assert s.total_trades == 0
        assert s.config_subset == {}

    def test_enabled(self):
        s = SniperStatus(
            enabled=True, total_trades=5,
            capital_in_flight=1000, capital_in_positions=2000,
            config_subset={"max_position": 25, "arb_min_edge": 3.0},
        )
        assert s.capital_in_flight + s.capital_in_positions == 3000


class TestCycleDiff:
    def test_no_changes(self):
        d = CycleDiff(has_changes=False)
        assert d.price_moves == []
        assert d.volume_spikes == []

    def test_with_changes(self):
        d = CycleDiff(
            elapsed_seconds=60.0,
            price_moves=["MKT-A +3c (40->43)"],
            volume_spikes=["EVT-1 +100 vol"],
            has_changes=True,
        )
        assert d.has_changes is True
        assert len(d.price_moves) == 1


class TestJsonSerialization:
    """Verify all models can be serialized to JSON and back."""

    def test_market_state_json(self):
        m = MarketSnapshot(ticker="MKT-A", title="A", yes_bid=40, yes_ask=45)
        e = EventSnapshot(event_ticker="EVT-1", title="Test", markets={"MKT-A": m})
        ms = MarketState(events=[e], total_events=1, total_markets=1)
        json_str = ms.model_dump_json()
        ms2 = MarketState.model_validate_json(json_str)
        assert ms2.events[0].markets["MKT-A"].yes_bid == 40

    def test_portfolio_json(self):
        p = Position(ticker="MKT-A", side="yes", quantity=5, unrealized_pnl_cents=25)
        ps = PortfolioState(balance_cents=50000, positions=[p], total_positions=1)
        json_str = ps.model_dump_json()
        ps2 = PortfolioState.model_validate_json(json_str)
        assert ps2.positions[0].ticker == "MKT-A"

    def test_order_result_json(self):
        r = OrderResult(order_id="ord-001", status="resting", new_balance_cents=49800)
        json_str = r.model_dump_json()
        r2 = OrderResult.model_validate_json(json_str)
        assert r2.new_balance_cents == 49800

    def test_arb_result_json(self):
        leg = ArbLegResult(ticker="MKT-A", side="yes", contracts=5, price_cents=40)
        r = ArbResult(status="completed", legs=[leg], new_balance_cents=49700)
        json_str = r.model_dump_json()
        r2 = ArbResult.model_validate_json(json_str)
        assert r2.legs[0].ticker == "MKT-A"
