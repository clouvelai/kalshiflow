"""
CausalModel - Structured causal knowledge for event-driven trading.

Separate from EventUnderstanding (static context, 4h cache, full rebuild).
CausalModel = "What's DRIVING prices NOW?" (dynamic state, incremental updates).

Data structures:
  CausalModel (per event)
  ├── drivers: List[Driver]       — What's moving prices
  ├── catalysts: List[Catalyst]   — Upcoming events that could shift drivers
  ├── entity_links                — Who affects which markets
  ├── dominant_narrative           — 1-2 sentence summary
  ├── consensus_direction          — bullish/bearish/mixed/unclear
  └── uncertainty_level            — 0-1

Driver lifecycle:
  ACTIVE → (2h without validation) → STALE → (prune) → removed
  ACTIVE → (3+ contradictions > validations) → INVALIDATED
"""

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.causal_model")

STALE_THRESHOLD_SECONDS = 2 * 60 * 60  # 2 hours without validation = stale


@dataclass
class MarketLink:
    """How a driver affects a specific market."""
    market_ticker: str = ""
    direction: str = "neutral"  # bullish / bearish / neutral
    magnitude: float = 0.5     # 0-1, how strongly this driver affects this market
    mechanism: str = ""        # Brief explanation


@dataclass
class Driver:
    """A causal factor driving prices in an event."""
    id: str = ""
    name: str = ""
    direction: str = "neutral"   # bullish / bearish / neutral / ambiguous
    confidence: float = 0.5      # 0-1
    status: str = "active"       # active / stale / invalidated
    market_links: List[MarketLink] = field(default_factory=list)
    evidence_sources: List[str] = field(default_factory=list)
    first_seen_at: float = 0.0
    last_validated_at: float = 0.0
    validation_count: int = 0
    contradiction_count: int = 0

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "direction": self.direction,
            "confidence": round(self.confidence, 2),
            "status": self.status,
            "market_links": [asdict(ml) for ml in self.market_links],
            "evidence_sources": self.evidence_sources[-3:],  # Last 3
            "first_seen_at": self.first_seen_at,
            "last_validated_at": self.last_validated_at,
            "validation_count": self.validation_count,
        }

    def to_compact(self) -> Dict:
        """Minimal representation for get_events_summary (~15 tokens)."""
        return {
            "id": self.id,
            "name": self.name,
            "dir": self.direction[:1],  # b/n/a
            "conf": round(self.confidence, 1),
            "status": self.status[0],   # a/s/i
        }


@dataclass
class Catalyst:
    """An upcoming event that could shift drivers."""
    id: str = ""
    name: str = ""
    type: str = "expected"     # scheduled / expected / conditional
    expected_date: str = ""    # Human-readable
    expected_ts: float = 0.0   # Unix timestamp
    affected_drivers: List[str] = field(default_factory=list)  # Driver IDs
    affected_markets: List[str] = field(default_factory=list)  # Market tickers
    occurred: bool = False
    magnitude: float = 0.5     # 0-1, expected impact

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_compact(self) -> Dict:
        return {
            "name": self.name,
            "type": self.type[0],  # s/e/c
            "date": self.expected_date,
            "occurred": self.occurred,
        }


@dataclass
class EntityMarketLink:
    """Maps an entity (person, org) to markets they affect."""
    entity_name: str = ""
    entity_role: str = ""
    market_ticker: str = ""
    relationship: str = ""   # e.g. "decision maker", "subject"
    strength: float = 0.5    # 0-1


@dataclass
class CausalModel:
    """Per-event causal model maintained across Captain cycles."""
    event_ticker: str = ""
    drivers: List[Driver] = field(default_factory=list)
    catalysts: List[Catalyst] = field(default_factory=list)
    entity_links: List[EntityMarketLink] = field(default_factory=list)
    dominant_narrative: str = ""
    consensus_direction: str = "unclear"  # bullish / bearish / mixed / unclear
    uncertainty_level: float = 0.5        # 0-1
    built_at: float = 0.0
    last_cycle_at: float = 0.0
    cycle_count: int = 0

    # --- Mutation methods ---

    def add_driver(
        self,
        name: str,
        direction: str = "neutral",
        confidence: float = 0.5,
        market_links: Optional[List[Dict]] = None,
        evidence: str = "",
    ) -> Driver:
        """Add a new driver. Returns the created Driver."""
        now = time.time()
        slug = hashlib.md5(name.lower().encode()).hexdigest()[:8]
        driver_id = f"D-{slug}"

        # Check for duplicate
        existing = self.get_driver(driver_id)
        if existing:
            # Treat as validation instead
            self.validate_driver(driver_id, evidence)
            return existing

        links = []
        if market_links:
            for ml in market_links:
                links.append(MarketLink(
                    market_ticker=ml.get("market_ticker", ""),
                    direction=ml.get("direction", direction),
                    magnitude=ml.get("magnitude", 0.5),
                    mechanism=ml.get("mechanism", ""),
                ))

        driver = Driver(
            id=driver_id,
            name=name,
            direction=direction,
            confidence=min(1.0, max(0.0, confidence)),
            status="active",
            market_links=links,
            evidence_sources=[evidence] if evidence else [],
            first_seen_at=now,
            last_validated_at=now,
            validation_count=1,
        )
        self.drivers.append(driver)
        return driver

    def validate_driver(self, driver_id: str, evidence: str = "") -> bool:
        """Mark a driver as still valid. Returns True if found."""
        driver = self.get_driver(driver_id)
        if not driver:
            return False
        driver.last_validated_at = time.time()
        driver.validation_count += 1
        if driver.status == "stale":
            driver.status = "active"
        if evidence:
            driver.evidence_sources.append(evidence)
            # Keep last 10 evidence sources
            driver.evidence_sources = driver.evidence_sources[-10:]
        return True

    def invalidate_driver(self, driver_id: str, reason: str = "") -> bool:
        """Explicitly invalidate a driver."""
        driver = self.get_driver(driver_id)
        if not driver:
            return False
        driver.status = "invalidated"
        driver.contradiction_count += 1
        if reason:
            driver.evidence_sources.append(f"INVALIDATED: {reason}")
        return True

    def contradict_driver(self, driver_id: str, evidence: str = "") -> bool:
        """Record a contradiction against a driver."""
        driver = self.get_driver(driver_id)
        if not driver:
            return False
        driver.contradiction_count += 1
        if evidence:
            driver.evidence_sources.append(f"CONTRA: {evidence}")
        # Auto-invalidate if contradictions exceed validations
        if driver.contradiction_count > driver.validation_count:
            driver.status = "invalidated"
        return True

    def add_catalyst(
        self,
        name: str,
        type: str = "expected",
        expected_date: str = "",
        expected_ts: float = 0.0,
        affected_drivers: Optional[List[str]] = None,
        affected_markets: Optional[List[str]] = None,
        magnitude: float = 0.5,
    ) -> Catalyst:
        """Add a new catalyst."""
        slug = hashlib.md5(name.lower().encode()).hexdigest()[:8]
        catalyst_id = f"C-{slug}"

        # Check for duplicate
        existing = next((c for c in self.catalysts if c.id == catalyst_id), None)
        if existing:
            return existing

        catalyst = Catalyst(
            id=catalyst_id,
            name=name,
            type=type,
            expected_date=expected_date,
            expected_ts=expected_ts,
            affected_drivers=affected_drivers or [],
            affected_markets=affected_markets or [],
            magnitude=min(1.0, max(0.0, magnitude)),
        )
        self.catalysts.append(catalyst)
        return catalyst

    def mark_catalyst_occurred(self, catalyst_id: str, magnitude: float = 0.0) -> bool:
        """Mark a catalyst as having occurred."""
        catalyst = next((c for c in self.catalysts if c.id == catalyst_id), None)
        if not catalyst:
            return False
        catalyst.occurred = True
        if magnitude > 0:
            catalyst.magnitude = magnitude
        return True

    def prune_stale(self) -> int:
        """Mark drivers not validated in 2h as stale, remove invalidated. Returns count pruned."""
        now = time.time()
        pruned = 0
        remaining = []
        for driver in self.drivers:
            if driver.status == "invalidated":
                pruned += 1
                continue
            if driver.status == "active" and (now - driver.last_validated_at) > STALE_THRESHOLD_SECONDS:
                driver.status = "stale"
            remaining.append(driver)
        self.drivers = remaining
        return pruned

    def increment_cycle(self) -> None:
        """Called at start of each Captain cycle."""
        self.last_cycle_at = time.time()
        self.cycle_count += 1

    def get_driver(self, driver_id: str) -> Optional[Driver]:
        """Find a driver by ID."""
        return next((d for d in self.drivers if d.id == driver_id), None)

    @property
    def active_drivers(self) -> List[Driver]:
        return [d for d in self.drivers if d.status == "active"]

    @property
    def next_catalyst(self) -> Optional[Catalyst]:
        """Next unoccurred catalyst by expected_ts."""
        pending = [c for c in self.catalysts if not c.occurred and c.expected_ts > 0]
        if not pending:
            return None
        return min(pending, key=lambda c: c.expected_ts)

    @property
    def imminent_catalysts(self) -> List[Catalyst]:
        """Catalysts within 4 hours."""
        now = time.time()
        four_hours = 4 * 60 * 60
        return [
            c for c in self.catalysts
            if not c.occurred and c.expected_ts > 0 and (c.expected_ts - now) < four_hours
        ]

    # --- Serialization ---

    def to_dict(self) -> Dict:
        """Full representation for get_event_snapshot() (~200-400 tokens)."""
        return {
            "event_ticker": self.event_ticker,
            "drivers": [d.to_dict() for d in self.drivers if d.status != "invalidated"],
            "catalysts": [c.to_dict() for c in self.catalysts],
            "entity_links": [asdict(el) for el in self.entity_links],
            "dominant_narrative": self.dominant_narrative,
            "consensus_direction": self.consensus_direction,
            "uncertainty_level": round(self.uncertainty_level, 2),
            "built_at": self.built_at,
            "cycle_count": self.cycle_count,
        }

    def to_compact_dict(self) -> Dict:
        """Compact representation for get_events_summary() (~50-80 tokens)."""
        active = self.active_drivers
        top_drivers = sorted(active, key=lambda d: d.confidence, reverse=True)[:3]
        nc = self.next_catalyst

        result = {
            "consensus": self.consensus_direction,
            "uncertainty": round(self.uncertainty_level, 1),
            "narrative": self.dominant_narrative[:100] if self.dominant_narrative else "",
            "top_drivers": [d.to_compact() for d in top_drivers],
        }
        if nc:
            hours_until = (nc.expected_ts - time.time()) / 3600 if nc.expected_ts > 0 else None
            result["next_catalyst"] = {
                "name": nc.name,
                "hours_until": round(hours_until, 1) if hours_until else None,
            }
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> "CausalModel":
        """Reconstruct from serialized dict."""
        model = cls(
            event_ticker=data.get("event_ticker", ""),
            dominant_narrative=data.get("dominant_narrative", ""),
            consensus_direction=data.get("consensus_direction", "unclear"),
            uncertainty_level=data.get("uncertainty_level", 0.5),
            built_at=data.get("built_at", 0.0),
            cycle_count=data.get("cycle_count", 0),
        )

        for d_data in data.get("drivers", []):
            links = [MarketLink(**ml) for ml in d_data.get("market_links", [])]
            driver = Driver(
                id=d_data.get("id", ""),
                name=d_data.get("name", ""),
                direction=d_data.get("direction", "neutral"),
                confidence=d_data.get("confidence", 0.5),
                status=d_data.get("status", "active"),
                market_links=links,
                evidence_sources=d_data.get("evidence_sources", []),
                first_seen_at=d_data.get("first_seen_at", 0.0),
                last_validated_at=d_data.get("last_validated_at", 0.0),
                validation_count=d_data.get("validation_count", 0),
                contradiction_count=d_data.get("contradiction_count", 0),
            )
            model.drivers.append(driver)

        for c_data in data.get("catalysts", []):
            catalyst = Catalyst(**{k: v for k, v in c_data.items()})
            model.catalysts.append(catalyst)

        for el_data in data.get("entity_links", []):
            model.entity_links.append(EntityMarketLink(**el_data))

        return model


class CausalModelBuilder:
    """Builds initial CausalModel from event data using a single Haiku LLM call.

    Pipeline:
    1. Gather context: EventMeta, understanding, micro signals, candlestick trends, lifecycle
    2. Single Haiku call → structured JSON
    3. Derive entity-market links from participants (pure Python)
    4. Fallback: convert flat key_factors into neutral drivers if LLM fails
    """

    BUILD_PROMPT = """Analyze this prediction market event and identify the CAUSAL DRIVERS of price movement.

EVENT DATA:
{context}

Return JSON (no markdown):
{{
    "drivers": [
        {{
            "name": "short descriptive name",
            "direction": "bullish|bearish|neutral|ambiguous",
            "confidence": 0.0-1.0,
            "market_links": [
                {{"market_ticker": "TICKER", "direction": "bullish|bearish", "magnitude": 0.0-1.0, "mechanism": "brief explanation"}}
            ],
            "evidence": "what supports this driver"
        }}
    ],
    "catalysts": [
        {{
            "name": "upcoming event name",
            "type": "scheduled|expected|conditional",
            "expected_date": "human-readable date/time",
            "affected_markets": ["TICKER1"],
            "magnitude": 0.0-1.0
        }}
    ],
    "dominant_narrative": "1-2 sentences: what story is driving prices right now",
    "consensus_direction": "bullish|bearish|mixed|unclear",
    "uncertainty_level": 0.0-1.0
}}

Rules:
- Max 5 drivers, max 3 catalysts
- Drivers should explain WHY prices are where they are
- Confidence reflects how certain we are this driver is active
- magnitude reflects how much this driver moves the specific market
- Use actual market tickers from the data
- Focus on ACTIONABLE drivers, not background context"""

    def __init__(self):
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            from .mentions_models import get_extraction_llm
            self._llm = get_extraction_llm(model="haiku", temperature=0.2, max_tokens=1000)
        return self._llm

    def _build_context(self, event_meta, understanding, lifecycle) -> str:
        """Build context string for LLM prompt."""
        lines = []
        lines.append(f"Event: {event_meta.title}")
        lines.append(f"Ticker: {event_meta.event_ticker}")
        lines.append(f"Markets: {len(event_meta.markets)}")
        lines.append(f"Mutually exclusive: {event_meta.mutually_exclusive}")

        # Market prices
        for ticker, m in event_meta.markets.items():
            mid = f"{m.yes_mid:.0f}c" if m.yes_mid is not None else "N/A"
            spread = f"{m.spread}c" if m.spread is not None else "N/A"
            lines.append(f"  {ticker}: {m.title[:50]} | mid={mid} spread={spread}")

        # Understanding context
        if understanding:
            if understanding.get("trading_summary"):
                lines.append(f"Trading summary: {understanding['trading_summary'][:200]}")
            for kf in understanding.get("key_factors", [])[:5]:
                lines.append(f"  Factor: {kf[:100]}")
            if understanding.get("participants"):
                names = [p.get("name", "") for p in understanding["participants"][:5]]
                lines.append(f"Participants: {', '.join(names)}")
            if understanding.get("news_articles"):
                lines.append(f"News ({len(understanding['news_articles'])} articles):")
                for a in understanding["news_articles"][:3]:
                    lines.append(f"  - {a.get('title', '')[:80]}")

        # Micro signals
        total_whale = sum(m.micro.whale_trade_count for m in event_meta.markets.values())
        total_vol = sum(m.micro.volume_5m for m in event_meta.markets.values())
        if total_whale > 0 or total_vol > 0:
            lines.append(f"Microstructure: whales={total_whale}, vol_5m={total_vol}")

        # Candlestick trends
        if event_meta.candlesticks:
            cs = event_meta.candlestick_summary()
            for ticker, info in list(cs.items())[:3]:
                lines.append(
                    f"  Trend {ticker}: {info.get('price_trend', 'flat')}, "
                    f"current={info.get('price_current')}c, 7d_avg={info.get('price_7d_avg')}c"
                )

        # Lifecycle
        if lifecycle:
            lines.append(
                f"Lifecycle: {lifecycle.get('stage', 'unknown')} "
                f"(action={lifecycle.get('recommended_action', 'N/A')})"
            )

        return "\n".join(lines)

    async def build(self, event_meta) -> CausalModel:
        """Build a CausalModel for an event.

        Args:
            event_meta: EventMeta instance with understanding and lifecycle populated

        Returns:
            CausalModel instance
        """
        understanding = getattr(event_meta, "understanding", None)
        lifecycle = getattr(event_meta, "lifecycle", None)

        context = self._build_context(event_meta, understanding, lifecycle)
        prompt = self.BUILD_PROMPT.format(context=context)

        model = CausalModel(
            event_ticker=event_meta.event_ticker,
            built_at=time.time(),
        )

        try:
            from .llm_schemas import CausalModelExtraction

            llm = self._get_llm()
            structured_llm = llm.with_structured_output(CausalModelExtraction)
            parsed = await structured_llm.ainvoke(prompt)

            # Build drivers
            for d_data in parsed.drivers[:5]:
                model.add_driver(
                    name=d_data.name,
                    direction=d_data.direction,
                    confidence=d_data.confidence,
                    market_links=[ml.model_dump() for ml in d_data.market_links],
                    evidence=d_data.evidence,
                )

            # Build catalysts
            for c_data in parsed.catalysts[:3]:
                model.add_catalyst(
                    name=c_data.name,
                    type=c_data.type,
                    expected_date=c_data.expected_date,
                    affected_markets=c_data.affected_markets,
                    magnitude=c_data.magnitude,
                )

            model.dominant_narrative = parsed.dominant_narrative
            model.consensus_direction = parsed.consensus_direction
            model.uncertainty_level = parsed.uncertainty_level

        except Exception as e:
            logger.warning(f"[CAUSAL] LLM build failed for {event_meta.event_ticker}: {e}")
            # Fallback: convert key_factors into neutral drivers
            self._build_fallback(model, event_meta, understanding)

        # Derive entity-market links from participants (pure Python)
        self._derive_entity_links(model, event_meta, understanding)

        logger.info(
            f"[CAUSAL] Built model for {event_meta.event_ticker}: "
            f"drivers={len(model.drivers)} catalysts={len(model.catalysts)} "
            f"entities={len(model.entity_links)} consensus={model.consensus_direction}"
        )
        return model

    def _build_fallback(self, model: CausalModel, event_meta, understanding) -> None:
        """Convert flat key_factors into neutral drivers."""
        if not understanding:
            return

        key_factors = understanding.get("key_factors", [])
        for i, factor in enumerate(key_factors[:5]):
            model.add_driver(
                name=factor[:60],
                direction="neutral",
                confidence=0.3,
                evidence="fallback from key_factors",
            )

        model.dominant_narrative = understanding.get("trading_summary", "")[:200]
        model.consensus_direction = "unclear"
        model.uncertainty_level = 0.7

    def _derive_entity_links(self, model: CausalModel, event_meta, understanding) -> None:
        """Derive entity-market links from participants."""
        if not understanding:
            return

        participants = understanding.get("participants", [])
        market_tickers = list(event_meta.markets.keys())

        for p in participants:
            name = p.get("name", "")
            role = p.get("role", "")
            if not name:
                continue

            # Link entity to all markets in the event with default strength
            for ticker in market_tickers:
                market = event_meta.markets.get(ticker)
                if not market:
                    continue

                # Check if entity name appears in market title
                name_lower = name.lower()
                title_lower = market.title.lower()
                strength = 0.8 if name_lower in title_lower else 0.3

                relationship = role if role else "participant"

                model.entity_links.append(EntityMarketLink(
                    entity_name=name,
                    entity_role=role,
                    market_ticker=ticker,
                    relationship=relationship,
                    strength=strength,
                ))
