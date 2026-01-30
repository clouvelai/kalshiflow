"""
Centralized Pydantic-style response schemas for all LLM calls.

All structured LLM responses should use these schemas for consistent
parsing, validation, and type safety across the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class LLMEntityItem:
    """Single entity from LLM extraction response."""

    name: str
    type: str  # PERSON, ORG, GPE, EVENT, POLICY, NORP
    sentiment: int = 0  # -100 to +100
    confidence: str = "medium"  # low, medium, high
    market_tickers: List[str] = field(default_factory=list)
    context: str = ""


@dataclass
class LLMEntityExtractionResponse:
    """Response from entity extraction LLM call."""

    entities: List[LLMEntityItem] = field(default_factory=list)

    @classmethod
    def from_json(cls, data: List[Dict[str, Any]]) -> "LLMEntityExtractionResponse":
        items = []
        for item in data:
            items.append(LLMEntityItem(
                name=item.get("name", ""),
                type=item.get("type", "UNKNOWN"),
                sentiment=int(item.get("sentiment", 0)),
                confidence=item.get("confidence", "medium"),
                market_tickers=item.get("market_tickers", []),
                context=item.get("context", ""),
            ))
        return cls(entities=items)


@dataclass
class LLMEventClassificationItem:
    """Classification of a single yes_sub_title within an event."""

    type: str  # person, organization, outcome
    aliases: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    related_entities: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)


@dataclass
class LLMEventClassificationResponse:
    """Response from event classification LLM call.

    Extended to include keywords, related_entities, and categories
    for objective entity generation.
    """

    event_entity: Optional[Dict[str, str]] = None
    classifications: Dict[str, LLMEventClassificationItem] = field(
        default_factory=dict
    )
    # Objective entity fields (populated for outcome-type events)
    keywords: List[str] = field(default_factory=list)
    related_entities: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "LLMEventClassificationResponse":
        event_entity = data.get("event_entity")
        classifications = {}
        for key, val in data.get("classifications", {}).items():
            if isinstance(val, dict):
                classifications[key] = LLMEventClassificationItem(
                    type=val.get("type", "person"),
                    aliases=val.get("aliases", []),
                    keywords=val.get("keywords", []),
                    related_entities=val.get("related_entities", []),
                    categories=val.get("categories", []),
                )
        return cls(
            event_entity=event_entity,
            classifications=classifications,
            keywords=data.get("keywords", []),
            related_entities=data.get("related_entities", []),
            categories=data.get("categories", []),
        )


@dataclass
class LLMObjectiveEntityResponse:
    """Response from objective entity generation LLM call."""

    keywords: List[str] = field(default_factory=list)
    related_entities: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "LLMObjectiveEntityResponse":
        return cls(
            keywords=data.get("keywords", []),
            related_entities=data.get("related_entities", []),
            categories=data.get("categories", []),
        )


@dataclass
class LLMTextCatResponse:
    """Response from text categorization LLM call."""

    categories: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "LLMTextCatResponse":
        return cls(categories=data)

    def above_threshold(self, threshold: float = 0.5) -> List[str]:
        """Return category names scoring above threshold."""
        return [cat for cat, score in self.categories.items() if score >= threshold]


@dataclass
class LLMMarketImpactResponse:
    """Response from market impact assessment LLM call."""

    market_ticker: str = ""
    price_impact: int = 0  # -100 to +100
    reasoning: str = ""
    side: str = "YES"  # YES or NO
    confidence: str = "medium"  # low, medium, high

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "LLMMarketImpactResponse":
        return cls(
            market_ticker=data.get("market_ticker", ""),
            price_impact=int(data.get("price_impact", 0)),
            reasoning=data.get("reasoning", ""),
            side=data.get("side", "YES"),
            confidence=data.get("confidence", "medium"),
        )

    @property
    def confidence_float(self) -> float:
        """Convert confidence label to float."""
        return {"low": 0.5, "medium": 0.7, "high": 0.9}.get(
            self.confidence.lower(), 0.5
        )
