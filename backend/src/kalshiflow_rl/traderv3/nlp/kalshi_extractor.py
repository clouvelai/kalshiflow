"""
KalshiExtractor - langextract wrapper with merged event specifications.

Single langextract call per post. Merges base extraction prompt with all
active event-specific instructions into one prompt. Returns structured
extractions that map to market tickers.

Usage:
    extractor = KalshiExtractor()
    result = await extractor.extract(
        title="Trump says tariffs next week",
        body="...",
        subreddit="politics",
        engagement_score=500,
        engagement_comments=120,
        event_configs=[...],
    )
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .prompt_instruction import BASE_EXTRACTION_PROMPT, INPUT_TEXT_TEMPLATE, MARKET_CONTEXT_TEMPLATE
from .examples import ExampleManager

logger = logging.getLogger("kalshiflow_rl.traderv3.nlp.kalshi_extractor")


@dataclass
class EventConfig:
    """In-memory representation of an event_configs row."""
    event_ticker: str
    event_title: str
    primary_entity: Optional[str] = None
    primary_entity_type: Optional[str] = None
    description: Optional[str] = None
    key_drivers: List[str] = field(default_factory=list)
    outcome_descriptions: Dict[str, str] = field(default_factory=dict)
    prompt_description: Optional[str] = None
    extraction_classes: List[Dict[str, Any]] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    watchlist: Dict[str, Any] = field(default_factory=dict)
    markets: List[Dict[str, Any]] = field(default_factory=list)
    is_active: bool = True
    research_version: int = 1


@dataclass
class ExtractionRow:
    """A single extraction ready for Supabase insert."""
    source_type: str
    source_id: str
    source_url: Optional[str]
    source_subreddit: Optional[str]
    extraction_class: str
    extraction_text: str
    attributes: Dict[str, Any]
    market_tickers: List[str]
    event_tickers: List[str]
    engagement_score: int = 0
    engagement_comments: int = 0
    source_created_at: Optional[str] = None


class KalshiExtractor:
    """
    langextract wrapper that merges base + event-specific specs into one call.

    For each incoming text (Reddit post, news article), produces a list of
    ExtractionRow objects ready for Supabase insertion.
    """

    def __init__(
        self,
        model_id: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
    ):
        self._model_id = model_id
        self._api_key = api_key or os.getenv("LANGEXTRACT_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self._example_manager = ExampleManager()
        self._initialized = False

    def _ensure_langextract(self):
        """Lazy import langextract to avoid startup cost if not needed."""
        if self._initialized:
            return
        try:
            import langextract as lx
            self._lx = lx
            self._initialized = True
            logger.info(f"langextract initialized with model={self._model_id}")
        except ImportError:
            raise ImportError(
                "langextract not installed. Run: pip install langextract"
            )

    async def extract(
        self,
        title: str,
        body: str,
        subreddit: str,
        engagement_score: int,
        engagement_comments: int,
        event_configs: List[EventConfig],
        source_type: str = "reddit_post",
        source_id: str = "",
        source_url: Optional[str] = None,
        source_created_at: Optional[str] = None,
    ) -> List[ExtractionRow]:
        """
        Single langextract call per post.

        Merges base instructions + all active event configs into one prompt,
        collects global + event-specific examples, and runs extraction.

        Args:
            title: Post/article title
            body: Full text content
            subreddit: Source subreddit (empty for news)
            engagement_score: Upvotes/likes at extraction time
            engagement_comments: Comment count at extraction time
            event_configs: All active EventConfig objects
            source_type: reddit_post | reddit_comment | news | video
            source_id: Unique source identifier
            source_url: Source URL
            source_created_at: ISO timestamp of source creation

        Returns:
            List of ExtractionRow objects ready for Supabase
        """
        self._ensure_langextract()
        lx = self._lx

        # 1. Format input text
        text = INPUT_TEXT_TEMPLATE.format(
            source_type=source_type,
            subreddit=subreddit or "unknown",
            score=engagement_score,
            comments=engagement_comments,
            title=title,
            body=body[:5000] if body else "",
        )

        # 2. Build merged prompt
        prompt = self._build_merged_prompt(event_configs)

        # 3. Collect examples
        examples = self._collect_examples(event_configs)

        # 4. Convert examples to langextract ExampleData objects
        lx_examples = []
        for ex in examples:
            try:
                extractions = [
                    lx.data.Extraction(
                        extraction_class=e["extraction_class"],
                        extraction_text=e["extraction_text"],
                        attributes=e.get("attributes", {}),
                    )
                    for e in ex.get("output", [])
                ]
                lx_examples.append(
                    lx.data.ExampleData(
                        text=ex["input"],
                        extractions=extractions,
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to convert example: {e}")

        # 5. Run langextract in thread (it's synchronous)
        try:
            result = await asyncio.to_thread(
                lx.extract,
                text,
                prompt,
                lx_examples,
                self._model_id,
                api_key=self._api_key,
                use_schema_constraints=True,
            )
        except Exception as e:
            logger.error(f"langextract extraction failed: {e}")
            return []

        # 6. Parse result into ExtractionRow objects
        rows = self._parse_result(
            result=result,
            event_configs=event_configs,
            source_type=source_type,
            source_id=source_id,
            source_url=source_url,
            source_subreddit=subreddit,
            engagement_score=engagement_score,
            engagement_comments=engagement_comments,
            source_created_at=source_created_at,
        )

        logger.info(
            f"Extracted {len(rows)} items from {source_type} {source_id} "
            f"({len([r for r in rows if r.extraction_class == 'market_signal'])} signals)"
        )

        return rows

    def _build_merged_prompt(self, event_configs: List[EventConfig]) -> str:
        """
        Merge base prompt + market context + event configs into one prompt.

        The base prompt is market-agnostic (extraction taxonomy + quality guidelines).
        Market-specific rules (OUT market semantics, ticker linking requirement) are
        injected here from MARKET_CONTEXT_TEMPLATE alongside the active market list.
        """
        prompt = BASE_EXTRACTION_PROMPT

        # Inject market-specific rules (only when markets exist)
        market_lines = []
        for ec in event_configs:
            for m in ec.markets:
                ticker = m.get("ticker", "")
                yes_sub = m.get("yes_sub_title", "")
                mtype = m.get("type", "")
                market_lines.append(
                    f"  {ticker} | {yes_sub} | event: {ec.event_ticker} | type: {mtype}"
                )

        if market_lines:
            prompt += MARKET_CONTEXT_TEMPLATE
            prompt += "\nACTIVE KALSHI MARKETS:\n"
            prompt += "\n".join(market_lines)

        # Add event-specific instructions
        for ec in event_configs:
            if ec.prompt_description:
                prompt += f"\n\n--- {ec.event_ticker}: {ec.event_title} ---\n"
                prompt += ec.prompt_description

                if ec.extraction_classes:
                    prompt += "\nEvent-specific extraction classes:\n"
                    for cls in ec.extraction_classes:
                        prompt += f"  - {cls.get('class_name', '')}: {cls.get('description', '')}\n"
                        attrs = cls.get("attributes", {})
                        if attrs:
                            for attr_name, attr_desc in attrs.items():
                                prompt += f"      {attr_name}: {attr_desc}\n"

        return prompt

    def _collect_examples(
        self, event_configs: List[EventConfig]
    ) -> List[Dict[str, Any]]:
        """
        Collect global + event-specific examples.
        Cap total to 5 to keep prompt manageable.
        """
        # Global examples (2 max)
        examples = self._example_manager.get_examples(max_examples=2)

        # Event-specific examples (1 per event, max 3 events)
        for ec in event_configs[:3]:
            event_ex = self._example_manager.get_event_examples(
                ec.examples, max_examples=1, event_ticker=ec.event_ticker
            )
            examples.extend(event_ex)

        return examples[:5]

    def _parse_result(
        self,
        result,
        event_configs: List[EventConfig],
        source_type: str,
        source_id: str,
        source_url: Optional[str],
        source_subreddit: Optional[str],
        engagement_score: int,
        engagement_comments: int,
        source_created_at: Optional[str],
    ) -> List[ExtractionRow]:
        """Parse langextract result into ExtractionRow objects."""
        rows = []

        # Build ticker→event lookup AND valid ticker set (API-sourced)
        ticker_to_events: Dict[str, List[str]] = {}
        valid_tickers: set = set()
        for ec in event_configs:
            for m in ec.markets:
                ticker = m.get("ticker", "")
                if ticker:
                    valid_tickers.add(ticker)
                    if ticker not in ticker_to_events:
                        ticker_to_events[ticker] = []
                    ticker_to_events[ticker].append(ec.event_ticker)

        dropped_count = 0

        try:
            # langextract returns ExtractionResult with .extractions attribute
            # Each extraction has extraction_class, extraction_text, attributes
            extractions = []
            if hasattr(result, "extractions"):
                extractions = result.extractions
            elif hasattr(result, "documents"):
                # Multi-document mode
                for doc in result.documents:
                    if hasattr(doc, "extractions"):
                        extractions.extend(doc.extractions)
            elif isinstance(result, list):
                extractions = result

            for ext in extractions:
                ext_class = getattr(ext, "extraction_class", "")
                ext_text = getattr(ext, "extraction_text", "")
                attrs = getattr(ext, "attributes", {})

                if isinstance(attrs, str):
                    import json
                    try:
                        attrs = json.loads(attrs)
                    except (json.JSONDecodeError, TypeError):
                        attrs = {}

                # Determine market_tickers and event_tickers
                market_tickers = []
                event_tickers = []

                if ext_class == "market_signal":
                    ticker = attrs.get("market_ticker", "")
                    if ticker and ticker in valid_tickers:
                        market_tickers = [ticker]
                        event_tickers = ticker_to_events.get(ticker, [])
                    else:
                        # Invalid or missing ticker — drop extraction entirely
                        if ticker:
                            dropped_count += 1
                        continue  # Skip — no fuzzy matching, no guessing

                rows.append(ExtractionRow(
                    source_type=source_type,
                    source_id=source_id,
                    source_url=source_url,
                    source_subreddit=source_subreddit,
                    extraction_class=ext_class,
                    extraction_text=ext_text,
                    attributes=attrs if isinstance(attrs, dict) else {},
                    market_tickers=market_tickers,
                    event_tickers=event_tickers,
                    engagement_score=engagement_score,
                    engagement_comments=engagement_comments,
                    source_created_at=source_created_at,
                ))

        except Exception as e:
            logger.error(f"Failed to parse langextract result: {e}")

        if dropped_count > 0:
            logger.warning(
                f"Dropped {dropped_count} market_signal extractions with invalid tickers "
                f"(not in {len(valid_tickers)} API-sourced tickers)"
            )

        return rows
