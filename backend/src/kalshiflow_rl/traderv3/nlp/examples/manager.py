"""
Example Manager for the KalshiExtractor.

Loads, scores, and selects the best examples for langextract calls.
Manages three sources:
1. Python-defined synthetic examples (one_shot_examples_real.py) — stable, market-agnostic
2. DB examples (extraction_examples table) — learning cycle, quality-scored
3. Event-specific examples (event_configs.examples) — per-event, from understand_event()
"""

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("kalshiflow_rl.traderv3.nlp.examples.manager")


class ExampleManager:
    """Manages langextract examples for the extraction pipeline."""

    def __init__(self, quality_threshold: float = 0.7):
        self._real_examples: List[Dict[str, Any]] = []
        self._claude_examples: List[Dict[str, Any]] = []
        self._db_examples: List[Dict[str, Any]] = []
        self._loaded = False
        self._db_loaded = False
        self._quality_threshold = quality_threshold
        self._supabase_client = None

    def _ensure_loaded(self) -> None:
        """Lazy-load Python-defined examples on first access."""
        if self._loaded:
            return
        self._loaded = True

        try:
            from .one_shot_examples_real import REAL_EXAMPLES
            self._real_examples = REAL_EXAMPLES
            logger.info(f"Loaded {len(self._real_examples)} real examples")
        except ImportError:
            logger.warning("No real examples found")

        try:
            from .one_shot_examples_claude import CLAUDE_EXAMPLES
            self._claude_examples = CLAUDE_EXAMPLES
            logger.info(f"Loaded {len(self._claude_examples)} claude examples")
        except ImportError:
            logger.debug("No claude examples found")

    def _get_supabase(self):
        """Get or create cached Supabase client."""
        if self._supabase_client is None:
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")
            if url and key:
                try:
                    from supabase import create_client
                    self._supabase_client = create_client(url, key)
                except Exception as e:
                    logger.warning(f"Could not create Supabase client: {e}")
        return self._supabase_client

    def _load_db_examples(self) -> None:
        """Load high-quality examples from the extraction_examples table."""
        if self._db_loaded:
            return
        self._db_loaded = True

        supabase = self._get_supabase()
        if not supabase:
            return

        try:
            result = supabase.table("extraction_examples") \
                .select("input_text, extractions, quality_score, event_ticker") \
                .gte("quality_score", self._quality_threshold) \
                .order("quality_score", desc=True) \
                .limit(10) \
                .execute()

            if result.data:
                for row in result.data:
                    self._db_examples.append({
                        "input": row["input_text"],
                        "output": row["extractions"],
                        "_quality_score": row.get("quality_score", 0.5),
                        "_event_ticker": row.get("event_ticker"),
                    })
                logger.info(
                    f"Loaded {len(self._db_examples)} DB examples "
                    f"(quality >= {self._quality_threshold})"
                )
        except Exception as e:
            logger.warning(f"Could not load DB examples: {e}")

    def get_examples(self, max_examples: int = 3) -> List[Dict[str, Any]]:
        """
        Get the best global examples for langextract.

        Priority order:
        1. DB examples with quality_score >= threshold (learned, validated)
        2. Python-defined real examples (synthetic, stable)
        3. Claude-generated examples

        Args:
            max_examples: Maximum number of examples to return

        Returns:
            List of example dicts with 'input' and 'output' keys
        """
        self._ensure_loaded()
        self._load_db_examples()

        # DB examples first (highest quality, learned from production),
        # then Python-defined, then claude-generated
        # Filter out event-specific DB examples (those serve event_configs)
        global_db = [
            ex for ex in self._db_examples
            if not ex.get("_event_ticker")
        ]
        combined = global_db + self._real_examples + self._claude_examples
        return combined[:max_examples]

    def get_event_examples(
        self,
        event_examples_json: List[Dict[str, Any]],
        max_examples: int = 1,
        event_ticker: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get event-specific examples from event_configs.examples JSON,
        supplemented by DB examples for the same event.

        Args:
            event_examples_json: Examples stored in event_configs table
            max_examples: Maximum number to return
            event_ticker: Optional event ticker to fetch DB examples for

        Returns:
            List of example dicts
        """
        examples = list(event_examples_json) if event_examples_json else []

        # Supplement with DB examples for this event
        if event_ticker and len(examples) < max_examples:
            self._load_db_examples()
            event_db = [
                ex for ex in self._db_examples
                if ex.get("_event_ticker") == event_ticker
            ]
            examples.extend(event_db)

        return examples[:max_examples]

    @property
    def total_count(self) -> int:
        """Total number of loaded examples."""
        self._ensure_loaded()
        self._load_db_examples()
        return len(self._real_examples) + len(self._claude_examples) + len(self._db_examples)
