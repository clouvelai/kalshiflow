"""
Relation Extractor - LLM-based extraction of relations between named entities.

Uses GPT-4o-mini to identify directional relationships between entities
discovered during entity extraction. Domain-specific labels are tuned for
political/event prediction markets.

Relation Labels:
- SUPPORTS: Subject endorses, defends, or promotes object
- OPPOSES: Subject criticizes, blocks, or works against object
- CAUSES: Subject's action/event leads to object's outcome
- AFFECTED_BY: Subject is impacted by object's actions
- MEMBER_OF: Subject belongs to or is part of object

Cost: ~$0.0005/call (GPT-4o-mini, ~300 output tokens)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from openai import AsyncOpenAI

from ..schemas.kb_schemas import EntityRelation, RelationLabel

logger = logging.getLogger("kalshiflow_rl.traderv3.nlp.relation_extractor")

# Valid relation labels
VALID_LABELS = {label.value for label in RelationLabel}

# Prompt template for relation extraction
RELATION_EXTRACTION_PROMPT = """Given this text and the entities found in it, identify relations between entity pairs.

Text: "{text}"

Entities found:
{entity_list}

For each pair of entities that have a clear relationship in the text, output a JSON object.
Use ONLY these relation labels:
- SUPPORTS: Subject endorses, defends, or promotes object (e.g., "Trump backs Bondi")
- OPPOSES: Subject criticizes, blocks, or works against object (e.g., "Democrats oppose shutdown")
- CAUSES: Subject's action/event leads to object's outcome (e.g., "ICE raid triggers protests")
- AFFECTED_BY: Subject is impacted by object's actions (e.g., "Markets affected by Fed decision")
- MEMBER_OF: Subject belongs to or is part of object (e.g., "Bondi, Trump's AG pick")

Rules:
- Only extract relations clearly stated or strongly implied in the text
- Each relation is directional: subject → relation → object
- Skip pairs with no clear relationship
- Maximum 5 relations per text

Return a JSON array. Each element:
{{"subject": "entity name", "relation": "LABEL", "object": "entity name", "confidence": 0.5-1.0}}

If no relations found, return: []
"""


def _normalize_entity_id(name: str) -> str:
    """Convert entity name to a normalized ID."""
    return name.lower().strip().replace(" ", "_").replace("'", "").replace('"', "")


class RelationExtractor:
    """Async LLM-based relation extractor.

    Identifies directional relationships between named entities using
    GPT-4o-mini with domain-specific prompt and labels.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        timeout: float = 15.0,
        max_relations: int = 5,
    ):
        self._model = model
        self._timeout = timeout
        self._max_relations = max_relations
        self._client: Optional[AsyncOpenAI] = None
        self._total_calls = 0
        self._total_relations = 0

    def _get_client(self) -> AsyncOpenAI:
        """Get or create async OpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    async def extract_relations(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        source_post_id: str = "",
    ) -> List[EntityRelation]:
        """Extract relations between entities in the given text.

        Args:
            text: Source text containing the entities
            entities: List of entity dicts with at least "name" or "canonical_name"
            source_post_id: Reddit post ID for provenance

        Returns:
            List of EntityRelation objects
        """
        # Need at least 2 entities for a relation
        if len(entities) < 2:
            return []

        # Build entity list string
        entity_names = []
        for e in entities:
            name = ""
            if isinstance(e, dict):
                name = e.get("canonical_name", "") or e.get("name", "") or e.get("entity_id", "")
            elif isinstance(e, str):
                name = e
            if name:
                entity_names.append(name)

        if len(entity_names) < 2:
            return []

        # Deduplicate
        entity_names = list(dict.fromkeys(entity_names))
        entity_list_str = "\n".join(f"- {name}" for name in entity_names)

        prompt = RELATION_EXTRACTION_PROMPT.format(
            text=text[:1500],  # Cap text length for cost control
            entity_list=entity_list_str,
        )

        try:
            client = self._get_client()
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.1,
                ),
                timeout=self._timeout,
            )

            self._total_calls += 1
            raw = response.choices[0].message.content.strip()
            relations = self._parse_response(raw, source_post_id)

            self._total_relations += len(relations)
            if relations:
                logger.info(
                    f"[relation_extractor] Extracted {len(relations)} relations "
                    f"from {len(entity_names)} entities"
                )
            return relations

        except asyncio.TimeoutError:
            logger.warning("[relation_extractor] LLM call timed out")
            return []
        except Exception as e:
            logger.warning(f"[relation_extractor] LLM call failed: {e}")
            return []

    def _parse_response(
        self,
        raw: str,
        source_post_id: str = "",
    ) -> List[EntityRelation]:
        """Parse LLM JSON response into EntityRelation objects."""
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:])
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        # Handle trailing commas (common LLM output issue)
        text = text.replace(",]", "]").replace(",}", "}")

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.debug(f"[relation_extractor] Failed to parse JSON: {text[:200]}")
            return []

        if not isinstance(data, list):
            return []

        relations = []
        for item in data[:self._max_relations]:
            if not isinstance(item, dict):
                continue

            subject = item.get("subject", "").strip()
            relation = item.get("relation", "").strip().upper()
            obj = item.get("object", "").strip()
            confidence = item.get("confidence", 0.7)

            # Validate
            if not subject or not obj or not relation:
                continue
            if relation not in VALID_LABELS:
                continue
            if not isinstance(confidence, (int, float)):
                confidence = 0.7
            confidence = max(0.0, min(1.0, float(confidence)))

            relations.append(EntityRelation(
                subject_entity_id=_normalize_entity_id(subject),
                subject_name=subject,
                relation=relation,
                object_entity_id=_normalize_entity_id(obj),
                object_name=obj,
                confidence=confidence,
                source_post_id=source_post_id,
                context_snippet="",
                created_at=time.time(),
            ))

        return relations

    def get_stats(self) -> Dict[str, Any]:
        """Get extractor statistics."""
        return {
            "total_calls": self._total_calls,
            "total_relations": self._total_relations,
            "model": self._model,
        }
