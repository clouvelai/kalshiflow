"""
Truth Social distiller.

Takes transient raw post text and returns structured distilled signals.

Safety goals:
- Do not persist raw post content in memory longer than necessary.
- Return short, one-sentence claims, not full post bodies.

NOTE: Only works when LLM is enabled (requires OPENAI_API_KEY).
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from ..state.event_research_context import SemanticFrame
from .truth_social_signal_store import DistilledTruthSignal, TruthPostMeta

logger = logging.getLogger("kalshiflow_rl.traderv3.services.truth_social_distiller")


@dataclass(frozen=True)
class PostForDistillation:
    meta: TruthPostMeta
    content: str  # transient raw text, not stored


class _SignalOut(BaseModel):
    signal_id: str = Field(..., description="Unique id, preferably derived from post id + short hash")
    created_at: float
    author_handle: str
    is_verified: bool
    engagement_score: float
    claim: str = Field(..., description="One sentence. Must be <= 200 chars. No verbatim long quotes.")
    claim_type: str = Field(..., description="intent/announcement/denial/rumor/quote")
    entities: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning_short: str = Field(..., description="<= 180 chars")
    source_url: str


class _DistillationOut(BaseModel):
    signals: List[_SignalOut] = Field(default_factory=list)


def _distillation_enabled() -> bool:
    flag = (os.getenv("TRUTHSOCIAL_DISTILLATION_ENABLED", "auto") or "auto").strip().lower()
    if flag in ("0", "false", "no", "n", "off"):
        return False
    if flag == "auto":
        return bool(os.getenv("OPENAI_API_KEY"))
    return True


def _clean_text(s: str, max_len: int = 900) -> str:
    if not s:
        return ""
    # Strip any remaining HTML
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len]


class TruthSocialDistillerLLM:
    """
    LLM-only distiller for Truth Social posts.
    
    Requires OPENAI_API_KEY to be set. If LLM is not available,
    returns empty list (no signals produced).
    """

    def __init__(
        self,
        *,
        openai_model: Optional[str] = None,
        temperature: float = 0.2,
    ):
        self._enabled = _distillation_enabled()
        self._model = openai_model or os.getenv("TRUTHSOCIAL_DISTILLATION_MODEL", "gpt-4o-mini")
        self._temperature = float(temperature)

        self._llm = None
        if self._enabled and os.getenv("OPENAI_API_KEY"):
            try:
                from langchain_openai import ChatOpenAI

                self._llm = ChatOpenAI(
                    model=self._model,
                    temperature=self._temperature,
                    api_key=os.getenv("OPENAI_API_KEY"),
                )
                logger.info(f"TruthSocialDistillerLLM: LLM initialized (model={self._model})")
            except Exception as e:
                logger.warning(f"TruthSocialDistillerLLM: failed to init LLM: {e}")
                self._llm = None
        else:
            if not self._enabled:
                logger.info("TruthSocialDistillerLLM: Distillation disabled via TRUTHSOCIAL_DISTILLATION_ENABLED")
            elif not os.getenv("OPENAI_API_KEY"):
                logger.info("TruthSocialDistillerLLM: OPENAI_API_KEY not set, distillation disabled")

    def is_available(self) -> bool:
        """Check if LLM is available for distillation."""
        return self._llm is not None

    async def distill(
        self,
        *,
        posts: Sequence[PostForDistillation],
        semantic_frame: Optional[SemanticFrame] = None,
        max_signals_per_post: int = 1,
    ) -> List[DistilledTruthSignal]:
        """
        Distill posts into signals using LLM.
        
        Returns empty list if LLM is not available.
        """
        if not posts:
            return []
        
        if self._llm is None:
            logger.debug("TruthSocialDistillerLLM: LLM not available, skipping distillation")
            return []
        
        return await self._distill_llm(posts, semantic_frame=semantic_frame, max_signals_per_post=max_signals_per_post)

    async def _distill_llm(
        self,
        posts: Sequence[PostForDistillation],
        *,
        semantic_frame: Optional[SemanticFrame],
        max_signals_per_post: int,
    ) -> List[DistilledTruthSignal]:
        from langchain_core.prompts import ChatPromptTemplate

        # Trim payload size aggressively.
        compact_posts: List[Dict[str, Any]] = []
        for p in posts[:25]:
            compact_posts.append(
                {
                    "post_id": p.meta.post_id,
                    "created_at": p.meta.created_at,
                    "author_handle": p.meta.author_handle,
                    "is_verified": p.meta.is_verified,
                    "engagement_score": p.meta.engagement_score,
                    "source_url": p.meta.source_url,
                    "content": _clean_text(p.content, max_len=700),
                }
            )

        frame_hint = ""
        if semantic_frame:
            try:
                frame_hint = semantic_frame.to_prompt_string()[:1200]
            except Exception:
                frame_hint = ""

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You extract *distilled narrative signals* from Truth Social posts.\n"
                    "Rules:\n"
                    "- Return short, one-sentence claims (<=200 chars). Do NOT output full post text.\n"
                    "- claim_type must be one of: intent, announcement, denial, rumor, quote.\n"
                    "- confidence is 0..1 and should be conservative.\n"
                    "- entities should be short strings: people/orgs/places/tickers mentioned.\n"
                    "- output at most {max_signals_per_post} signals per post.\n"
                    "- signal_id should be stable and unique per signal (use post_id).\n",
                ),
                (
                    "human",
                    "SEMANTIC FRAME (optional, for relevance):\n{frame_hint}\n\n"
                    "POSTS:\n{posts_json}\n\n"
                    "Return ONLY the structured output.",
                ),
            ]
        )

        chain = prompt | self._llm.with_structured_output(_DistillationOut)
        result: _DistillationOut = await chain.ainvoke(
            {
                "frame_hint": frame_hint or "(none)",
                "posts_json": compact_posts,
                "max_signals_per_post": int(max_signals_per_post),
            }
        )

        signals: List[DistilledTruthSignal] = []
        for s in (result.signals or [])[: 25 * max(1, int(max_signals_per_post))]:
            claim = (s.claim or "").strip()
            if not claim:
                continue
            # Hard clamp for safety
            if len(claim) > 200:
                claim = claim[:199].rstrip() + "…"
            reasoning_short = (s.reasoning_short or "").strip()
            if len(reasoning_short) > 180:
                reasoning_short = reasoning_short[:179].rstrip() + "…"

            signals.append(
                DistilledTruthSignal(
                    signal_id=s.signal_id,
                    created_at=float(s.created_at),
                    author_handle=s.author_handle,
                    is_verified=bool(s.is_verified),
                    engagement_score=float(s.engagement_score or 0.0),
                    claim=claim,
                    claim_type=(s.claim_type or "quote").strip().lower(),
                    entities=list(s.entities or []),
                    linked_roles=None,
                    confidence=float(s.confidence),
                    reasoning_short=reasoning_short,
                    source_url=s.source_url,
                )
            )

        return signals


# Global singleton (so cache service doesn't re-init per refresh)
_global_distiller: Optional[TruthSocialDistillerLLM] = None


def get_truth_social_distiller() -> TruthSocialDistillerLLM:
    global _global_distiller
    if _global_distiller is None:
        _global_distiller = TruthSocialDistillerLLM()
        logger.info("Initialized global TruthSocialDistillerLLM")
    return _global_distiller

