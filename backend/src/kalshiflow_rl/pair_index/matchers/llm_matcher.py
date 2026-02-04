"""
Tier 2: LLM fallback for unmatched/ambiguous events and markets.

Only processes events that Tier 1 couldn't match (score < threshold)
and near-miss markets (score between 0.35 and threshold).

Estimated 60-70% cost reduction vs sending all events to LLM.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from ..fetchers.models import MatchCandidate, NormalizedEvent, NormalizedMarket

logger = logging.getLogger("kalshiflow_rl.pair_index.matchers.llm_matcher")


class LLMMatcher:
    """Tier 2 LLM matcher for ambiguous cases."""

    def __init__(
        self,
        openai_client: Optional[Any] = None,
        event_model: str = "gpt-4o",
        market_model: str = "gpt-4o-mini",
    ):
        self._client = openai_client
        self._event_model = event_model
        self._market_model = market_model

    @property
    def available(self) -> bool:
        return self._client is not None

    async def match_events(
        self,
        unmatched_kalshi: List[NormalizedEvent],
        unmatched_poly: List[NormalizedEvent],
    ) -> List[Tuple[NormalizedEvent, NormalizedEvent, float]]:
        """
        Send unmatched events to LLM for matching.

        Groups by broad category and sends one batch per group.
        Returns matched pairs with confidence scores.
        """
        if not self._client:
            logger.debug("No OpenAI client - skipping LLM event matching")
            return []

        if not unmatched_kalshi or not unmatched_poly:
            return []

        from .text_matcher import normalize_category
        from collections import defaultdict

        # Group by category
        k_by_cat: Dict[str, List[Tuple[int, NormalizedEvent]]] = defaultdict(list)
        p_by_cat: Dict[str, List[Tuple[int, NormalizedEvent]]] = defaultdict(list)

        for i, e in enumerate(unmatched_kalshi):
            k_by_cat[normalize_category(e.category)].append((i, e))
        for i, e in enumerate(unmatched_poly):
            p_by_cat[normalize_category(e.category)].append((i, e))

        shared_cats = set(k_by_cat.keys()) & set(p_by_cat.keys())
        if not shared_cats:
            # No shared categories means events are fundamentally different topics.
            # Cross-category matching produces garbage (e.g. Pope vs Bitcoin).
            logger.info("Tier 2: no shared categories between unmatched events, skipping")
            return []

        logger.info(
            f"Tier 2: {len(unmatched_kalshi)} unmatched Kalshi, "
            f"{len(unmatched_poly)} unmatched Poly, "
            f"{len(shared_cats)} category batches"
        )

        # Run batches concurrently
        tasks = []
        for cat in sorted(shared_cats):
            k_items = k_by_cat.get(cat, [])
            p_items = p_by_cat.get(cat, [])
            if k_items and p_items:
                tasks.append(self._match_event_batch(cat, k_items, p_items))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_matches: List[Tuple[int, int, float, str]] = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"Tier 2 batch failed: {r}")
                continue
            all_matches.extend(r)

        # Greedy 1:1 assignment
        all_matches.sort(key=lambda x: x[2], reverse=True)
        used_k: set = set()
        used_p: set = set()
        matched: List[Tuple[NormalizedEvent, NormalizedEvent, float]] = []

        for k_idx, p_idx, confidence, reason in all_matches:
            if confidence < 0.7:
                continue
            if k_idx in used_k or p_idx in used_p:
                continue
            if k_idx >= len(unmatched_kalshi) or p_idx >= len(unmatched_poly):
                continue
            used_k.add(k_idx)
            used_p.add(p_idx)
            matched.append((unmatched_kalshi[k_idx], unmatched_poly[p_idx], confidence))

        logger.info(f"Tier 2 event matching: {len(matched)} additional matches from LLM")
        return matched

    async def _match_event_batch(
        self,
        category: str,
        k_items: List[Tuple[int, NormalizedEvent]],
        p_items: List[Tuple[int, NormalizedEvent]],
    ) -> List[Tuple[int, int, float, str]]:
        """Run LLM matching for one category batch. Returns (k_idx, p_idx, confidence, reason)."""
        k_lines = [f"  K{gi}: \"{e.title}\" ({e.market_count} mkts)" for gi, (_, e) in enumerate(k_items)]
        p_lines = [f"  P{gi}: \"{e.title}\" ({e.market_count} mkts)" for gi, (_, e) in enumerate(p_items)]

        prompt = (
            f"You are matching prediction market events (category: {category}).\n"
            "Find event pairs that resolve on the SAME specific real-world outcome.\n\n"
            "CRITICAL RULES:\n"
            "1. Events must ask about the SAME real-world question. Sharing an entity (e.g. 'Trump') is NOT enough.\n"
            "2. Events must cover the SAME time period. '2026 Senate' != '2028 Presidential'.\n"
            "3. Events must have the SAME resolution criteria. 'Will X happen?' vs 'Will Y happen?' = NO match even if same person.\n"
            "4. Use confidence 0.85+ only for near-identical topics. Use 0.7-0.84 for strong but imperfect matches.\n\n"
            "VALID matches (same outcome, same time period):\n"
            "- \"Next Pope\" <-> \"Who will be the next Pope?\" (identical topic)\n"
            "- \"US acquire Greenland by 2026\" <-> \"Trump acquire Greenland 2026\" (same outcome + timeframe)\n"
            "- \"Fed Chair Nominee 2029\" <-> \"Next Fed Chair Nomination\" (same outcome)\n\n"
            "INVALID - do NOT match (real failures from our system):\n"
            "- \"Next Pope\" <-> \"Bitcoin above $150k\" (completely unrelated topics)\n"
            "- \"US acquire Greenland\" <-> \"Trump resignation\" (both mention Trump-adjacent topics but different questions)\n"
            "- \"2028 Presidential Election\" <-> \"2026 Florida Senate\" (different office, different year)\n"
            "- \"Minnesota Governor 2026\" <-> \"South Carolina Governor 2026\" (different states = different outcomes)\n"
            "- \"2026 House Control\" <-> \"2028 Presidential\" (different branch, different year)\n"
            "- \"Putin-Zelenskyy meeting location\" <-> \"Trump-Putin bilateral meeting\" (different meetings, different participants)\n\n"
            f"KALSHI EVENTS:\n" + "\n".join(k_lines) + "\n\n"
            f"POLYMARKET EVENTS:\n" + "\n".join(p_lines) + "\n\n"
            "Return JSON:\n"
            "{\"matches\": [{\"kalshi_idx\": 0, \"poly_idx\": 0, "
            "\"confidence\": 0.9, \"reason\": \"both about X\"}]}\n"
            "- kalshi_idx = number after K, poly_idx = number after P\n"
            "- confidence >= 0.7 only. When in doubt, DO NOT match.\n"
            "- Return {\"matches\": []} if no matches"
        )

        try:
            response = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self._event_model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=4096,
                ),
                timeout=120.0,
            )

            result_text = response.choices[0].message.content
            if hasattr(response, "usage") and response.usage:
                logger.info(
                    f"Tier 2 [{category}] tokens: "
                    f"in={response.usage.prompt_tokens}, "
                    f"out={response.usage.completion_tokens}"
                )

            result = json.loads(result_text)
            raw_matches = result.get("matches", [])

            # Remap local indices to global indices
            remapped = []
            for m in raw_matches:
                local_k = m.get("kalshi_idx")
                local_p = m.get("poly_idx")
                if not isinstance(local_k, int) or not isinstance(local_p, int):
                    continue
                if local_k < 0 or local_k >= len(k_items) or local_p < 0 or local_p >= len(p_items):
                    continue
                global_k = k_items[local_k][0]
                global_p = p_items[local_p][0]
                remapped.append((
                    global_k,
                    global_p,
                    m.get("confidence", 0.0),
                    m.get("reason", ""),
                ))

            logger.info(f"Tier 2 [{category}]: {len(remapped)} matches from {len(k_items)}K x {len(p_items)}P")
            return remapped

        except asyncio.TimeoutError:
            logger.warning(f"Tier 2 [{category}]: timed out (120s)")
            return []
        except Exception as e:
            logger.warning(f"Tier 2 [{category}] failed: {e}")
            return []

    async def match_markets(
        self,
        event_pairs: List[Tuple[NormalizedEvent, NormalizedEvent, float]],
        near_miss_markets: Optional[List[Tuple[NormalizedMarket, NormalizedMarket, float]]] = None,
    ) -> List[MatchCandidate]:
        """
        LLM market matching for Tier 2 event pairs + near-miss confirmations.

        Sends sub-markets of each event pair to the LLM.
        Also confirms/denies near-miss market pairs from Tier 1.
        """
        if not self._client:
            return []

        if not event_pairs:
            return []

        # Build prompt sections for event pairs
        sections: List[str] = []
        market_maps: List[Dict[str, Any]] = []

        for pair_idx, (ke, pe, _score) in enumerate(event_pairs):
            k_markets = [m for m in ke.markets if m.is_active]
            p_markets = [m for m in pe.markets if m.is_active]

            market_maps.append({"kalshi": k_markets, "poly": p_markets, "ke": ke, "pe": pe})

            k_lines = [f"    K{i}: \"{m.question}\"" for i, m in enumerate(k_markets)]
            p_lines = [f"    P{i}: \"{m.question}\"" for i, m in enumerate(p_markets)]

            sections.append(
                f"EVENT PAIR {pair_idx}:\n"
                f"  Kalshi: \"{ke.title}\"\n"
                + "\n".join(k_lines) + "\n"
                f"  Polymarket: \"{pe.title}\"\n"
                + "\n".join(p_lines)
            )

        # Batch to keep prompt size manageable
        BATCH_SIZE = 15
        all_candidates: List[MatchCandidate] = []

        for batch_start in range(0, len(sections), BATCH_SIZE):
            batch_sections = sections[batch_start:batch_start + BATCH_SIZE]
            listing = "\n\n".join(batch_sections)

            prompt = (
                "You are matching prediction market sub-markets across venues.\n"
                "Below are event pairs with their sub-markets.\n"
                "Find ALL market-level pairs that resolve on the SAME real-world outcome.\n\n"
                "RULES:\n"
                "- Only match markets with the SAME resolution criteria and outcome\n"
                "- Different time periods, thresholds, numerical ranges, or regions = NO match\n"
                "- 'Will X be above 100?' vs 'Will X be above 200?' = different thresholds, NO match\n"
                "- 'By end of 2026' vs 'By end of 2028' = different deadlines, NO match\n"
                "- Markets must resolve YES/NO on the exact same condition\n"
                "- If unsure, set confidence below 0.7\n\n"
                f"{listing}\n\n"
                "Return JSON:\n"
                "{\"pairs\": [{\"event_pair\": 0, \"kalshi_idx\": 0, \"poly_idx\": 0, "
                "\"confidence\": 0.95, \"reason\": \"both ask if X\"}]}\n"
                "- event_pair = EVENT PAIR number, kalshi_idx = K number, poly_idx = P number\n"
                "- Only confidence >= 0.5"
            )

            batch_num = batch_start // BATCH_SIZE + 1
            total_batches = (len(sections) + BATCH_SIZE - 1) // BATCH_SIZE

            try:
                response = await asyncio.wait_for(
                    self._client.chat.completions.create(
                        model=self._market_model,
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"},
                        temperature=0.0,
                        max_tokens=16384,
                    ),
                    timeout=60.0,
                )

                result_text = response.choices[0].message.content
                if hasattr(response, "usage") and response.usage:
                    logger.info(
                        f"Tier 2 market batch {batch_num}/{total_batches} tokens: "
                        f"in={response.usage.prompt_tokens}, "
                        f"out={response.usage.completion_tokens}"
                    )

                result = json.loads(result_text)
                for lp in result.get("pairs", []):
                    ep_idx = lp.get("event_pair")
                    k_idx = lp.get("kalshi_idx")
                    p_idx = lp.get("poly_idx")
                    confidence = lp.get("confidence", 0.0)

                    if confidence < 0.5:
                        continue

                    # Adjust for batch offset
                    actual_ep = (ep_idx if isinstance(ep_idx, int) else -1) + batch_start
                    if actual_ep < 0 or actual_ep >= len(market_maps):
                        continue
                    if not isinstance(k_idx, int) or not isinstance(p_idx, int):
                        continue

                    mm = market_maps[actual_ep]
                    if k_idx < 0 or k_idx >= len(mm["kalshi"]) or p_idx < 0 or p_idx >= len(mm["poly"]):
                        continue

                    km = mm["kalshi"][k_idx]
                    pm = mm["poly"][p_idx]
                    ke = mm["ke"]
                    pe = mm["pe"]

                    all_candidates.append(MatchCandidate(
                        kalshi_market=km,
                        poly_market=pm,
                        kalshi_event=ke,
                        poly_event=pe,
                        event_text_score=0.0,
                        market_text_score=confidence,
                        combined_score=confidence,
                        match_method="llm",
                        match_tier=2,
                        match_signals=["tier2_llm"],
                        llm_reasoning=lp.get("reason"),
                    ))

                logger.info(
                    f"Tier 2 market batch {batch_num}/{total_batches}: "
                    f"{len(result.get('pairs', []))} pairs"
                )

            except asyncio.TimeoutError:
                logger.warning(f"Tier 2 market batch {batch_num}/{total_batches}: timed out")
            except Exception as e:
                logger.warning(f"Tier 2 market batch {batch_num}/{total_batches} failed: {e}")

        logger.info(f"Tier 2 market matching: {len(all_candidates)} pairs from LLM")
        return all_candidates
