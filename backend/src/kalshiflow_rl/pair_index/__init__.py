"""
Pair Index - Standalone, self-maintaining cross-venue market matching.

Deterministic-first matching pipeline:
  Tier 1: Text normalization + entity extraction + fuzzy matching (60-80%)
  Tier 2: LLM fallback for ambiguous cases (20-40%)

Usage:
    # CLI one-shot
    uv run python scripts/build_pair_index.py build

    # Library
    from kalshiflow_rl.pair_index import PairIndexBuilder
    builder = PairIndexBuilder(...)
    result = await builder.build()

    # Background refresh (inside trader)
    result = await builder.refresh()
"""

from .builder import PairIndexBuilder, BuildResult

__all__ = ["PairIndexBuilder", "BuildResult"]
