import { useMemo } from 'react';

/**
 * usePairIndex - Merges static pair index with live spread prices.
 *
 * The pair index (hierarchy, volume, metadata) changes rarely (~5 min).
 * Spread prices change constantly. This hook joins them by pair_id.
 *
 * @param {Object|null} pairIndex - Static index from pair_index_snapshot
 * @param {Map} spreads - Live spread data from spread_update messages
 * @returns {Array} Merged event groups with live prices
 */
export const usePairIndex = (pairIndex, spreads) => {
  return useMemo(() => {
    if (!pairIndex?.events) return [];

    return pairIndex.events.map(event => {
      const pairs = event.pairs.map(pair => {
        const liveSpread = spreads.get(pair.pair_id);
        return {
          ...pair,
          ...(liveSpread || {}),
        };
      });

      // Compute event-level aggregates from live data
      const livePairs = pairs.filter(p => p.spread_cents != null);
      const avgSpread = livePairs.length > 0
        ? Math.round(livePairs.reduce((sum, p) => sum + (p.spread_cents || 0), 0) / livePairs.length)
        : null;
      const bestSpread = livePairs.length > 0
        ? livePairs.reduce((best, p) => Math.abs(p.spread_cents || 0) > Math.abs(best) ? p.spread_cents : best, 0)
        : null;
      const tradeablePairCount = pairs.filter(p => p.tradeable === true).length;

      return {
        ...event,
        pairs,
        avg_spread: avgSpread,
        best_spread: bestSpread,
        live_pair_count: livePairs.length,
        tradeable_pair_count: tradeablePairCount,
      };
    });
  }, [pairIndex, spreads]);
};

export default usePairIndex;
