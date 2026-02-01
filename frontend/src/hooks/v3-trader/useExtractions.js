import { useMemo } from 'react';

/**
 * useExtractions - Event-centric derived views from extraction data.
 *
 * Takes raw extractions, marketSignals, and eventConfigs from useV3WebSocket
 * and produces organized indexes for the Agent page UI.
 *
 * @param {Object} params
 * @param {Array} params.extractions - All extraction objects from WebSocket
 * @param {Array} params.marketSignals - market_signal class extractions
 * @param {Array} params.eventConfigs - Active event configurations
 */
export const useExtractions = ({ extractions = [], marketSignals = [], eventConfigs = [] }) => {

  // Build lookup maps from event configs
  const { configByEvent, marketToEvent } = useMemo(() => {
    const cbe = {};
    const mte = {};
    for (const cfg of eventConfigs) {
      cbe[cfg.event_ticker] = cfg;
      for (const m of cfg.markets || []) {
        const ticker = typeof m === 'string' ? m : m.ticker;
        if (ticker) {
          mte[ticker] = cfg.event_ticker;
        }
      }
    }
    return { configByEvent: cbe, marketToEvent: mte };
  }, [eventConfigs]);

  // Group extractions by extraction_class
  const byClass = useMemo(() => {
    const result = {};
    for (const ext of extractions) {
      const cls = ext.extraction_class || 'unknown';
      if (!result[cls]) result[cls] = [];
      result[cls].push(ext);
    }
    return result;
  }, [extractions]);

  // Group extractions by source_id (post)
  const bySource = useMemo(() => {
    const result = {};
    for (const ext of extractions) {
      const sid = ext.source_id;
      if (!sid) continue;
      if (!result[sid]) result[sid] = [];
      result[sid].push(ext);
    }
    return result;
  }, [extractions]);

  // Group extractions by market_ticker (flattened from market_tickers array)
  const byMarket = useMemo(() => {
    const result = {};
    for (const ext of extractions) {
      for (const ticker of ext.market_tickers || []) {
        if (!result[ticker]) result[ticker] = [];
        result[ticker].push(ext);
      }
    }
    return result;
  }, [extractions]);

  // Group extractions by event_ticker (from event_tickers or via marketToEvent lookup)
  const byEvent = useMemo(() => {
    const result = {};
    for (const ext of extractions) {
      // Collect event tickers from the extraction itself
      const eventTickers = new Set(ext.event_tickers || []);
      // Also resolve via market_tickers -> marketToEvent
      for (const mt of ext.market_tickers || []) {
        const et = marketToEvent[mt];
        if (et) eventTickers.add(et);
      }
      for (const et of eventTickers) {
        if (!result[et]) result[et] = [];
        result[et].push(ext);
      }
    }
    return result;
  }, [extractions, marketToEvent]);

  // Aggregated signals per market (matches backend get_extraction_signals() shape)
  const aggregatedSignals = useMemo(() => {
    const result = {};

    for (const sig of marketSignals) {
      for (const ticker of sig.market_tickers || []) {
        if (!result[ticker]) {
          result[ticker] = {
            market_ticker: ticker,
            occurrence_count: 0,
            unique_sources: new Set(),
            directions: [],
            magnitudes: [],
            max_engagement: 0,
            recent_extractions: [],
          };
        }
        const agg = result[ticker];
        agg.occurrence_count += 1;
        if (sig.source_id) agg.unique_sources.add(sig.source_id);

        // Extract direction/magnitude from attributes
        const attrs = sig.attributes || {};
        if (attrs.direction) agg.directions.push(attrs.direction);
        if (attrs.magnitude != null) agg.magnitudes.push(Number(attrs.magnitude));
        if ((sig.engagement_score || 0) > agg.max_engagement) {
          agg.max_engagement = sig.engagement_score || 0;
        }
        // Keep last 3 recent extractions
        if (agg.recent_extractions.length < 3) {
          agg.recent_extractions.push(sig);
        }
      }
    }

    // Compute consensus fields
    for (const agg of Object.values(result)) {
      // Consensus direction
      const dirCounts = {};
      for (const d of agg.directions) {
        dirCounts[d] = (dirCounts[d] || 0) + 1;
      }
      const sortedDirs = Object.entries(dirCounts).sort((a, b) => b[1] - a[1]);
      agg.consensus = sortedDirs[0]?.[0] || null;
      agg.consensus_strength = agg.directions.length > 0
        ? Math.round((sortedDirs[0]?.[1] || 0) / agg.directions.length * 100)
        : 0;

      // Average magnitude
      agg.avg_magnitude = agg.magnitudes.length > 0
        ? Math.round(agg.magnitudes.reduce((s, v) => s + v, 0) / agg.magnitudes.length)
        : 0;

      // Convert Set to count
      agg.unique_sources_count = agg.unique_sources.size;
      delete agg.unique_sources;
      delete agg.directions;
      delete agg.magnitudes;
    }

    return result;
  }, [marketSignals]);

  // Summary stats
  const stats = useMemo(() => {
    const classes = {};
    for (const ext of extractions) {
      const cls = ext.extraction_class || 'unknown';
      classes[cls] = (classes[cls] || 0) + 1;
    }
    return {
      totalExtractions: extractions.length,
      marketSignalCount: classes['market_signal'] || 0,
      entityMentionCount: classes['entity_mention'] || 0,
      contextFactorCount: classes['context_factor'] || 0,
      uniqueMarkets: Object.keys(byMarket).length,
      uniqueEvents: Object.keys(byEvent).length,
    };
  }, [extractions, byMarket, byEvent]);

  return {
    byEvent,
    byMarket,
    byClass,
    bySource,
    aggregatedSignals,
    stats,
    configByEvent,
    marketToEvent,
  };
};

export default useExtractions;
