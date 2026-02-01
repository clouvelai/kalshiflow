import React, { useState, useCallback, useMemo, memo } from 'react';
import useRelativeTime from '../../../hooks/v3-trader/useRelativeTime';
import {
  Brain,
  Sparkles,
  ArrowRight,
  BarChart3,
  Activity,
  TrendingUp,
  TrendingDown,
  Zap,
  RefreshCw,
  MessageSquare,
  GitBranch,
  ChevronDown,
  ChevronRight,
  Wrench,
  BookOpen,
  AlertCircle,
  CheckCircle,
  XCircle,
  Clock,
  DollarSign,
  Filter,
  Layers,
  Newspaper,
  Globe,
  ExternalLink,
  ThumbsUp,
  ThumbsDown,
  Minus,
} from 'lucide-react';
import V3Header from '../layout/V3Header';
import renderThinkingMarkdown from '../../../utils/renderThinkingMarkdown';
import { useV3WebSocket } from '../../../hooks/v3-trader/useV3WebSocket';
import { useDeepAgent } from '../../../hooks/v3-trader/useDeepAgent';
import { useExtractions } from '../../../hooks/v3-trader/useExtractions';
import { CostPanel } from '../panels/DeepAgentPanel';

const EMPTY_POSITIONS = [];

/**
 * Pipeline Stage Card - Visualizes a stage in the data pipeline
 */
const PipelineStage = memo(({
  icon: Icon,
  title,
  count,
  color = 'gray',
  isActive = false,
  description
}) => {
  const colorClasses = {
    orange: 'from-orange-900/30 to-orange-950/20 border-orange-700/30 text-orange-400',
    violet: 'from-violet-900/30 to-violet-950/20 border-violet-700/30 text-violet-400',
    cyan: 'from-cyan-900/30 to-cyan-950/20 border-cyan-700/30 text-cyan-400',
    emerald: 'from-emerald-900/30 to-emerald-950/20 border-emerald-700/30 text-emerald-400',
    gray: 'from-gray-800/30 to-gray-900/20 border-gray-700/30 text-gray-400',
  };

  return (
    <div className={`
      flex-1 min-w-[140px] p-4 rounded-xl border
      bg-gradient-to-br ${colorClasses[color]}
      ${isActive ? 'ring-1 ring-offset-1 ring-offset-gray-950' : ''}
      transition-all duration-300
    `}>
      <div className="flex items-center gap-2 mb-2">
        <Icon className={`w-4 h-4 ${isActive ? 'animate-pulse' : ''}`} />
        <span className="text-xs font-semibold uppercase tracking-wider">{title}</span>
      </div>
      <div className="text-2xl font-mono font-bold text-white mb-1">
        {count}
      </div>
      <div className="text-[10px] text-gray-500">
        {description}
      </div>
    </div>
  );
});

PipelineStage.displayName = 'PipelineStage';

// === Extraction class colors ===
const CLASS_COLORS = {
  market_signal: { border: 'border-l-cyan-400', bg: 'bg-cyan-900/20', text: 'text-cyan-400', badge: 'bg-cyan-900/30 text-cyan-400 border-cyan-600/30', label: 'SIGNAL' },
  entity_mention: { border: 'border-l-violet-400', bg: 'bg-violet-900/20', text: 'text-violet-400', badge: 'bg-violet-900/30 text-violet-400 border-violet-600/30', label: 'ENTITY' },
  context_factor: { border: 'border-l-amber-400', bg: 'bg-amber-900/20', text: 'text-amber-400', badge: 'bg-amber-900/30 text-amber-400 border-amber-600/30', label: 'CONTEXT' },
  custom: { border: 'border-l-pink-400', bg: 'bg-pink-900/20', text: 'text-pink-400', badge: 'bg-pink-900/30 text-pink-400 border-pink-600/30', label: 'CUSTOM' },
};

const getClassStyle = (cls) => CLASS_COLORS[cls] || CLASS_COLORS.custom;

/**
 * ExtractionCard - Single extraction with class-specific styling
 */
const ExtractionCard = memo(({ extraction }) => {
  const cls = extraction.extraction_class || 'custom';
  const style = getClassStyle(cls);
  const attrs = extraction.attributes || {};

  return (
    <div className={`p-3 rounded-lg border border-gray-700/30 border-l-2 ${style.border} bg-gray-800/30 hover:border-gray-600/40 transition-colors`}>
      <div className="flex items-center justify-between mb-1.5">
        <span className={`px-1.5 py-0.5 text-[9px] font-bold rounded border ${style.badge}`}>
          {style.label}
        </span>
        <div className="flex items-center gap-2">
          {extraction.source_subreddit && (
            <span className="text-[9px] text-orange-400">r/{extraction.source_subreddit}</span>
          )}
          {extraction.engagement_score > 0 && (
            <span className="text-[9px] text-gray-500">↑{extraction.engagement_score}</span>
          )}
        </div>
      </div>
      <div className="text-xs text-gray-300 line-clamp-2 mb-1.5" title={extraction.extraction_text}>
        {extraction.extraction_text}
      </div>
      {/* Class-specific attributes */}
      {cls === 'market_signal' && attrs.direction && (
        <div className="flex items-center gap-2 mb-1">
          <span className={`text-[10px] font-bold ${attrs.direction === 'BULLISH' || attrs.direction === 'bullish' ? 'text-emerald-400' : attrs.direction === 'BEARISH' || attrs.direction === 'bearish' ? 'text-rose-400' : 'text-gray-400'}`}>
            {attrs.direction?.toUpperCase()}
          </span>
          {attrs.magnitude != null && (
            <span className="text-[10px] text-gray-500">mag: {attrs.magnitude}</span>
          )}
        </div>
      )}
      {cls === 'entity_mention' && attrs.entity_name && (
        <div className="text-[10px] text-violet-400 mb-1">{attrs.entity_name} {attrs.sentiment ? `(${attrs.sentiment})` : ''}</div>
      )}
      {/* Market ticker badges */}
      {extraction.market_tickers?.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-1.5">
          {extraction.market_tickers.slice(0, 3).map(ticker => (
            <span key={ticker} className="px-1.5 py-0.5 text-[8px] font-mono bg-gray-700/50 text-gray-300 rounded">
              {ticker}
            </span>
          ))}
          {extraction.market_tickers.length > 3 && (
            <span className="text-[8px] text-gray-500">+{extraction.market_tickers.length - 3}</span>
          )}
        </div>
      )}
    </div>
  );
});

ExtractionCard.displayName = 'ExtractionCard';

/**
 * ExtractionFeedPanel - Scrollable feed with filter tabs
 */
const ExtractionFeedPanel = memo(({ extractions }) => {
  const [filter, setFilter] = useState('all');
  const [expanded, setExpanded] = useState(true);

  const filtered = useMemo(() => {
    if (filter === 'all') return extractions;
    if (filter === 'signals') return extractions.filter(e => e.extraction_class === 'market_signal');
    if (filter === 'entities') return extractions.filter(e => e.extraction_class === 'entity_mention');
    if (filter === 'context') return extractions.filter(e => e.extraction_class === 'context_factor');
    if (filter === 'custom') return extractions.filter(e => !['market_signal', 'entity_mention', 'context_factor'].includes(e.extraction_class));
    return extractions;
  }, [extractions, filter]);

  const filterTabs = [
    { key: 'all', label: 'All', count: extractions.length },
    { key: 'signals', label: 'Signals', count: extractions.filter(e => e.extraction_class === 'market_signal').length },
    { key: 'entities', label: 'Entities', count: extractions.filter(e => e.extraction_class === 'entity_mention').length },
    { key: 'context', label: 'Context', count: extractions.filter(e => e.extraction_class === 'context_factor').length },
    { key: 'custom', label: 'Custom', count: extractions.filter(e => !['market_signal', 'entity_mention', 'context_factor'].includes(e.extraction_class)).length },
  ];

  return (
    <div className="bg-gray-900/50 rounded-2xl border border-gray-800/50 overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between p-4 hover:bg-gray-800/30 transition-colors"
      >
        <div className="flex items-center gap-2">
          <Zap className="w-4 h-4 text-cyan-400" />
          <span className="text-sm font-semibold text-gray-300">Extraction Stream</span>
          <span className="px-2 py-0.5 bg-cyan-900/30 text-cyan-400 text-[10px] font-bold rounded-full">
            {extractions.length}
          </span>
        </div>
        {expanded ? <ChevronDown className="w-4 h-4 text-gray-500" /> : <ChevronRight className="w-4 h-4 text-gray-500" />}
      </button>
      {expanded && (
        <div className="px-4 pb-4">
          {/* Filter tabs */}
          <div className="flex gap-1 mb-3 overflow-x-auto">
            {filterTabs.map(tab => (
              <button
                key={tab.key}
                onClick={() => setFilter(tab.key)}
                className={`px-2 py-1 text-[10px] font-medium rounded-md whitespace-nowrap transition-colors ${
                  filter === tab.key
                    ? 'bg-gray-700/60 text-white'
                    : 'text-gray-500 hover:text-gray-300 hover:bg-gray-800/50'
                }`}
              >
                {tab.label} {tab.count > 0 && <span className="font-mono ml-0.5">{tab.count}</span>}
              </button>
            ))}
          </div>
          {/* Feed */}
          <div className="space-y-2 max-h-[400px] overflow-y-auto">
            {filtered.length === 0 ? (
              <div className="text-center py-6 text-gray-500 text-sm">
                No extractions yet...
              </div>
            ) : (
              filtered.slice(0, 50).map((ext, idx) => (
                <ExtractionCard key={`${ext.source_id}_${ext.extraction_class}_${idx}`} extraction={ext} />
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
});

ExtractionFeedPanel.displayName = 'ExtractionFeedPanel';

/**
 * EventMarketRow - One market within an event showing live prices + signals
 */
const EventMarketRow = memo(({ market, marketPrices, signalAgg, position, isQuiet }) => {
  const ticker = typeof market === 'string' ? market : market.ticker;
  const displayTitle = typeof market === 'string' ? null : (market.yes_sub_title || market.title);
  const title = displayTitle;
  const prices = marketPrices?.[ticker];

  const yesBid = prices?.yes_bid;
  const yesAsk = prices?.yes_ask;
  const spread = prices?.spread;
  const volume = prices?.volume_24h || prices?.volume;
  const volume24h = typeof market === 'string' ? null : market.volume_24h;

  // Time-to-close formatting
  const timeToClose = useMemo(() => {
    const seconds = typeof market === 'string' ? null : market.time_to_close_seconds;
    if (seconds == null || seconds <= 0) return null;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h`;
    return `${Math.floor(seconds / 86400)}d`;
  }, [market]);

  // Consensus badge
  const consensus = signalAgg?.consensus;
  const consensusBg = consensus === 'BULLISH' || consensus === 'bullish'
    ? 'bg-emerald-900/30 text-emerald-400 border-emerald-600/30'
    : consensus === 'BEARISH' || consensus === 'bearish'
      ? 'bg-rose-900/30 text-rose-400 border-rose-600/30'
      : 'bg-gray-700/40 text-gray-400 border-gray-600/30';

  // Position info
  const posData = position;
  const hasPosData = posData && (posData.total_contracts || posData.market_exposure);

  if (isQuiet && !hasPosData) {
    return (
      <div className="flex items-center justify-between px-3 py-1.5 bg-gray-800/20 rounded-lg border border-gray-800/30 opacity-60">
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono text-gray-400">{ticker}</span>
          {title && <span className="text-[9px] text-gray-600 truncate max-w-[200px]">{title}</span>}
        </div>
        <div className="flex items-center gap-3 text-[10px] text-gray-600">
          {yesBid != null && <span>YES {yesBid}c/{yesAsk}c</span>}
          {spread != null && <span>Sp {spread}c</span>}
          {timeToClose && <span className="text-gray-500">{timeToClose}</span>}
          {volume24h != null && volume24h > 0 && (
            <span className="text-gray-600 font-mono">{volume24h > 1000 ? `${(volume24h / 1000).toFixed(1)}k` : volume24h}</span>
          )}
          <span className="text-gray-700 italic">Awaiting signals...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="px-3 py-2.5 bg-gray-800/30 rounded-lg border border-gray-700/30 hover:border-gray-600/40 transition-colors">
      {/* Market header row */}
      <div className="flex items-center justify-between mb-1.5">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-gray-200">{ticker}</span>
          {title && <span className="text-[9px] text-gray-500 truncate max-w-[200px]">{title}</span>}
        </div>
        <div className="flex items-center gap-3 text-[10px]">
          {yesBid != null && (
            <span className="text-gray-300">
              YES <span className="font-mono text-emerald-400">{yesBid}c</span>/<span className="font-mono">{yesAsk}c</span>
            </span>
          )}
          {yesBid != null && (
            <span className="text-gray-400">
              NO <span className="font-mono text-rose-400">{100 - (yesAsk || 0)}c</span>/<span className="font-mono">{100 - (yesBid || 0)}c</span>
            </span>
          )}
          {spread != null && <span className="text-gray-500">Spread {spread}c</span>}
          {volume != null && <span className="text-gray-600">{volume > 1000 ? `${(volume / 1000).toFixed(1)}k` : volume} vol</span>}
          {timeToClose && (
            <span className="text-gray-500 flex items-center gap-0.5">
              <Clock className="w-3 h-3" />{timeToClose}
            </span>
          )}
        </div>
      </div>

      {/* Signal summary */}
      {signalAgg && signalAgg.occurrence_count > 0 && (
        <div className="flex items-center gap-2 mb-1.5">
          <span className={`px-1.5 py-0.5 text-[9px] font-bold rounded border ${consensusBg}`}>
            {consensus?.toUpperCase() || 'MIXED'}
          </span>
          <span className="text-[10px] text-gray-400">
            {signalAgg.occurrence_count} signal{signalAgg.occurrence_count > 1 ? 's' : ''}, {signalAgg.unique_sources_count} source{signalAgg.unique_sources_count > 1 ? 's' : ''}, consensus {signalAgg.consensus_strength}%
          </span>
          {signalAgg.avg_magnitude > 0 && (
            <span className="text-[10px] text-gray-500">mag {signalAgg.avg_magnitude}</span>
          )}
        </div>
      )}

      {/* Recent extraction snippets */}
      {signalAgg?.recent_extractions?.length > 0 && (
        <div className="space-y-1 mt-1">
          {signalAgg.recent_extractions.slice(0, 2).map((ext, i) => (
            <div key={i} className="flex items-start gap-1.5">
              <span className="text-gray-700 text-[10px] mt-0.5">&gt;</span>
              <span className="text-[10px] text-gray-400 line-clamp-1">
                {ext.extraction_text?.slice(0, 80)}{ext.extraction_text?.length > 80 ? '...' : ''}
              </span>
              {ext.source_subreddit && (
                <span className="text-[9px] text-orange-400/60 whitespace-nowrap">r/{ext.source_subreddit}</span>
              )}
              {ext.engagement_score > 0 && (
                <span className="text-[9px] text-gray-600 whitespace-nowrap">↑{ext.engagement_score}</span>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Position row */}
      {hasPosData && (
        <div className="flex items-center gap-2 mt-1.5 pt-1.5 border-t border-gray-700/20 text-[10px]">
          <span className="text-gray-500">Position:</span>
          <span className={`font-bold ${posData.market_position > 0 ? 'text-emerald-400' : posData.market_position < 0 ? 'text-rose-400' : 'text-gray-400'}`}>
            {posData.total_contracts || Math.abs(posData.market_position || 0)} {posData.position_side || (posData.market_position > 0 ? 'YES' : 'NO')}
          </span>
          {posData.realized_pnl != null && (
            <span className={`font-mono ${posData.realized_pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
              {posData.realized_pnl >= 0 ? '+' : ''}${(posData.realized_pnl / 100).toFixed(2)}
            </span>
          )}
        </div>
      )}
    </div>
  );
});

EventMarketRow.displayName = 'EventMarketRow';

/**
 * EventCard - Event header + list of EventMarketRows
 */
const EventCard = memo(({ config, eventExtractions, aggregatedSignals, marketPrices, positionsDetails }) => {
  const [collapsed, setCollapsed] = useState(false);

  const markets = config.markets || [];
  const eventTicker = config.event_ticker;

  // Build position lookup from positionsDetails array
  const positionLookup = useMemo(() => {
    const lookup = {};
    for (const pos of positionsDetails || []) {
      if (pos.ticker || pos.market_ticker) {
        lookup[pos.ticker || pos.market_ticker] = pos;
      }
    }
    return lookup;
  }, [positionsDetails]);

  // Sort markets: those with signals first
  const sortedMarkets = useMemo(() => {
    return [...markets].sort((a, b) => {
      const aTicker = typeof a === 'string' ? a : a.ticker;
      const bTicker = typeof b === 'string' ? b : b.ticker;
      const aCount = aggregatedSignals[aTicker]?.occurrence_count || 0;
      const bCount = aggregatedSignals[bTicker]?.occurrence_count || 0;
      return bCount - aCount;
    });
  }, [markets, aggregatedSignals]);

  // Total signals for this event
  const totalSignals = useMemo(() => {
    return markets.reduce((sum, m) => {
      const ticker = typeof m === 'string' ? m : m.ticker;
      return sum + (aggregatedSignals[ticker]?.occurrence_count || 0);
    }, 0);
  }, [markets, aggregatedSignals]);

  // YES sum risk calculation
  const yesSum = useMemo(() => {
    return markets.reduce((sum, m) => {
      const ticker = typeof m === 'string' ? m : m.ticker;
      const price = marketPrices?.[ticker];
      return sum + (price?.yes_bid || price?.last_price || 0);
    }, 0);
  }, [markets, marketPrices]);

  const riskLevel = yesSum > 120 ? 'HIGH' : yesSum > 90 ? 'ELEVATED' : 'NORMAL';
  const riskColor = riskLevel === 'HIGH' ? 'text-rose-400' : riskLevel === 'ELEVATED' ? 'text-amber-400' : 'text-gray-500';

  // Category badge styling
  const categoryColors = {
    politics: 'bg-indigo-900/30 text-indigo-400 border-indigo-600/30',
    economics: 'bg-teal-900/30 text-teal-400 border-teal-600/30',
    finance: 'bg-emerald-900/30 text-emerald-400 border-emerald-600/30',
    science: 'bg-blue-900/30 text-blue-400 border-blue-600/30',
    sports: 'bg-orange-900/30 text-orange-400 border-orange-600/30',
    entertainment: 'bg-pink-900/30 text-pink-400 border-pink-600/30',
  };
  const category = config.category?.toLowerCase();
  const categoryClass = categoryColors[category] || 'bg-gray-700/30 text-gray-400 border-gray-600/30';

  // Total volume across markets
  const totalVolume = useMemo(() => {
    return markets.reduce((sum, m) => {
      const ticker = typeof m === 'string' ? m : m.ticker;
      const vol = m.volume_24h || marketPrices?.[ticker]?.volume_24h || marketPrices?.[ticker]?.volume || 0;
      return sum + vol;
    }, 0);
  }, [markets, marketPrices]);

  return (
    <div className="bg-gray-900/50 rounded-xl border border-gray-800/50 overflow-hidden">
      {/* Event header */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="w-full flex items-center justify-between p-4 hover:bg-gray-800/30 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-cyan-400" />
            <span className="text-sm font-semibold text-white">{eventTicker}</span>
          </div>
          {config.event_title && (
            <span className="text-xs text-gray-400 truncate max-w-[300px]">{config.event_title}</span>
          )}
          {config.category && (
            <span className={`px-1.5 py-0.5 text-[9px] font-bold uppercase rounded border ${categoryClass}`}>
              {config.category}
            </span>
          )}
          <span className={`px-1.5 py-0.5 text-[9px] rounded border ${
            config._source === 'researched'
              ? 'bg-cyan-900/20 text-cyan-400 border-cyan-600/30'
              : 'bg-gray-700/30 text-gray-500 border-gray-600/30'
          }`}>
            {config._source === 'researched' ? 'Researched' : 'Discovered'}
          </span>
          <span className="px-1.5 py-0.5 text-[9px] bg-gray-700/50 text-gray-400 rounded">
            {markets.length} market{markets.length !== 1 ? 's' : ''}
          </span>
          {totalVolume > 0 && (
            <span className="text-[9px] text-gray-500 font-mono">
              {totalVolume > 1000 ? `${(totalVolume / 1000).toFixed(1)}k` : totalVolume} vol
            </span>
          )}
          {totalSignals > 0 && (
            <span className="px-1.5 py-0.5 text-[9px] bg-cyan-900/30 text-cyan-400 rounded border border-cyan-600/30">
              {totalSignals} signal{totalSignals !== 1 ? 's' : ''}
            </span>
          )}
        </div>
        {collapsed ? <ChevronRight className="w-4 h-4 text-gray-500" /> : <ChevronDown className="w-4 h-4 text-gray-500" />}
      </button>

      {!collapsed && (
        <div className="px-4 pb-4 space-y-2">
          {sortedMarkets.map((market) => {
            const ticker = typeof market === 'string' ? market : market.ticker;
            const sig = aggregatedSignals[ticker];
            const isQuiet = !sig || sig.occurrence_count === 0;
            return (
              <EventMarketRow
                key={ticker}
                market={market}
                marketPrices={marketPrices}
                signalAgg={sig}
                position={positionLookup[ticker]}
                isQuiet={isQuiet}
              />
            );
          })}

          {/* Event-level risk footer */}
          {markets.length > 1 && yesSum > 0 && (
            <div className="flex items-center justify-between px-3 py-1.5 mt-1 border-t border-gray-800/30 text-[10px]">
              <span className="text-gray-600">
                Event Risk: YES_sum = <span className="font-mono text-gray-400">{yesSum}c</span>
              </span>
              <span className={`font-bold ${riskColor}`}>{riskLevel}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
});

EventCard.displayName = 'EventCard';

/**
 * EventIntelligenceBoard - Scrollable list of EventCards
 */
const EventIntelligenceBoard = memo(({ eventConfigs, byEvent, aggregatedSignals, marketPrices, positionsDetails }) => {
  // Sort events by total signal count (most active first)
  const sortedConfigs = useMemo(() => {
    return [...eventConfigs].sort((a, b) => {
      const aSignals = (a.markets || []).reduce((sum, m) => {
        const ticker = typeof m === 'string' ? m : m.ticker;
        return sum + (aggregatedSignals[ticker]?.occurrence_count || 0);
      }, 0);
      const bSignals = (b.markets || []).reduce((sum, m) => {
        const ticker = typeof m === 'string' ? m : m.ticker;
        return sum + (aggregatedSignals[ticker]?.occurrence_count || 0);
      }, 0);
      return bSignals - aSignals;
    });
  }, [eventConfigs, aggregatedSignals]);

  const totalSignals = Object.values(aggregatedSignals).reduce((s, a) => s + (a.occurrence_count || 0), 0);

  return (
    <div className="bg-gray-900/50 rounded-2xl border border-gray-800/50 p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-gradient-to-br from-cyan-900/40 to-blue-900/30 border border-cyan-700/30">
            <Layers className="w-5 h-5 text-cyan-400" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-white">Event Intelligence</h2>
            <p className="text-xs text-gray-500">Extraction signals + live market data per event</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-gray-500">{eventConfigs.length} events</span>
          {totalSignals > 0 && (
            <span className="px-2 py-0.5 bg-cyan-900/30 text-cyan-400 text-[10px] font-bold rounded border border-cyan-600/30">
              {totalSignals} signals
            </span>
          )}
        </div>
      </div>

      <div className="space-y-3 max-h-[600px] overflow-y-auto pr-1">
        {sortedConfigs.length === 0 ? (
          <div className="text-center py-12">
            <Layers className="w-10 h-10 text-gray-700 mx-auto mb-3" />
            <div className="text-gray-500 text-sm font-medium mb-1">Waiting for market discovery...</div>
            <div className="text-gray-600 text-xs">
              Events appear automatically once the trader discovers active markets
            </div>
          </div>
        ) : (
          sortedConfigs.map(config => (
            <EventCard
              key={config.event_ticker}
              config={config}
              eventExtractions={byEvent[config.event_ticker] || []}
              aggregatedSignals={aggregatedSignals}
              marketPrices={marketPrices}
              positionsDetails={positionsDetails}
            />
          ))
        )}
      </div>
    </div>
  );
});

EventIntelligenceBoard.displayName = 'EventIntelligenceBoard';

/**
 * Reddit Post Card - Shows a Reddit post with extraction class pills
 */
const RedditPostCard = memo(({ post, postExtractions = [] }) => {
  const formatTime = (utc) => {
    if (!utc) return '';
    const date = new Date(utc * 1000);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    if (diffMins < 1) return 'just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };

  return (
    <div className="p-3 bg-gray-800/30 rounded-lg border border-gray-700/30 hover:border-orange-700/30 transition-colors">
      <div className="flex items-center justify-between mb-1.5">
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-orange-400 font-medium">r/{post.subreddit}</span>
          {post.score > 0 && (
            <span className="text-[10px] text-gray-500">↑{post.score}</span>
          )}
        </div>
        {post.created_utc && (
          <span className="text-[9px] text-gray-600">{formatTime(post.created_utc)}</span>
        )}
      </div>
      <div className="text-xs text-gray-300 line-clamp-2 mb-1.5" title={post.title}>
        {post.title}
      </div>
      {/* Extraction class pills */}
      {postExtractions.length > 0 && (
        <div className="flex flex-wrap gap-1 mt-2 pt-2 border-t border-gray-700/30">
          {postExtractions.slice(0, 5).map((ext, idx) => {
            const style = getClassStyle(ext.extraction_class);
            return (
              <span
                key={idx}
                className={`px-1.5 py-0.5 text-[8px] font-bold rounded border ${style.badge}`}
                title={ext.extraction_text?.slice(0, 100)}
              >
                {style.label}
              </span>
            );
          })}
          {postExtractions.length > 5 && (
            <span className="text-[8px] text-gray-500">+{postExtractions.length - 5}</span>
          )}
        </div>
      )}
    </div>
  );
});

RedditPostCard.displayName = 'RedditPostCard';

/**
 * Agent Status Header - Shows running state, cycle count, P&L
 */
const AgentStatusHeader = memo(({ agentState, settlements, trades }) => {
  const isRunning = agentState.status === 'active' || agentState.status === 'started';

  const totalPnL = useMemo(() => {
    return settlements.reduce((sum, s) => sum + (s.pnlCents || 0), 0) / 100;
  }, [settlements]);

  const pnlColor = totalPnL >= 0 ? 'text-emerald-400' : 'text-rose-400';
  const pnlBg = totalPnL >= 0 ? 'bg-emerald-900/20' : 'bg-rose-900/20';

  return (
    <div className="flex items-center justify-between p-3 bg-gray-800/50 rounded-xl border border-gray-700/40">
      <div className="flex items-center gap-4">
        <div className={`
          flex items-center gap-2 px-3 py-1.5 rounded-lg border
          ${isRunning
            ? 'bg-emerald-900/30 border-emerald-600/40 text-emerald-400'
            : 'bg-gray-800/50 border-gray-700/40 text-gray-400'}
        `}>
          {isRunning ? (
            <>
              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
              <span className="text-xs font-bold">RUNNING</span>
            </>
          ) : (
            <>
              <Activity className="w-3.5 h-3.5" />
              <span className="text-xs font-bold">STOPPED</span>
            </>
          )}
        </div>

        <div className="flex items-center gap-2 text-xs text-gray-400">
          <Clock className="w-3.5 h-3.5" />
          <span>Cycle <span className="font-mono text-white">{agentState.cycleCount}</span></span>
        </div>

        <div className="flex items-center gap-2 text-xs text-gray-400">
          <BarChart3 className="w-3.5 h-3.5" />
          <span>Trades <span className="font-mono text-white">{trades.length}</span></span>
        </div>
      </div>

      <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg ${pnlBg}`}>
        <DollarSign className="w-3.5 h-3.5 text-gray-400" />
        <span className={`text-sm font-mono font-bold ${pnlColor}`}>
          {totalPnL >= 0 ? '+' : ''}{totalPnL.toFixed(2)}
        </span>
        <span className="text-[10px] text-gray-500">P&L</span>
      </div>
    </div>
  );
});

AgentStatusHeader.displayName = 'AgentStatusHeader';

/**
 * Thinking Stream - Real-time agent reasoning with pulse animation
 */
const ThinkingStream = memo(({ thinking, isRunning }) => {
  if (!thinking.text) {
    return (
      <div className="p-4 bg-gray-800/30 rounded-xl border border-gray-700/30">
        <div className="flex items-center gap-2 mb-2">
          <Brain className="w-4 h-4 text-violet-400" />
          <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Thinking</span>
        </div>
        <div className="text-sm text-gray-500 italic">
          Waiting for next cycle...
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 bg-gradient-to-br from-violet-900/20 to-violet-950/10 rounded-xl border border-violet-700/30">
      <div className="flex items-center gap-2 mb-2">
        <Brain className="w-4 h-4 text-violet-400 animate-pulse" />
        <span className="text-xs font-semibold text-violet-300 uppercase tracking-wider">Thinking</span>
        <span className="ml-auto text-[10px] text-gray-500 font-mono">Cycle {thinking.cycle}</span>
      </div>
      <div className="text-sm text-gray-200 leading-relaxed">
        {renderThinkingMarkdown(thinking.text)}
      </div>
      {isRunning && (
        <div className="mt-2 flex items-center gap-2">
          <div className="w-2 h-2 bg-violet-400 rounded-full animate-pulse" />
          <span className="text-[10px] text-violet-400">Processing...</span>
        </div>
      )}
    </div>
  );
});

ThinkingStream.displayName = 'ThinkingStream';

/**
 * Tool Call Card
 */
const ToolCallCard = memo(({ toolCall }) => {
  const toolIcons = {
    get_price_impacts: Sparkles,
    get_markets: BarChart3,
    trade: TrendingUp,
    read_memory: BookOpen,
    write_memory: BookOpen,
    think: Brain,
    get_session_state: Activity,
  };

  const Icon = toolIcons[toolCall.tool] || Wrench;
  const durationMs = toolCall.durationMs;
  const isSlow = durationMs != null && durationMs > 5000;

  return (
    <div className="flex items-start gap-2 p-2 bg-gray-800/40 rounded-lg">
      <Icon className="w-3.5 h-3.5 text-cyan-400 mt-0.5 flex-shrink-0" />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-cyan-300">{toolCall.tool}()</span>
          {durationMs != null && (
            <span className={`text-[10px] font-mono ${isSlow ? 'text-amber-400' : 'text-gray-600'}`}>
              {durationMs >= 1000 ? `${(durationMs / 1000).toFixed(1)}s` : `${durationMs}ms`}
            </span>
          )}
          {toolCall.outputPreview && (
            <span className="text-[10px] text-gray-500 truncate">
              {toolCall.outputPreview}
            </span>
          )}
        </div>
      </div>
    </div>
  );
});

ToolCallCard.displayName = 'ToolCallCard';

/**
 * Tool Calls Panel
 */
const ToolCallsPanel = memo(({ toolCalls, defaultCollapsed = true }) => {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);

  return (
    <div className="bg-gray-900/50 rounded-xl border border-gray-800/50 overflow-hidden">
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="w-full flex items-center justify-between p-3 hover:bg-gray-800/30 transition-colors"
      >
        <div className="flex items-center gap-2">
          <Wrench className="w-4 h-4 text-cyan-400" />
          <span className="text-xs font-semibold text-gray-300">Tool Calls</span>
          <span className="px-2 py-0.5 bg-cyan-900/30 text-cyan-400 text-[10px] font-bold rounded-full">
            {toolCalls.length}
          </span>
        </div>
        {collapsed ? <ChevronRight className="w-4 h-4 text-gray-500" /> : <ChevronDown className="w-4 h-4 text-gray-500" />}
      </button>
      {!collapsed && (
        <div className="px-3 pb-3 space-y-1.5 max-h-[200px] overflow-y-auto">
          {toolCalls.length === 0 ? (
            <div className="text-center py-4 text-gray-500 text-xs">No tool calls yet...</div>
          ) : (
            toolCalls.slice(0, 15).map((tc) => (
              <ToolCallCard key={tc.id} toolCall={tc} />
            ))
          )}
        </div>
      )}
    </div>
  );
});

ToolCallsPanel.displayName = 'ToolCallsPanel';

/**
 * Trade Card
 */
const TradeCard = memo(({ trade }) => {
  const isYes = trade.side?.toLowerCase() === 'yes';
  const isSell = trade.action === 'sell';
  const sideColor = isYes ? 'text-emerald-400' : 'text-rose-400';
  const sideBg = isYes ? 'bg-emerald-900/20' : 'bg-rose-900/20';

  return (
    <div className={`flex items-center justify-between p-2.5 rounded-lg border ${sideBg} border-gray-700/30`}>
      <div className="flex items-center gap-2">
        <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
          isSell ? 'bg-amber-900/30 text-amber-400' : 'bg-blue-900/30 text-blue-400'
        }`}>
          {isSell ? 'SELL' : 'BUY'}
        </span>
        <div className={`text-xs font-bold ${sideColor}`}>{isYes ? 'YES' : 'NO'}</div>
        <span className="text-xs font-mono text-gray-300">{trade.ticker}</span>
        <span className="text-[10px] text-gray-500">{trade.contracts}x @ {trade.priceCents}c</span>
      </div>
      <span className="text-[10px] text-gray-500">${((trade.contracts * trade.priceCents) / 100).toFixed(2)}</span>
    </div>
  );
});

TradeCard.displayName = 'TradeCard';

/**
 * Settlement Card
 */
const SettlementCard = memo(({ settlement }) => {
  const isWin = settlement.result === 'win';
  const isLoss = settlement.result === 'loss';
  const pnlCents = settlement.pnlCents || 0;
  const ResultIcon = isWin ? CheckCircle : isLoss ? XCircle : AlertCircle;
  const resultColor = isWin ? 'text-emerald-400' : isLoss ? 'text-rose-400' : 'text-gray-400';
  const resultBg = isWin ? 'bg-emerald-900/20' : isLoss ? 'bg-rose-900/20' : 'bg-gray-800/30';

  return (
    <div className={`flex items-center justify-between p-2.5 rounded-lg border ${resultBg} border-gray-700/30`}>
      <div className="flex items-center gap-2">
        <ResultIcon className={`w-3.5 h-3.5 ${resultColor}`} />
        <span className="text-xs font-mono text-gray-300">{settlement.ticker}</span>
        <span className="text-[10px] text-gray-500 capitalize">{settlement.result}</span>
      </div>
      <span className={`text-xs font-mono font-bold ${resultColor}`}>
        {pnlCents >= 0 ? '+' : ''}${(pnlCents / 100).toFixed(2)}
      </span>
    </div>
  );
});

SettlementCard.displayName = 'SettlementCard';

/**
 * Trades & Settlements Panel
 */
const TradesPanel = memo(({ trades, settlements }) => {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="bg-gray-900/50 rounded-xl border border-gray-800/50 overflow-hidden">
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="w-full flex items-center justify-between p-3 hover:bg-gray-800/30 transition-colors"
      >
        <div className="flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-emerald-400" />
          <span className="text-xs font-semibold text-gray-300">Trades</span>
          <span className="px-2 py-0.5 bg-emerald-900/30 text-emerald-400 text-[10px] font-bold rounded-full">{trades.length}</span>
          {settlements.length > 0 && (
            <span className="px-2 py-0.5 bg-gray-700/30 text-gray-400 text-[10px] font-bold rounded-full">{settlements.length} settled</span>
          )}
        </div>
        {collapsed ? <ChevronRight className="w-4 h-4 text-gray-500" /> : <ChevronDown className="w-4 h-4 text-gray-500" />}
      </button>
      {!collapsed && (
        <div className="px-3 pb-3 space-y-1.5 max-h-[250px] overflow-y-auto">
          {trades.length === 0 && settlements.length === 0 ? (
            <div className="text-center py-4 text-gray-500 text-xs">No trades yet...</div>
          ) : (
            <>
              {settlements.slice(0, 5).map((s) => (<SettlementCard key={s.id} settlement={s} />))}
              {trades.slice(0, 10).map((t) => (<TradeCard key={t.id} trade={t} />))}
            </>
          )}
        </div>
      )}
    </div>
  );
});

TradesPanel.displayName = 'TradesPanel';

/**
 * Learnings Panel
 */
const LearningsPanel = memo(({ learnings }) => {
  const [collapsed, setCollapsed] = useState(true);

  return (
    <div className="bg-gray-900/50 rounded-xl border border-gray-800/50 overflow-hidden">
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="w-full flex items-center justify-between p-3 hover:bg-gray-800/30 transition-colors"
      >
        <div className="flex items-center gap-2">
          <BookOpen className="w-4 h-4 text-amber-400" />
          <span className="text-xs font-semibold text-gray-300">Learnings</span>
          <span className="px-2 py-0.5 bg-amber-900/30 text-amber-400 text-[10px] font-bold rounded-full">{learnings.length}</span>
        </div>
        {collapsed ? <ChevronRight className="w-4 h-4 text-gray-500" /> : <ChevronDown className="w-4 h-4 text-gray-500" />}
      </button>
      {!collapsed && (
        <div className="px-3 pb-3 space-y-2 max-h-[200px] overflow-y-auto">
          {learnings.length === 0 ? (
            <div className="text-center py-4 text-gray-500 text-xs">No learnings recorded yet...</div>
          ) : (
            learnings.slice(0, 10).map((learning) => (
              <div key={learning.id} className="p-2 bg-gray-800/40 rounded-lg">
                <div className="text-xs text-gray-300">{learning.content}</div>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
});

LearningsPanel.displayName = 'LearningsPanel';

/**
 * GdeltArticleRow - Single GDELT article with link
 */
const GdeltArticleRow = memo(({ article }) => {
  const toneColor = article.tone > 1.5 ? 'text-emerald-400' : article.tone < -1.5 ? 'text-rose-400' : 'text-gray-400';

  // Extract domain from URL for display
  const domain = (() => {
    try {
      return new URL(article.url).hostname.replace('www.', '');
    } catch {
      return article.source || 'unknown';
    }
  })();

  return (
    <div className="flex items-start gap-2 py-1.5 border-b border-gray-700/20 last:border-b-0">
      <ExternalLink className="w-3 h-3 text-gray-500 mt-0.5 flex-shrink-0" />
      <div className="flex-1 min-w-0">
        <a
          href={article.url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-[11px] text-blue-400 hover:text-blue-300 hover:underline truncate block"
          title={article.url}
        >
          {domain}
        </a>
        <div className="flex items-center gap-2 mt-0.5">
          <span className={`text-[9px] font-mono ${toneColor}`}>
            tone {article.tone > 0 ? '+' : ''}{article.tone}
          </span>
          {article.key_persons?.length > 0 && (
            <span className="text-[9px] text-gray-600 truncate">
              {article.key_persons.slice(0, 2).join(', ')}
            </span>
          )}
        </div>
      </div>
    </div>
  );
});

GdeltArticleRow.displayName = 'GdeltArticleRow';

/**
 * GdeltEventTripleRow - Actor → Event → Actor triple with Goldstein coloring
 */
const GdeltEventTripleRow = memo(({ triple }) => {
  const gs = triple.goldstein;
  const gsColor = gs == null ? 'text-gray-500' : gs > 2 ? 'text-emerald-400' : gs < -2 ? 'text-rose-400' : 'text-gray-400';
  const qcLabel = triple.quad_class_label || '';
  const qcColor = triple.quad_class <= 2 ? 'bg-emerald-900/20 text-emerald-400 border-emerald-700/30' : 'bg-rose-900/20 text-rose-400 border-rose-700/30';

  return (
    <div className="flex items-center gap-1.5 py-1 border-b border-gray-700/20 last:border-b-0 text-[10px]">
      <span className="text-violet-400 truncate max-w-[90px]" title={triple.actor1?.name}>
        {triple.actor1?.name || '?'}
      </span>
      <ArrowRight className="w-2.5 h-2.5 text-gray-600 flex-shrink-0" />
      <span className="text-cyan-400 font-mono flex-shrink-0" title={`CAMEO ${triple.event?.code}: ${triple.event?.description}`}>
        {triple.event?.description || triple.event?.code || '?'}
      </span>
      <ArrowRight className="w-2.5 h-2.5 text-gray-600 flex-shrink-0" />
      <span className="text-orange-400 truncate max-w-[90px]" title={triple.actor2?.name}>
        {triple.actor2?.name || '?'}
      </span>
      <span className="flex-1" />
      {gs != null && (
        <span className={`font-mono flex-shrink-0 ${gsColor}`}>
          {gs > 0 ? '+' : ''}{gs.toFixed(1)}
        </span>
      )}
      {qcLabel && (
        <span className={`px-1 py-0.5 text-[8px] rounded border flex-shrink-0 ${qcColor}`}>
          {qcLabel.split(' ')[0]}
        </span>
      )}
      {triple.mentions > 0 && (
        <span className="text-gray-600 flex-shrink-0">{triple.mentions}m</span>
      )}
    </div>
  );
});

GdeltEventTripleRow.displayName = 'GdeltEventTripleRow';

/**
 * GdeltQueryCard - Single GDELT query result
 */
const GdeltQueryCard = memo(({ result }) => {
  const [expanded, setExpanded] = useState(false);
  const isTimeline = result.source === 'volume_timeline';
  const isEvents = result.source === 'events';
  const tone = result.toneSummary;
  const avgTone = tone?.avg_tone || 0;
  const toneColor = avgTone > 1.5 ? 'text-emerald-400' : avgTone < -1.5 ? 'text-rose-400' : 'text-gray-400';
  const toneBg = avgTone > 1.5 ? 'bg-emerald-900/20' : avgTone < -1.5 ? 'bg-rose-900/20' : 'bg-gray-800/30';

  // Source label mapping
  const sourceLabel = { gkg: 'GKG', doc_api: 'DOC', volume_timeline: 'TREND', events: 'EVENTS' }[result.source] || '';
  const sourceColor = { gkg: 'text-amber-500 border-amber-700/30', doc_api: 'text-blue-400 border-blue-700/30', volume_timeline: 'text-purple-400 border-purple-700/30', events: 'text-red-400 border-red-700/30' }[result.source] || 'text-gray-400 border-gray-600/30';

  return (
    <div className="bg-gray-800/30 rounded-lg border border-gray-700/30 overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-3 py-2.5 hover:bg-gray-800/50 transition-colors"
      >
        <div className="flex items-center gap-2 min-w-0">
          {isEvents
            ? <Activity className="w-3.5 h-3.5 text-red-400 flex-shrink-0" />
            : isTimeline
              ? <TrendingUp className="w-3.5 h-3.5 text-purple-400 flex-shrink-0" />
              : <Globe className="w-3.5 h-3.5 text-blue-400 flex-shrink-0" />
          }
          <span className="text-[11px] text-gray-300 truncate">
            {(result.actorNames?.length > 0 ? result.actorNames : result.searchTerms).join(', ')}
          </span>
          {sourceLabel && (
            <span className={`text-[8px] px-1 py-0.5 rounded border ${sourceColor} flex-shrink-0`}>
              {sourceLabel}
            </span>
          )}
          {result.cached && (
            <span className="text-[8px] text-gray-600 flex-shrink-0">cached</span>
          )}
        </div>
        <div className="flex items-center gap-2 flex-shrink-0 ml-2">
          {isTimeline ? (
            /* Timeline data points badge */
            <span className={`px-1.5 py-0.5 text-[9px] font-bold rounded ${
              result.dataPoints >= 10
                ? 'bg-purple-900/30 text-purple-400 border border-purple-600/30'
                : 'bg-gray-700/50 text-gray-300 border border-gray-600/30'
            }`}>
              {result.dataPoints} pts
            </span>
          ) : isEvents ? (
            <>
              {/* Event count badge */}
              <span className={`px-1.5 py-0.5 text-[9px] font-bold rounded ${
                result.eventCount >= 5
                  ? 'bg-red-900/30 text-red-400 border border-red-600/30'
                  : result.eventCount > 0
                    ? 'bg-gray-700/50 text-gray-300 border border-gray-600/30'
                    : 'bg-gray-800/50 text-gray-500 border border-gray-700/30'
              }`}>
                {result.eventCount} events
              </span>
              {/* Goldstein indicator */}
              {result.goldsteinSummary && (
                <span className={`flex items-center gap-0.5 px-1.5 py-0.5 text-[9px] rounded ${
                  result.goldsteinSummary.avg > 2 ? 'bg-emerald-900/20 text-emerald-400'
                  : result.goldsteinSummary.avg < -2 ? 'bg-rose-900/20 text-rose-400'
                  : 'bg-gray-800/30 text-gray-400'
                }`}>
                  GS {result.goldsteinSummary.avg > 0 ? '+' : ''}{result.goldsteinSummary.avg?.toFixed(1) || '0'}
                </span>
              )}
            </>
          ) : (
            <>
              {/* Article count badge */}
              <span className={`px-1.5 py-0.5 text-[9px] font-bold rounded ${
                result.articleCount >= 5
                  ? 'bg-blue-900/30 text-blue-400 border border-blue-600/30'
                  : result.articleCount > 0
                    ? 'bg-gray-700/50 text-gray-300 border border-gray-600/30'
                    : 'bg-gray-800/50 text-gray-500 border border-gray-700/30'
              }`}>
                {result.articleCount} articles
              </span>
              {/* Source diversity */}
              {result.sourceDiversity > 0 && (
                <span className="text-[9px] text-gray-500">
                  {result.sourceDiversity} src
                </span>
              )}
              {/* Tone indicator */}
              <span className={`flex items-center gap-0.5 px-1.5 py-0.5 text-[9px] rounded ${toneBg} ${toneColor}`}>
                {avgTone > 1.5 ? <ThumbsUp className="w-2.5 h-2.5" /> : avgTone < -1.5 ? <ThumbsDown className="w-2.5 h-2.5" /> : <Minus className="w-2.5 h-2.5" />}
                {avgTone !== 0 ? (avgTone > 0 ? '+' : '') + avgTone.toFixed(1) : '0'}
              </span>
            </>
          )}
          <span className="text-[9px] text-gray-600 font-mono">{result.timestamp}</span>
          {expanded ? <ChevronDown className="w-3 h-3 text-gray-500" /> : <ChevronRight className="w-3 h-3 text-gray-500" />}
        </div>
      </button>

      {expanded && (
        <div className="px-3 pb-3 pt-1 border-t border-gray-700/20 space-y-3">
          {/* Volume timeline visualization */}
          {isTimeline && result.timeline?.length > 0 && (
            <div>
              <span className="text-[9px] text-gray-500 uppercase tracking-wider">Coverage Volume</span>
              <div className="mt-1 flex items-end gap-px h-10">
                {(() => {
                  const values = result.timeline.map(p => p.value || 0);
                  const max = Math.max(...values, 1);
                  return values.map((v, i) => (
                    <div
                      key={i}
                      className="flex-1 bg-purple-500/60 rounded-t-sm min-w-[2px]"
                      style={{ height: `${Math.max((v / max) * 100, 2)}%` }}
                      title={`${result.timeline[i]?.date}: ${v}`}
                    />
                  ));
                })()}
              </div>
              <div className="flex justify-between mt-1">
                <span className="text-[8px] text-gray-600">{result.timeline[0]?.date}</span>
                <span className="text-[8px] text-gray-600">{result.timeline[result.timeline.length - 1]?.date}</span>
              </div>
            </div>
          )}

          {/* === Events-specific expanded content === */}
          {isEvents && (
            <>
              {/* QuadClass summary bar */}
              {result.quadClassSummary && Object.keys(result.quadClassSummary).length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  <span className="text-[9px] text-gray-500 mr-1">QuadClass:</span>
                  {Object.entries(result.quadClassSummary).map(([qc, data]) => (
                    <span key={qc} className={`px-1.5 py-0.5 text-[9px] rounded border ${
                      parseInt(qc) <= 2
                        ? 'bg-emerald-900/20 text-emerald-400 border-emerald-700/30'
                        : 'bg-rose-900/20 text-rose-400 border-rose-700/30'
                    }`}>
                      {data.label} ({data.count})
                    </span>
                  ))}
                </div>
              )}

              {/* Goldstein summary */}
              {result.goldsteinSummary && (
                <div className="flex items-center gap-3 text-[10px]">
                  <span className="text-gray-500">Goldstein:</span>
                  <span className={result.goldsteinSummary.avg > 0 ? 'text-emerald-400' : result.goldsteinSummary.avg < 0 ? 'text-rose-400' : 'text-gray-400'}>
                    avg {result.goldsteinSummary.avg > 0 ? '+' : ''}{result.goldsteinSummary.avg?.toFixed(1)}
                  </span>
                  <span className="text-gray-600">
                    range [{result.goldsteinSummary.min?.toFixed(1)}, {result.goldsteinSummary.max?.toFixed(1)}]
                  </span>
                  <span className="text-emerald-400/70">{result.goldsteinSummary.positive_count} pos</span>
                  <span className="text-rose-400/70">{result.goldsteinSummary.negative_count} neg</span>
                </div>
              )}

              {/* Event triples */}
              {result.topEventTriples?.length > 0 && (
                <div>
                  <span className="text-[9px] text-gray-500 uppercase tracking-wider">Event Triples</span>
                  <div className="mt-1">
                    {result.topEventTriples.slice(0, 8).map((triple, i) => (
                      <GdeltEventTripleRow key={i} triple={triple} />
                    ))}
                  </div>
                </div>
              )}

              {/* CAMEO code distribution */}
              {result.eventCodeDistribution?.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  <span className="text-[9px] text-gray-500 mr-1">CAMEO:</span>
                  {result.eventCodeDistribution.slice(0, 8).map((ec, i) => (
                    <span key={i} className="px-1.5 py-0.5 text-[8px] bg-gray-700/50 text-gray-400 rounded" title={`Code ${ec.code}`}>
                      {ec.description?.toLowerCase()} ({ec.count})
                    </span>
                  ))}
                </div>
              )}

              {/* Top actors */}
              {result.topActors?.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  <span className="text-[9px] text-gray-500 mr-1">Actors:</span>
                  {result.topActors.slice(0, 6).map((a, i) => (
                    <span key={i} className={`px-1.5 py-0.5 text-[9px] rounded border ${
                      a.avg_goldstein > 1 ? 'bg-emerald-900/20 text-emerald-400 border-emerald-700/30'
                      : a.avg_goldstein < -1 ? 'bg-rose-900/20 text-rose-400 border-rose-700/30'
                      : 'bg-gray-700/50 text-gray-400 border-gray-600/30'
                    }`}>
                      {a.name} ({a.count})
                    </span>
                  ))}
                </div>
              )}

              {/* Geo hotspots */}
              {result.geoHotspots?.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  <span className="text-[9px] text-gray-500 mr-1">Geo:</span>
                  {result.geoHotspots.slice(0, 4).map((g, i) => (
                    <span key={i} className="px-1.5 py-0.5 text-[8px] bg-gray-700/50 text-gray-400 rounded">
                      {g.location} ({g.count})
                    </span>
                  ))}
                </div>
              )}
            </>
          )}

          {/* === News-specific expanded content (non-events, non-timeline) === */}

          {/* Tone breakdown */}
          {!isTimeline && !isEvents && (tone?.positive_count > 0 || tone?.negative_count > 0) && (
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1">
                <ThumbsUp className="w-3 h-3 text-emerald-400" />
                <span className="text-[10px] text-emerald-400 font-mono">{tone.positive_count}</span>
              </div>
              <div className="flex items-center gap-1">
                <Minus className="w-3 h-3 text-gray-400" />
                <span className="text-[10px] text-gray-400 font-mono">{tone.neutral_count}</span>
              </div>
              <div className="flex items-center gap-1">
                <ThumbsDown className="w-3 h-3 text-rose-400" />
                <span className="text-[10px] text-rose-400 font-mono">{tone.negative_count}</span>
              </div>
            </div>
          )}

          {/* Key entities row */}
          {!isEvents && (result.keyPersons?.length > 0 || result.keyOrganizations?.length > 0) && (
            <div className="space-y-1.5">
              {result.keyPersons?.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  <span className="text-[9px] text-gray-500 mr-1">People:</span>
                  {result.keyPersons.slice(0, 5).map((p, i) => (
                    <span key={i} className="px-1.5 py-0.5 text-[9px] bg-violet-900/30 text-violet-400 rounded border border-violet-700/30">
                      {p.person} ({p.count})
                    </span>
                  ))}
                </div>
              )}
              {result.keyOrganizations?.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  <span className="text-[9px] text-gray-500 mr-1">Orgs:</span>
                  {result.keyOrganizations.slice(0, 5).map((o, i) => (
                    <span key={i} className="px-1.5 py-0.5 text-[9px] bg-cyan-900/30 text-cyan-400 rounded border border-cyan-700/30">
                      {o.org} ({o.count})
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Key themes */}
          {!isEvents && result.keyThemes?.length > 0 && (
            <div className="flex flex-wrap gap-1">
              <span className="text-[9px] text-gray-500 mr-1">Themes:</span>
              {result.keyThemes.slice(0, 8).map((t, i) => (
                <span key={i} className="px-1.5 py-0.5 text-[8px] bg-gray-700/50 text-gray-400 rounded">
                  {t.theme.toLowerCase().replace(/_/g, ' ')}
                </span>
              ))}
            </div>
          )}

          {/* Top articles */}
          {!isEvents && result.topArticles?.length > 0 && (
            <div>
              <span className="text-[9px] text-gray-500 uppercase tracking-wider">Top Articles</span>
              <div className="mt-1">
                {result.topArticles.slice(0, 5).map((article, i) => (
                  <GdeltArticleRow key={i} article={article} />
                ))}
              </div>
            </div>
          )}

          {/* Duration */}
          {result.durationMs != null && (
            <div className="text-[9px] text-gray-600">
              Query took {result.durationMs >= 1000 ? `${(result.durationMs / 1000).toFixed(1)}s` : `${result.durationMs}ms`}
            </div>
          )}
        </div>
      )}
    </div>
  );
});

GdeltQueryCard.displayName = 'GdeltQueryCard';

/**
 * GdeltNewsPanel - GDELT News Intelligence panel showing recent query results
 */
const GdeltNewsPanel = memo(({ gdeltResults }) => {
  const [collapsed, setCollapsed] = useState(false);

  // Aggregate stats
  const totalArticles = gdeltResults.reduce((sum, r) => sum + (r.articleCount || 0), 0);
  const avgTone = gdeltResults.length > 0
    ? gdeltResults.reduce((sum, r) => sum + (r.toneSummary?.avg_tone || 0), 0) / gdeltResults.length
    : 0;

  if (gdeltResults.length === 0) return null;

  return (
    <div className="bg-gray-900/50 rounded-2xl border border-blue-800/30 overflow-hidden">
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="w-full flex items-center justify-between p-4 hover:bg-gray-800/30 transition-colors"
      >
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-blue-900/30 border border-blue-700/30">
            <Newspaper className="w-4 h-4 text-blue-400" />
          </div>
          <span className="text-sm font-semibold text-gray-300">GDELT News Intelligence</span>
          <span className="px-2 py-0.5 bg-blue-900/30 text-blue-400 text-[10px] font-bold rounded-full border border-blue-600/30">
            {gdeltResults.length} queries
          </span>
          <span className="text-[10px] text-gray-500">
            {totalArticles} total articles
          </span>
        </div>
        <div className="flex items-center gap-2">
          {avgTone !== 0 && (
            <span className={`text-[10px] font-mono ${avgTone > 0 ? 'text-emerald-400' : avgTone < 0 ? 'text-rose-400' : 'text-gray-400'}`}>
              avg tone {avgTone > 0 ? '+' : ''}{avgTone.toFixed(1)}
            </span>
          )}
          {collapsed ? <ChevronRight className="w-4 h-4 text-gray-500" /> : <ChevronDown className="w-4 h-4 text-gray-500" />}
        </div>
      </button>
      {!collapsed && (
        <div className="px-4 pb-4 space-y-2 max-h-[500px] overflow-y-auto">
          {gdeltResults.map((result) => (
            <GdeltQueryCard key={result.id} result={result} />
          ))}
        </div>
      )}
    </div>
  );
});

GdeltNewsPanel.displayName = 'GdeltNewsPanel';

/**
 * AgentPage - Dedicated full-page view for the Deep Agent
 *
 * Displays the Extraction Intelligence -> Market Signal Pipeline:
 * 1. Reddit posts streaming in with extraction class pills
 * 2. Extraction stream (all classes)
 * 3. Event Intelligence Board (events + markets + signals + prices)
 * 4. Agent activity, thinking, trades
 */
const AgentPage = () => {
  const [showPosts, setShowPosts] = useState(true);

  // Live-updating timestamps: forces re-render every 30s
  useRelativeTime(30000);

  // Initialize deep agent hook first to get processMessage
  const {
    agentState,
    thinking,
    toolCalls,
    trades,
    settlements,
    learnings,
    memoryUpdates,
    processMessage,
    isRunning: agentIsRunning,
    gdeltResults,
  } = useDeepAgent({ useV3WebSocketState: true });

  // Wire deep agent message processing to WebSocket
  const handleMessage = useCallback((type, message, context) => {
    processMessage(type, message);
  }, [processMessage]);

  const {
    wsStatus,
    currentState,
    tradingState,
    strategyStatus,
    entityRedditPosts,
    entitySystemActive,
    redditAgentHealth,
    extractions,
    marketSignals,
    eventConfigs,
    trackedMarkets,
  } = useV3WebSocket({ onMessage: handleMessage });

  // Extraction-centric derived views
  const {
    byEvent,
    bySource,
    aggregatedSignals,
    stats: extractionStats,
  } = useExtractions({ extractions, marketSignals, eventConfigs });

  // Auto-generate event configs from tracked markets, merge with researched configs
  const mergedEventConfigs = useMemo(() => {
    // Group tracked markets by event_ticker
    const discoveredEvents = {};
    for (const market of trackedMarkets) {
      if (!market.event_ticker || market.status !== 'active') continue;
      if (!discoveredEvents[market.event_ticker]) {
        discoveredEvents[market.event_ticker] = {
          event_ticker: market.event_ticker,
          event_title: market.event_title || market.event_ticker,
          category: market.category,
          markets: [],
          _source: 'discovery',
        };
      }
      discoveredEvents[market.event_ticker].markets.push({
        ticker: market.ticker,
        title: market.title || market.yes_sub_title || '',
        subtitle: market.subtitle || '',
        yes_sub_title: market.yes_sub_title,
        close_ts: market.close_ts,
        time_to_close_seconds: market.time_to_close_seconds,
        volume_24h: market.volume_24h,
      });
    }

    // Merge: actual eventConfigs override discovery-derived ones
    const configsByEvent = {};
    for (const cfg of Object.values(discoveredEvents)) {
      configsByEvent[cfg.event_ticker] = cfg;
    }
    for (const cfg of eventConfigs) {
      configsByEvent[cfg.event_ticker] = { ...configsByEvent[cfg.event_ticker], ...cfg, _source: 'researched' };
    }
    return Object.values(configsByEvent);
  }, [trackedMarkets, eventConfigs]);

  // Build merged price map (tracked market data as fallback)
  const mergedMarketPrices = useMemo(() => {
    const prices = { ...(tradingState?.market_prices || {}) };
    for (const market of trackedMarkets) {
      if (!prices[market.ticker] && market.yes_bid != null) {
        prices[market.ticker] = {
          yes_bid: market.yes_bid,
          yes_ask: market.yes_ask,
          spread: (market.yes_ask || 0) - (market.yes_bid || 0),
          volume: market.volume,
          volume_24h: market.volume_24h,
          last_price: market.price,
        };
      }
    }
    return prices;
  }, [tradingState?.market_prices, trackedMarkets]);

  // Get deep agent strategy data
  const deepAgentStrategy = strategyStatus?.strategies?.deep_agent;
  const isAgentRunning = agentIsRunning || deepAgentStrategy?.running || entitySystemActive;

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950">
      <V3Header
        wsStatus={wsStatus}
        currentState={currentState}
        balance={tradingState?.balance}
      />

      <div className="max-w-7xl mx-auto px-6 py-6">
        {/* Page Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="p-3 rounded-2xl bg-gradient-to-br from-violet-900/50 to-violet-950/30 border border-violet-700/30">
                <Brain className={`w-8 h-8 text-violet-400 ${isAgentRunning ? 'animate-pulse' : ''}`} />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">Deep Agent</h1>
                <p className="text-sm text-gray-400 mt-0.5">
                  Extraction Intelligence → Market Signal Pipeline
                </p>
              </div>
            </div>

            <div className={`
              flex items-center gap-2 px-4 py-2 rounded-lg border
              ${isAgentRunning
                ? 'bg-emerald-900/30 border-emerald-600/40 text-emerald-400'
                : 'bg-gray-800/50 border-gray-700/40 text-gray-400'}
            `}>
              {isAgentRunning ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span className="font-semibold">ACTIVE</span>
                </>
              ) : (
                <>
                  <Activity className="w-4 h-4" />
                  <span className="font-semibold">INACTIVE</span>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Pipeline Visualization */}
        <div className="mb-8">
          <div className="flex items-center gap-2 mb-4">
            <GitBranch className="w-4 h-4 text-gray-500" />
            <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">
              Data Pipeline
            </h2>
          </div>

          <div className="flex items-center gap-3">
            <PipelineStage
              icon={MessageSquare}
              title="Sources"
              count={entityRedditPosts.length}
              color="orange"
              isActive={entityRedditPosts.length > 0}
              description="Posts ingested"
            />
            <ArrowRight className="w-5 h-5 text-gray-600 flex-shrink-0" />
            <PipelineStage
              icon={Zap}
              title="Extractions"
              count={extractionStats.totalExtractions}
              color="violet"
              isActive={extractions.length > 0}
              description={`${extractionStats.marketSignalCount} signals, ${extractionStats.entityMentionCount} entities`}
            />
            <ArrowRight className="w-5 h-5 text-gray-600 flex-shrink-0" />
            <PipelineStage
              icon={Sparkles}
              title="Market Signals"
              count={extractionStats.marketSignalCount}
              color="cyan"
              isActive={marketSignals.length > 0}
              description={`${extractionStats.uniqueMarkets} markets with signals`}
            />
            <ArrowRight className="w-5 h-5 text-gray-600 flex-shrink-0" />
            <PipelineStage
              icon={Newspaper}
              title="GDELT"
              count={gdeltResults.length}
              color="cyan"
              isActive={gdeltResults.length > 0}
              description={`${gdeltResults.reduce((s, r) => s + (r.articleCount || 0), 0)} articles found`}
            />
            <ArrowRight className="w-5 h-5 text-gray-600 flex-shrink-0" />
            <PipelineStage
              icon={Layers}
              title="Markets"
              count={trackedMarkets.length}
              color="cyan"
              isActive={trackedMarkets.length > 0}
              description={`${mergedEventConfigs.length} events`}
            />
            <ArrowRight className="w-5 h-5 text-gray-600 flex-shrink-0" />
            <PipelineStage
              icon={Brain}
              title="Agent Cycles"
              count={agentState.cycleCount || 0}
              color="emerald"
              isActive={isAgentRunning}
              description="Deep agent cycles"
            />
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-12 gap-6">
          {/* Left Column - Source Feed + Extraction Stream */}
          <div className="col-span-4 space-y-6">
            {/* Reddit Posts */}
            <div className="bg-gray-900/50 rounded-2xl border border-gray-800/50 overflow-hidden">
              <button
                onClick={() => setShowPosts(!showPosts)}
                className="w-full flex items-center justify-between p-4 hover:bg-gray-800/30 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <MessageSquare className="w-4 h-4 text-orange-400" />
                  <span className="text-sm font-semibold text-gray-300">Source Feed</span>
                  <span className="px-2 py-0.5 bg-orange-900/30 text-orange-400 text-[10px] font-bold rounded-full">
                    {entityRedditPosts.length}
                  </span>
                  {/* Health Status */}
                  {redditAgentHealth.health === 'healthy' ? (
                    <span className="flex items-center gap-1 px-1.5 py-0.5 bg-emerald-900/30 text-emerald-400 text-[9px] rounded-full border border-emerald-700/30">
                      <CheckCircle className="w-3 h-3" />
                      <span className="font-medium">Live</span>
                    </span>
                  ) : redditAgentHealth.health === 'degraded' ? (
                    <span className="flex items-center gap-1 px-1.5 py-0.5 bg-amber-900/30 text-amber-400 text-[9px] rounded-full border border-amber-700/30">
                      <AlertCircle className="w-3 h-3" />
                      <span className="font-medium">Partial</span>
                    </span>
                  ) : redditAgentHealth.health === 'unhealthy' ? (
                    <span className="flex items-center gap-1 px-1.5 py-0.5 bg-red-900/30 text-red-400 text-[9px] rounded-full border border-red-700/30">
                      <XCircle className="w-3 h-3" />
                      <span className="font-medium">Offline</span>
                    </span>
                  ) : (
                    <span className="flex items-center gap-1 px-1.5 py-0.5 bg-gray-800/50 text-gray-500 text-[9px] rounded-full border border-gray-700/30">
                      <Clock className="w-3 h-3" />
                      <span className="font-medium">...</span>
                    </span>
                  )}
                </div>
                {showPosts ? <ChevronDown className="w-4 h-4 text-gray-500" /> : <ChevronRight className="w-4 h-4 text-gray-500" />}
              </button>
              {showPosts && (
                <div className="px-4 pb-4 space-y-2 max-h-[300px] overflow-y-auto">
                  {entityRedditPosts.length === 0 ? (
                    <div className="text-center py-6 text-gray-500 text-sm">
                      {redditAgentHealth.health === 'unhealthy' ? (
                        <div className="space-y-2">
                          <XCircle className="w-8 h-8 text-red-500 mx-auto" />
                          <div className="text-red-400 font-medium">Reddit API not connected</div>
                          <div className="text-[11px] text-gray-600">
                            {redditAgentHealth.lastError || 'Check REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET env vars'}
                          </div>
                        </div>
                      ) : redditAgentHealth.health === 'degraded' ? (
                        <div className="space-y-2">
                          <AlertCircle className="w-8 h-8 text-amber-500 mx-auto" />
                          <div className="text-amber-400 font-medium">Partially connected</div>
                          <div className="text-[11px] text-gray-600">
                            Reddit: yes | Extractor: {redditAgentHealth.extractorAvailable ? 'yes' : 'no'} | Supabase: {redditAgentHealth.supabaseAvailable ? 'yes' : 'no'}
                          </div>
                        </div>
                      ) : redditAgentHealth.health === 'healthy' ? (
                        <div className="space-y-2">
                          <RefreshCw className="w-6 h-6 text-orange-400 mx-auto animate-spin" />
                          <div>Listening to r/{redditAgentHealth.subreddits?.join(', r/') || 'politics, news'}...</div>
                        </div>
                      ) : (
                        <div className="space-y-2">
                          <Clock className="w-6 h-6 text-gray-500 mx-auto" />
                          <div>Waiting for Reddit posts...</div>
                        </div>
                      )}
                    </div>
                  ) : (
                    entityRedditPosts.slice(0, 10).map((post) => (
                      <RedditPostCard
                        key={post.post_id}
                        post={post}
                        postExtractions={bySource[post.post_id] || []}
                      />
                    ))
                  )}
                </div>
              )}
            </div>

            {/* Extraction Stream */}
            <ExtractionFeedPanel extractions={extractions} />
          </div>

          {/* Right Column - Agent + Event Intelligence */}
          <div className="col-span-8 space-y-4">
            {/* Agent Status Header - Sticky */}
            <div className="sticky top-0 z-10">
              <AgentStatusHeader
                agentState={agentState}
                settlements={settlements}
                trades={trades}
              />
            </div>

            {/* Event Intelligence Board */}
            <EventIntelligenceBoard
              eventConfigs={mergedEventConfigs}
              byEvent={byEvent}
              aggregatedSignals={aggregatedSignals}
              marketPrices={mergedMarketPrices}
              positionsDetails={tradingState?.positions_details || EMPTY_POSITIONS}
            />

            {/* GDELT News Intelligence */}
            {gdeltResults.length > 0 && (
              <GdeltNewsPanel gdeltResults={gdeltResults} />
            )}

            {/* LLM Cost Panel */}
            <CostPanel costData={agentState.costData} />

            {/* Thinking Stream */}
            <ThinkingStream thinking={thinking} isRunning={isAgentRunning} />

            {/* Tool Calls */}
            <ToolCallsPanel toolCalls={toolCalls} defaultCollapsed={true} />

            {/* Trades & Settlements */}
            <TradesPanel trades={trades} settlements={settlements} />

            {/* Learnings */}
            <LearningsPanel learnings={learnings} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default memo(AgentPage);
