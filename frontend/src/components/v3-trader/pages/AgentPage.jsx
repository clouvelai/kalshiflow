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
  Layers,
  Newspaper,
  Globe,
  ExternalLink,
  ThumbsUp,
  ThumbsDown,
  Minus,
  ShieldOff,
  Target,
  FileText,
  ListChecks,
  Circle,
  CheckCircle2,
} from 'lucide-react';
import V3Header from '../layout/V3Header';
import renderThinkingMarkdown from '../../../utils/renderThinkingMarkdown';
import { useV3WebSocket } from '../../../hooks/v3-trader/useV3WebSocket';
import { useDeepAgent } from '../../../hooks/v3-trader/useDeepAgent';
import { useExtractions } from '../../../hooks/v3-trader/useExtractions';
import { CostPanel } from '../panels/DeepAgentPanel';
import { WatchdogToast } from '../ui';

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
  description,
  testId
}) => {
  const colorClasses = {
    orange: 'from-orange-900/30 to-orange-950/20 border-orange-700/30 text-orange-400',
    violet: 'from-violet-900/30 to-violet-950/20 border-violet-700/30 text-violet-400',
    cyan: 'from-cyan-900/30 to-cyan-950/20 border-cyan-700/30 text-cyan-400',
    emerald: 'from-emerald-900/30 to-emerald-950/20 border-emerald-700/30 text-emerald-400',
    gray: 'from-gray-800/30 to-gray-900/20 border-gray-700/30 text-gray-400',
  };

  return (
    <div {...(testId ? { 'data-testid': testId } : {})} className={`
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
          <span className={`text-[10px] font-bold ${{ bullish: 'text-emerald-400', bearish: 'text-rose-400' }[attrs.direction.toLowerCase()] || 'text-gray-400'}`}>
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

  const filterCounts = useMemo(() => {
    let signals = 0, entities = 0, context = 0, custom = 0;
    for (const e of extractions) {
      switch (e.extraction_class) {
        case 'market_signal': signals++; break;
        case 'entity_mention': entities++; break;
        case 'context_factor': context++; break;
        default: custom++; break;
      }
    }
    return { signals, entities, context, custom };
  }, [extractions]);

  const filterTabs = [
    { key: 'all', label: 'All', count: extractions.length },
    { key: 'signals', label: 'Signals', count: filterCounts.signals },
    { key: 'entities', label: 'Entities', count: filterCounts.entities },
    { key: 'context', label: 'Context', count: filterCounts.context },
    { key: 'custom', label: 'Custom', count: filterCounts.custom },
  ];

  return (
    <div data-testid="extraction-feed-panel" className="bg-gray-900/50 rounded-2xl border border-gray-800/50 overflow-hidden">
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
          <div className="space-y-2 max-h-[400px] overflow-y-auto scrollbar-dark">
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
const EventMarketRow = memo(({ market, marketPrices, signalAgg, position, tradeFlow }) => {
  const ticker = typeof market === 'string' ? market : market.ticker;
  const title = typeof market === 'string' ? null : (market.yes_sub_title || market.title);
  const prices = marketPrices?.[ticker];

  const yesBid = prices?.yes_bid;
  const yesAsk = prices?.yes_ask;
  const spread = prices?.spread;
  const volume = prices?.volume_24h || prices?.volume;
  const volume24h = typeof market === 'string' ? null : market.volume_24h;

  const hasSignals = signalAgg && signalAgg.occurrence_count > 0;

  // Time-to-close formatting
  const timeToCloseSeconds = typeof market === 'string' ? null : market.time_to_close_seconds;
  const timeToClose = useMemo(() => {
    if (timeToCloseSeconds == null || timeToCloseSeconds <= 0) return null;
    if (timeToCloseSeconds < 3600) return `${Math.floor(timeToCloseSeconds / 60)}m`;
    if (timeToCloseSeconds < 86400) return `${Math.floor(timeToCloseSeconds / 3600)}h`;
    return `${Math.floor(timeToCloseSeconds / 86400)}d`;
  }, [timeToCloseSeconds]);

  // Trade flow metrics
  const totalTrades = tradeFlow?.total_trades || 0;
  const yesRatio = totalTrades > 0 ? (tradeFlow?.yes_trades || 0) / totalTrades : 0;
  const priceDrop = tradeFlow?.price_drop || 0;

  // Consensus badge
  const consensus = signalAgg?.consensus;
  const consensusStyles = {
    bullish: 'bg-emerald-900/30 text-emerald-400 border-emerald-600/30',
    bearish: 'bg-rose-900/30 text-rose-400 border-rose-600/30',
  };
  const consensusBg = consensusStyles[consensus?.toLowerCase()] || 'bg-gray-700/40 text-gray-400 border-gray-600/30';

  const hasPosData = position && (position.total_contracts || position.market_exposure);

  // Visual hierarchy: signal markets are brighter with cyan accent
  const tickerColor = hasSignals ? 'text-gray-200' : 'text-gray-400';
  const metricColor = hasSignals ? 'text-gray-300' : 'text-gray-500';
  const metricDimColor = hasSignals ? 'text-gray-400' : 'text-gray-600';
  const displayVol = volume || volume24h || 0;

  return (
    <div
      data-testid={`market-row-${ticker}`}
      className={`px-3 py-2 rounded-lg border transition-colors ${
        hasSignals
          ? 'bg-gray-800/30 border-gray-700/30 border-l-2 border-l-cyan-500 hover:border-gray-600/40'
          : 'bg-gray-800/15 border-gray-800/20 hover:border-gray-700/30'
      }`}
    >
      {/* Vitals row - always shown for ALL markets */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 min-w-0">
          <span className={`text-[10px] font-mono font-semibold ${tickerColor} shrink-0`}>{ticker}</span>
          {title && <span className={`text-[9px] ${hasSignals ? 'text-gray-500' : 'text-gray-600'} truncate max-w-[200px]`}>{title}</span>}
        </div>
        <div className="flex items-center gap-2.5 text-[10px] shrink-0">
          {/* YES / NO prices */}
          {yesBid != null && (
            <span className={metricColor}>
              Y <span className="font-mono text-emerald-400">{yesBid}c</span>
            </span>
          )}
          {yesBid != null && (
            <span className={metricDimColor}>
              N <span className="font-mono text-rose-400">{100 - (yesAsk || 0)}c</span>
            </span>
          )}
          {/* Spread */}
          {spread != null && (
            <span className={`font-mono ${
              spread <= 3 ? 'text-emerald-500/70' : spread <= 7 ? 'text-amber-500/70' : 'text-rose-500/70'
            }`}>
              Sp {spread}c
            </span>
          )}
          {/* Flow ratio mini-bar */}
          {totalTrades > 0 && (
            <div className="flex items-center gap-1">
              <div className="w-8 h-1.5 rounded-full overflow-hidden bg-gray-700/50 flex">
                <div
                  className="h-full bg-emerald-500/70"
                  style={{ width: `${yesRatio * 100}%` }}
                />
                <div
                  className="h-full bg-rose-500/70"
                  style={{ width: `${(1 - yesRatio) * 100}%` }}
                />
              </div>
              <span className={`font-mono ${metricDimColor}`}>{totalTrades}</span>
            </div>
          )}
          {/* Price movement */}
          {priceDrop !== 0 && (
            <span className={`font-mono ${priceDrop > 0 ? 'text-rose-400/70' : 'text-emerald-400/70'}`}>
              {priceDrop > 0 ? '-' : '+'}{Math.abs(priceDrop)}c
            </span>
          )}
          {/* Volume */}
          {displayVol > 0 && (
            <span className={`font-mono ${metricDimColor}`}>
              {displayVol > 1000 ? `${(displayVol / 1000).toFixed(1)}k` : displayVol}
            </span>
          )}
          {/* Time to close */}
          {timeToClose && (
            <span className={`${metricDimColor} flex items-center gap-0.5`}>
              <Clock className="w-2.5 h-2.5" />{timeToClose}
            </span>
          )}
        </div>
      </div>

      {/* Signal summary - only for markets with signals */}
      {hasSignals && (
        <div className="flex items-center gap-2 mt-1.5">
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

      {/* Recent extraction snippets - only for markets with signals */}
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
          <span className={`font-bold ${position.market_position > 0 ? 'text-emerald-400' : position.market_position < 0 ? 'text-rose-400' : 'text-gray-400'}`}>
            {position.total_contracts || Math.abs(position.market_position || 0)} {position.position_side || (position.market_position > 0 ? 'YES' : 'NO')}
          </span>
          {position.realized_pnl != null && (
            <span className={`font-mono ${position.realized_pnl >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
              {position.realized_pnl >= 0 ? '+' : ''}${(position.realized_pnl / 100).toFixed(2)}
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
const EventCard = memo(({ config, eventExtractions, aggregatedSignals, marketPrices, positionsDetails, tradeFlowStates }) => {
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

  // Event-level flow stats from tradeFlowStates
  const eventFlowStats = useMemo(() => {
    let yesTrades = 0, noTrades = 0, activeCount = 0;
    const spreads = [];
    for (const m of markets) {
      const ticker = typeof m === 'string' ? m : m.ticker;
      const flow = tradeFlowStates?.[ticker];
      if (flow && flow.total_trades > 0) {
        yesTrades += flow.yes_trades || 0;
        noTrades += flow.no_trades || 0;
        activeCount++;
      }
      const sp = marketPrices?.[ticker]?.spread;
      if (sp != null) spreads.push(sp);
    }
    const total = yesTrades + noTrades;
    const yesRatio = total > 0 ? yesTrades / total : 0;
    const avgSpread = spreads.length > 0 ? Math.round(spreads.reduce((a, b) => a + b, 0) / spreads.length) : null;
    return { yesTrades, noTrades, total, yesRatio, activeCount, avgSpread };
  }, [markets, tradeFlowStates, marketPrices]);

  return (
    <div data-testid={`event-card-${eventTicker}`} className="bg-gray-900/50 rounded-xl border border-gray-800/50 overflow-hidden">
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
          {/* Event Summary Bar */}
          <div className="flex items-center gap-3 px-3 py-1.5 bg-gray-800/20 rounded-lg border border-gray-800/30 text-[10px]">
            {/* YES sum + risk badge */}
            {yesSum > 0 && (
              <span className="text-gray-500">
                Sum <span className="font-mono text-gray-400">{yesSum}c</span>
                {' '}<span className={`font-bold ${riskColor}`}>[{riskLevel}]</span>
              </span>
            )}
            <span className="text-gray-700">|</span>
            {/* Total volume */}
            <span className="text-gray-500">
              Vol <span className="font-mono text-gray-400">
                {totalVolume > 1000 ? `${(totalVolume / 1000).toFixed(1)}k` : totalVolume || '0'}
              </span>
            </span>
            <span className="text-gray-700">|</span>
            {/* Flow ratio bar */}
            <span className="flex items-center gap-1 text-gray-500">
              Flow
              {eventFlowStats.total > 0 ? (
                <>
                  <div className="w-12 h-1.5 rounded-full overflow-hidden bg-gray-700/50 flex">
                    <div className="h-full bg-emerald-500/70" style={{ width: `${eventFlowStats.yesRatio * 100}%` }} />
                    <div className="h-full bg-rose-500/70" style={{ width: `${(1 - eventFlowStats.yesRatio) * 100}%` }} />
                  </div>
                  <span className="font-mono text-gray-400">{Math.round(eventFlowStats.yesRatio * 100)}%Y</span>
                </>
              ) : (
                <span className="text-gray-600">--</span>
              )}
            </span>
            <span className="text-gray-700">|</span>
            {/* Active markets */}
            <span className="text-gray-500">
              Active <span className="font-mono text-gray-400">{eventFlowStats.activeCount}/{markets.length}</span>
            </span>
            {/* Avg spread */}
            {eventFlowStats.avgSpread != null && (
              <>
                <span className="text-gray-700">|</span>
                <span className="text-gray-500">
                  Avg Sp <span className={`font-mono ${
                    eventFlowStats.avgSpread <= 3 ? 'text-emerald-500/70' : eventFlowStats.avgSpread <= 7 ? 'text-amber-500/70' : 'text-rose-500/70'
                  }`}>{eventFlowStats.avgSpread}c</span>
                </span>
              </>
            )}
          </div>

          {sortedMarkets.map((market) => {
            const ticker = typeof market === 'string' ? market : market.ticker;
            const sig = aggregatedSignals[ticker];
            return (
              <EventMarketRow
                key={ticker}
                market={market}
                marketPrices={marketPrices}
                signalAgg={sig}
                position={positionLookup[ticker]}
                tradeFlow={tradeFlowStates?.[ticker]}
              />
            );
          })}
        </div>
      )}
    </div>
  );
});

EventCard.displayName = 'EventCard';

/**
 * EventIntelligenceBoard - Scrollable list of EventCards
 */
const EventIntelligenceBoard = memo(({ eventConfigs, byEvent, aggregatedSignals, marketPrices, positionsDetails, tradeFlowStates }) => {
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
    <div data-testid="event-intelligence-board" className="bg-gray-900/50 rounded-2xl border border-gray-800/50 p-6">
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

      <div className="space-y-3 max-h-[600px] overflow-y-auto pr-1 scrollbar-dark">
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
              tradeFlowStates={tradeFlowStates}
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
const AgentStatusHeader = memo(({ agentState, settlements, trades, watchdog, errors = [], cycleCountdown }) => {
  const isRunning = agentState.status === 'active' || agentState.status === 'started';

  const totalPnL = useMemo(() => {
    // 1. Settlement P&L (from market settlements / reflections)
    const settlementPnl = settlements.reduce((sum, s) => sum + (s.pnlCents || 0), 0);

    // 2. Trade-based realized P&L: match SELL trades to BUY trades on same ticker.
    //    For each SELL, find the most recent BUY on the same ticker+side to compute
    //    the entry-to-exit price difference. Skip tickers already counted in settlements.
    const settledTickers = new Set(settlements.map(s => s.ticker));
    const buysByTickerSide = {};
    const sellTrades = [];

    for (const t of trades) {
      const key = `${t.ticker}_${t.side}`;
      if (t.action === 'sell') {
        sellTrades.push(t);
      } else {
        // BUY trade - store for matching
        if (!buysByTickerSide[key]) buysByTickerSide[key] = [];
        buysByTickerSide[key].push(t);
      }
    }

    let tradePnl = 0;
    for (const sell of sellTrades) {
      if (settledTickers.has(sell.ticker)) continue; // Already counted in settlements
      const key = `${sell.ticker}_${sell.side}`;
      const buys = buysByTickerSide[key];
      if (!buys || buys.length === 0) continue;

      // Match with the earliest unmatched BUY for this ticker+side
      const buy = buys.shift();
      const buyPrice = buy.priceCents || buy.limitPriceCents || 0;
      const sellPrice = sell.priceCents || sell.limitPriceCents || 0;
      const matchedContracts = Math.min(sell.contracts || 0, buy.contracts || 0);

      if (buyPrice > 0 && sellPrice > 0 && matchedContracts > 0) {
        // P&L = (sell_price - buy_price) * contracts for long positions
        tradePnl += (sellPrice - buyPrice) * matchedContracts;
      }
    }

    return (settlementPnl + tradePnl) / 100;
  }, [settlements, trades]);

  const pnlColor = totalPnL >= 0 ? 'text-emerald-400' : 'text-rose-400';
  const pnlBg = totalPnL >= 0 ? 'bg-emerald-900/20' : 'bg-rose-900/20';

  return (
    <div data-testid="agent-status-header" className="flex items-center justify-between p-3 bg-gray-800/50 rounded-xl border border-gray-700/40">
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

        <div className="flex items-center gap-2 text-xs text-gray-400" data-testid="agent-cycle-count">
          <Clock className="w-3.5 h-3.5" />
          <span>Cycle <span className="font-mono text-white">{agentState.cycleCount}</span></span>
          {cycleCountdown != null && cycleCountdown > 0 && (
            <span className="font-mono text-gray-500">
              ({Math.floor(cycleCountdown / 60)}:{String(cycleCountdown % 60).padStart(2, '0')})
            </span>
          )}
        </div>

        <div className="flex items-center gap-2 text-xs text-gray-400" data-testid="agent-trade-count">
          <BarChart3 className="w-3.5 h-3.5" />
          <span>Trades <span className="font-mono text-white">{trades.length}</span></span>
        </div>

        {/* Watchdog indicators */}
        {watchdog?.permanentlyStopped && (
          <div className="flex items-center gap-1.5 px-2 py-1 rounded-lg bg-rose-900/30 border border-rose-600/40">
            <ShieldOff className="w-3.5 h-3.5 text-rose-400" />
            <span className="text-[10px] font-bold text-rose-400">CIRCUIT BREAK</span>
          </div>
        )}
        {watchdog && !watchdog.permanentlyStopped && watchdog.restartsThisHour > 0 && (
          <div className="flex items-center gap-1.5 px-2 py-1 rounded-lg bg-amber-900/30 border border-amber-600/40">
            <AlertCircle className="w-3.5 h-3.5 text-amber-400" />
            <span className="text-[10px] font-bold text-amber-400">
              {watchdog.restartsThisHour}/{watchdog.maxRestartsPerHour}
            </span>
          </div>
        )}
        {errors.length > 0 && !watchdog?.permanentlyStopped && !(watchdog?.restartsThisHour > 0) && (
          <div className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-rose-900/20 border border-rose-700/30">
            <span className="text-[9px] font-bold text-rose-400">{errors.length} err</span>
          </div>
        )}
      </div>

      <div data-testid="agent-pnl-value" className={`flex items-center gap-2 px-3 py-1.5 rounded-lg ${pnlBg}`}>
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
const ThinkingStream = memo(({ thinking, isRunning, activeToolCall }) => {
  // No thinking text — show active tool or idle state
  if (!thinking.text) {
    if (activeToolCall) {
      return (
        <div data-testid="agent-thinking-stream" className="p-4 bg-gradient-to-br from-cyan-900/20 to-cyan-950/10 rounded-xl border border-cyan-700/30">
          <div className="flex items-center gap-2 mb-2">
            <Wrench className="w-4 h-4 text-cyan-400 animate-pulse" />
            <span className="text-xs font-semibold text-cyan-300 uppercase tracking-wider">Executing Tool</span>
            <span className="ml-auto text-[10px] text-gray-500 font-mono">Cycle {activeToolCall.cycle}</span>
          </div>
          <div className="flex items-center gap-2">
            <RefreshCw className="w-3.5 h-3.5 text-cyan-400 animate-spin" />
            <span className="text-sm font-mono text-cyan-300">{activeToolCall.tool}()</span>
          </div>
        </div>
      );
    }
    return (
      <div data-testid="agent-thinking-stream" className="p-4 bg-gray-800/30 rounded-xl border border-gray-700/30">
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
    <div data-testid="agent-thinking-stream" className="p-4 bg-gradient-to-br from-violet-900/20 to-violet-950/10 rounded-xl border border-violet-700/30">
      <div className="flex items-center gap-2 mb-2">
        <Brain className="w-4 h-4 text-violet-400 animate-pulse" />
        <span className="text-xs font-semibold text-violet-300 uppercase tracking-wider">Thinking</span>
        <span className="ml-auto text-[10px] text-gray-500 font-mono">Cycle {thinking.cycle}</span>
      </div>
      <div data-testid="agent-thinking-text" className="text-sm text-gray-200 leading-relaxed">
        {renderThinkingMarkdown(thinking.text)}
        {thinking.streaming && <span className="inline-block w-0.5 h-4 bg-violet-400 ml-0.5 animate-pulse align-middle" />}
      </div>
      {activeToolCall && (
        <div className="mt-2 flex items-center gap-2 px-2 py-1.5 bg-cyan-900/20 rounded-lg border border-cyan-800/30">
          <RefreshCw className="w-3 h-3 text-cyan-400 animate-spin" />
          <span className="text-[10px] font-mono text-cyan-300">{activeToolCall.tool}()</span>
        </div>
      )}
      {isRunning && !activeToolCall && !thinking.streaming && (
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
    <div data-testid="agent-tool-calls-panel" className="bg-gray-900/50 rounded-xl border border-gray-800/50 overflow-hidden">
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="w-full flex items-center justify-between p-3 hover:bg-gray-800/30 transition-colors"
      >
        <div className="flex items-center gap-2">
          <Wrench className="w-4 h-4 text-cyan-400" />
          <span className="text-xs font-semibold text-gray-300">Tool Calls</span>
          <span data-testid="agent-tool-calls-count" className="px-2 py-0.5 bg-cyan-900/30 text-cyan-400 text-[10px] font-bold rounded-full">
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
  // Use priceCents (avg fill price) when available, fall back to limitPriceCents
  const displayPrice = trade.priceCents || trade.limitPriceCents || 0;
  const notionalCost = displayPrice > 0 ? ((trade.contracts * displayPrice) / 100).toFixed(2) : '—';

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
        <span className="text-[10px] text-gray-500">{trade.contracts}x @ {displayPrice}c</span>
      </div>
      <span className="text-[10px] text-gray-500">${notionalCost}</span>
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

  // Filter out trades that already have a corresponding settlement to avoid visual duplicates
  const settledTickers = useMemo(() => {
    const set = new Set();
    for (const s of settlements) {
      set.add(s.ticker);
    }
    return set;
  }, [settlements]);

  // Show unsettled trades only when there are settlements to avoid confusion
  const displayTrades = useMemo(() => {
    if (settlements.length === 0) return trades;
    return trades.filter(t => !settledTickers.has(t.ticker));
  }, [trades, settlements, settledTickers]);

  return (
    <div data-testid="agent-trades-panel" className="bg-gray-900/50 rounded-xl border border-gray-800/50 overflow-hidden">
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
              {displayTrades.slice(0, 10).map((t) => (<TradeCard key={t.id} trade={t} />))}
            </>
          )}
        </div>
      )}
    </div>
  );
});

TradesPanel.displayName = 'TradesPanel';

/**
 * Memory Panel - Tabbed view of all memory file updates
 */
const MEMORY_TABS = [
  { key: 'learnings.md', label: 'Learnings', icon: BookOpen, color: 'amber' },
  { key: 'strategy.md', label: 'Strategy', icon: Target, color: 'violet' },
  { key: 'mistakes.md', label: 'Mistakes', icon: AlertCircle, color: 'rose' },
  { key: 'patterns.md', label: 'Patterns', icon: TrendingUp, color: 'emerald' },
  { key: 'cycle_journal.md', label: 'Journal', icon: FileText, color: 'blue' },
  { key: 'golden_rules.md', label: 'Rules', icon: Zap, color: 'yellow' },
];

const TAB_COLORS = {
  amber:   { active: 'bg-amber-900/40 text-amber-300 border-amber-600/40', badge: 'bg-amber-900/30 text-amber-400' },
  violet:  { active: 'bg-violet-900/40 text-violet-300 border-violet-600/40', badge: 'bg-violet-900/30 text-violet-400' },
  rose:    { active: 'bg-rose-900/40 text-rose-300 border-rose-600/40', badge: 'bg-rose-900/30 text-rose-400' },
  emerald: { active: 'bg-emerald-900/40 text-emerald-300 border-emerald-600/40', badge: 'bg-emerald-900/30 text-emerald-400' },
  blue:    { active: 'bg-blue-900/40 text-blue-300 border-blue-600/40', badge: 'bg-blue-900/30 text-blue-400' },
  yellow:  { active: 'bg-yellow-900/40 text-yellow-300 border-yellow-600/40', badge: 'bg-yellow-900/30 text-yellow-400' },
};

/**
 * TodoPanel - Agent's self-managed task list
 */
const PRIORITY_STYLES = {
  high: { dot: 'text-rose-400', bg: 'bg-rose-900/20 text-rose-300 border-rose-700/30' },
  medium: { dot: 'text-amber-400', bg: 'bg-amber-900/20 text-amber-300 border-amber-700/30' },
  low: { dot: 'text-gray-500', bg: 'bg-gray-800/30 text-gray-400 border-gray-700/30' },
};

const TodoPanel = memo(({ todos }) => {
  if (!todos || todos.length === 0) return null;

  const pending = todos.filter(t => t.status !== 'done');
  const done = todos.filter(t => t.status === 'done');

  return (
    <div className="bg-gray-900/50 rounded-xl border border-gray-800/50 p-4">
      <div className="flex items-center gap-2 mb-3">
        <ListChecks className="w-4 h-4 text-cyan-400" />
        <span className="text-sm font-semibold text-gray-300">Agent TODO List</span>
        <span className="px-2 py-0.5 bg-cyan-900/30 text-cyan-400 text-[10px] font-bold rounded-full">
          {pending.length} pending
        </span>
        {done.length > 0 && (
          <span className="px-2 py-0.5 bg-emerald-900/30 text-emerald-400 text-[10px] font-bold rounded-full">
            {done.length} done
          </span>
        )}
      </div>
      <div className="space-y-1.5">
        {pending.map((item, i) => {
          const style = PRIORITY_STYLES[item.priority] || PRIORITY_STYLES.medium;
          return (
            <div key={`todo-${i}`} className="flex items-start gap-2 group">
              <Circle className={`w-3.5 h-3.5 mt-0.5 flex-shrink-0 ${style.dot}`} />
              <span className="text-xs text-gray-300 leading-relaxed flex-1">{item.task}</span>
              <span className={`text-[9px] px-1.5 py-0.5 rounded border flex-shrink-0 ${style.bg}`}>
                {item.priority}
              </span>
            </div>
          );
        })}
        {done.map((item, i) => (
          <div key={`done-${i}`} className="flex items-start gap-2 opacity-50">
            <CheckCircle2 className="w-3.5 h-3.5 mt-0.5 flex-shrink-0 text-emerald-500" />
            <span className="text-xs text-gray-500 leading-relaxed line-through flex-1">{item.task}</span>
          </div>
        ))}
      </div>
    </div>
  );
});

const MemoryPanel = memo(({ memoryByFile }) => {
  const [collapsed, setCollapsed] = useState(true);
  const [activeTab, setActiveTab] = useState('learnings.md');

  const totalCount = useMemo(() =>
    Object.values(memoryByFile).reduce((sum, arr) => sum + arr.length, 0),
    [memoryByFile],
  );

  const entries = memoryByFile[activeTab] || [];
  const activeTabMeta = MEMORY_TABS.find(t => t.key === activeTab);

  return (
    <div data-testid="agent-memory-panel" className="bg-gray-900/50 rounded-xl border border-gray-800/50 overflow-hidden">
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="w-full flex items-center justify-between p-3 hover:bg-gray-800/30 transition-colors"
      >
        <div className="flex items-center gap-2">
          <BookOpen className="w-4 h-4 text-violet-400" />
          <span className="text-xs font-semibold text-gray-300">Memory</span>
          {totalCount > 0 && (
            <span className="px-2 py-0.5 bg-violet-900/30 text-violet-400 text-[10px] font-bold rounded-full">{totalCount}</span>
          )}
        </div>
        {collapsed ? <ChevronRight className="w-4 h-4 text-gray-500" /> : <ChevronDown className="w-4 h-4 text-gray-500" />}
      </button>
      {!collapsed && (
        <div className="px-3 pb-3">
          {/* Tab bar */}
          <div className="flex gap-1 mb-3 overflow-x-auto pb-1">
            {MEMORY_TABS.map(tab => {
              const count = (memoryByFile[tab.key] || []).length;
              const isActive = activeTab === tab.key;
              const colors = TAB_COLORS[tab.color];
              const Icon = tab.icon;
              return (
                <button
                  key={tab.key}
                  onClick={() => setActiveTab(tab.key)}
                  className={`flex items-center gap-1 px-2 py-1 text-[10px] font-medium rounded-md whitespace-nowrap transition-colors border ${
                    isActive
                      ? colors.active
                      : 'border-transparent text-gray-500 hover:text-gray-300 hover:bg-gray-800/50'
                  }`}
                >
                  <Icon className="w-3 h-3" />
                  {tab.label}
                  {count > 0 && (
                    <span className={`ml-0.5 px-1 py-px text-[9px] font-mono rounded ${isActive ? colors.badge : 'text-gray-500'}`}>
                      {count}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
          {/* Tab content */}
          <div className="space-y-2 max-h-[250px] overflow-y-auto scrollbar-dark">
            {entries.length === 0 ? (
              <div className="text-center py-4 text-gray-500 text-xs">No updates this session</div>
            ) : (
              entries.slice(0, 20).map(entry => (
                <div key={entry.id} className="p-2 bg-gray-800/40 rounded-lg">
                  <div className="text-xs text-gray-300">{entry.contentPreview || entry.content}</div>
                  {entry.timestamp && (
                    <div className="text-[9px] text-gray-600 mt-1">
                      {new Date(entry.timestamp).toLocaleTimeString()}
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
});

MemoryPanel.displayName = 'MemoryPanel';

/**
 * ErrorHistoryPanel - Collapsible list of recent agent errors
 */
const ErrorHistoryPanel = memo(({ errors }) => {
  const [collapsed, setCollapsed] = useState(true);

  if (errors.length === 0) return null;

  return (
    <div data-testid="agent-error-history-panel" className="bg-gray-900/50 rounded-xl border border-rose-800/30 overflow-hidden">
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="w-full flex items-center justify-between p-3 hover:bg-gray-800/30 transition-colors"
      >
        <div className="flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-rose-400" />
          <span className="text-xs font-semibold text-gray-300">Errors</span>
          <span className="px-2 py-0.5 bg-rose-900/30 text-rose-400 text-[10px] font-bold rounded-full">{errors.length}</span>
        </div>
        {collapsed ? <ChevronRight className="w-4 h-4 text-gray-500" /> : <ChevronDown className="w-4 h-4 text-gray-500" />}
      </button>
      {!collapsed && (
        <div className="px-3 pb-3 space-y-1.5 max-h-[200px] overflow-y-auto">
          {errors.map((error) => (
            <div key={error.id} className="flex items-start gap-2 p-2 bg-gray-800/40 rounded-lg">
              <span className={`px-1 py-0.5 text-[8px] font-bold rounded border flex-shrink-0 mt-0.5 ${
                error.severity === 'critical'
                  ? 'bg-rose-900/30 text-rose-400 border-rose-600/30'
                  : 'bg-amber-900/30 text-amber-400 border-amber-600/30'
              }`}>
                {error.severity === 'critical' ? 'CRIT' : 'WARN'}
              </span>
              <div className="min-w-0 flex-1">
                <div className="text-[11px] text-gray-300 break-words">{error.message}</div>
                <div className="text-[9px] text-gray-600 mt-0.5">{error.timestamp}</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
});

ErrorHistoryPanel.displayName = 'ErrorHistoryPanel';

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
  const isIntel = result.source === 'news_intelligence';
  const tone = result.toneSummary;
  const avgTone = tone?.avg_tone || 0;
  const toneColor = avgTone > 1.5 ? 'text-emerald-400' : avgTone < -1.5 ? 'text-rose-400' : 'text-gray-400';
  const toneBg = avgTone > 1.5 ? 'bg-emerald-900/20' : avgTone < -1.5 ? 'bg-rose-900/20' : 'bg-gray-800/30';

  // News intelligence data (only populated for source === 'news_intelligence')
  const intel = result.intelligence || {};
  const intelSentiment = intel.sentiment || {};
  const intelFreshness = intel.freshness || {};
  const intelSources = intel.source_analysis || {};
  const intelDevs = intel.key_developments || [];

  // Sentiment label and styling for news_intelligence collapsed row
  const sentimentLabel = {
    strongly_positive: 'Positive', positive: 'Positive',
    strongly_negative: 'Negative', negative: 'Negative',
    mixed: 'Mixed', neutral: 'Neutral',
  }[intelSentiment.overall] || '';
  const sentimentColor = {
    strongly_positive: 'text-emerald-400', positive: 'text-emerald-400',
    strongly_negative: 'text-rose-400', negative: 'text-rose-400',
    mixed: 'text-amber-400', neutral: 'text-gray-400',
  }[intelSentiment.overall] || 'text-gray-400';
  const sentimentBg = {
    strongly_positive: 'bg-emerald-900/20', positive: 'bg-emerald-900/20',
    strongly_negative: 'bg-rose-900/20', negative: 'bg-rose-900/20',
    mixed: 'bg-amber-900/20', neutral: 'bg-gray-800/30',
  }[intelSentiment.overall] || 'bg-gray-800/30';
  const trendArrow = { improving: '\u2197', stable: '\u2192', deteriorating: '\u2198' }[intelSentiment.trend] || '';

  // Source label mapping
  const sourceLabel = { gkg: 'GKG', doc_api: 'DOC', volume_timeline: 'TREND', events: 'EVENTS', news_intelligence: 'INTEL' }[result.source] || '';
  const sourceColor = { gkg: 'text-amber-500 border-amber-700/30', doc_api: 'text-blue-400 border-blue-700/30', volume_timeline: 'text-purple-400 border-purple-700/30', events: 'text-red-400 border-red-700/30', news_intelligence: 'text-teal-400 border-teal-700/30' }[result.source] || 'text-gray-400 border-gray-600/30';

  return (
    <div className={`bg-gray-800/30 rounded-lg border overflow-hidden ${isIntel ? 'border-teal-800/30' : 'border-gray-700/30'}`}>
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-3 py-2.5 hover:bg-gray-800/50 transition-colors"
      >
        <div className="flex items-center gap-2 min-w-0">
          {(() => {
            const iconMap = {
              events: [Activity, 'text-red-400'],
              volume_timeline: [TrendingUp, 'text-purple-400'],
              news_intelligence: [Sparkles, 'text-teal-400'],
            };
            const [Icon, color] = iconMap[result.source] || [Globe, 'text-blue-400'];
            return <Icon className={`w-3.5 h-3.5 ${color} flex-shrink-0`} />;
          })()}
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
          {isIntel ? (
            /* --- News Intelligence collapsed badges --- */
            <>
              {/* Article count */}
              <span className={`px-1.5 py-0.5 text-[9px] font-bold rounded ${
                result.articleCount >= 5
                  ? 'bg-teal-900/30 text-teal-400 border border-teal-600/30'
                  : result.articleCount > 0
                    ? 'bg-gray-700/50 text-gray-300 border border-gray-600/30'
                    : 'bg-gray-800/50 text-gray-500 border border-gray-700/30'
              }`}>
                {result.articleCount} articles
              </span>
              {/* Sentiment + trend arrow */}
              {sentimentLabel && (
                <span className={`flex items-center gap-0.5 px-1.5 py-0.5 text-[9px] rounded ${sentimentBg} ${sentimentColor}`}>
                  {sentimentLabel}
                  {trendArrow && <span className="ml-0.5">{trendArrow}</span>}
                </span>
              )}
              {/* Freshness: breaking or age */}
              {intelFreshness.is_breaking ? (
                <span className="px-1.5 py-0.5 text-[9px] font-bold rounded bg-rose-900/30 text-rose-400 border border-rose-600/30 animate-pulse">
                  BREAKING
                </span>
              ) : intelFreshness.newest_article_age_minutes != null && (
                <span className="text-[9px] text-gray-500 flex items-center gap-0.5">
                  <Clock className="w-2.5 h-2.5" />
                  {intelFreshness.newest_article_age_minutes < 60
                    ? `${intelFreshness.newest_article_age_minutes}m ago`
                    : `${Math.round(intelFreshness.newest_article_age_minutes / 60)}h ago`}
                </span>
              )}
            </>
          ) : isTimeline ? (
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

          {/* === News Intelligence expanded content === */}
          {isIntel && (
            <>
              {/* Narrative summary */}
              {intel.narrative_summary && (
                <div className="text-[11px] text-gray-300 leading-relaxed">
                  {intel.narrative_summary}
                </div>
              )}

              {/* Key developments */}
              {intelDevs.length > 0 && (
                <div>
                  <span className="text-[9px] text-gray-500 uppercase tracking-wider">Key Developments</span>
                  <div className="mt-1.5 space-y-1.5">
                    {intelDevs.slice(0, 5).map((dev, i) => {
                      const recencyStyle = {
                        breaking: 'bg-rose-900/30 text-rose-400 border-rose-600/30',
                        recent: 'bg-amber-900/30 text-amber-400 border-amber-600/30',
                        developing: 'bg-blue-900/30 text-blue-400 border-blue-600/30',
                        stale: 'bg-gray-700/50 text-gray-500 border-gray-600/30',
                      }[dev.recency] || 'bg-gray-700/50 text-gray-500 border-gray-600/30';
                      return (
                        <div key={i} className="flex items-start gap-2">
                          <span className={`px-1.5 py-0.5 text-[8px] font-bold rounded border flex-shrink-0 mt-0.5 ${recencyStyle}`}>
                            {(dev.recency || 'unknown').toUpperCase()}
                          </span>
                          <div className="flex-1 min-w-0">
                            <span className="text-[10px] text-gray-300">{dev.headline}</span>
                            {dev.source_count > 0 && (
                              <span className="text-[9px] text-gray-600 ml-1.5">
                                {dev.source_count} sources
                              </span>
                            )}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Source analysis */}
              {(intelSources.total_articles > 0 || intelSources.notable_sources?.length > 0) && (
                <div>
                  <span className="text-[9px] text-gray-500 uppercase tracking-wider">Sources</span>
                  <div className="mt-1 flex items-center flex-wrap gap-1.5">
                    {intelSources.total_articles > 0 && (
                      <span className="text-[10px] text-gray-400">
                        {intelSources.total_articles} articles from {intelSources.unique_sources || 0} sources
                      </span>
                    )}
                    {intelSources.geographic_spread && intelSources.geographic_spread !== 'domestic' && (
                      <span className="px-1.5 py-0.5 text-[8px] bg-blue-900/20 text-blue-400 rounded border border-blue-700/30">
                        {intelSources.geographic_spread}
                      </span>
                    )}
                  </div>
                  {intelSources.notable_sources?.length > 0 && (
                    <div className="mt-1 flex flex-wrap gap-1">
                      {intelSources.notable_sources.slice(0, 6).map((src, i) => (
                        <span key={i} className="px-1.5 py-0.5 text-[8px] bg-gray-700/50 text-gray-400 rounded">
                          {src}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Freshness detail */}
              <div className="flex items-center gap-3 text-[10px]">
                {intelFreshness.newest_article_age_minutes != null && (
                  <span className="text-gray-400 flex items-center gap-1">
                    <Clock className="w-3 h-3 text-gray-500" />
                    Newest: {intelFreshness.newest_article_age_minutes < 60
                      ? `${intelFreshness.newest_article_age_minutes}m ago`
                      : `${Math.round(intelFreshness.newest_article_age_minutes / 60)}h ago`}
                  </span>
                )}
                {intelFreshness.volume_trend && (
                  <span className={`flex items-center gap-0.5 ${
                    intelFreshness.volume_trend === 'surging' ? 'text-emerald-400'
                    : intelFreshness.volume_trend === 'declining' ? 'text-amber-400'
                    : 'text-gray-500'
                  }`}>
                    {intelFreshness.volume_trend === 'surging' ? '\u2191' : intelFreshness.volume_trend === 'declining' ? '\u2193' : '\u2192'}
                    {' '}{intelFreshness.volume_trend}
                  </span>
                )}
                {intelFreshness.coverage_window_hours > 0 && (
                  <span className="text-gray-600">
                    {intelFreshness.coverage_window_hours}h window
                  </span>
                )}
                {intelSentiment.confidence && (
                  <span className={`flex items-center gap-1 ${
                    intelSentiment.confidence === 'high' ? 'text-emerald-400'
                    : intelSentiment.confidence === 'low' ? 'text-gray-600'
                    : 'text-gray-400'
                  }`}>
                    <Circle className={`w-1.5 h-1.5 fill-current`} />
                    {intelSentiment.confidence} confidence
                  </span>
                )}
              </div>

              {/* Duration */}
              {result.durationMs != null && (
                <div className="text-[9px] text-gray-600">
                  Analysis took {result.durationMs >= 1000 ? `${(result.durationMs / 1000).toFixed(1)}s` : `${result.durationMs}ms`}
                </div>
              )}
            </>
          )}

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

          {/* === News-specific expanded content (non-events, non-timeline, non-intel) === */}

          {/* Tone breakdown */}
          {!isTimeline && !isEvents && !isIntel && (tone?.positive_count > 0 || tone?.negative_count > 0) && (
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
          {!isEvents && !isIntel && (result.keyPersons?.length > 0 || result.keyOrganizations?.length > 0) && (
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
          {!isEvents && !isIntel && result.keyThemes?.length > 0 && (
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
          {!isEvents && !isIntel && result.topArticles?.length > 0 && (
            <div>
              <span className="text-[9px] text-gray-500 uppercase tracking-wider">Top Articles</span>
              <div className="mt-1">
                {result.topArticles.slice(0, 5).map((article, i) => (
                  <GdeltArticleRow key={i} article={article} />
                ))}
              </div>
            </div>
          )}

          {/* Duration (for non-intel sources) */}
          {!isIntel && result.durationMs != null && (
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
    <div data-testid="agent-gdelt-panel" className="bg-gray-900/50 rounded-2xl border border-blue-800/30 overflow-hidden">
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
    activeToolCall,
    toolCalls,
    trades,
    settlements,
    learnings,
    memoryByFile,
    errors,
    memoryUpdates,
    processMessage,
    isRunning: agentIsRunning,
    gdeltResults,
    todos,
    redditHistoricDigest,
    watchdog,
    newWatchdogEvent,
    dismissWatchdogEvent,
    cycleCountdown,
  } = useDeepAgent();

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
    tradeFlowStates,
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
    <div data-testid="agent-page" className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950">
      <WatchdogToast event={newWatchdogEvent} onDismiss={dismissWatchdogEvent} />

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

            <div data-testid="agent-running-status" className={`
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

          <div className="flex items-center gap-3 overflow-x-auto pb-2">
            <PipelineStage
              icon={MessageSquare}
              title="Sources"
              count={entityRedditPosts.length}
              color="orange"
              isActive={entityRedditPosts.length > 0}
              description="Posts ingested"
              testId="pipeline-sources"
            />
            <ArrowRight className="w-5 h-5 text-gray-600 flex-shrink-0" />
            <PipelineStage
              icon={Zap}
              title="Extractions"
              count={extractionStats.totalExtractions}
              color="violet"
              isActive={extractions.length > 0}
              description={`${extractionStats.marketSignalCount} signals, ${extractionStats.entityMentionCount} entities`}
              testId="pipeline-extractions"
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
              testId="pipeline-markets"
            />
            <ArrowRight className="w-5 h-5 text-gray-600 flex-shrink-0" />
            <PipelineStage
              icon={Brain}
              title="Agent Cycles"
              count={agentState.cycleCount || 0}
              color="emerald"
              isActive={isAgentRunning}
              description="Deep agent cycles"
              testId="pipeline-agent-cycles"
            />
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-12 gap-6">
          {/* Left Column - Source Feed + Extraction Stream */}
          <div className="col-span-4 space-y-6">
            {/* Reddit Historic Digest Summary */}
            {redditHistoricDigest && redditHistoricDigest.status === 'completed' && (
              <div className="bg-gray-900/50 rounded-2xl border border-orange-800/30 p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Newspaper className="w-4 h-4 text-orange-400" />
                  <span className="text-sm font-semibold text-gray-300">Reddit Daily Digest</span>
                  <span className="px-2 py-0.5 bg-orange-900/30 text-orange-400 text-[10px] font-bold rounded-full border border-orange-600/30">
                    {redditHistoricDigest.posts_processed} posts
                  </span>
                  <span className="px-2 py-0.5 bg-cyan-900/30 text-cyan-400 text-[10px] font-bold rounded-full border border-cyan-600/30">
                    {redditHistoricDigest.extractions_created} extractions
                  </span>
                </div>
                <div className="text-[10px] text-gray-500">
                  Completed in {redditHistoricDigest.duration_seconds}s
                  {redditHistoricDigest.timestamp && (
                    <span className="ml-2">
                      at {new Date(redditHistoricDigest.timestamp).toLocaleTimeString()}
                    </span>
                  )}
                </div>
              </div>
            )}

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
                  {(() => {
                    const healthBadges = {
                      healthy: { bg: 'bg-emerald-900/30 text-emerald-400 border-emerald-700/30', Icon: CheckCircle, label: 'Live' },
                      degraded: { bg: 'bg-amber-900/30 text-amber-400 border-amber-700/30', Icon: AlertCircle, label: 'Partial' },
                      unhealthy: { bg: 'bg-red-900/30 text-red-400 border-red-700/30', Icon: XCircle, label: 'Offline' },
                    };
                    const badge = healthBadges[redditAgentHealth.health] || { bg: 'bg-gray-800/50 text-gray-500 border-gray-700/30', Icon: Clock, label: '...' };
                    return (
                      <span className={`flex items-center gap-1 px-1.5 py-0.5 text-[9px] rounded-full border ${badge.bg}`}>
                        <badge.Icon className="w-3 h-3" />
                        <span className="font-medium">{badge.label}</span>
                      </span>
                    );
                  })()}
                </div>
                {showPosts ? <ChevronDown className="w-4 h-4 text-gray-500" /> : <ChevronRight className="w-4 h-4 text-gray-500" />}
              </button>
              {showPosts && (
                <div className="px-4 pb-4 space-y-2 max-h-[300px] overflow-y-auto scrollbar-dark">
                  {entityRedditPosts.length === 0 ? (
                    <div className="text-center py-6 text-gray-500 text-sm">
                      {(() => {
                        switch (redditAgentHealth.health) {
                          case 'unhealthy':
                            return (
                              <div className="space-y-2">
                                <XCircle className="w-8 h-8 text-red-500 mx-auto" />
                                <div className="text-red-400 font-medium">Reddit API not connected</div>
                                <div className="text-[11px] text-gray-600">
                                  {redditAgentHealth.lastError || 'Check REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET env vars'}
                                </div>
                              </div>
                            );
                          case 'degraded':
                            return (
                              <div className="space-y-2">
                                <AlertCircle className="w-8 h-8 text-amber-500 mx-auto" />
                                <div className="text-amber-400 font-medium">Partially connected</div>
                                <div className="text-[11px] text-gray-600">
                                  Reddit: yes | Extractor: {redditAgentHealth.extractorAvailable ? 'yes' : 'no'} | Supabase: {redditAgentHealth.supabaseAvailable ? 'yes' : 'no'}
                                </div>
                              </div>
                            );
                          case 'healthy':
                            return (
                              <div className="space-y-2">
                                <RefreshCw className="w-6 h-6 text-orange-400 mx-auto animate-spin" />
                                <div>Listening to r/{redditAgentHealth.subreddits?.join(', r/') || 'politics, news'}...</div>
                              </div>
                            );
                          default:
                            return (
                              <div className="space-y-2">
                                <Clock className="w-6 h-6 text-gray-500 mx-auto" />
                                <div>Waiting for Reddit posts...</div>
                              </div>
                            );
                        }
                      })()}
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
            <div className="sticky top-0 z-10 backdrop-blur-sm bg-gray-950/80 rounded-xl">
              <AgentStatusHeader
                agentState={agentState}
                settlements={settlements}
                trades={trades}
                watchdog={watchdog}
                errors={errors}
                cycleCountdown={cycleCountdown}
              />
            </div>

            {/* Agent TODO List */}
            <TodoPanel todos={todos} />

            {/* Event Intelligence Board */}
            <EventIntelligenceBoard
              eventConfigs={mergedEventConfigs}
              byEvent={byEvent}
              aggregatedSignals={aggregatedSignals}
              marketPrices={mergedMarketPrices}
              positionsDetails={tradingState?.positions_details || EMPTY_POSITIONS}
              tradeFlowStates={tradeFlowStates}
            />

            {/* GDELT News Intelligence */}
            {gdeltResults.length > 0 && (
              <GdeltNewsPanel gdeltResults={gdeltResults} />
            )}

            {/* LLM Cost Panel */}
            <CostPanel costData={agentState.costData} />

            {/* Thinking Stream */}
            <ThinkingStream thinking={thinking} isRunning={isAgentRunning} activeToolCall={activeToolCall} />

            {/* Tool Calls */}
            <ToolCallsPanel toolCalls={toolCalls} defaultCollapsed={true} />

            {/* Trades & Settlements */}
            <TradesPanel trades={trades} settlements={settlements} />

            {/* Error History */}
            <ErrorHistoryPanel errors={errors} />

            {/* Memory */}
            <MemoryPanel memoryByFile={memoryByFile} />
          </div>
        </div>
      </div>
    </div>
  );
};

/**
 * Error Boundary for AgentPage - catches render errors and displays fallback UI
 * instead of crashing the entire application.
 */
class AgentPageErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('[AgentPage] Render error caught by boundary:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-gray-950 flex items-center justify-center">
          <div className="max-w-lg mx-auto p-8 bg-gray-900/80 rounded-2xl border border-rose-700/40">
            <div className="flex items-center gap-3 mb-4">
              <AlertCircle className="w-6 h-6 text-rose-400" />
              <h2 className="text-lg font-bold text-white">Agent Page Error</h2>
            </div>
            <p className="text-sm text-gray-400 mb-4">
              The Deep Agent page encountered a rendering error. This is typically caused by
              unexpected data shapes from the WebSocket connection.
            </p>
            <pre className="text-xs text-rose-300 bg-gray-950/50 p-3 rounded-lg overflow-x-auto mb-4">
              {this.state.error?.message || 'Unknown error'}
            </pre>
            <button
              onClick={() => {
                this.setState({ hasError: false, error: null });
              }}
              className="px-4 py-2 bg-violet-600 hover:bg-violet-500 text-white text-sm font-semibold rounded-lg transition-colors"
            >
              Retry
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

const AgentPageWithBoundary = memo(() => (
  <AgentPageErrorBoundary>
    <AgentPage />
  </AgentPageErrorBoundary>
));

AgentPageWithBoundary.displayName = 'AgentPageWithBoundary';

export default AgentPageWithBoundary;
