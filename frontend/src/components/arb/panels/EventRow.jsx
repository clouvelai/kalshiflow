import React, { memo, useState, useCallback } from 'react';
import { ChevronDown, ChevronRight, Shield, ShieldOff } from 'lucide-react';
import EdgeBadge from '../ui/EdgeBadge';

/**
 * ProbSumBadge - Color-coded probability sum indicator
 *
 * Green near 100, red when deviating significantly.
 */
const ProbSumBadge = memo(({ value, label }) => {
  if (value == null) return <span className="text-[10px] text-gray-600 font-mono">--</span>;

  const deviation = Math.abs(value - 100);
  let colorClass;
  if (deviation < 3) colorClass = 'text-gray-400';
  else if (deviation < 8) colorClass = 'text-amber-400';
  else colorClass = 'text-emerald-400';

  return (
    <span className={`text-xs font-mono font-bold ${colorClass}`} title={label}>
      {typeof value === 'number' ? value.toFixed(0) : value}c
    </span>
  );
});
ProbSumBadge.displayName = 'ProbSumBadge';

/**
 * MiniDepth - Compact orderbook depth display (top 3 levels)
 */
const MiniDepth = memo(({ yesLevels = [], noLevels = [] }) => {
  if (yesLevels.length === 0 && noLevels.length === 0) return null;

  const maxSize = Math.max(
    ...yesLevels.slice(0, 3).map(l => l[1] || 0),
    ...noLevels.slice(0, 3).map(l => l[1] || 0),
    1,
  );

  return (
    <div className="grid grid-cols-2 gap-1 mb-1.5">
      {/* Bids (YES) */}
      <div className="space-y-px">
        {yesLevels.slice(0, 3).map((level, i) => (
          <div key={i} className="flex items-center gap-1 text-[9px]">
            <span className="font-mono text-cyan-400/70 w-6 text-right">{level[0]}</span>
            <div className="flex-1 h-2 bg-gray-800/50 rounded-sm overflow-hidden">
              <div
                className="h-full bg-cyan-500/20 rounded-sm"
                style={{ width: `${(level[1] / maxSize) * 100}%` }}
              />
            </div>
            <span className="font-mono text-gray-600 w-6 text-right">{level[1]}</span>
          </div>
        ))}
      </div>
      {/* Asks (NO -> derived YES asks) */}
      <div className="space-y-px">
        {noLevels.slice(0, 3).map((level, i) => (
          <div key={i} className="flex items-center gap-1 text-[9px]">
            <span className="font-mono text-gray-600 w-6 text-left">{level[1]}</span>
            <div className="flex-1 h-2 bg-gray-800/50 rounded-sm overflow-hidden">
              <div
                className="h-full bg-red-500/20 rounded-sm ml-auto"
                style={{ width: `${(level[1] / maxSize) * 100}%` }}
              />
            </div>
            <span className="font-mono text-red-400/60 w-6 text-left">{100 - level[0]}</span>
          </div>
        ))}
      </div>
    </div>
  );
});
MiniDepth.displayName = 'MiniDepth';

/**
 * MarketCard - Single market within an event.
 *
 * Shows BBO, mini depth, last trade, volume, and freshness.
 */
const MarketCard = memo(({ market }) => {
  const {
    ticker,
    title,
    yes_bid,
    yes_ask,
    spread,
    source,
    freshness_seconds,
    volume_24h,
    yes_levels,
    no_levels,
    last_trade_price,
    last_trade_side,
    trade_count,
    volume_delta_total,
  } = market;

  const fmtPrice = (cents) => cents != null ? `${cents}c` : '--';

  const fmtVol = (v) => {
    if (v == null || v <= 0) return '--';
    if (v >= 1000000) return `${(v / 1000000).toFixed(1)}M`;
    if (v >= 1000) return `${(v / 1000).toFixed(0)}k`;
    return String(v);
  };

  const freshnessColor = () => {
    if (freshness_seconds == null || freshness_seconds === Infinity) return 'text-gray-600';
    if (freshness_seconds < 5) return 'text-emerald-500';
    if (freshness_seconds < 30) return 'text-amber-500';
    return 'text-red-400';
  };

  return (
    <div className="
      bg-gradient-to-br from-gray-900/70 via-gray-900/50 to-gray-950/60
      rounded-xl border border-gray-800/50 p-3
      transition-all duration-200
      hover:border-gray-700/60 hover:bg-gray-900/60
    ">
      {/* Title + last trade */}
      <div className="flex items-start justify-between gap-2 mb-2">
        <p className="text-sm text-gray-300 leading-snug line-clamp-2 flex-1">
          {title || ticker}
        </p>
        {last_trade_price != null && (
          <span className={`text-xs font-mono font-bold flex-shrink-0 ${
            last_trade_side === 'yes' ? 'text-emerald-400' : last_trade_side === 'no' ? 'text-red-400' : 'text-gray-400'
          }`}>
            {last_trade_price}c
          </span>
        )}
      </div>

      {/* Price row */}
      <div className="grid grid-cols-3 gap-2 mb-1.5">
        <div>
          <span className="text-[9px] text-gray-500 uppercase tracking-wider font-semibold block mb-0.5">Bid</span>
          <span className="font-mono text-base text-cyan-400 font-bold">
            {fmtPrice(yes_bid)}
          </span>
        </div>
        <div>
          <span className="text-[9px] text-gray-500 uppercase tracking-wider font-semibold block mb-0.5">Ask</span>
          <span className="font-mono text-base text-cyan-400 font-bold">
            {fmtPrice(yes_ask)}
          </span>
        </div>
        <div className="flex flex-col items-center justify-center">
          <span className="text-[9px] text-gray-500 uppercase tracking-wider font-semibold block mb-0.5">Spread</span>
          <span className="font-mono text-sm text-gray-400">
            {spread != null ? `${spread}c` : '--'}
          </span>
        </div>
      </div>

      {/* Mini orderbook depth */}
      <MiniDepth yesLevels={yes_levels} noLevels={no_levels} />

      {/* Volume / activity row */}
      <div className="flex items-center gap-3 text-[10px] text-gray-500 font-mono mb-1">
        {volume_24h > 0 && (
          <span title="24h volume">Vol {fmtVol(volume_24h)}</span>
        )}
        {trade_count > 0 && (
          <span title="Trade count">{trade_count} trades</span>
        )}
        {volume_delta_total > 0 && (
          <span className="text-cyan-500/60" title="Session volume delta">+{fmtVol(volume_delta_total)}</span>
        )}
      </div>

      {/* Footer: source + ticker */}
      <div className="flex items-center justify-between text-[10px] text-gray-600 font-mono border-t border-gray-800/40 pt-1.5 mt-1">
        <div className="flex items-center gap-2">
          <span className={`uppercase ${source === 'ws' ? 'text-emerald-500' : source === 'api' ? 'text-amber-500' : 'text-gray-600'}`}>
            {source || 'none'}
          </span>
          {freshness_seconds != null && freshness_seconds !== Infinity && (
            <span className={freshnessColor()}>
              {freshness_seconds < 60 ? `${Math.round(freshness_seconds)}s` : `${Math.floor(freshness_seconds / 60)}m`}
            </span>
          )}
        </div>
        <span className="text-gray-600">{ticker}</span>
      </div>
    </div>
  );
});
MarketCard.displayName = 'MarketCard';

/**
 * EventRow - Collapsible event group with nested MarketCards.
 *
 * Shows prob sums, edge badges, data coverage, and expandable market cards.
 */
const EventRow = ({ event, isSelected, onSelectEvent }) => {
  const [expanded, setExpanded] = useState(false);

  const {
    event_ticker,
    title,
    market_count,
    markets_with_data,
    all_markets_have_data,
    sum_yes_mid,
    long_edge,
    short_edge,
    markets = [],
  } = event;

  const handleRowClick = useCallback(() => {
    if (onSelectEvent) onSelectEvent(event_ticker);
  }, [onSelectEvent, event_ticker]);

  const handleExpandClick = useCallback((e) => {
    e.stopPropagation();
    setExpanded(prev => !prev);
  }, []);

  // Which edge to display (best of long/short)
  const bestEdge = (long_edge != null && long_edge > 0) ? long_edge
    : (short_edge != null && short_edge > 0) ? short_edge
    : null;
  const edgeDirection = (long_edge != null && long_edge > 0) ? 'long'
    : (short_edge != null && short_edge > 0) ? 'short'
    : null;

  return (
    <div className={`
      border rounded-xl overflow-hidden transition-colors duration-150
      ${isSelected
        ? 'border-cyan-500/30 bg-cyan-950/10'
        : 'border-gray-800/50'
      }
    `}>
      {/* Summary Row */}
      <button
        onClick={handleRowClick}
        className={`
          w-full flex items-center gap-3 px-4 py-3
          transition-all duration-150 text-left
          ${isSelected
            ? 'bg-gradient-to-r from-cyan-900/20 to-gray-950/60 hover:from-cyan-900/30'
            : 'bg-gradient-to-r from-gray-900/80 to-gray-950/60 hover:from-gray-800/80 hover:to-gray-900/60'
          }
        `}
      >
        {/* Expand icon */}
        <span
          className="text-gray-500 flex-shrink-0 hover:text-gray-300 p-0.5"
          onClick={handleExpandClick}
          role="button"
          tabIndex={-1}
        >
          {expanded
            ? <ChevronDown className="w-4 h-4" />
            : <ChevronRight className="w-4 h-4" />
          }
        </span>

        {/* Title */}
        <span className="text-sm text-gray-200 font-medium truncate flex-1 min-w-0">
          {title || event_ticker}
        </span>

        {/* Data coverage indicator */}
        <span className="flex-shrink-0" title={all_markets_have_data ? 'All markets have data' : 'Missing data'}>
          {all_markets_have_data
            ? <Shield className="w-3.5 h-3.5 text-emerald-500/70" />
            : <ShieldOff className="w-3.5 h-3.5 text-gray-600" />
          }
        </span>

        {/* Markets with data / total */}
        <span className="text-xs font-mono flex-shrink-0 w-14 text-right">
          <span className={markets_with_data === market_count ? 'text-emerald-400' : 'text-amber-400'}>
            {markets_with_data ?? 0}
          </span>
          <span className="text-gray-600">/{market_count ?? 0}</span>
        </span>

        {/* Prob sum */}
        <span className="flex-shrink-0 w-14 text-right">
          <ProbSumBadge value={sum_yes_mid} label="Prob sum (mid)" />
        </span>

        {/* Edge badge */}
        <span className="flex-shrink-0 w-16 text-right">
          {bestEdge != null
            ? <EdgeBadge edgeCents={bestEdge} direction={edgeDirection} />
            : <span className="text-[10px] text-gray-600 font-mono">--</span>
          }
        </span>
      </button>

      {/* Expanded: MarketCards */}
      {expanded && (
        <div className="px-4 py-3 bg-gray-950/40 border-t border-gray-800/30">
          {markets.length === 0 ? (
            <p className="text-xs text-gray-600 text-center py-2">No markets in this event</p>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {markets.map(market => (
                <MarketCard key={market.ticker} market={market} />
              ))}
            </div>
          )}
          {/* Event ticker label */}
          <div className="mt-2 pt-2 border-t border-gray-800/30 flex items-center justify-between">
            <span className="font-mono text-[10px] text-gray-600">{event_ticker}</span>
            <span className="text-[10px] text-gray-600">
              {markets_with_data ?? 0}/{market_count ?? 0} with data
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default memo(EventRow);
