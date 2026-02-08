import React, { memo, useState, useCallback } from 'react';
import { ChevronDown, ChevronRight, Shield, ShieldOff } from 'lucide-react';
import EdgeBadge from '../ui/EdgeBadge';
import EventUnderstandingCard from './EventUnderstandingCard';

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
 * EventStructurePanel - Compact event context when expanded.
 *
 * Shows understanding card (compact mode) when available,
 * falls back to basic metadata display.
 */
const EventStructurePanel = memo(({ event }) => {
  const {
    event_ticker,
    title,
    category,
    series_ticker,
    mutually_exclusive,
    markets = [],
    understanding,
  } = event;

  // Use understanding compact card only if it has meaningful content beyond just the settlement rule
  const hasRichUnderstanding = understanding && (
    understanding.trading_summary || understanding.event_summary ||
    (understanding.participants && understanding.participants.length > 0) ||
    (understanding.key_factors && understanding.key_factors.length > 0)
  );

  if (hasRichUnderstanding) {
    return (
      <div className="bg-gray-900/50 rounded-lg border border-gray-800/40 p-3 space-y-2">
        <EventUnderstandingCard understanding={understanding} compact />
        {/* Market list */}
        <div className="border-t border-gray-800/30 pt-2">
          <div className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold mb-1">
            Markets ({markets.length} outcomes)
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-1">
            {markets.map((m) => (
              <div key={m.ticker} className="flex items-center gap-1.5 text-[11px]">
                <span className="text-gray-500">-</span>
                <span className="text-gray-400 truncate">{m.title || m.ticker}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  // Fallback: basic metadata (also used when understanding is sparse)
  const closeTime = markets[0]?.close_time || understanding?.close_time;
  const formattedExpiry = closeTime
    ? new Date(closeTime).toLocaleString([], {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      })
    : null;

  // Settlement rule from understanding (if available)
  const settlementText = understanding?.settlement_summary;

  return (
    <div className="bg-gray-900/50 rounded-lg border border-gray-800/40 p-3 space-y-2">
      <div>
        <h4 className="text-sm font-medium text-gray-200 mb-1">{title || event_ticker}</h4>
        <div className="flex flex-wrap items-center gap-2 text-[10px]">
          {category && (
            <span className="bg-gray-800/60 text-gray-400 px-2 py-0.5 rounded">{category}</span>
          )}
          {understanding?.domain && understanding.domain !== 'generic' && (
            <span className="bg-gray-800/60 text-gray-400 px-2 py-0.5 rounded">{understanding.domain}</span>
          )}
          {series_ticker && (
            <span className="text-gray-500 font-mono">Series: {series_ticker}</span>
          )}
          <span className={`px-2 py-0.5 rounded ${
            mutually_exclusive
              ? 'bg-emerald-900/30 text-emerald-400/80'
              : 'bg-amber-900/30 text-amber-400/80'
          }`}>
            {mutually_exclusive ? 'Mutually Exclusive' : 'Independent'}
          </span>
        </div>
        {settlementText && (
          <p className="text-[10px] text-gray-500 mt-1.5 leading-relaxed line-clamp-2">
            {settlementText}
          </p>
        )}
      </div>
      <div>
        <div className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold mb-1">
          Markets ({markets.length} outcomes)
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-1">
          {markets.map((m) => (
            <div key={m.ticker} className="flex items-center gap-1.5 text-[11px]">
              <span className="text-gray-500">-</span>
              <span className="text-gray-400 truncate">{m.title || m.ticker}</span>
            </div>
          ))}
        </div>
      </div>
      {formattedExpiry && (
        <div className="text-[10px] text-gray-500">
          <span className="text-gray-600">Expires:</span>{' '}
          <span className="text-gray-400">{formattedExpiry}</span>
        </div>
      )}
    </div>
  );
});
EventStructurePanel.displayName = 'EventStructurePanel';

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
    mutually_exclusive,
    event_type,
    markets = [],
  } = event;

  // Independent events don't have meaningful spread arb edges
  const isIndependent = event_type === 'independent' || !mutually_exclusive;

  const handleRowClick = useCallback(() => {
    if (onSelectEvent) onSelectEvent(event_ticker);
    setExpanded(true);
  }, [onSelectEvent, event_ticker]);

  const handleExpandClick = useCallback((e) => {
    e.stopPropagation();
    setExpanded(prev => !prev);
    // Also select the event so the details panel appears
    if (onSelectEvent) onSelectEvent(event_ticker);
  }, [onSelectEvent, event_ticker]);

  // Which edge to display (best of long/short) - only for mutually exclusive events
  const bestEdge = isIndependent ? null
    : (long_edge != null && long_edge > 0) ? long_edge
    : (short_edge != null && short_edge > 0) ? short_edge
    : null;
  const edgeDirection = isIndependent ? null
    : (long_edge != null && long_edge > 0) ? 'long'
    : (short_edge != null && short_edge > 0) ? 'short'
    : null;

  return (
    <div
      data-testid={`event-row-${event_ticker}`}
      data-event-ticker={event_ticker}
      data-selected={isSelected}
      data-edge={bestEdge}
      data-edge-direction={edgeDirection}
      className={`
        border rounded-xl overflow-hidden transition-colors duration-150
        ${isSelected
          ? 'border-cyan-500/30 bg-cyan-950/10'
          : 'border-gray-800/50'
        }
      `}
    >
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

        {/* Image avatar + Title */}
        {event.image_url && (
          <img
            src={event.image_url}
            alt=""
            className="w-5 h-5 rounded object-cover opacity-80 flex-shrink-0"
            onError={(e) => { e.target.style.display = 'none'; }}
          />
        )}
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
          {isIndependent
            ? <span className="text-[10px] text-amber-500/60 font-mono" title="Independent outcomes - no spread arb">indep</span>
            : bestEdge != null
              ? <EdgeBadge edgeCents={bestEdge} direction={edgeDirection} />
              : <span className="text-[10px] text-gray-600 font-mono">--</span>
          }
        </span>
      </button>

      {/* Expanded: Event Structure Panel */}
      {expanded && (
        <div className="px-4 py-3 bg-gray-950/40 border-t border-gray-800/30">
          <EventStructurePanel event={event} />
        </div>
      )}
    </div>
  );
};

export default memo(EventRow);
