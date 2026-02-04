import React, { memo, useState, useCallback } from 'react';
import { ChevronDown, ChevronRight, Shield, ShieldOff } from 'lucide-react';
import PairCard from './PairCard';
import SpreadBadge from '../ui/SpreadBadge';

/**
 * EventRow - Collapsible event group with nested pair cards.
 *
 * Shows: event title, pair count, volume badge, avg/best spread, expand toggle.
 * Expanded: nested PairCard components with live prices.
 * Clickable: fires onSelectEvent for EventDetailsPanel.
 */
const EventRow = ({ event, isSelected, onSelectEvent }) => {
  const [expanded, setExpanded] = useState(false);

  const {
    event_ticker,
    title,
    volume_24h,
    market_count,
    is_tradeable,
    tradeable_pair_count,
    pairs = [],
    avg_spread,
    best_spread,
    live_pair_count = 0,
  } = event;

  const formatVolume = (vol) => {
    if (vol <= 0) return '--';
    if (vol >= 1000000) return `$${(vol / 1000000).toFixed(1)}M`;
    if (vol >= 1000) return `$${(vol / 1000).toFixed(0)}k`;
    return `$${vol}`;
  };
  const volumeDisplay = formatVolume(volume_24h);

  const handleRowClick = useCallback((e) => {
    // Select the event for details panel
    if (onSelectEvent) {
      onSelectEvent(event_ticker);
    }
  }, [onSelectEvent, event_ticker]);

  const handleExpandClick = useCallback((e) => {
    e.stopPropagation();
    setExpanded(prev => !prev);
  }, []);

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

        {/* Tradeable indicator */}
        <span className="flex-shrink-0" title={is_tradeable ? 'Trading enabled' : 'View only'}>
          {is_tradeable
            ? <Shield className="w-3.5 h-3.5 text-emerald-500/70" />
            : <ShieldOff className="w-3.5 h-3.5 text-gray-600" />
          }
        </span>

        {/* Tradeable / total pair count */}
        <span className="text-xs font-mono flex-shrink-0 w-14 text-right">
          {tradeable_pair_count != null ? (
            <>
              <span className={tradeable_pair_count > 0 ? 'text-emerald-400' : 'text-red-400'}>
                {tradeable_pair_count}
              </span>
              <span className="text-gray-600">/{market_count}</span>
            </>
          ) : (
            <span className="text-gray-400">{market_count} pair{market_count !== 1 ? 's' : ''}</span>
          )}
        </span>

        {/* Volume badge */}
        <span className="text-xs text-cyan-400/80 font-mono flex-shrink-0 w-16 text-right">
          {volumeDisplay}
        </span>

        {/* Average spread */}
        <span className="flex-shrink-0 w-16 text-right">
          {avg_spread != null
            ? <SpreadBadge spreadCents={avg_spread} />
            : <span className="text-[10px] text-gray-600 font-mono">--</span>
          }
        </span>
      </button>

      {/* Expanded: pair cards */}
      {expanded && (
        <div className="px-4 py-3 bg-gray-950/40 border-t border-gray-800/30">
          {pairs.length === 0 ? (
            <p className="text-xs text-gray-600 text-center py-2">No pairs in this event</p>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {pairs.map(pair => (
                <PairCard
                  key={pair.pair_id}
                  pair={pair}
                />
              ))}
            </div>
          )}
          {/* Event ticker label */}
          <div className="mt-2 pt-2 border-t border-gray-800/30 flex items-center justify-between">
            <span className="font-mono text-[10px] text-gray-600">{event_ticker}</span>
            <span className="text-[10px] text-gray-600">
              {live_pair_count}/{market_count} live
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default memo(EventRow);
