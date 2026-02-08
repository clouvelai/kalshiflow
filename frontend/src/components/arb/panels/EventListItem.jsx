import React, { memo } from 'react';
import { Shield, ShieldOff } from 'lucide-react';
import EdgeBadge from '../ui/EdgeBadge';

const EventListItem = memo(({ event, isSelected, onSelect }) => {
  const {
    event_ticker, title, market_count, markets_with_data,
    all_markets_have_data, long_edge, short_edge,
    mutually_exclusive, event_type,
  } = event;

  const isIndependent = event_type === 'independent' || !mutually_exclusive;
  const bestEdge = isIndependent ? null
    : (long_edge != null && long_edge > 0) ? long_edge
    : (short_edge != null && short_edge > 0) ? short_edge
    : null;
  const edgeDirection = isIndependent ? null
    : (long_edge != null && long_edge > 0) ? 'long'
    : (short_edge != null && short_edge > 0) ? 'short'
    : null;

  return (
    <button
      onClick={() => onSelect(event_ticker)}
      data-testid={`event-row-${event_ticker}`}
      className={`w-full text-left px-3 py-2 transition-colors rounded-md group ${
        isSelected
          ? 'bg-cyan-950/20 border-l-2 border-l-cyan-400'
          : 'hover:bg-gray-800/30 border-l-2 border-l-transparent'
      }`}
    >
      <div className="flex items-center gap-2 min-w-0">
        {/* Edge badge */}
        <span className="flex-shrink-0 w-12">
          {isIndependent
            ? <span className="text-[9px] text-amber-500/60 font-mono">ind</span>
            : bestEdge != null
              ? <EdgeBadge edgeCents={bestEdge} direction={edgeDirection} />
              : <span className="text-[9px] text-gray-600 font-mono">--</span>
          }
        </span>

        {/* Title */}
        <span className={`text-[11px] truncate flex-1 min-w-0 ${
          isSelected ? 'text-gray-200' : 'text-gray-400 group-hover:text-gray-300'
        }`}>
          {title || event_ticker}
        </span>

        {/* Coverage */}
        <span className="flex-shrink-0 flex items-center gap-1">
          {all_markets_have_data
            ? <Shield className="w-3 h-3 text-emerald-500/60" />
            : <ShieldOff className="w-3 h-3 text-gray-700" />
          }
          <span className="text-[9px] font-mono text-gray-600 tabular-nums">
            {markets_with_data ?? 0}/{market_count ?? 0}
          </span>
        </span>
      </div>
    </button>
  );
});

EventListItem.displayName = 'EventListItem';

export default EventListItem;
