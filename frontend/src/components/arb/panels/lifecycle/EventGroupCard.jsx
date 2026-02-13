import React, { useState } from 'react';

/**
 * EventGroupCard - Expandable card grouping markets under a parent event.
 */
const EventGroupCard = ({ event, markets = [] }) => {
  const [expanded, setExpanded] = useState(false);

  if (!event) return null;

  const dotColorMap = {
    pending: 'bg-yellow-400',
    active: 'bg-green-400',
    deactivated: 'bg-orange-400',
    determined: 'bg-gray-400',
    settled: 'bg-gray-600',
  };

  const statusDot = (status) => (
    <span className={`inline-block w-1.5 h-1.5 rounded-full ${dotColorMap[status] || 'bg-gray-500'}`} />
  );

  return (
    <div className="bg-gray-800/60 rounded-lg border border-gray-700/50">
      <button
        className="w-full p-2.5 flex items-center gap-2 text-left hover:bg-gray-700/30 transition-colors rounded-lg"
        onClick={() => setExpanded(!expanded)}
      >
        <span className={`text-xs ${expanded ? 'rotate-90' : ''} transition-transform text-gray-500`}>
          &#9654;
        </span>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5">
            {statusDot(event.status)}
            <span className="text-xs text-gray-200 truncate">{event.title || event.event_ticker}</span>
          </div>
          <div className="flex items-center gap-2 mt-0.5">
            <span className="text-[10px] text-gray-500">{event.market_count || 0} markets</span>
            {event.mutually_exclusive && (
              <span className="text-[10px] bg-cyan-500/15 text-cyan-400 px-1 rounded">ME</span>
            )}
            {event.time_to_close_seconds != null && event.time_to_close_seconds > 0 && (
              <span className="text-[10px] text-amber-400 font-mono">
                closes {Math.floor(event.time_to_close_seconds / 3600)}h
              </span>
            )}
          </div>
        </div>
      </button>

      {expanded && markets.length > 0 && (
        <div className="px-2.5 pb-2.5 space-y-1">
          {markets.map(m => (
            <div key={m.ticker} className="flex items-center gap-2 py-1 px-2 bg-gray-900/50 rounded text-[11px]">
              {statusDot(m.status)}
              <span className="text-gray-300 truncate flex-1">{m.yes_sub_title || m.title || m.ticker}</span>
              {m.price > 0 && (
                <span className="text-gray-400 font-mono">{m.price}c</span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default EventGroupCard;
