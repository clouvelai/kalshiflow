import React, { memo } from 'react';
import { Layers } from 'lucide-react';
import EventRow from './EventRow';

/**
 * EventIndexPanel - Event index for single-event arb.
 *
 * Shows events sorted by edge, each expandable to reveal
 * nested MarketCards with orderbook depth and trade data.
 */
const EventIndexPanel = ({ events, selectedEventTicker, onSelectEvent }) => {
  const totalEvents = events?.length || 0;
  const totalMarkets = events?.reduce((sum, e) => sum + (e.market_count || 0), 0) || 0;

  return (
    <div className="
      bg-gradient-to-br from-gray-900/90 via-gray-950/80 to-black/90
      rounded-2xl border border-cyan-500/10 shadow-lg shadow-cyan-500/5
      overflow-hidden
    ">
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-3 border-b border-gray-800/50">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-cyan-400" />
          <h3 className="text-sm font-semibold text-gray-200">
            Single-Event Arb
          </h3>
          <span className="text-[10px] font-mono text-gray-500 bg-gray-800/60 rounded-full px-2 py-0.5">
            {totalEvents} events / {totalMarkets} markets
          </span>
        </div>
      </div>

      {/* Event list */}
      <div className="p-3 space-y-2 max-h-[70vh] overflow-y-auto">
        {totalEvents === 0 ? (
          <div className="text-center py-8">
            <Layers className="w-8 h-8 text-gray-700 mx-auto mb-2" />
            <p className="text-sm text-gray-500">No events loaded yet</p>
            <p className="text-[10px] text-gray-600 mt-1">
              Waiting for event data from backend...
            </p>
          </div>
        ) : (
          events.map(event => (
            <EventRow
              key={event.event_ticker}
              event={event}
              isSelected={event.event_ticker === selectedEventTicker}
              onSelectEvent={onSelectEvent}
            />
          ))
        )}
      </div>
    </div>
  );
};

export default memo(EventIndexPanel);
