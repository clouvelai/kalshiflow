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
    <div
      id="event-index-panel"
      data-testid="event-index-panel"
      data-event-count={totalEvents}
      data-market-count={totalMarkets}
      className="
        bg-gradient-to-br from-gray-900/80 via-gray-950/70 to-black/80
        rounded-xl border border-cyan-500/8 shadow-lg shadow-cyan-500/3
        overflow-hidden
      "
    >
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-3 border-b border-gray-800/30">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-cyan-400/80" />
          <h3 className="text-[12px] font-semibold text-gray-200 uppercase tracking-wider">
            Events
          </h3>
          <span data-testid="event-stats" className="text-[10px] font-mono text-gray-500 tabular-nums bg-gray-800/40 rounded-full px-2 py-0.5">
            {totalEvents} events / {totalMarkets} markets
          </span>
        </div>
      </div>

      {/* Event list */}
      <div id="event-list" data-testid="event-list" className="p-3 space-y-2 max-h-[70vh] overflow-y-auto">
        {totalEvents === 0 ? (
          <div data-testid="no-events-placeholder" className="text-center py-8">
            <Layers className="w-6 h-6 text-gray-700 mx-auto mb-2" />
            <p className="text-[11px] text-gray-500">No events loaded yet</p>
            <p className="text-[10px] text-gray-600 mt-1">
              Waiting for event data...
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
