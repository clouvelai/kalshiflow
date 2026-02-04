import React, { memo } from 'react';
import { Layers, RefreshCw } from 'lucide-react';
import EventRow from './EventRow';

/**
 * EventIndexPanel - Hierarchical event index with collapsible groups.
 *
 * Shows events sorted by volume, each expandable to reveal
 * nested PairCards with live prices.
 */
const EventIndexPanel = ({ events, pairIndex, selectedEventTicker, onSelectEvent }) => {
  const totalEvents = events?.length || 0;
  const totalPairs = pairIndex?.total_pairs || 0;

  // Format scan timing
  const lastScan = pairIndex?.last_scan_at;
  const nextScan = pairIndex?.next_scan_at;
  const lastScanAge = lastScan ? Math.floor((Date.now() / 1000 - lastScan)) : null;
  const nextScanIn = nextScan ? Math.max(0, Math.floor(nextScan - Date.now() / 1000)) : null;

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
          <h3 className="text-sm font-semibold text-gray-200">Event Index</h3>
          <span className="text-[10px] font-mono text-gray-500 bg-gray-800/60 rounded-full px-2 py-0.5">
            {totalEvents} events / {totalPairs} pairs
          </span>
        </div>
        <div className="flex items-center gap-2 text-[10px] text-gray-600 font-mono">
          {lastScanAge != null && (
            <span title="Time since last scan">
              <RefreshCw className="w-3 h-3 inline mr-1" />
              {lastScanAge < 60 ? `${lastScanAge}s ago` : `${Math.floor(lastScanAge / 60)}m ago`}
            </span>
          )}
          {nextScanIn != null && (
            <span className="text-gray-700">
              next {nextScanIn < 60 ? `${nextScanIn}s` : `${Math.floor(nextScanIn / 60)}m`}
            </span>
          )}
        </div>
      </div>

      {/* Event list */}
      <div className="p-3 space-y-2 max-h-[70vh] overflow-y-auto">
        {totalEvents === 0 ? (
          <div className="text-center py-8">
            <Layers className="w-8 h-8 text-gray-700 mx-auto mb-2" />
            <p className="text-sm text-gray-500">No events indexed yet</p>
            <p className="text-[10px] text-gray-600 mt-1">
              Pair discovery scan runs every {pairIndex?.next_scan_at ? '5' : '--'} minutes
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
