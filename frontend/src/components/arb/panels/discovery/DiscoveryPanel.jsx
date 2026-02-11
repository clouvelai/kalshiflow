import React, { memo, useState, useMemo, useEffect } from 'react';
import {
  Search, Eye, Layers, RefreshCw, BarChart3, Bookmark, XCircle,
} from 'lucide-react';

/**
 * Format volume for display: 1234567 -> "1.2M", 12345 -> "12.3K", 999 -> "999"
 */
function formatVolume(vol) {
  if (!vol) return '0';
  if (vol >= 1_000_000) return `${(vol / 1_000_000).toFixed(1)}M`;
  if (vol >= 1_000) return `${(vol / 1_000).toFixed(1)}K`;
  return String(vol);
}

const EVICTION_DISPLAY_MS = 60_000; // Show evicted events for 60s

/**
 * DiscoveryPanel - Shows top events by volume.
 *
 * Simple flat list of discovered events ranked by 24h volume,
 * with monitored status indicators and eviction notices.
 */
const DiscoveryPanel = memo(({ discoveryState, events }) => {
  const [filter, setFilter] = useState('');
  const [, forceUpdate] = useState(0);

  const {
    events: discoveredEvents = [],
    stats,
    lastFetch,
    recentEvictions = [],
  } = discoveryState || {};

  // Auto-expire eviction notices
  useEffect(() => {
    if (recentEvictions.length === 0) return;
    const oldest = recentEvictions[recentEvictions.length - 1];
    const age = Date.now() - (oldest?.evicted_at || 0);
    const delay = Math.max(100, EVICTION_DISPLAY_MS - age);
    const timer = setTimeout(() => forceUpdate(n => n + 1), delay);
    return () => clearTimeout(timer);
  }, [recentEvictions]);

  // Filter out expired eviction notices
  const activeEvictions = useMemo(() => {
    const cutoff = Date.now() - EVICTION_DISPLAY_MS;
    return recentEvictions.filter(e => (e.evicted_at || 0) > cutoff);
  }, [recentEvictions, /* forceUpdate triggers re-render */]);

  // Monitored event tickers from the events Map (arb index)
  const monitoredTickers = useMemo(() => {
    if (!events) return new Set();
    return new Set(events.keys());
  }, [events]);

  // Filter events by search
  const filteredEvents = useMemo(() => {
    if (!filter) return discoveredEvents;
    const q = filter.toLowerCase();
    return discoveredEvents.filter(e =>
      (e.title || '').toLowerCase().includes(q) ||
      (e.event_ticker || '').toLowerCase().includes(q) ||
      (e.category || '').toLowerCase().includes(q)
    );
  }, [discoveredEvents, filter]);

  const totalEvents = discoveredEvents.length;
  const totalMonitored = discoveredEvents.filter(e => monitoredTickers.has(e.event_ticker)).length;
  const targetCount = stats?.event_count || 10;
  const evictedTotal = stats?.total_events_evicted || 0;

  const lastFetchAge = lastFetch ? Math.round((Date.now() - lastFetch) / 1000) : null;
  const lastFetchDisplay = lastFetchAge != null
    ? lastFetchAge < 60 ? `${lastFetchAge}s ago`
      : lastFetchAge < 3600 ? `${Math.floor(lastFetchAge / 60)}m ago`
      : `${Math.floor(lastFetchAge / 3600)}h ago`
    : '--';

  if (totalEvents === 0 && activeEvictions.length === 0) {
    return (
      <div className="flex-1 min-h-0 flex flex-col items-center justify-center px-6 py-12">
        <Layers className="w-8 h-8 text-gray-700 mb-3" />
        <span className="text-[13px] text-gray-500 font-medium mb-1">Discovering Events</span>
        <span className="text-[11px] text-gray-600 text-center max-w-[300px]">
          Scanning Kalshi for the top {targetCount} events by volume...
        </span>
      </div>
    );
  }

  return (
    <div className="flex-1 min-h-0 flex flex-col">
      {/* Header bar */}
      <div className="flex items-center gap-3 px-3 py-2 border-b border-gray-800/30 shrink-0">
        <div className="flex items-center gap-1.5">
          <BarChart3 className="w-3.5 h-3.5 text-cyan-400/60" />
          <span className="text-[11px] font-semibold text-gray-300">
            Top {targetCount} by Volume
          </span>
        </div>
        <div className="flex items-center gap-2 ml-auto">
          <span className="text-[9px] text-gray-500 font-mono tabular-nums">
            {totalEvents} events
          </span>
          {totalMonitored > 0 && (
            <span className="flex items-center gap-0.5 text-[9px] text-emerald-400/80 font-mono tabular-nums">
              <Eye className="w-2.5 h-2.5" />
              {totalMonitored}
            </span>
          )}
          {evictedTotal > 0 && (
            <span className="flex items-center gap-0.5 text-[9px] text-gray-500/60 font-mono tabular-nums" title={`${evictedTotal} events evicted (settled/closed)`}>
              <XCircle className="w-2.5 h-2.5" />
              {evictedTotal}
            </span>
          )}
          {stats?.total_events_scanned > 0 && (
            <span className="text-[8px] text-gray-600 font-mono">
              ({stats.total_events_scanned} scanned)
            </span>
          )}
          <div className="flex items-center gap-1" title={`Last fetch: ${lastFetchDisplay}`}>
            <RefreshCw className="w-2.5 h-2.5 text-gray-600" />
            <span className="text-[8px] text-gray-600 font-mono">{lastFetchDisplay}</span>
          </div>
        </div>
      </div>

      {/* Search filter */}
      {totalEvents > 5 && (
        <div className="px-3 py-1.5 border-b border-gray-800/20 shrink-0">
          <div className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-gray-900/40 border border-gray-800/25">
            <Search className="w-3 h-3 text-gray-600 shrink-0" />
            <input
              type="text"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="Filter events..."
              className="bg-transparent border-none outline-none text-[10px] text-gray-300 placeholder-gray-600 w-full"
            />
          </div>
        </div>
      )}

      {/* Events list */}
      <div className="flex-1 min-h-0 overflow-y-auto px-3 py-2 space-y-0.5">
        {filteredEvents.map((event, idx) => {
          const monitored = monitoredTickers.has(event.event_ticker);
          const isSeed = event.source === 'seed';
          return (
            <div
              key={event.event_ticker}
              className={`flex items-center gap-2 px-2.5 py-1.5 rounded-md transition-colors ${
                monitored ? 'bg-cyan-500/5 hover:bg-cyan-500/8' : 'bg-gray-900/10 hover:bg-gray-800/15'
              }`}
            >
              {/* Rank number */}
              <span className="text-[9px] text-gray-600 font-mono tabular-nums w-4 text-right shrink-0">
                {isSeed ? '' : `${idx + 1}`}
              </span>

              {/* Monitored dot */}
              <div className={`w-1.5 h-1.5 rounded-full shrink-0 ${
                monitored ? 'bg-emerald-500' : 'bg-gray-600'
              }`} />

              {/* Title */}
              <span className="text-[10px] text-gray-300 truncate flex-1" title={event.title || event.event_ticker}>
                {event.title || event.event_ticker}
              </span>

              {/* Seed badge */}
              {isSeed && (
                <Bookmark className="w-2.5 h-2.5 text-amber-400/50 shrink-0" />
              )}

              {/* Volume */}
              <span className="text-[9px] font-mono text-gray-500 shrink-0 tabular-nums" title={`Volume 24h: ${(event.volume_24h || 0).toLocaleString()}`}>
                {formatVolume(event.volume_24h)}
              </span>

              {/* Market count */}
              <span className="text-[9px] font-mono text-gray-600 shrink-0 tabular-nums">
                {event.market_count || 0}m
              </span>

              {/* Monitored eye */}
              {monitored && (
                <Eye className="w-2.5 h-2.5 text-emerald-400/60 shrink-0" />
              )}
            </div>
          );
        })}
        {filteredEvents.length === 0 && filter && (
          <div className="py-4 text-center">
            <span className="text-[10px] text-gray-600">No events match "{filter}"</span>
          </div>
        )}

        {/* Recently evicted events */}
        {activeEvictions.length > 0 && (
          <div className="mt-3 pt-2 border-t border-gray-800/20">
            <div className="flex items-center gap-1.5 mb-1.5 px-1">
              <XCircle className="w-3 h-3 text-gray-600" />
              <span className="text-[9px] text-gray-500 font-semibold uppercase tracking-wider">
                Recently Settled
              </span>
            </div>
            {activeEvictions.map(ev => {
              const age = Math.round((Date.now() - ev.evicted_at) / 1000);
              const ageStr = age < 60 ? `${age}s ago` : `${Math.floor(age / 60)}m ago`;
              return (
                <div
                  key={ev.event_ticker}
                  className="flex items-center gap-2 px-2.5 py-1 rounded-md bg-gray-900/5 opacity-50"
                >
                  <span className="text-[9px] text-gray-700 font-mono tabular-nums w-4 text-right shrink-0">
                    --
                  </span>
                  <div className="w-1.5 h-1.5 rounded-full shrink-0 bg-red-400/40" />
                  <span className="text-[10px] text-gray-500 truncate flex-1 line-through" title={ev.title || ev.event_ticker}>
                    {ev.title || ev.event_ticker}
                  </span>
                  <span className="text-[8px] font-mono text-gray-600 shrink-0">
                    {ageStr}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
});

DiscoveryPanel.displayName = 'DiscoveryPanel';

export default DiscoveryPanel;
