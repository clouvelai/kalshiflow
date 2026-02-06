import React, { memo, useMemo, useState } from 'react';
import { BookOpen } from 'lucide-react';
import EdgeBadge from '../ui/EdgeBadge';
import EventTradeFeed from '../ui/EventTradeFeed';
import MarketOrderbook from '../ui/MarketOrderbook';

/**
 * SingleArbDetailsView - Event details for single-event arb mode.
 *
 * Shows prob sums, edges, signals, market table with depth,
 * orderbook for selected market, and trade feed.
 */
const SingleArbDetailsView = memo(({ event, eventTrades = [] }) => {
  const [selectedMarket, setSelectedMarket] = useState(null);

  const {
    event_ticker,
    title,
    category,
    series_ticker,
    mutually_exclusive,
    event_type,
    market_count,
    markets_with_data,
    all_markets_have_data,
    sum_yes_bid,
    sum_yes_ask,
    sum_yes_mid,
    long_edge,
    short_edge,
    signals,
    markets: rawMarkets,
  } = event;

  // Independent events don't have meaningful spread arb edges
  const isIndependent = event_type === 'independent' || !mutually_exclusive;

  // Defensive default: ensure markets is always an object
  const markets = rawMarkets || {};
  const marketList = Object.values(markets);

  const formatVol = (v) => {
    if (v == null || v <= 0) return '--';
    if (v >= 1000000) return `$${(v / 1000000).toFixed(1)}M`;
    if (v >= 1000) return `$${(v / 1000).toFixed(0)}k`;
    return `$${Math.round(v)}`;
  };

  // Filter trades for this event
  const filteredTrades = useMemo(() =>
    eventTrades.filter(t => t.event_ticker === event_ticker),
    [eventTrades, event_ticker]
  );

  // Selected market data for orderbook view
  const selectedMarketData = selectedMarket ? markets[selectedMarket] : null;

  return (
    <>
      {/* Header */}
      <div className="px-5 py-3 border-b border-gray-800/50 flex-shrink-0">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-[10px] font-mono text-gray-500">{event_ticker}</span>
          {category && (
            <span className="text-[10px] font-mono bg-gray-800/60 text-gray-400 rounded-full px-2 py-0.5">
              {category}
            </span>
          )}
          {mutually_exclusive ? (
            <span className="text-[10px] font-mono bg-emerald-900/30 text-emerald-400/80 rounded-full px-2 py-0.5">
              mut. excl.
            </span>
          ) : (
            <span className="text-[10px] font-mono bg-amber-900/30 text-amber-400/80 rounded-full px-2 py-0.5">
              independent
            </span>
          )}
        </div>
        <h3 className="text-sm font-semibold text-gray-200 leading-tight">
          {title || event_ticker}
        </h3>
      </div>

      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto px-5 py-3 space-y-3">
        {/* Probability Sum Breakdown + Signals */}
        <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/30">
          <div className="text-[10px] font-semibold text-cyan-400/80 uppercase tracking-wider mb-2">
            Probability Sums &amp; Signals
          </div>
          <div className="space-y-1.5 text-xs text-gray-300">
            <div className="flex justify-between">
              <span className="text-gray-500">Sum YES Bid</span>
              <span className="font-mono">{sum_yes_bid != null ? `${sum_yes_bid}c` : '--'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Sum YES Ask</span>
              <span className="font-mono">{sum_yes_ask != null ? `${sum_yes_ask}c` : '--'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Sum YES Mid</span>
              <span className="font-mono font-bold text-cyan-400">
                {sum_yes_mid != null ? `${sum_yes_mid.toFixed(1)}c` : '--'}
              </span>
            </div>
            {signals?.deviation != null && (
              <div className="flex justify-between">
                <span className="text-gray-500">Deviation</span>
                <span className={`font-mono ${signals.deviation > 5 ? 'text-amber-400' : 'text-gray-400'}`}>
                  {signals.deviation.toFixed(1)}c
                </span>
              </div>
            )}
            {isIndependent ? (
              <div className="pt-1 border-t border-gray-700/20">
                <div className="flex items-center gap-2 text-amber-500/70">
                  <span className="text-[10px]">No spread arb - independent outcomes</span>
                </div>
              </div>
            ) : (
              <>
                <div className="flex justify-between items-center pt-1 border-t border-gray-700/20">
                  <span className="text-gray-500">Long Edge (after fees)</span>
                  {long_edge != null
                    ? <EdgeBadge edgeCents={long_edge} direction="long" />
                    : <span className="font-mono text-gray-600">--</span>
                  }
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-500">Short Edge (after fees)</span>
                  {short_edge != null
                    ? <EdgeBadge edgeCents={short_edge} direction="short" />
                    : <span className="font-mono text-gray-600">--</span>
                  }
                </div>
              </>
            )}
            <div className="flex justify-between items-center pt-1 border-t border-gray-700/20">
              <span className="text-gray-500">Data Coverage</span>
              <span className="font-mono">
                <span className={all_markets_have_data ? 'text-emerald-400' : 'text-amber-400'}>
                  {markets_with_data}
                </span>
                <span className="text-gray-600">/{market_count}</span>
                {all_markets_have_data && <span className="text-emerald-500 ml-1">&#10003;</span>}
              </span>
            </div>
            {signals?.widest_spread_ticker && (
              <div className="flex justify-between">
                <span className="text-gray-500">Widest Spread</span>
                <span className="font-mono text-amber-400/70 text-[10px]">{signals.widest_spread_ticker}</span>
              </div>
            )}
            {signals?.most_active_ticker && (
              <div className="flex justify-between">
                <span className="text-gray-500">Most Active</span>
                <span className="font-mono text-cyan-400/70 text-[10px]">{signals.most_active_ticker}</span>
              </div>
            )}
            {series_ticker && (
              <div className="flex justify-between">
                <span className="text-gray-500">Series</span>
                <span className="font-mono">{series_ticker}</span>
              </div>
            )}
          </div>
        </div>

        {/* Markets list with price, volume, last trade */}
        {marketList.length > 0 && (
          <div className="bg-gray-800/20 rounded-lg p-3 border border-gray-800/30">
            <div className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider mb-2">
              Markets ({marketList.length})
              <span className="text-gray-600 font-normal ml-2">click for orderbook</span>
            </div>
            <div className="space-y-1">
              {/* Header row */}
              <div className="flex items-center text-[9px] text-gray-600 uppercase tracking-wider font-semibold mb-1">
                <span className="flex-1 min-w-0">Outcome</span>
                <span className="w-10 text-right">Bid</span>
                <span className="w-10 text-right">Ask</span>
                <span className="w-10 text-right">Mid</span>
                <span className="w-10 text-right">Last</span>
                <span className="w-12 text-right">Vol</span>
                <span className="w-10 text-right">Src</span>
              </div>
              {marketList.map((m) => (
                <div
                  key={m.ticker}
                  className={`flex items-center text-[11px] cursor-pointer rounded px-1 -mx-1 transition-colors ${
                    selectedMarket === m.ticker ? 'bg-cyan-900/20' : 'hover:bg-gray-800/30'
                  }`}
                  onClick={() => setSelectedMarket(selectedMarket === m.ticker ? null : m.ticker)}
                >
                  <span className="text-gray-400 truncate flex-1 min-w-0 mr-2">
                    {m.title || m.ticker}
                  </span>
                  <span className="w-10 text-right font-mono text-gray-500">
                    {m.yes_bid != null ? `${m.yes_bid}` : '--'}
                  </span>
                  <span className="w-10 text-right font-mono text-gray-500">
                    {m.yes_ask != null ? `${m.yes_ask}` : '--'}
                  </span>
                  <span className="w-10 text-right font-mono text-cyan-400/70">
                    {m.yes_mid != null ? `${m.yes_mid.toFixed(0)}` : '--'}
                  </span>
                  <span className={`w-10 text-right font-mono ${
                    m.last_trade_side === 'yes' ? 'text-emerald-400/70' :
                    m.last_trade_side === 'no' ? 'text-red-400/70' : 'text-gray-500'
                  }`}>
                    {m.last_trade_price != null ? `${m.last_trade_price}` : m.last_price != null ? `${m.last_price}` : '--'}
                  </span>
                  <span className="w-12 text-right font-mono text-gray-600 text-[10px]">
                    {formatVol(m.volume_24h)}
                  </span>
                  <span className={`w-10 text-right font-mono text-[10px] ${
                    m.source === 'ws' ? 'text-emerald-500' : m.source === 'api' ? 'text-amber-500' : 'text-gray-600'
                  }`}>
                    {m.source || '--'}
                  </span>
                </div>
              ))}
              {/* Sum row */}
              <div className="flex items-center text-[11px] border-t border-gray-700/30 pt-1 mt-1 font-bold">
                <span className="text-gray-400 flex-1 min-w-0">TOTAL</span>
                <span className="w-10 text-right font-mono text-gray-400">
                  {sum_yes_bid != null ? `${sum_yes_bid}` : '--'}
                </span>
                <span className="w-10 text-right font-mono text-gray-400">
                  {sum_yes_ask != null ? `${sum_yes_ask}` : '--'}
                </span>
                <span className="w-10 text-right font-mono text-cyan-400">
                  {sum_yes_mid != null ? `${sum_yes_mid.toFixed(0)}` : '--'}
                </span>
                <span className="w-10" />
                <span className="w-12" />
                <span className="w-10 text-right font-mono text-[10px] text-gray-600">
                  /100
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Orderbook for selected market */}
        {selectedMarketData && (
          <MarketOrderbook market={selectedMarketData} />
        )}

        {/* Trade feed */}
        <EventTradeFeed trades={filteredTrades} />
      </div>
    </>
  );
});

const EventDetailsPanel = ({ selectedEventTicker, events, eventTrades = [] }) => {
  const event = useMemo(() => {
    if (!selectedEventTicker || !events) return null;
    return events.get(selectedEventTicker) || null;
  }, [selectedEventTicker, events]);

  if (!selectedEventTicker) {
    return (
      <div className="
        bg-gradient-to-br from-gray-900/90 via-gray-950/80 to-black/90
        rounded-2xl border border-cyan-500/10 shadow-lg shadow-cyan-500/5
        h-full flex flex-col items-center justify-center min-h-[420px]
      ">
        <BookOpen className="w-8 h-8 text-gray-700 mb-3" />
        <p className="text-sm text-gray-500">Select an event from the index</p>
        <p className="text-[10px] text-gray-600 mt-1">Click any event row to view details</p>
      </div>
    );
  }

  if (!event) {
    return (
      <div className="
        bg-gradient-to-br from-gray-900/90 via-gray-950/80 to-black/90
        rounded-2xl border border-cyan-500/10 shadow-lg shadow-cyan-500/5
        h-full flex flex-col items-center justify-center min-h-[420px]
      ">
        <div className="animate-pulse space-y-3 w-3/4">
          <div className="h-3 bg-gray-800 rounded w-1/3" />
          <div className="h-5 bg-gray-800 rounded w-2/3" />
          <div className="h-20 bg-gray-800/50 rounded" />
        </div>
        <p className="text-[10px] text-gray-600 mt-4">
          Waiting for event data...
        </p>
      </div>
    );
  }

  return (
    <div className="
      bg-gradient-to-br from-gray-900/90 via-gray-950/80 to-black/90
      rounded-2xl border border-cyan-500/10 shadow-lg shadow-cyan-500/5
      overflow-hidden h-full flex flex-col
    ">
      <SingleArbDetailsView event={event} eventTrades={eventTrades} />
    </div>
  );
};

export default memo(EventDetailsPanel);
