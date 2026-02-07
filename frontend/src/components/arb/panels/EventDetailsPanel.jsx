import React, { memo, useMemo, useState } from 'react';
import { List, BarChart2 } from 'lucide-react';
import EdgeBadge from '../ui/EdgeBadge';
import EventTradeFeed from '../ui/EventTradeFeed';
import MarketOrderbook from '../ui/MarketOrderbook';
import EventUnderstandingCard from './EventUnderstandingCard';

/**
 * TabButton - Orderbook/trades tab toggle.
 */
const TabButton = memo(({ active, onClick, icon: Icon, label }) => (
  <button
    onClick={onClick}
    className={`flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-medium rounded-lg transition-colors ${
      active
        ? 'bg-cyan-500/12 text-cyan-400 border border-cyan-500/20'
        : 'text-gray-500 hover:text-gray-300 hover:bg-gray-800/30'
    }`}
  >
    <Icon className="w-3.5 h-3.5" />
    {label}
  </button>
));
TabButton.displayName = 'TabButton';

/**
 * SingleArbDetailsView - Full event detail with markets, orderbook, trades.
 */
const SingleArbDetailsView = memo(({ event, eventTrades = [], arbTrades = [], positionsByTicker = {} }) => {
  const [selectedMarket, setSelectedMarket] = useState(null);
  const [activeTab, setActiveTab] = useState('orderbook');

  const {
    event_ticker, title, category, series_ticker,
    mutually_exclusive, event_type,
    market_count, markets_with_data, all_markets_have_data,
    sum_yes_bid, sum_yes_ask, sum_yes_mid,
    long_edge, short_edge, signals,
    markets: rawMarkets,
  } = event;

  const isIndependent = event_type === 'independent' || !mutually_exclusive;
  const markets = rawMarkets || {};
  const marketList = Object.values(markets);

  const formatVol = (v) => {
    if (v == null || v <= 0) return '--';
    if (v >= 1000000) return `$${(v / 1000000).toFixed(1)}M`;
    if (v >= 1000) return `$${(v / 1000).toFixed(0)}k`;
    return `$${Math.round(v)}`;
  };

  const hasPositions = useMemo(() =>
    marketList.some(m => positionsByTicker[m.ticker]),
    [marketList, positionsByTicker]
  );

  const positionSummary = useMemo(() => {
    if (!hasPositions) return null;
    let marketsWithPos = 0;
    let totalUnrealized = 0;
    for (const m of marketList) {
      const pos = positionsByTicker[m.ticker];
      if (pos) {
        marketsWithPos++;
        totalUnrealized += pos.unrealized_pnl || 0;
      }
    }
    return { marketsWithPos, totalUnrealized };
  }, [marketList, positionsByTicker, hasPositions]);

  const getFreshnessDisplay = (seconds) => {
    if (seconds == null || seconds >= 9999) return { text: '--', color: 'text-gray-600' };
    if (seconds < 5) return { text: `${Math.round(seconds)}s`, color: 'text-emerald-500' };
    if (seconds < 30) return { text: `${Math.round(seconds)}s`, color: 'text-amber-500' };
    if (seconds < 60) return { text: `${Math.round(seconds)}s`, color: 'text-red-400' };
    return { text: `${Math.floor(seconds / 60)}m`, color: 'text-red-400' };
  };

  const mergedTrades = useMemo(() => {
    const eventMarkets = new Set(Object.keys(markets));
    const publicTrades = (eventTrades || [])
      .filter(t => t.event_ticker === event_ticker || eventMarkets.has(t.market_ticker))
      .map(t => ({ ...t, source: 'public' }));
    const captainTrades = (arbTrades || [])
      .filter(t => eventMarkets.has(t.market_ticker))
      .map(t => ({ ...t, source: 'captain' }));
    return [...publicTrades, ...captainTrades].sort((a, b) => (b.ts || 0) - (a.ts || 0));
  }, [event_ticker, markets, eventTrades, arbTrades]);

  const selectedMarketData = selectedMarket ? markets[selectedMarket] : null;

  return (
    <>
      {/* Header */}
      <div className="px-5 py-3 border-b border-gray-800/30 shrink-0">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-[10px] font-mono text-gray-500">{event_ticker}</span>
          {category && (
            <span className="text-[9px] font-mono bg-gray-800/40 text-gray-400 rounded-full px-2 py-0.5">
              {category}
            </span>
          )}
          <span className={`text-[9px] font-mono rounded-full px-2 py-0.5 ${
            mutually_exclusive
              ? 'bg-emerald-500/10 text-emerald-400/70'
              : 'bg-amber-500/10 text-amber-400/70'
          }`}>
            {mutually_exclusive ? 'mut. excl.' : 'independent'}
          </span>
        </div>
        <h3 className="text-sm font-medium text-gray-200 leading-tight">
          {title || event_ticker}
        </h3>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto px-5 py-3 space-y-3">
        {/* Understanding - only show if it has meaningful content */}
        {event.understanding && (
          event.understanding.trading_summary || event.understanding.event_summary ||
          (event.understanding.participants && event.understanding.participants.length > 0) ||
          (event.understanding.key_factors && event.understanding.key_factors.length > 0)
        ) && (
          <EventUnderstandingCard understanding={event.understanding} />
        )}

        {/* Prob Sums & Signals */}
        <div className="bg-gray-800/20 rounded-lg p-3 border border-gray-700/15">
          <div className="text-[9px] font-semibold text-cyan-400/70 uppercase tracking-wider mb-2">
            Probability Sums &amp; Signals
          </div>
          <div className="space-y-1.5 text-[11px] text-gray-300">
            <div className="flex justify-between">
              <span className="text-gray-500">Sum YES Bid</span>
              <span className="font-mono tabular-nums">{sum_yes_bid != null ? `${sum_yes_bid}c` : '--'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Sum YES Ask</span>
              <span className="font-mono tabular-nums">{sum_yes_ask != null ? `${sum_yes_ask}c` : '--'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Sum YES Mid</span>
              <span className="font-mono font-semibold text-cyan-400 tabular-nums">
                {sum_yes_mid != null ? `${sum_yes_mid.toFixed(1)}c` : '--'}
              </span>
            </div>
            {signals?.deviation != null && (
              <div className="flex justify-between">
                <span className="text-gray-500">Deviation</span>
                <span className={`font-mono tabular-nums ${signals.deviation > 5 ? 'text-amber-400' : 'text-gray-400'}`}>
                  {signals.deviation.toFixed(1)}c
                </span>
              </div>
            )}
            {isIndependent ? (
              <div className="pt-1.5 border-t border-gray-700/15">
                <span className="text-[10px] text-amber-500/60">No spread arb - independent outcomes</span>
              </div>
            ) : (
              <>
                <div className="flex justify-between items-center pt-1.5 border-t border-gray-700/15">
                  <span className="text-gray-500">Long Edge</span>
                  {long_edge != null ? <EdgeBadge edgeCents={long_edge} direction="long" /> : <span className="font-mono text-gray-600">--</span>}
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-500">Short Edge</span>
                  {short_edge != null ? <EdgeBadge edgeCents={short_edge} direction="short" /> : <span className="font-mono text-gray-600">--</span>}
                </div>
              </>
            )}
            <div className="flex justify-between items-center pt-1.5 border-t border-gray-700/15">
              <span className="text-gray-500">Data Coverage</span>
              <span className="font-mono tabular-nums">
                <span className={all_markets_have_data ? 'text-emerald-400' : 'text-amber-400'}>{markets_with_data}</span>
                <span className="text-gray-600">/{market_count}</span>
                {all_markets_have_data && <span className="text-emerald-500/70 ml-1">&#10003;</span>}
              </span>
            </div>
            {signals?.widest_spread_ticker && (
              <div className="flex justify-between">
                <span className="text-gray-500">Widest Spread</span>
                <span className="font-mono text-amber-400/60 text-[10px]">{signals.widest_spread_ticker}</span>
              </div>
            )}
            {signals?.most_active_ticker && (
              <div className="flex justify-between">
                <span className="text-gray-500">Most Active</span>
                <span className="font-mono text-cyan-400/60 text-[10px]">{signals.most_active_ticker}</span>
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

        {/* Markets Table */}
        {marketList.length > 0 && (
          <div className="bg-gray-800/15 rounded-lg p-3 border border-gray-800/20">
            <div className="text-[9px] font-semibold text-gray-500 uppercase tracking-wider mb-2">
              Markets ({marketList.length})
              <span className="text-gray-600 font-normal ml-2">click for orderbook</span>
            </div>
            <div className="space-y-0.5 overflow-x-auto">
              {/* Header */}
              <div className="flex items-center text-[8px] text-gray-600 uppercase tracking-wider font-semibold mb-1 min-w-max">
                <span className="flex-1 min-w-[120px]">Outcome</span>
                <span className="w-10 text-right">Bid</span>
                <span className="w-10 text-right">Ask</span>
                <span className="w-10 text-right">Mid</span>
                <span className="w-12 text-right">Vol</span>
                <span className="w-12 text-right">Age</span>
                {hasPositions && (
                  <>
                    <span className="w-10 text-right">Qty</span>
                    <span className="w-12 text-right">Avg</span>
                    <span className="w-14 text-right">P&L</span>
                  </>
                )}
              </div>
              {marketList.map((m) => {
                const pos = positionsByTicker[m.ticker];
                const freshness = getFreshnessDisplay(m.freshness_seconds);
                return (
                  <div
                    key={m.ticker}
                    className={`flex items-center text-[11px] cursor-pointer rounded px-1 -mx-1 transition-colors min-w-max ${
                      selectedMarket === m.ticker ? 'bg-cyan-900/15' : 'hover:bg-gray-800/20'
                    }`}
                    onClick={() => setSelectedMarket(selectedMarket === m.ticker ? null : m.ticker)}
                  >
                    <div className="flex-1 min-w-[120px] mr-2">
                      <span className="text-gray-400 truncate block">{m.title || m.ticker}</span>
                      <span className="text-[9px] text-gray-600 font-mono">{m.ticker}</span>
                    </div>
                    <span className="w-10 text-right font-mono text-gray-500 tabular-nums">{m.yes_bid != null ? `${m.yes_bid}` : '--'}</span>
                    <span className="w-10 text-right font-mono text-gray-500 tabular-nums">{m.yes_ask != null ? `${m.yes_ask}` : '--'}</span>
                    <span className="w-10 text-right font-mono text-cyan-400/60 tabular-nums">{m.yes_mid != null ? `${m.yes_mid.toFixed(0)}` : '--'}</span>
                    <span className="w-12 text-right font-mono text-gray-600 text-[10px] tabular-nums">{formatVol(m.volume_24h)}</span>
                    <span className={`w-12 text-right font-mono text-[10px] tabular-nums ${freshness.color}`}>{freshness.text}</span>
                    {hasPositions && (
                      <>
                        <span className="w-10 text-right font-mono text-gray-400 tabular-nums">{pos?.qty || '--'}</span>
                        <span className="w-12 text-right font-mono text-gray-500 tabular-nums">{pos?.avg_cost != null ? `${pos.avg_cost}c` : '--'}</span>
                        <span className={`w-14 text-right font-mono tabular-nums ${
                          pos?.unrealized_pnl > 0 ? 'text-emerald-400' : pos?.unrealized_pnl < 0 ? 'text-red-400' : 'text-gray-500'
                        }`}>
                          {pos?.unrealized_pnl != null ? `${pos.unrealized_pnl >= 0 ? '+' : ''}$${pos.unrealized_pnl.toFixed(2)}` : '--'}
                        </span>
                      </>
                    )}
                  </div>
                );
              })}
              {/* Sum row */}
              <div className="flex items-center text-[11px] border-t border-gray-700/20 pt-1 mt-1 font-semibold min-w-max">
                <span className="text-gray-400 flex-1 min-w-[120px]">TOTAL</span>
                <span className="w-10 text-right font-mono text-gray-400 tabular-nums">{sum_yes_bid != null ? `${sum_yes_bid}` : '--'}</span>
                <span className="w-10 text-right font-mono text-gray-400 tabular-nums">{sum_yes_ask != null ? `${sum_yes_ask}` : '--'}</span>
                <span className="w-10 text-right font-mono text-cyan-400 tabular-nums">{sum_yes_mid != null ? `${sum_yes_mid.toFixed(0)}` : '--'}</span>
                <span className="w-12" />
                <span className="w-12 text-right font-mono text-[10px] text-gray-600">/100</span>
                {hasPositions && <><span className="w-10" /><span className="w-12" /><span className="w-14" /></>}
              </div>
              {positionSummary && (
                <div className="flex items-center justify-between text-[10px] border-t border-gray-700/20 pt-1 mt-1">
                  <span className="text-gray-500">
                    Positions: {positionSummary.marketsWithPos}/{marketList.length} markets
                  </span>
                  <span className={`font-mono font-semibold tabular-nums ${positionSummary.totalUnrealized >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {positionSummary.totalUnrealized >= 0 ? '+' : ''}${positionSummary.totalUnrealized.toFixed(2)}
                  </span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Orderbook / Trades Tabs */}
        <div className="bg-gray-800/15 rounded-lg border border-gray-800/20 overflow-hidden">
          <div className="flex items-center gap-2 px-3 py-2 border-b border-gray-800/20 bg-gray-900/20">
            <TabButton active={activeTab === 'orderbook'} onClick={() => setActiveTab('orderbook')} icon={BarChart2} label="Orderbook" />
            <TabButton active={activeTab === 'trades'} onClick={() => setActiveTab('trades')} icon={List} label={`Trades (${mergedTrades.length})`} />
            {selectedMarket && (
              <span className="ml-auto text-[10px] text-gray-500 font-mono">{selectedMarket}</span>
            )}
          </div>
          <div className="p-3">
            {activeTab === 'orderbook' ? (
              selectedMarketData ? (
                <MarketOrderbook market={selectedMarketData} />
              ) : (
                <div className="text-center py-6 text-gray-600">
                  <BarChart2 className="w-5 h-5 mx-auto mb-1.5 opacity-40" />
                  <p className="text-[11px]">Select a market to view orderbook</p>
                </div>
              )
            ) : (
              <EventTradeFeed trades={mergedTrades} arbTrades={arbTrades} showSource={true} />
            )}
          </div>
        </div>
      </div>
    </>
  );
});

const EventDetailsPanel = ({ selectedEventTicker, events, eventTrades = [], arbTrades = [], tradingState }) => {
  const event = useMemo(() => {
    if (!selectedEventTicker || !events) return null;
    return events.get(selectedEventTicker) || null;
  }, [selectedEventTicker, events]);

  const positionsByTicker = useMemo(() => {
    const map = {};
    (tradingState?.positions || []).forEach(p => { map[p.ticker] = p; });
    return map;
  }, [tradingState?.positions]);

  if (!event) {
    return (
      <div className="bg-gradient-to-br from-gray-900/80 to-gray-950/80 rounded-2xl border border-cyan-500/8 shadow-lg shadow-cyan-500/3 flex flex-col items-center justify-center min-h-[200px]">
        <div className="animate-pulse space-y-3 w-3/4">
          <div className="h-3 bg-gray-800/60 rounded w-1/3" />
          <div className="h-5 bg-gray-800/40 rounded w-2/3" />
          <div className="h-20 bg-gray-800/20 rounded" />
        </div>
        <p className="text-[10px] text-gray-600 mt-4">Waiting for event data...</p>
      </div>
    );
  }

  return (
    <div className="bg-gradient-to-br from-gray-900/80 to-gray-950/80 rounded-2xl border border-cyan-500/8 shadow-lg shadow-cyan-500/3 overflow-hidden flex flex-col">
      <SingleArbDetailsView event={event} eventTrades={eventTrades} arbTrades={arbTrades} positionsByTicker={positionsByTicker} />
    </div>
  );
};

export default memo(EventDetailsPanel);
