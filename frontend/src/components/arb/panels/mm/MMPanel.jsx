import React, { memo, useMemo } from 'react';
import {
  TrendingUp, Activity, AlertTriangle, Package, BarChart3,
  ArrowUpDown, Zap, Pause,
} from 'lucide-react';

/**
 * MMPanel - Market Maker dashboard tab for the Captain UX.
 *
 * Shows quote engine state, per-market quotes + inventory,
 * trade log (fills/placements/pulls), and performance metrics.
 * Uses the same styling patterns as DiscoveryPanel / LifecycleTimelinePanel.
 */

// ─── Helpers ────────────────────────────────────────────────────────────────

const formatCents = (c) => {
  if (c == null) return '--';
  const abs = Math.abs(c);
  if (abs >= 10000) return `$${(c / 100).toFixed(0)}`;
  if (abs >= 100) return `$${(c / 100).toFixed(2)}`;
  return `${c.toFixed?.(1) ?? c}c`;
};

const formatPnl = (c) => {
  if (c == null) return '--';
  const sign = c >= 0 ? '+' : '';
  return `${sign}${formatCents(c)}`;
};

const formatTime = (ts) => {
  if (!ts) return '';
  const d = new Date(typeof ts === 'number' && ts < 1e12 ? ts * 1000 : ts);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
};

// ─── Quote Status Bar ───────────────────────────────────────────────────────

const QuoteStatusBar = memo(({ quoteState }) => {
  if (!quoteState) return null;

  const isPulled = quoteState.quotes_pulled;
  const isStorm = quoteState.fill_storm_active;
  const activeQuotes = quoteState.active_quotes || 0;
  const cycles = quoteState.total_requote_cycles || 0;
  const multiplier = quoteState.spread_multiplier || 1;

  return (
    <div className="flex items-center gap-3 px-3 py-2 border-b border-gray-800/30 shrink-0">
      <div className="flex items-center gap-1.5">
        <TrendingUp className="w-3.5 h-3.5 text-fuchsia-400/60" />
        <span className="text-[11px] font-semibold text-gray-300">Quote Engine</span>
      </div>

      <div className="flex items-center gap-2 ml-auto">
        {/* Status pill */}
        {isPulled ? (
          <span className="flex items-center gap-1 px-1.5 py-0.5 rounded-full bg-red-500/10 border border-red-500/15">
            <Pause className="w-2.5 h-2.5 text-red-400" />
            <span className="text-[8px] font-semibold text-red-400 uppercase tracking-wider">Pulled</span>
          </span>
        ) : (
          <span className="flex items-center gap-1 px-1.5 py-0.5 rounded-full bg-fuchsia-500/10 border border-fuchsia-500/15">
            <Activity className="w-2.5 h-2.5 text-fuchsia-400" />
            <span className="text-[8px] font-semibold text-fuchsia-400 uppercase tracking-wider">Active</span>
          </span>
        )}

        {/* Storm indicator */}
        {isStorm && (
          <span className="flex items-center gap-0.5 px-1.5 py-0.5 rounded-full bg-amber-500/10 border border-amber-500/15 animate-pulse">
            <Zap className="w-2.5 h-2.5 text-amber-400" />
            <span className="text-[8px] font-semibold text-amber-400 uppercase">Storm</span>
          </span>
        )}

        {/* Spread multiplier */}
        {multiplier > 1 && (
          <span className="text-[9px] font-mono text-amber-400/80 tabular-nums">{multiplier.toFixed(1)}x</span>
        )}

        {/* Active quotes count */}
        <span className="text-[9px] text-gray-500 font-mono tabular-nums">
          {activeQuotes} quotes
        </span>

        {/* Cycle count */}
        <span className="text-[9px] text-gray-600 font-mono tabular-nums">
          #{cycles}
        </span>
      </div>
    </div>
  );
});
QuoteStatusBar.displayName = 'QuoteStatusBar';

// ─── Market Card ────────────────────────────────────────────────────────────

const MarketCard = memo(({ ticker, market }) => {
  if (!market) return null;

  const position = market.position || 0;
  const pnl = market.unrealized_pnl_cents || 0;
  const posColor = position > 0 ? 'text-emerald-400' : position < 0 ? 'text-red-400' : 'text-gray-500';
  const pnlColor = pnl > 0 ? 'text-emerald-400' : pnl < 0 ? 'text-red-400' : 'text-gray-500';

  // Our quotes
  const hasBid = market.our_bid_price != null;
  const hasAsk = market.our_ask_price != null;

  return (
    <div className="rounded-lg bg-gray-800/40 border border-gray-800/25 overflow-hidden hover:border-fuchsia-500/15 transition-colors">
      {/* Market header */}
      <div className="flex items-center gap-2 px-2.5 py-1.5">
        <span className="text-[10px] text-gray-300 truncate flex-1 font-medium" title={ticker}>
          {market.title || ticker}
        </span>
        {market.vpin > 0.7 && (
          <span className={`text-[8px] font-mono px-1 py-px rounded ${
            market.vpin > 0.9 ? 'bg-red-500/10 text-red-400' : 'bg-amber-500/10 text-amber-400'
          }`}>
            VPIN {(market.vpin * 100).toFixed(0)}%
          </span>
        )}
      </div>

      {/* Quote + orderbook row */}
      <div className="flex items-center gap-1 px-2.5 pb-1.5">
        {/* Our bid */}
        <div className={`flex-1 text-center rounded py-0.5 ${hasBid ? 'bg-emerald-500/8' : 'bg-gray-900/30'}`}>
          <div className="text-[8px] text-gray-600 uppercase">Bid</div>
          <div className={`text-[11px] font-mono tabular-nums ${hasBid ? 'text-emerald-400' : 'text-gray-600'}`}>
            {hasBid ? market.our_bid_price : '--'}
          </div>
          {hasBid && (
            <div className="text-[8px] font-mono text-gray-600 tabular-nums">x{market.our_bid_size}</div>
          )}
        </div>

        {/* Spread / Fair Value */}
        <div className="flex flex-col items-center px-1">
          {market.fair_value != null && (
            <span className="text-[8px] font-mono text-fuchsia-400/60 tabular-nums">
              FV {Math.round(market.fair_value)}
            </span>
          )}
          {market.spread != null && (
            <span className="text-[8px] font-mono text-gray-600 tabular-nums">
              {market.spread}c
            </span>
          )}
        </div>

        {/* Our ask */}
        <div className={`flex-1 text-center rounded py-0.5 ${hasAsk ? 'bg-red-500/8' : 'bg-gray-900/30'}`}>
          <div className="text-[8px] text-gray-600 uppercase">Ask</div>
          <div className={`text-[11px] font-mono tabular-nums ${hasAsk ? 'text-red-400' : 'text-gray-600'}`}>
            {hasAsk ? market.our_ask_price : '--'}
          </div>
          {hasAsk && (
            <div className="text-[8px] font-mono text-gray-600 tabular-nums">x{market.our_ask_size}</div>
          )}
        </div>
      </div>

      {/* Position row */}
      {position !== 0 && (
        <div className="flex items-center gap-2 px-2.5 py-1 border-t border-gray-800/20">
          <span className="text-[8px] text-gray-600 uppercase">Pos</span>
          <span className={`text-[10px] font-mono tabular-nums ${posColor}`}>
            {position > 0 ? '+' : ''}{position}
          </span>
          <span className="text-[8px] text-gray-600 uppercase ml-auto">uPnL</span>
          <span className={`text-[10px] font-mono tabular-nums ${pnlColor}`}>
            {formatPnl(pnl)}
          </span>
        </div>
      )}
    </div>
  );
});
MarketCard.displayName = 'MarketCard';

// ─── Inventory Summary ──────────────────────────────────────────────────────

const InventorySummary = memo(({ inventory, mmSnapshot }) => {
  // Compute totals from snapshot events
  const totals = useMemo(() => {
    if (!mmSnapshot?.events) return null;
    let totalPos = 0;
    let totalUnrealized = 0;
    let totalRealized = 0;
    Object.values(mmSnapshot.events).forEach(ev => {
      totalPos += ev.total_position_contracts || 0;
      totalUnrealized += ev.total_unrealized_pnl_cents || 0;
      totalRealized += ev.total_realized_pnl_cents || 0;
    });
    return { totalPos, totalUnrealized, totalRealized };
  }, [mmSnapshot]);

  if (!totals) return null;

  const netPnl = totals.totalRealized + totals.totalUnrealized;
  const netColor = netPnl > 0 ? 'text-emerald-400' : netPnl < 0 ? 'text-red-400' : 'text-gray-400';

  return (
    <div className="flex items-center gap-3 px-3 py-1.5 border-b border-gray-800/20 bg-gray-900/20 shrink-0">
      <div className="flex items-center gap-1">
        <Package className="w-3 h-3 text-gray-500" />
        <span className="text-[9px] text-gray-500 uppercase tracking-wider">Inventory</span>
      </div>
      <div className="flex items-center gap-3 ml-auto">
        <div className="flex items-center gap-1">
          <span className="text-[8px] text-gray-600">POS</span>
          <span className="text-[10px] font-mono text-gray-300 tabular-nums">{totals.totalPos}</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="text-[8px] text-gray-600">REAL</span>
          <span className={`text-[10px] font-mono tabular-nums ${totals.totalRealized >= 0 ? 'text-emerald-400/80' : 'text-red-400/80'}`}>
            {formatPnl(totals.totalRealized)}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <span className="text-[8px] text-gray-600">NET</span>
          <span className={`text-[10px] font-mono font-semibold tabular-nums ${netColor}`}>
            {formatPnl(netPnl)}
          </span>
        </div>
      </div>
    </div>
  );
});
InventorySummary.displayName = 'InventorySummary';

// ─── Performance Section ────────────────────────────────────────────────────

const PerformanceSection = memo(({ performance }) => {
  if (!performance) return null;

  const totalFills = (performance.total_fills_bid || 0) + (performance.total_fills_ask || 0);
  const netPnl = (performance.spread_captured_cents || 0) -
                 (performance.adverse_selection_cents || 0) -
                 (performance.fees_paid_cents || 0);
  const netColor = netPnl > 0 ? 'text-emerald-400' : netPnl < 0 ? 'text-red-400' : 'text-gray-400';

  return (
    <div className="px-3 py-2 border-b border-gray-800/20 shrink-0">
      <div className="flex items-center gap-1.5 mb-1.5">
        <BarChart3 className="w-3 h-3 text-gray-500" />
        <span className="text-[9px] text-gray-500 uppercase tracking-wider font-semibold">Performance</span>
      </div>
      <div className="grid grid-cols-4 gap-2">
        <Stat label="Fills" value={totalFills} sublabel={`${performance.total_fills_bid || 0}b / ${performance.total_fills_ask || 0}a`} />
        <Stat label="Spread" value={formatCents(performance.spread_captured_cents)} color="text-emerald-400/80" />
        <Stat label="Adverse" value={formatCents(performance.adverse_selection_cents)} color="text-red-400/80" />
        <Stat label="Net P&L" value={formatPnl(netPnl)} color={netColor} bold />
      </div>
    </div>
  );
});
PerformanceSection.displayName = 'PerformanceSection';

const Stat = memo(({ label, value, sublabel, color = 'text-gray-300', bold = false }) => (
  <div>
    <div className="text-[8px] text-gray-600 uppercase tracking-wider">{label}</div>
    <div className={`text-[11px] font-mono tabular-nums ${color} ${bold ? 'font-semibold' : ''}`}>{value}</div>
    {sublabel && <div className="text-[8px] text-gray-600 font-mono">{sublabel}</div>}
  </div>
));
Stat.displayName = 'Stat';

// ─── Trade Log ──────────────────────────────────────────────────────────────

const TRADE_TYPE_STYLES = {
  quote_placed: { bg: 'bg-fuchsia-500/8', text: 'text-fuchsia-400/80', label: 'PLACED' },
  quote_filled: { bg: 'bg-emerald-500/10', text: 'text-emerald-400/80', label: 'FILLED' },
  quotes_pulled: { bg: 'bg-red-500/10', text: 'text-red-400/80', label: 'PULLED' },
};

const TradeLogSection = memo(({ tradeLog }) => {
  if (!tradeLog || tradeLog.length === 0) return null;

  // Show max 30 entries
  const entries = tradeLog.slice(0, 30);

  return (
    <div className="flex-1 min-h-0 flex flex-col">
      <div className="flex items-center gap-1.5 px-3 py-1.5 border-b border-gray-800/20 shrink-0">
        <ArrowUpDown className="w-3 h-3 text-gray-500" />
        <span className="text-[9px] text-gray-500 uppercase tracking-wider font-semibold">Trade Log</span>
        <span className="text-[8px] text-gray-600 font-mono ml-auto">{tradeLog.length}</span>
      </div>
      <div className="flex-1 min-h-0 overflow-y-auto px-3 py-1 space-y-0.5">
        {entries.map((entry, i) => {
          const style = TRADE_TYPE_STYLES[entry.type] || TRADE_TYPE_STYLES.quote_placed;
          return (
            <div key={`${entry.type}-${entry.timestamp}-${i}`} className="flex items-center gap-1.5 py-0.5">
              <span className={`text-[8px] font-semibold px-1 py-px rounded ${style.bg} ${style.text} uppercase tracking-wider shrink-0`}>
                {style.label}
              </span>
              {entry.market_ticker && (
                <span className="text-[9px] text-gray-400 font-mono truncate max-w-[120px]">
                  {entry.market_ticker}
                </span>
              )}
              {entry.quote_side && (
                <span className={`text-[8px] font-mono ${entry.quote_side === 'bid' ? 'text-emerald-400/60' : 'text-red-400/60'}`}>
                  {entry.quote_side}
                </span>
              )}
              {entry.price_cents != null && (
                <span className="text-[9px] font-mono text-gray-400 tabular-nums">{entry.price_cents}c</span>
              )}
              {entry.size != null && (
                <span className="text-[9px] font-mono text-gray-600 tabular-nums">x{entry.size || entry.count || ''}</span>
              )}
              {entry.cancelled != null && (
                <span className="text-[9px] text-gray-500">{entry.cancelled} cancelled</span>
              )}
              <span className="text-[8px] text-gray-700 font-mono ml-auto tabular-nums shrink-0">
                {formatTime(entry.timestamp)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
});
TradeLogSection.displayName = 'TradeLogSection';

// ─── Main MMPanel ───────────────────────────────────────────────────────────

const MMPanel = memo(({ mmSnapshot, mmQuoteState, mmInventory, mmTradeLog, mmPerformance }) => {
  // Flatten all markets from all events for card display
  const allMarkets = useMemo(() => {
    if (!mmSnapshot?.events) return [];
    const result = [];
    Object.values(mmSnapshot.events).forEach(ev => {
      if (ev.markets) {
        Object.entries(ev.markets).forEach(([ticker, market]) => {
          result.push({ ticker, market, eventTicker: ev.event_ticker, eventTitle: ev.title });
        });
      }
    });
    return result;
  }, [mmSnapshot]);

  const eventCount = mmSnapshot?.total_events || 0;
  const marketCount = mmSnapshot?.total_markets || 0;

  if (!mmSnapshot) {
    return (
      <div className="flex-1 min-h-0 flex flex-col items-center justify-center px-6 py-12">
        <TrendingUp className="w-8 h-8 text-gray-700 mb-3" />
        <span className="text-[13px] text-gray-500 font-medium mb-1">Market Maker</span>
        <span className="text-[11px] text-gray-600 text-center max-w-[300px]">
          Waiting for quote engine data...
        </span>
      </div>
    );
  }

  return (
    <div className="flex-1 min-h-0 flex flex-col">
      <QuoteStatusBar quoteState={mmQuoteState} />
      <InventorySummary inventory={mmInventory} mmSnapshot={mmSnapshot} />
      <PerformanceSection performance={mmPerformance} />

      {/* Markets header */}
      <div className="flex items-center gap-2 px-3 py-1.5 border-b border-gray-800/20 shrink-0">
        <span className="text-[9px] text-gray-500 uppercase tracking-wider font-semibold">Markets</span>
        <span className="text-[8px] text-gray-600 font-mono tabular-nums ml-auto">
          {eventCount} events / {marketCount} markets
        </span>
      </div>

      {/* Scrollable content: market cards + trade log */}
      <div className="flex-1 min-h-0 overflow-y-auto">
        {/* Market cards */}
        <div className="px-3 py-2 space-y-1.5">
          {allMarkets.map(({ ticker, market }) => (
            <MarketCard key={ticker} ticker={ticker} market={market} />
          ))}
          {allMarkets.length === 0 && (
            <div className="py-4 text-center">
              <span className="text-[10px] text-gray-600">No markets active</span>
            </div>
          )}
        </div>

        {/* Trade log inline */}
        <TradeLogSection tradeLog={mmTradeLog} />
      </div>
    </div>
  );
});

MMPanel.displayName = 'MMPanel';

export default MMPanel;
