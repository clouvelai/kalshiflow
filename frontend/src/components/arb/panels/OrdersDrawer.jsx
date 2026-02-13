import React, { memo, useState, useMemo, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  ScrollText, Wallet, TrendingUp, TrendingDown, Crosshair,
  Briefcase, HeartPulse, ChevronDown, ChevronRight,
  AlertTriangle, CheckCircle, Clock, Trash2, ArrowUpRight, ArrowDownRight, Minus,
} from 'lucide-react';
import SidebarCollapseToggle from '../layout/SidebarCollapseToggle';
import { StatusBadge } from '../ui/EventTradeFeed';
import { fmtTime } from '../utils/formatters';

const fmtCents = (c) => {
  if (c == null) return '--';
  const dollars = (c / 100).toFixed(2);
  return `$${dollars}`;
};

const MetricsStrip = memo(({ tradingState }) => {
  const balance = tradingState?.balance ?? 0;
  const pnl = tradingState?.pnl;
  const posCount = tradingState?.position_count ?? tradingState?.positions?.length ?? 0;
  const orderCount = tradingState?.order_count ?? tradingState?.open_orders?.length ?? 0;
  const pnlValue = pnl?.total_cents ?? pnl?.realized_cents ?? null;
  const pnlColor = pnlValue == null ? 'text-gray-500' : pnlValue >= 0 ? 'text-emerald-400' : 'text-red-400';
  const PnlIcon = pnlValue != null && pnlValue >= 0 ? TrendingUp : TrendingDown;

  return (
    <div className="grid grid-cols-2 gap-x-3 gap-y-1 px-3 py-2 border-b border-gray-800/30 shrink-0">
      <div className="flex items-center gap-1.5">
        <Wallet className="w-3 h-3 text-gray-600" />
        <span className="text-[10px] font-mono text-gray-400">{fmtCents(balance)}</span>
      </div>
      <div className="flex items-center gap-1.5">
        <PnlIcon className={`w-3 h-3 ${pnlColor}`} />
        <span className={`text-[10px] font-mono ${pnlColor}`}>
          {pnlValue != null ? fmtCents(pnlValue) : '--'}
        </span>
      </div>
      <div className="flex items-center gap-1.5">
        <span className="text-[9px] text-gray-600">POS</span>
        <span className="text-[10px] font-mono text-gray-400 tabular-nums">{posCount}</span>
      </div>
      <div className="flex items-center gap-1.5">
        <span className="text-[9px] text-gray-600">ORD</span>
        <span className="text-[10px] font-mono text-amber-400/70 tabular-nums">{orderCount}</span>
      </div>
    </div>
  );
});
MetricsStrip.displayName = 'MetricsStrip';

const SIDEBAR_WIDTH = 340;
const SIDEBAR_COLLAPSED = 40;

// ─── Tab definitions ─────────────────────────────────────────────────────────

const MAIN_TABS = [
  { key: 'orders', label: 'Orders', Icon: ScrollText },
  { key: 'positions', label: 'Positions', Icon: Briefcase },
  { key: 'health', label: 'Health', Icon: HeartPulse },
];

// ─── Orders Tab ──────────────────────────────────────────────────────────────

const ORDER_SUBTABS = ['All', 'Resting', 'Filled', 'Failed'];

const statusToTab = (status) => {
  if (!status) return 'All';
  const s = status.toLowerCase();
  if (s === 'executed' || s === 'filled' || s === 'partial') return 'Filled';
  if (s === 'resting' || s === 'placed' || s === 'pending') return 'Resting';
  if (s === 'cancelled' || s === 'expired' || s === 'canceled') return 'Failed';
  return 'All';
};

const OrderRow = memo(({ trade, isSniper }) => {
  const isBuy = trade.action?.toLowerCase() !== 'sell' && trade.side?.toLowerCase() !== 'no';
  return (
    <div className="flex items-center gap-2 px-2.5 py-1.5 border-b border-gray-800/15 hover:bg-gray-800/20 transition-colors">
      <span className="text-[10px] font-mono text-gray-500 tabular-nums whitespace-nowrap shrink-0">
        {fmtTime(trade.timestamp)}
      </span>
      {isSniper && (
        <Crosshair className="w-2.5 h-2.5 text-cyan-400 shrink-0" title="Sniper order" />
      )}
      <span className={`text-[10px] font-semibold shrink-0 ${isBuy ? 'text-emerald-400' : 'text-red-400'}`}>
        {isBuy ? 'BUY' : 'SELL'}
      </span>
      <span className="text-[10px] font-mono text-gray-300 truncate min-w-0 flex-1">
        {trade.kalshi_ticker || trade.pair_id || '--'}
      </span>
      <span className="text-[10px] font-mono text-gray-400 tabular-nums shrink-0">
        {trade.contracts || '--'}@{trade.price_cents != null ? `${trade.price_cents}c` : '--'}
      </span>
      <StatusBadge status={trade.status} />
    </div>
  );
});
OrderRow.displayName = 'OrderRow';

const OrdersTab = memo(({ trades, sniperOrderIds }) => {
  const [activeSubTab, setActiveSubTab] = useState('All');
  const scrollRef = useRef(null);
  const prevCountRef = useRef(trades.length);

  useEffect(() => {
    if (trades.length > prevCountRef.current && scrollRef.current) {
      scrollRef.current.scrollTop = 0;
    }
    prevCountRef.current = trades.length;
  }, [trades.length]);

  const counts = useMemo(() => {
    const c = { All: trades.length, Resting: 0, Filled: 0, Failed: 0 };
    trades.forEach(t => {
      const tab = statusToTab(t.status);
      if (tab !== 'All') c[tab]++;
    });
    return c;
  }, [trades]);

  const filtered = useMemo(() => {
    if (activeSubTab === 'All') return trades;
    return trades.filter(t => statusToTab(t.status) === activeSubTab);
  }, [trades, activeSubTab]);

  return (
    <>
      <div className="flex items-center gap-3 px-3 py-1.5 border-b border-gray-800/20 shrink-0">
        <span className="text-[9px] font-mono text-gray-600 tabular-nums">{counts.Resting} resting</span>
        <span className="text-[9px] font-mono text-gray-600 tabular-nums">{counts.Filled} filled</span>
        <span className="text-[9px] font-mono text-gray-600 tabular-nums">{counts.Failed} failed</span>
      </div>
      <div className="flex items-center gap-1 px-2 py-1 border-b border-gray-800/30 shrink-0">
        {ORDER_SUBTABS.map(tab => (
          <button
            key={tab}
            onClick={() => setActiveSubTab(tab)}
            className={`px-2 py-0.5 rounded text-[10px] font-medium transition-colors ${
              activeSubTab === tab
                ? 'bg-gray-700/50 text-gray-200'
                : 'text-gray-600 hover:text-gray-400 hover:bg-gray-800/30'
            }`}
          >
            {tab} ({counts[tab]})
          </button>
        ))}
      </div>
      <div ref={scrollRef} className="flex-1 overflow-y-auto min-h-0">
        {filtered.length === 0 ? (
          <div className="flex items-center justify-center py-8 text-[11px] text-gray-700">No orders</div>
        ) : (
          filtered.map(trade => (
            <OrderRow key={trade.id || trade.order_id} trade={trade} isSniper={sniperOrderIds.includes(trade.order_id)} />
          ))
        )}
      </div>
    </>
  );
});
OrdersTab.displayName = 'OrdersTab';

// ─── Positions Tab ───────────────────────────────────────────────────────────

const PositionRow = memo(({ pos, marketPrices }) => {
  const side = pos.position > 0 ? 'YES' : 'NO';
  const sideColor = side === 'YES' ? 'text-emerald-400' : 'text-red-400';
  const qty = Math.abs(pos.position || 0);
  const ticker = pos.ticker || pos.market_ticker || '--';

  // Try to compute P&L from market prices
  const currentPrice = marketPrices?.[ticker];
  let pnl = null;
  if (currentPrice != null && pos.market_exposure != null) {
    const value = side === 'YES' ? qty * currentPrice : qty * (100 - currentPrice);
    pnl = value - Math.abs(pos.market_exposure);
  }
  const pnlColor = pnl == null ? 'text-gray-500' : pnl >= 0 ? 'text-emerald-400' : 'text-red-400';

  return (
    <div className="flex items-center gap-2 px-2.5 py-1.5 border-b border-gray-800/15 hover:bg-gray-800/20 transition-colors">
      <span className={`text-[10px] font-semibold w-6 shrink-0 ${sideColor}`}>{side}</span>
      <span className="text-[10px] font-mono text-gray-300 truncate min-w-0 flex-1">{ticker}</span>
      <span className="text-[10px] font-mono text-gray-400 tabular-nums shrink-0">{qty}</span>
      {pnl != null && (
        <span className={`text-[10px] font-mono tabular-nums shrink-0 ${pnlColor}`}>
          {pnl >= 0 ? '+' : ''}{(pnl / 100).toFixed(2)}
        </span>
      )}
    </div>
  );
});
PositionRow.displayName = 'PositionRow';

const PositionsTab = memo(({ tradingState }) => {
  const positions = tradingState?.positions || [];
  const marketPrices = tradingState?.market_prices || {};
  const eventExposure = tradingState?.event_exposure;

  // Group positions by event_ticker
  const grouped = useMemo(() => {
    const groups = {};
    positions.forEach(pos => {
      if ((pos.position || 0) === 0) return;
      const et = pos.event_ticker || 'ungrouped';
      if (!groups[et]) groups[et] = [];
      groups[et].push(pos);
    });
    return groups;
  }, [positions]);

  const groupKeys = Object.keys(grouped);

  if (groupKeys.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-[11px] text-gray-700">
        No open positions
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto min-h-0">
      {eventExposure && (
        <div className="px-3 py-1.5 border-b border-gray-800/20">
          <span className="text-[9px] text-gray-600">Event Exposure: </span>
          <span className="text-[10px] font-mono text-gray-400">{fmtCents(eventExposure.total_cents)}</span>
        </div>
      )}
      {groupKeys.map(et => (
        <div key={et}>
          <div className="px-2.5 py-1 bg-gray-900/40 border-b border-gray-800/20">
            <span className="text-[9px] font-semibold text-gray-500 uppercase tracking-wider">{et}</span>
          </div>
          {grouped[et].map(pos => (
            <PositionRow key={pos.ticker || pos.market_ticker} pos={pos} marketPrices={marketPrices} />
          ))}
        </div>
      ))}
    </div>
  );
});
PositionsTab.displayName = 'PositionsTab';

// ─── Health Tab ──────────────────────────────────────────────────────────────

const STATUS_COLORS = {
  healthy: { bg: 'bg-emerald-500/15', text: 'text-emerald-400', dot: 'bg-emerald-500' },
  warning: { bg: 'bg-amber-500/15', text: 'text-amber-400', dot: 'bg-amber-500' },
  critical: { bg: 'bg-red-500/15', text: 'text-red-400', dot: 'bg-red-500' },
};

const TREND_ICONS = {
  rising: ArrowUpRight,
  falling: ArrowDownRight,
  stable: Minus,
};

const ACTIVITY_STYLES = {
  stale_order_cancelled: { color: 'text-amber-400', Icon: Trash2 },
  settlement_discovered: { color: 'text-emerald-400', Icon: CheckCircle },
  low_balance: { color: 'text-red-400', Icon: AlertTriangle },
  order_group_cleaned: { color: 'text-cyan-400', Icon: Trash2 },
};

const ActivityEntry = memo(({ entry }) => {
  const style = ACTIVITY_STYLES[entry.type] || { color: 'text-gray-400', Icon: Clock };
  const Icon = style.Icon;
  const ts = entry.ts ? new Date(entry.ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '';

  return (
    <div className="flex items-start gap-2 px-2.5 py-1.5 border-b border-gray-800/10">
      <Icon className={`w-3 h-3 mt-0.5 shrink-0 ${style.color}`} />
      <div className="min-w-0 flex-1">
        <p className="text-[10px] text-gray-300 leading-tight">{entry.message}</p>
      </div>
      <span className="text-[9px] font-mono text-gray-600 tabular-nums shrink-0">{ts}</span>
    </div>
  );
});
ActivityEntry.displayName = 'ActivityEntry';

const HealthTab = memo(({ accountHealth }) => {
  if (!accountHealth) {
    return (
      <div className="flex-1 flex items-center justify-center text-[11px] text-gray-700">
        Waiting for health data...
      </div>
    );
  }

  const h = accountHealth;
  const statusStyle = STATUS_COLORS[h.status] || STATUS_COLORS.healthy;
  const TrendIcon = TREND_ICONS[h.balance_trend] || Minus;
  const trendColor = h.balance_trend === 'rising' ? 'text-emerald-400' : h.balance_trend === 'falling' ? 'text-red-400' : 'text-gray-500';

  return (
    <div className="flex-1 overflow-y-auto min-h-0">
      {/* Status banner */}
      <div className={`flex items-center gap-2 px-3 py-2 ${statusStyle.bg} border-b border-gray-800/20`}>
        <div className={`w-2 h-2 rounded-full ${statusStyle.dot}`} />
        <span className={`text-[11px] font-semibold uppercase ${statusStyle.text}`}>{h.status}</span>
        {h.drawdown_pct > 0 && (
          <span className="text-[10px] font-mono text-gray-400 ml-auto">-{h.drawdown_pct}% drawdown</span>
        )}
      </div>

      {/* Balance card */}
      <div className="px-3 py-2 border-b border-gray-800/20">
        <div className="flex items-center justify-between">
          <span className="text-[9px] text-gray-600 uppercase">Balance</span>
          <div className="flex items-center gap-1">
            <TrendIcon className={`w-3 h-3 ${trendColor}`} />
            <span className="text-[10px] font-mono text-gray-300">{fmtCents(h.balance_cents)}</span>
          </div>
        </div>
        <div className="flex items-center gap-3 mt-1">
          <span className="text-[9px] text-gray-600">Peak: <span className="text-gray-500 font-mono">{fmtCents(h.balance_peak_cents)}</span></span>
        </div>
      </div>

      {/* Settlements */}
      <div className="px-3 py-2 border-b border-gray-800/20">
        <div className="flex items-center justify-between">
          <span className="text-[9px] text-gray-600 uppercase">Settlements</span>
          <span className="text-[10px] font-mono text-gray-400">{h.settlement_count_session || 0}</span>
        </div>
        {h.total_realized_pnl_cents != null && (
          <div className="mt-0.5 group relative">
            <span className="text-[9px] text-gray-600">Settlement P&L: </span>
            <span className={`text-[10px] font-mono ${h.total_realized_pnl_cents >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
              {h.total_realized_pnl_cents >= 0 ? '+' : ''}{fmtCents(h.total_realized_pnl_cents)}
            </span>
            <span className="invisible group-hover:visible absolute left-0 top-full mt-1 z-50 px-2 py-1 text-[9px] text-gray-300 bg-gray-900 border border-gray-700 rounded shadow-lg whitespace-nowrap">
              Net profit/loss from settled positions only
            </span>
          </div>
        )}
        {h.recent_settlements?.length > 0 && (
          <div className="mt-1.5 space-y-0.5">
            {h.recent_settlements.slice(0, 10).map((s, i) => {
              const pnl = s.pnl_cents ?? s.revenue_cents;
              return (
                <div key={i} className="flex items-center gap-2 text-[9px]">
                  <span className="font-mono text-gray-400 truncate flex-1">{s.ticker || '--'}</span>
                  <span className={`font-semibold ${s.result === 'yes' ? 'text-emerald-400' : s.result === 'no' ? 'text-red-400' : 'text-gray-500'}`}>
                    {(s.result || '').toUpperCase()}
                  </span>
                  <span className={`font-mono tabular-nums ${pnl == null ? 'text-gray-500' : pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {pnl != null && pnl >= 0 ? '+' : ''}{fmtCents(pnl)}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Hygiene counters */}
      {(h.stale_orders_cleaned > 0 || h.orphaned_groups_cleaned > 0) && (
        <div className="px-3 py-1.5 border-b border-gray-800/20 flex items-center gap-3">
          {h.stale_orders_cleaned > 0 && (
            <span className="text-[9px] text-amber-400/70">
              <Trash2 className="w-2.5 h-2.5 inline mr-0.5" />{h.stale_orders_cleaned} stale orders
            </span>
          )}
          {h.orphaned_groups_cleaned > 0 && (
            <span className="text-[9px] text-cyan-400/70">
              <Trash2 className="w-2.5 h-2.5 inline mr-0.5" />{h.orphaned_groups_cleaned} groups
            </span>
          )}
        </div>
      )}

      {/* Stale positions warning */}
      {h.stale_positions?.length > 0 && (
        <div className="px-3 py-2 border-b border-gray-800/20 bg-amber-500/5">
          <div className="flex items-center gap-1.5 mb-1">
            <AlertTriangle className="w-3 h-3 text-amber-400" />
            <span className="text-[10px] font-semibold text-amber-400">Stale Positions</span>
          </div>
          {h.stale_positions.map((sp, i) => (
            <div key={i} className="text-[9px] text-gray-400 font-mono pl-4">
              {sp.ticker} ({sp.side} x{sp.quantity}) — {sp.reason}
            </div>
          ))}
        </div>
      )}

      {/* Activity log */}
      <div className="px-2.5 py-1 bg-gray-900/30 border-b border-gray-800/20">
        <span className="text-[9px] font-semibold text-gray-500 uppercase tracking-wider">Activity</span>
      </div>
      {h.activity_log?.length > 0 ? (
        h.activity_log.map((entry, i) => <ActivityEntry key={i} entry={entry} />)
      ) : (
        <div className="flex items-center justify-center py-4 text-[10px] text-gray-700">No activity yet</div>
      )}
    </div>
  );
});
HealthTab.displayName = 'HealthTab';

// ─── Main Sidebar ────────────────────────────────────────────────────────────

const OrdersSidebar = memo(({ trades = [], tradingState, collapsed, onToggle, sniperOrderIds = [], accountHealth }) => {
  const [activeMainTab, setActiveMainTab] = useState('orders');

  const posCount = tradingState?.positions?.filter(p => p.position !== 0)?.length ?? 0;
  const restingCount = useMemo(() => trades.filter(t => statusToTab(t.status) === 'Resting').length, [trades]);

  return (
    <motion.div
      className="shrink-0 flex flex-col bg-gray-950/60 border-l border-gray-800/40 overflow-hidden"
      animate={{ width: collapsed ? SIDEBAR_COLLAPSED : SIDEBAR_WIDTH }}
      transition={{ type: 'spring', stiffness: 400, damping: 30 }}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-2 py-1.5 border-b border-gray-800/30 shrink-0">
        {!collapsed && (
          <span className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider pl-1">
            {MAIN_TABS.find(t => t.key === activeMainTab)?.label || 'Orders'}
          </span>
        )}
        <SidebarCollapseToggle collapsed={collapsed} onToggle={onToggle} side="right" />
      </div>

      {collapsed ? (
        <div className="flex flex-col items-center gap-3 py-3">
          {MAIN_TABS.map(tab => {
            const count = tab.key === 'orders' ? restingCount : tab.key === 'positions' ? posCount : 0;
            return (
              <button
                key={tab.key}
                onClick={() => { setActiveMainTab(tab.key); onToggle(); }}
                className="flex flex-col items-center gap-0.5"
                title={tab.label}
              >
                <tab.Icon className={`w-4 h-4 ${activeMainTab === tab.key ? 'text-gray-300' : 'text-gray-600'}`} />
                {count > 0 && (
                  <span className="text-[8px] font-mono text-amber-400/70 tabular-nums">{count}</span>
                )}
              </button>
            );
          })}
        </div>
      ) : (
        <>
          {/* Account metrics (shared) */}
          <MetricsStrip tradingState={tradingState} />

          {/* Main tab bar */}
          <div className="flex items-center gap-0.5 px-2 py-1 border-b border-gray-800/30 shrink-0">
            {MAIN_TABS.map(tab => (
              <button
                key={tab.key}
                onClick={() => setActiveMainTab(tab.key)}
                className={`flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium transition-colors ${
                  activeMainTab === tab.key
                    ? 'bg-gray-700/50 text-gray-200'
                    : 'text-gray-600 hover:text-gray-400 hover:bg-gray-800/30'
                }`}
              >
                <tab.Icon className="w-3 h-3" />
                {tab.label}
              </button>
            ))}
          </div>

          {/* Tab content */}
          {activeMainTab === 'orders' && (
            <OrdersTab trades={trades} sniperOrderIds={sniperOrderIds} />
          )}
          {activeMainTab === 'positions' && (
            <PositionsTab tradingState={tradingState} />
          )}
          {activeMainTab === 'health' && (
            <HealthTab accountHealth={accountHealth} />
          )}
        </>
      )}
    </motion.div>
  );
});

OrdersSidebar.displayName = 'OrdersSidebar';

export default OrdersSidebar;
