import React, { memo, useMemo, useState } from 'react';
import { Briefcase, ChevronDown, ChevronRight } from 'lucide-react';

/**
 * PositionPanel - Portfolio overview grouped by event with expand/collapse.
 *
 * Positions arrive server-side filtered (only tracked events).
 * Groups by event_ticker, shows aggregate cost/value/P&L per event,
 * and individual market positions when expanded.
 */

const formatPnL = (cents) => {
  const d = (cents || 0) / 100;
  const prefix = d >= 0 ? '+' : '';
  return `${prefix}$${Math.abs(d).toFixed(2)}`;
};

const formatDollars = (cents) => `$${((cents || 0) / 100).toFixed(2)}`;

const pnlColor = (v) => v > 0 ? 'text-emerald-400' : v < 0 ? 'text-red-400' : 'text-gray-500';

/**
 * Risk badge from event_exposure data.
 */
const RiskBadge = memo(function RiskBadge({ exposure }) {
  if (!exposure) return null;
  const label = exposure === 'ARBITRAGE' ? 'ARB'
    : exposure === 'HIGH_RISK' ? 'RISK'
    : exposure === 'GUARANTEED_LOSS' ? 'LOSS'
    : null;
  if (!label) return null;
  const color = exposure === 'ARBITRAGE' ? 'bg-emerald-500/10 text-emerald-400'
    : exposure === 'HIGH_RISK' ? 'bg-amber-500/10 text-amber-400'
    : 'bg-red-500/10 text-red-400';
  return (
    <span className={`text-[8px] font-semibold uppercase px-1.5 py-0.5 rounded ${color}`}>
      {label}
    </span>
  );
});

/**
 * MarketPositionRow - Single position within an expanded event group.
 */
const MarketPositionRow = memo(function MarketPositionRow({ pos }) {
  const qty = Math.abs(pos.position || pos.quantity || 0);
  const avgPrice = pos.total_cost && qty > 0 ? Math.round(pos.total_cost / qty) : (pos.avg_price ?? 0);
  const mktPrice = pos.current_value && qty > 0 ? Math.round(pos.current_value / qty) : (pos.market_price ?? 0);
  const unrealizedPnL = pos.unrealized_pnl ?? 0;
  const sideColor = pos.side === 'yes'
    ? 'bg-blue-500/10 text-blue-300/80'
    : 'bg-orange-500/10 text-orange-300/80';

  return (
    <div className="flex items-center gap-2 text-[10px] py-1 px-2 hover:bg-gray-800/20 transition-colors">
      <span className="font-mono text-gray-400 truncate flex-1 min-w-0" title={pos.ticker}>
        {pos.ticker}
      </span>
      <span className={`px-1.5 py-px rounded text-[8px] font-semibold uppercase shrink-0 ${sideColor}`}>
        {pos.side}
      </span>
      <span className="font-mono text-gray-300 tabular-nums w-8 text-right shrink-0">{qty}</span>
      <span className="font-mono text-gray-500 tabular-nums w-10 text-right shrink-0">{avgPrice}c</span>
      <span className="font-mono text-cyan-400/70 tabular-nums w-10 text-right shrink-0">{mktPrice}c</span>
      <span className={`font-mono font-semibold tabular-nums w-16 text-right shrink-0 ${pnlColor(unrealizedPnL)}`}>
        {formatPnL(unrealizedPnL)}
      </span>
    </div>
  );
});

/**
 * EventPositionGroup - Collapsible event group with aggregate data.
 */
const EventPositionGroup = memo(function EventPositionGroup({ eventTicker, title, positions, aggregate, riskExposure, expanded, onToggle }) {
  return (
    <div className="border-b border-gray-700/15 last:border-0">
      {/* Header row */}
      <button
        onClick={onToggle}
        className="w-full flex items-center gap-2 px-3 py-2 hover:bg-gray-800/20 transition-colors text-left"
      >
        {expanded
          ? <ChevronDown className="w-3 h-3 text-gray-500 shrink-0" />
          : <ChevronRight className="w-3 h-3 text-gray-500 shrink-0" />
        }
        <span className="text-[11px] text-gray-200 truncate flex-1 min-w-0" title={title || eventTicker}>
          {title || eventTicker}
        </span>
        <span className="text-[9px] text-gray-600 font-mono tabular-nums shrink-0">
          {positions.length} pos
        </span>
        <RiskBadge exposure={riskExposure} />
        <span className="text-[10px] text-gray-500 font-mono tabular-nums w-16 text-right shrink-0">
          {formatDollars(aggregate.totalCost)}
        </span>
        <span className="text-[10px] text-gray-400 font-mono tabular-nums w-16 text-right shrink-0">
          {formatDollars(aggregate.totalValue)}
        </span>
        <span className={`text-[10px] font-mono font-semibold tabular-nums w-16 text-right shrink-0 ${pnlColor(aggregate.totalPnL)}`}>
          {formatPnL(aggregate.totalPnL)}
        </span>
      </button>

      {/* Expanded positions */}
      <div
        className="grid transition-all duration-200 ease-in-out"
        style={{ gridTemplateRows: expanded ? '1fr' : '0fr' }}
      >
        <div className="overflow-hidden">
          <div className="pl-6 pr-3 pb-1">
            {positions.map(pos => (
              <MarketPositionRow key={pos.ticker} pos={pos} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
});

const PositionPanel = ({ tradingState, events }) => {
  const positions = tradingState?.positions || [];
  const eventExposure = tradingState?.event_exposure;

  // Group positions by event_ticker and compute aggregates
  const { groups, totalPnL, eventCount } = useMemo(() => {
    if (positions.length === 0) return { groups: [], totalPnL: 0, eventCount: 0 };

    const byEvent = {};
    for (const pos of positions) {
      const et = pos.event_ticker || 'unknown';
      if (!byEvent[et]) byEvent[et] = [];
      byEvent[et].push(pos);
    }

    let total = 0;
    const grouped = Object.entries(byEvent).map(([eventTicker, posArr]) => {
      const totalCost = posArr.reduce((s, p) => s + (p.total_cost || 0), 0);
      const totalValue = posArr.reduce((s, p) => s + (p.current_value || 0), 0);
      const totalPnL = posArr.reduce((s, p) => s + (p.unrealized_pnl || 0), 0);
      total += totalPnL;

      // Get event title from events Map
      const eventData = events?.get(eventTicker);
      const title = eventData?.title || eventTicker;

      // Get risk exposure for this event
      let riskExposure = null;
      if (eventExposure) {
        const groups = eventExposure.event_groups || eventExposure;
        if (Array.isArray(groups)) {
          const match = groups.find(g => g.event_ticker === eventTicker);
          riskExposure = match?.exposure_type || null;
        }
      }

      return {
        eventTicker,
        title,
        positions: posArr,
        aggregate: { totalCost, totalValue, totalPnL },
        riskExposure,
      };
    });

    // Sort by |totalPnL| descending
    grouped.sort((a, b) => Math.abs(b.aggregate.totalPnL) - Math.abs(a.aggregate.totalPnL));

    return { groups: grouped, totalPnL: total, eventCount: grouped.length };
  }, [positions, events, eventExposure]);

  // Expand/collapse state - auto-expand events with <= 2 positions
  const [expanded, setExpanded] = useState(() => {
    const initial = new Set();
    for (const g of groups) {
      if (g.positions.length <= 2) initial.add(g.eventTicker);
    }
    return initial;
  });

  const toggleExpand = (eventTicker) => {
    setExpanded(prev => {
      const next = new Set(prev);
      if (next.has(eventTicker)) next.delete(eventTicker);
      else next.add(eventTicker);
      return next;
    });
  };

  return (
    <div className="
      bg-gradient-to-br from-gray-900/60 via-gray-900/40 to-gray-950/60
      backdrop-blur-sm rounded-xl
      border border-gray-800/30
      shadow-lg shadow-black/10
      flex flex-col h-[400px]
    ">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800/20 shrink-0">
        <div className="flex items-center gap-2">
          <Briefcase className="w-4 h-4 text-violet-400/70" />
          <span className="text-[11px] font-semibold text-gray-200 uppercase tracking-wider">Positions</span>
          {positions.length > 0 && (
            <span className="text-[10px] text-gray-500 font-mono tabular-nums">
              {eventCount} event{eventCount !== 1 ? 's' : ''} / {positions.length} pos
            </span>
          )}
        </div>
        {positions.length > 0 && (
          <span className={`font-mono text-sm font-semibold tabular-nums ${pnlColor(totalPnL)}`}>
            {formatPnL(totalPnL)}
          </span>
        )}
      </div>

      {/* Body - scrollable */}
      <div className="flex-1 overflow-y-auto min-h-0">
        {positions.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-600">
            <Briefcase className="w-6 h-6 mb-2 opacity-30" />
            <span className="text-[11px]">No open positions</span>
          </div>
        ) : (
          <div>
            {/* Column headers */}
            <div className="flex items-center gap-2 px-3 py-1.5 text-[8px] text-gray-600 uppercase tracking-wider border-b border-gray-800/20 sticky top-0 bg-gray-900/90 backdrop-blur-sm z-10">
              <span className="w-3 shrink-0" />
              <span className="flex-1">Event</span>
              <span className="w-10 shrink-0" />
              <span className="w-16 text-right shrink-0">Cost</span>
              <span className="w-16 text-right shrink-0">Value</span>
              <span className="w-16 text-right shrink-0">P&L</span>
            </div>
            {groups.map(g => (
              <EventPositionGroup
                key={g.eventTicker}
                eventTicker={g.eventTicker}
                title={g.title}
                positions={g.positions}
                aggregate={g.aggregate}
                riskExposure={g.riskExposure}
                expanded={expanded.has(g.eventTicker)}
                onToggle={() => toggleExpand(g.eventTicker)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default memo(PositionPanel);
