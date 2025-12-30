import React, { useState, useEffect, useRef, memo } from 'react';
import { ChevronRight, ChevronDown, Briefcase, TrendingUp, TrendingDown, Radio, Wifi, WifiOff } from 'lucide-react';
import { formatPnLCurrency, formatCentsAsCurrency, formatRelativeTime, getSideClasses, getPnLColor } from '../../../utils/v3-trader';

/**
 * Check if a WebSocket update is recent (within threshold seconds)
 */
const isRecentWsUpdate = (lastWsUpdateTime, thresholdSeconds = 3) => {
  if (!lastWsUpdateTime) return false;
  const now = Date.now() / 1000; // Convert to seconds
  return (now - lastWsUpdateTime) < thresholdSeconds;
};

/**
 * Check if a market has closed (close time is in the past)
 */
const isMarketClosed = (closeTime) => {
  if (!closeTime) return false;
  const closeDate = new Date(closeTime);
  return closeDate.getTime() < Date.now();
};

/**
 * LiveIndicator - Shows when data is being updated in real-time via WebSocket
 */
const LiveIndicator = memo(({ isLive, lastUpdateTime }) => {
  const [pulseKey, setPulseKey] = useState(0);

  // Trigger pulse animation when lastUpdateTime changes
  useEffect(() => {
    if (lastUpdateTime) {
      setPulseKey(prev => prev + 1);
    }
  }, [lastUpdateTime]);

  if (!isLive) {
    return (
      <span className="inline-flex items-center gap-1 text-gray-500" title="No recent WebSocket updates">
        <WifiOff className="w-3 h-3" />
      </span>
    );
  }

  return (
    <span
      key={pulseKey}
      className="inline-flex items-center gap-1 text-emerald-400"
      title="Live WebSocket data"
    >
      <span className="relative flex h-2 w-2">
        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
        <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
      </span>
    </span>
  );
});

LiveIndicator.displayName = 'LiveIndicator';

/**
 * Calculate position metrics for a single position
 * Centralized to avoid code duplication
 */
const calcPositionMetrics = (pos) => {
  const qty = Math.abs(pos.position || 0);
  if (qty === 0) {
    return { qty: 0, costPerContract: 0, valuePerContract: 0, totalCost: 0, totalValue: 0, unrealizedPnL: 0 };
  }

  const costPerContract = Math.round((pos.total_traded || 0) / qty);
  const kalshiValuePerContract = Math.round((pos.market_exposure || 0) / qty);

  let valuePerContract;
  if (pos.market_last > 0) {
    valuePerContract = pos.side === 'no' ? (100 - pos.market_last) : pos.market_last;
  } else if (pos.market_bid > 0) {
    valuePerContract = pos.market_bid;
  } else {
    valuePerContract = kalshiValuePerContract;
  }

  const totalCost = costPerContract * qty;
  const totalValue = valuePerContract * qty;
  const unrealizedPnL = totalValue - totalCost;

  return { qty, costPerContract, valuePerContract, totalCost, totalValue, unrealizedPnL };
};

/**
 * PositionRow - Memoized row component for position table
 */
const PositionRow = memo(({ pos, index, isRecentlyChanged }) => {
  const { qty, costPerContract, valuePerContract, unrealizedPnL } = calcPositionMetrics(pos);
  const pnlPercent = costPerContract > 0
    ? (((valuePerContract - costPerContract) / costPerContract) * 100).toFixed(1)
    : 0;

  // Check if this position has recent WebSocket price updates
  const hasLiveData = isRecentWsUpdate(pos.last_ws_update_time, 5);
  const isWsSource = pos.price_source === 'ws_ticker';

  // Check if market has closed (pending settlement)
  const marketClosed = isMarketClosed(pos.market_close_time);

  return (
    <tr
      className={`border-b border-gray-700/30 transition-all duration-500
        ${marketClosed
          ? 'opacity-50 bg-gray-900/30'
          : isRecentlyChanged
            ? 'border-l-2 border-l-emerald-400/70 bg-emerald-900/10 hover:bg-gray-800/50'
            : hasLiveData
              ? 'border-l-2 border-l-emerald-500/40 bg-emerald-900/5 hover:bg-gray-800/50'
              : 'border-l-2 border-l-transparent hover:bg-gray-800/50'
        }`}
    >
      <td className="px-3 py-2 font-mono text-xs">
        <div className="flex items-center gap-1.5">
          {/* Live indicator for WebSocket updates (hide for closed markets) */}
          {!marketClosed && <LiveIndicator isLive={hasLiveData} lastUpdateTime={pos.last_ws_update_time} />}
          {marketClosed && <span className="w-2 h-2 rounded-full bg-yellow-500/50" title="Pending settlement" />}
          <span className={marketClosed ? 'text-gray-500' : 'text-gray-300'}>{pos.ticker}</span>
        </div>
      </td>
      <td className="px-3 py-2 text-center">
        <span className={`px-2 py-0.5 rounded text-xs font-bold uppercase ${
          marketClosed
            ? 'bg-gray-800/50 text-gray-500 border border-gray-700/50'
            : pos.side === 'yes'
              ? 'bg-green-900/30 text-green-400 border border-green-700/50'
              : 'bg-red-900/30 text-red-400 border border-red-700/50'
        }`}>
          {pos.side}
        </span>
      </td>
      <td className={`px-3 py-2 text-right font-mono ${marketClosed ? 'text-gray-500' : 'text-gray-300'}`}>{qty}</td>
      <td className={`px-3 py-2 text-right font-mono ${marketClosed ? 'text-gray-600' : 'text-gray-400'}`}>{costPerContract}c</td>
      <td className={`px-3 py-2 text-right font-mono ${marketClosed ? 'text-gray-500' : hasLiveData ? 'text-emerald-300' : 'text-gray-300'}`}>
        {valuePerContract}c
        {!marketClosed && hasLiveData && <span className="ml-1 text-emerald-500 text-[10px]">LIVE</span>}
      </td>
      <td className={`px-3 py-2 text-right font-mono ${marketClosed ? 'text-gray-600' : getPnLColor(valuePerContract - costPerContract)}`}>
        {valuePerContract - costPerContract >= 0 ? '+' : ''}{valuePerContract - costPerContract}c
      </td>
      <td className={`px-3 py-2 text-right font-mono font-bold ${marketClosed ? 'text-gray-500' : getPnLColor(unrealizedPnL)}`}>
        {formatPnLCurrency(unrealizedPnL)}
        <span className="text-xs ml-1 opacity-70">({pnlPercent}%)</span>
      </td>
      <td className="px-3 py-2 text-center font-mono text-xs">
        {pos.market_close_time ? (
          marketClosed ? (
            <span className="px-2 py-0.5 rounded bg-yellow-900/30 text-yellow-500 border border-yellow-700/30 text-[10px] font-medium">
              SETTLING
            </span>
          ) : (
            <span className="text-gray-400" title={new Date(pos.market_close_time).toLocaleString()}>
              {formatRelativeTime(pos.market_close_time)}
            </span>
          )
        ) : (
          <span className="text-gray-600">-</span>
        )}
      </td>
    </tr>
  );
});

PositionRow.displayName = 'PositionRow';

/**
 * SideSummaryCard - Compact summary card for YES/NO positions
 * Note: Not memoized because positions array reference stays same but ticker data changes via WebSocket
 */
const SideSummaryCard = ({ side, positions }) => {
  const isYes = side === 'yes';
  const totalQty = positions.reduce((sum, p) => sum + Math.abs(p.position || 0), 0);
  const aggregated = positions.reduce((acc, pos) => {
    const metrics = calcPositionMetrics(pos);
    acc.totalCost += metrics.totalCost;
    acc.totalValue += metrics.totalValue;
    acc.unrealizedPnL += metrics.unrealizedPnL;
    return acc;
  }, { totalCost: 0, totalValue: 0, unrealizedPnL: 0 });

  const pnlPercent = aggregated.totalCost > 0
    ? ((aggregated.unrealizedPnL / aggregated.totalCost) * 100).toFixed(1)
    : 0;

  return (
    <div className={`flex-1 rounded-lg border p-3 ${
      isYes
        ? 'bg-green-900/20 border-green-700/40'
        : 'bg-red-900/20 border-red-700/40'
    }`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className={`text-sm font-bold uppercase ${isYes ? 'text-green-400' : 'text-red-400'}`}>
            {side}
          </span>
          <span className="text-xs text-gray-500">({positions.length})</span>
        </div>
        {aggregated.unrealizedPnL >= 0 ? (
          <TrendingUp className={`w-4 h-4 ${isYes ? 'text-green-500' : 'text-red-500'}`} />
        ) : (
          <TrendingDown className={`w-4 h-4 ${isYes ? 'text-green-500' : 'text-red-500'}`} />
        )}
      </div>

      <div className="space-y-1 text-xs">
        <div className="flex justify-between">
          <span className="text-gray-500">Contracts</span>
          <span className="font-mono text-gray-300">{totalQty.toLocaleString()}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-500">Cost Basis</span>
          <span className="font-mono text-gray-400">{formatCentsAsCurrency(aggregated.totalCost)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-500">Mkt Value</span>
          <span className="font-mono text-gray-300">{formatCentsAsCurrency(aggregated.totalValue)}</span>
        </div>
        <div className={`flex justify-between pt-1 border-t ${isYes ? 'border-green-800/50' : 'border-red-800/50'}`}>
          <span className="text-gray-500">P&L</span>
          <span className={`font-mono font-bold ${getPnLColor(aggregated.unrealizedPnL)}`}>
            {formatPnLCurrency(aggregated.unrealizedPnL)}
            <span className="ml-1 opacity-70">({pnlPercent}%)</span>
          </span>
        </div>
      </div>
    </div>
  );
};

/**
 * PortfolioSummaryBar - Total portfolio summary statistics
 * Note: Not memoized because positions array reference stays same but ticker data changes via WebSocket
 */
const PortfolioSummaryBar = ({ positions }) => {
  const totals = positions.reduce((acc, pos) => {
    const metrics = calcPositionMetrics(pos);
    acc.totalCost += metrics.totalCost;
    acc.totalValue += metrics.totalValue;
    acc.unrealizedPnL += metrics.unrealizedPnL;
    acc.totalContracts += metrics.qty;
    return acc;
  }, { totalCost: 0, totalValue: 0, unrealizedPnL: 0, totalContracts: 0 });

  const pnlPercent = totals.totalCost > 0
    ? ((totals.unrealizedPnL / totals.totalCost) * 100).toFixed(1)
    : 0;

  return (
    <div className="flex items-center justify-between px-3 py-2 bg-gray-800/50 rounded-lg border border-gray-700/50 text-xs">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-1.5">
          <span className="text-gray-500">Total Cost:</span>
          <span className="font-mono text-gray-300">{formatCentsAsCurrency(totals.totalCost)}</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-gray-500">Mkt Value:</span>
          <span className="font-mono text-gray-300">{formatCentsAsCurrency(totals.totalValue)}</span>
        </div>
      </div>
      <div className="flex items-center gap-1.5">
        <span className="text-gray-500">Unrealized P&L:</span>
        <span className={`font-mono font-bold ${getPnLColor(totals.unrealizedPnL)}`}>
          {formatPnLCurrency(totals.unrealizedPnL)}
        </span>
        <span className={`font-mono ${getPnLColor(totals.unrealizedPnL)} opacity-70`}>
          ({pnlPercent}%)
        </span>
      </div>
    </div>
  );
};

/**
 * PositionListPanel - Detailed position breakdown with per-position P&L
 */
const PositionListPanel = ({ positions, positionListener, sessionUpdates }) => {
  const [isExpanded, setIsExpanded] = useState(true);
  const [changedTickers, setChangedTickers] = useState(new Set());
  const prevPositionsRef = useRef({});
  const isFirstRender = useRef(true);

  // Detect position changes and trigger subtle highlight animation
  useEffect(() => {
    if (!positions || positions.length === 0) return;

    if (isFirstRender.current) {
      isFirstRender.current = false;
      const newRef = {};
      positions.forEach(pos => { newRef[pos.ticker] = { ...pos }; });
      prevPositionsRef.current = newRef;
      return;
    }

    const changed = new Set();
    positions.forEach(pos => {
      const prev = prevPositionsRef.current[pos.ticker];
      if (prev) {
        if (prev.position !== pos.position ||
            prev.total_traded !== pos.total_traded ||
            prev.unrealized_pnl !== pos.unrealized_pnl ||
            prev.market_bid !== pos.market_bid ||
            prev.market_last !== pos.market_last) {
          changed.add(pos.ticker);
        }
      } else {
        changed.add(pos.ticker);
      }
    });

    const newRef = {};
    positions.forEach(pos => {
      newRef[pos.ticker] = { ...pos };
    });
    prevPositionsRef.current = newRef;

    if (changed.size > 0) {
      setChangedTickers(changed);
      const timeout = setTimeout(() => setChangedTickers(new Set()), 2000);
      return () => clearTimeout(timeout);
    }
  }, [positions]);

  if (!positions || positions.length === 0) {
    return null;
  }

  // Sort positions: open markets by close time (soonest first), then settling at bottom
  const sortedPositions = [...positions].sort((a, b) => {
    const now = Date.now();
    const aTime = a.market_close_time ? new Date(a.market_close_time).getTime() : Infinity;
    const bTime = b.market_close_time ? new Date(b.market_close_time).getTime() : Infinity;
    const aClosed = aTime < now;
    const bClosed = bTime < now;

    // Settling positions go to bottom
    if (aClosed && !bClosed) return 1;
    if (!aClosed && bClosed) return -1;

    // Within same group, sort by close time (soonest first for open, most recent for settling)
    return aTime - bTime;
  });

  // Count settling positions for header display
  const settlingCount = positions.filter(p => isMarketClosed(p.market_close_time)).length;

  // Separate YES/NO positions for summary cards (use unsorted for consistent aggregation)
  const yesPositions = positions.filter(p => p.side === 'yes');
  const noPositions = positions.filter(p => p.side === 'no');

  // Count positions with live WebSocket data
  const liveDataCount = positions.filter(p => isRecentWsUpdate(p.last_ws_update_time, 5)).length;
  const hasAnyLiveData = liveDataCount > 0;

  return (
    <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl border border-gray-800 p-4 mb-4">
      {/* Header with expand/collapse */}
      <div
        className="flex items-center justify-between cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center space-x-2">
          <Briefcase className="w-4 h-4 text-purple-400" />
          <h3 className="text-sm font-bold text-gray-300 uppercase tracking-wider">Open Positions</h3>
          <span className="text-xs text-gray-500">
            ({positions.length - settlingCount}{settlingCount > 0 && <span className="text-yellow-500/70"> + {settlingCount} settling</span>})
          </span>
          {/* Position listener status indicator */}
          {positionListener && (
            <div className="flex items-center gap-1.5 ml-3 px-2 py-0.5 bg-gray-800/50 rounded-full">
              <span className={`w-1.5 h-1.5 rounded-full ${
                positionListener.connected
                  ? 'bg-emerald-400 animate-pulse'
                  : 'bg-yellow-400'
              }`} />
              <span className="text-xs text-gray-500">
                {positionListener.connected ? 'Live' : 'Polling'}
              </span>
              {positionListener.positions_received > 0 && (
                <span className="text-xs text-gray-600">
                  ({positionListener.positions_received} updates)
                </span>
              )}
            </div>
          )}
          {/* Live ticker data indicator - shows when WebSocket price updates are flowing */}
          {hasAnyLiveData && (
            <div className="flex items-center gap-1.5 ml-2 px-2 py-0.5 bg-emerald-900/30 rounded-full border border-emerald-800/30">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
              <span className="text-xs text-emerald-400 font-medium">
                {liveDataCount}/{positions.length} LIVE
              </span>
            </div>
          )}
          {/* Session updates count (fallback when no live data) */}
          {!hasAnyLiveData && sessionUpdates && sessionUpdates.count > 0 && (
            <div className="flex items-center gap-1.5 ml-2 px-2 py-0.5 bg-gray-800/50 rounded-full border border-gray-700/30">
              <WifiOff className="w-3 h-3 text-gray-500" />
              <span className="text-xs text-gray-500">
                {sessionUpdates.count} updated this session
              </span>
            </div>
          )}
        </div>
        {isExpanded ? (
          <ChevronDown className="w-4 h-4 text-gray-500" />
        ) : (
          <ChevronRight className="w-4 h-4 text-gray-500" />
        )}
      </div>

      {isExpanded && (
        <div className="mt-4 space-y-3">
          {/* YES/NO Summary Cards */}
          <div className="flex gap-3">
            <SideSummaryCard side="yes" positions={yesPositions} />
            <SideSummaryCard side="no" positions={noPositions} />
          </div>

          {/* Portfolio Summary Bar */}
          <PortfolioSummaryBar positions={positions} />

          {/* Position Table */}
          <div className="bg-gray-800/30 rounded-lg border border-gray-700/50 overflow-hidden max-h-[220px] overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 z-10">
                <tr className="bg-gray-900 border-b border-gray-700/50">
                  <th className="px-3 py-2 text-left text-xs text-gray-500 uppercase font-medium">Ticker</th>
                  <th className="px-3 py-2 text-center text-xs text-gray-500 uppercase font-medium">Side</th>
                  <th className="px-3 py-2 text-right text-xs text-gray-500 uppercase font-medium">Qty</th>
                  <th className="px-3 py-2 text-right text-xs text-gray-500 uppercase font-medium" title="Entry cost per contract">Cost/C</th>
                  <th className="px-3 py-2 text-right text-xs text-gray-500 uppercase font-medium" title="Current market bid">Value/C</th>
                  <th className="px-3 py-2 text-right text-xs text-gray-500 uppercase font-medium" title="Unrealized P&L per contract">Unreal/C</th>
                  <th className="px-3 py-2 text-right text-xs text-gray-500 uppercase font-medium">P&L</th>
                  <th className="px-3 py-2 text-center text-xs text-gray-500 uppercase font-medium" title="When market closes">Closes</th>
                </tr>
              </thead>
              <tbody>
                {sortedPositions.map((pos, index) => (
                  <PositionRow
                    key={pos.ticker || index}
                    pos={pos}
                    index={index}
                    isRecentlyChanged={changedTickers.has(pos.ticker)}
                  />
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default memo(PositionListPanel);
