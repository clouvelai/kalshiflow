import React, { useState, useEffect, useRef, memo, useMemo } from 'react';
import { ChevronRight, ChevronDown, Briefcase, TrendingUp, TrendingDown, Radio, Wifi, WifiOff, DollarSign, Layers, Activity, Clock } from 'lucide-react';
import { formatPnLCurrency, formatCentsAsCurrency, formatRelativeTime, getSideClasses, getPnLColor } from '../../../utils/v3-trader';

/**
 * Strategy configuration for display and styling
 */
const STRATEGY_CONFIG = {
  deep_agent: {
    label: 'DEEP',
    bgClass: 'bg-emerald-900/30',
    borderClass: 'border-emerald-700/40',
    textClass: 'text-emerald-400',
  },
};

/**
 * Get strategy display config, with fallback for unknown strategies
 */
const getStrategyConfig = (strategyId) => {
  if (!strategyId) {
    return null;
  }
  return STRATEGY_CONFIG[strategyId] || {
    label: strategyId.substring(0, 4).toUpperCase(),
    bgClass: 'bg-purple-900/30',
    borderClass: 'border-purple-700/40',
    textClass: 'text-purple-400',
  };
};

/**
 * StrategyBadge - Small strategy identifier badge for positions
 */
const StrategyBadge = memo(({ strategyId }) => {
  const config = getStrategyConfig(strategyId);
  if (!config) return null;

  return (
    <span className={`
      inline-flex items-center px-1 py-0.5 rounded text-[9px] font-bold
      ${config.bgClass} ${config.textClass} border ${config.borderClass}
    `} title={strategyId}>
      {config.label}
    </span>
  );
});

StrategyBadge.displayName = 'StrategyBadge';

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
 * StatBox - Enhanced stat display component with gradient and icon
 * Matches SettlementsPanel styling
 */
const StatBox = memo(({ label, value, subtitle, valueClass = 'text-white', icon: Icon, accentColor = 'gray' }) => {
  const accentStyles = {
    cyan: 'border-cyan-500/30 bg-gradient-to-br from-cyan-950/30 via-gray-900/50 to-gray-900/30',
    gray: 'border-gray-700/50 bg-gradient-to-br from-gray-800/40 via-gray-900/50 to-gray-900/30',
    green: 'border-green-500/30 bg-gradient-to-br from-green-950/30 via-gray-900/50 to-gray-900/30',
    red: 'border-red-500/30 bg-gradient-to-br from-red-950/30 via-gray-900/50 to-gray-900/30',
    emerald: 'border-emerald-500/30 bg-gradient-to-br from-emerald-950/30 via-gray-900/50 to-gray-900/30',
    purple: 'border-purple-500/30 bg-gradient-to-br from-purple-950/30 via-gray-900/50 to-gray-900/30',
  };

  return (
    <div className={`
      rounded-xl p-4 border backdrop-blur-sm
      transition-all duration-300 ease-out
      hover:scale-[1.02] hover:shadow-lg hover:shadow-black/20
      ${accentStyles[accentColor] || accentStyles.gray}
    `}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs text-gray-500 uppercase tracking-wider font-medium">{label}</span>
        {Icon && <Icon className="w-3.5 h-3.5 text-gray-600" />}
      </div>
      <div className={`text-2xl font-mono font-bold tracking-tight ${valueClass}`}>
        {value}
      </div>
      {subtitle && (
        <div className="text-xs text-gray-500 mt-1 font-mono">{subtitle}</div>
      )}
    </div>
  );
});

StatBox.displayName = 'StatBox';

/**
 * DecisionBox - Compact stat component
 * Matches SettlementsPanel styling
 */
const DecisionBox = memo(({ label, value, valueClass = 'text-gray-400', icon: Icon }) => (
  <div className="
    bg-gradient-to-b from-gray-800/40 to-gray-900/40
    rounded-lg p-3 border border-gray-700/40
    transition-all duration-200 ease-out
    hover:border-gray-600/50 hover:bg-gray-800/50
    text-center
  ">
    <div className="flex items-center justify-center space-x-1 mb-1.5">
      {Icon && <Icon className="w-3 h-3 text-gray-600" />}
      <span className="text-[10px] text-gray-500 uppercase tracking-wider font-medium">{label}</span>
    </div>
    <div className={`text-lg font-mono font-bold ${valueClass}`}>
      {value}
    </div>
  </div>
));

DecisionBox.displayName = 'DecisionBox';

/**
 * Calculate position metrics for a single position
 * Centralized to avoid code duplication
 *
 * CRITICAL: Uses backend-provided values (total_cost, current_value, unrealized_pnl)
 * instead of recalculating. The backend computes these from:
 * - total_cost: tracked filled orders
 * - current_value: current_price × quantity from live market data
 * - unrealized_pnl: current_value - total_cost
 */
const calcPositionMetrics = (pos) => {
  const qty = Math.abs(pos.position || 0);
  if (qty === 0) {
    return { qty: 0, costPerContract: 0, valuePerContract: 0, totalCost: 0, totalValue: 0, unrealizedPnL: 0, costBasisEstimated: true };
  }

  // Use backend-provided cost basis (computed from tracked filled orders)
  const totalCost = pos.total_cost ?? 0;
  const costPerContract = Math.round(totalCost / qty);

  // Use backend-provided current value (computed from market price × qty)
  const totalValue = pos.current_value ?? 0;
  const valuePerContract = Math.round(totalValue / qty);

  // Use backend-provided unrealized P&L (already computed correctly)
  const unrealizedPnL = pos.unrealized_pnl ?? (totalValue - totalCost);

  // Track if cost basis is estimated (for UI indicator)
  const costBasisEstimated = pos.cost_basis_estimated ?? true;

  return { qty, costPerContract, valuePerContract, totalCost, totalValue, unrealizedPnL, costBasisEstimated };
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
          ? 'border-l-2 border-l-amber-500/40 opacity-60 bg-amber-900/5'
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
          {marketClosed && <span className="w-2 h-2 rounded-full bg-amber-500/50" title="Pending settlement" />}
          <span className={marketClosed ? 'text-gray-500' : 'text-gray-300'}>{pos.ticker}</span>
          {/* Strategy badge */}
          <StrategyBadge strategyId={pos.strategy_id} />
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
        <div className="flex items-center justify-end gap-1.5">
          <span>{valuePerContract}c</span>
          {/* Per-contract unrealized P&L delta */}
          <span className={`text-[10px] ${marketClosed ? 'text-gray-600' : getPnLColor(valuePerContract - costPerContract)}`}>
            {valuePerContract - costPerContract >= 0 ? '+' : ''}{valuePerContract - costPerContract}c
          </span>
          {!marketClosed && hasLiveData && <span className="text-emerald-500 text-[10px]">LIVE</span>}
        </div>
      </td>
      <td className={`px-3 py-2 text-right font-mono font-bold ${marketClosed ? 'text-gray-500' : getPnLColor(unrealizedPnL)}`}>
        {formatPnLCurrency(unrealizedPnL)}
        <span className="text-xs ml-1 opacity-70">({pnlPercent}%)</span>
      </td>
      <td className="px-3 py-2 text-center font-mono text-xs">
        {pos.market_close_time ? (
          marketClosed ? (
            <span className="px-2 py-0.5 rounded bg-amber-900/30 text-amber-500 border border-amber-700/30 text-[10px] font-medium">
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
 * PositionListPanel - Detailed position breakdown with per-position P&L
 */
const PositionListPanel = ({ positions, positionListener, sessionUpdates }) => {
  const [isExpanded, setIsExpanded] = useState(true);
  const [sideFilter, setSideFilter] = useState('all'); // 'all', 'yes', 'no'
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

  // Calculate stats using useMemo
  const stats = useMemo(() => {
    if (!positions || positions.length === 0) {
      return {
        totalPnL: 0, totalCost: 0, totalValue: 0, totalContracts: 0,
        liveCount: 0, avgSize: 0, bestGain: 0, maxLoss: 0, settlingCount: 0, openCount: 0
      };
    }

    let totalPnL = 0, totalCost = 0, totalValue = 0, totalContracts = 0;
    let bestGain = -Infinity, maxLoss = Infinity;
    let liveCount = 0, settlingCount = 0;

    positions.forEach(pos => {
      const metrics = calcPositionMetrics(pos);
      totalPnL += metrics.unrealizedPnL;
      totalCost += metrics.totalCost;
      totalValue += metrics.totalValue;
      totalContracts += metrics.qty;

      if (metrics.unrealizedPnL > bestGain) bestGain = metrics.unrealizedPnL;
      if (metrics.unrealizedPnL < maxLoss) maxLoss = metrics.unrealizedPnL;

      if (isRecentWsUpdate(pos.last_ws_update_time, 5)) liveCount++;
      if (isMarketClosed(pos.market_close_time)) settlingCount++;
    });

    const avgSize = positions.length > 0 ? totalCost / positions.length : 0;

    return {
      totalPnL,
      totalCost,
      totalValue,
      totalContracts,
      liveCount,
      avgSize,
      bestGain: bestGain === -Infinity ? 0 : bestGain,
      maxLoss: maxLoss === Infinity ? 0 : maxLoss,
      settlingCount,
      openCount: positions.length - settlingCount
    };
  }, [positions]);

  // Sort positions: open markets by close time (soonest first), then settling at bottom
  const sortedPositions = useMemo(() => {
    if (!positions || positions.length === 0) return [];
    return [...positions].sort((a, b) => {
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
  }, [positions]);

  // Filter positions by side
  const filteredPositions = useMemo(() => {
    if (sideFilter === 'all') return sortedPositions;
    return sortedPositions.filter(p => p.side === sideFilter);
  }, [sortedPositions, sideFilter]);

  // Split into open and settling
  const openPositions = useMemo(() =>
    filteredPositions.filter(p => !isMarketClosed(p.market_close_time)),
    [filteredPositions]
  );
  const settlingPositions = useMemo(() =>
    filteredPositions.filter(p => isMarketClosed(p.market_close_time)),
    [filteredPositions]
  );

  if (!positions || positions.length === 0) {
    return null;
  }

  const hasAnyLiveData = stats.liveCount > 0;

  return (
    <div className="
      bg-gradient-to-br from-gray-900/70 via-gray-900/50 to-gray-950/70
      backdrop-blur-md rounded-2xl
      border border-gray-800/80
      shadow-xl shadow-black/20
      p-5
    ">
      {/* Header */}
      <div
        className="flex items-center justify-between mb-5 cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-lg bg-gradient-to-br from-purple-900/30 to-purple-950/20 border border-purple-800/30">
            <Briefcase className="w-4 h-4 text-purple-400" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-gray-200 uppercase tracking-wider">Open Positions</h3>
            <div className="flex items-center space-x-2 mt-0.5">
              <span className="font-mono text-[10px] text-gray-500">
                {positions.length} positions
              </span>
              {/* Live ticker data indicator */}
              {hasAnyLiveData && (
                <div className="flex items-center gap-1 px-1.5 py-0.5 bg-emerald-900/30 rounded border border-emerald-800/30">
                  <span className="relative flex h-1.5 w-1.5">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-500"></span>
                  </span>
                  <span className="text-[10px] text-emerald-400 font-medium">
                    {stats.liveCount} LIVE
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
        <div className="flex items-center space-x-3">
          <div className={`px-3 py-1 rounded-lg text-sm font-bold font-mono ${getPnLColor(stats.totalPnL)}`}>
            {formatPnLCurrency(stats.totalPnL)}
          </div>
          {isExpanded ? (
            <ChevronDown className="w-4 h-4 text-gray-400" />
          ) : (
            <ChevronRight className="w-4 h-4 text-gray-400" />
          )}
        </div>
      </div>

      {isExpanded && (
        <>
          {/* Stats Grid - Primary metrics (5 columns) */}
          <div className="grid grid-cols-5 gap-3 mb-4">
            <StatBox
              label="Unrealized P&L"
              value={formatPnLCurrency(stats.totalPnL)}
              icon={DollarSign}
              accentColor={stats.totalPnL >= 0 ? 'green' : 'red'}
              valueClass={stats.totalPnL >= 0 ? 'text-green-400' : 'text-red-400'}
            />
            <StatBox
              label="Total Exposure"
              value={formatCentsAsCurrency(stats.totalCost)}
              icon={Briefcase}
              accentColor="gray"
              valueClass="text-gray-300"
            />
            <StatBox
              label="Positions"
              value={positions.length.toString()}
              subtitle={stats.settlingCount > 0 ? `${stats.openCount} open, ${stats.settlingCount} settling` : null}
              icon={Layers}
              accentColor="cyan"
              valueClass="text-cyan-400"
            />
            <StatBox
              label="Live Data"
              value={`${stats.liveCount}/${positions.length}`}
              icon={Activity}
              accentColor="emerald"
              valueClass={stats.liveCount > 0 ? 'text-emerald-400' : 'text-gray-500'}
            />
            <StatBox
              label="Avg Size"
              value={formatCentsAsCurrency(stats.avgSize)}
              icon={TrendingUp}
              accentColor="purple"
              valueClass="text-purple-400"
            />
          </div>

          {/* Secondary Stats Grid (5 columns) */}
          <div className="grid grid-cols-5 gap-2 mb-5">
            <DecisionBox
              label="Best Gain"
              value={formatPnLCurrency(stats.bestGain)}
              valueClass={stats.bestGain > 0 ? 'text-green-400' : 'text-gray-500'}
              icon={TrendingUp}
            />
            <DecisionBox
              label="Max Loss"
              value={formatPnLCurrency(stats.maxLoss)}
              valueClass={stats.maxLoss < 0 ? 'text-red-400' : 'text-gray-500'}
              icon={TrendingDown}
            />
            <DecisionBox
              label="Contracts"
              value={stats.totalContracts.toLocaleString()}
              valueClass="text-gray-300"
            />
            <DecisionBox
              label="Mkt Value"
              value={formatCentsAsCurrency(stats.totalValue)}
              valueClass="text-gray-300"
            />
            <DecisionBox
              label="Settling"
              value={stats.settlingCount.toString()}
              valueClass={stats.settlingCount > 0 ? 'text-amber-400' : 'text-gray-500'}
              icon={Clock}
            />
          </div>

          {/* Position Table */}
          <div className="relative">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs text-gray-500 uppercase tracking-wider font-medium">Position Details</span>
              {/* Filter Toggle */}
              <div className="flex items-center gap-1">
                {['all', 'yes', 'no'].map(filter => (
                  <button
                    key={filter}
                    onClick={(e) => { e.stopPropagation(); setSideFilter(filter); }}
                    className={`px-2 py-1 rounded text-xs font-medium uppercase transition-all ${
                      sideFilter === filter
                        ? filter === 'yes' ? 'bg-green-900/50 text-green-400 border border-green-700/50'
                        : filter === 'no' ? 'bg-red-900/50 text-red-400 border border-red-700/50'
                        : 'bg-gray-700/50 text-gray-300 border border-gray-600/50'
                        : 'text-gray-500 hover:text-gray-400 border border-transparent'
                    }`}
                  >
                    {filter}
                  </button>
                ))}
              </div>
            </div>

            <div className="
              bg-gradient-to-b from-gray-800/30 to-gray-900/30
              rounded-xl border border-gray-700/40
              overflow-hidden
              max-h-[280px] overflow-y-auto
            ">
              <table className="w-full text-sm">
                <thead className="sticky top-0 z-10">
                  <tr className="bg-gray-900/80 border-b border-gray-700/40">
                    <th className="px-3 py-2.5 text-left text-[10px] text-gray-500 uppercase font-semibold tracking-wider">Ticker</th>
                    <th className="px-3 py-2.5 text-center text-[10px] text-gray-500 uppercase font-semibold tracking-wider">Side</th>
                    <th className="px-3 py-2.5 text-right text-[10px] text-gray-500 uppercase font-semibold tracking-wider">Qty</th>
                    <th className="px-3 py-2.5 text-right text-[10px] text-gray-500 uppercase font-semibold tracking-wider" title="Entry cost per contract">Cost/C</th>
                    <th className="px-3 py-2.5 text-right text-[10px] text-gray-500 uppercase font-semibold tracking-wider" title="Current market price + unrealized per contract">Price</th>
                    <th className="px-3 py-2.5 text-right text-[10px] text-gray-500 uppercase font-semibold tracking-wider">P&L</th>
                    <th className="px-3 py-2.5 text-center text-[10px] text-gray-500 uppercase font-semibold tracking-wider" title="When market closes">Closes</th>
                  </tr>
                </thead>
                <tbody>
                  {openPositions.map((pos, index) => (
                    <PositionRow
                      key={pos.ticker || index}
                      pos={pos}
                      index={index}
                      isRecentlyChanged={changedTickers.has(pos.ticker)}
                    />
                  ))}
                  {/* Settling Section Divider */}
                  {settlingPositions.length > 0 && (
                    <tr className="bg-amber-900/10 border-y border-amber-700/30">
                      <td colSpan={7} className="px-3 py-2 text-xs text-amber-500 font-medium uppercase tracking-wider">
                        Pending Settlement ({settlingPositions.length})
                      </td>
                    </tr>
                  )}
                  {settlingPositions.map((pos, index) => (
                    <PositionRow
                      key={pos.ticker || `settling-${index}`}
                      pos={pos}
                      index={index}
                      isRecentlyChanged={changedTickers.has(pos.ticker)}
                    />
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default memo(PositionListPanel);
