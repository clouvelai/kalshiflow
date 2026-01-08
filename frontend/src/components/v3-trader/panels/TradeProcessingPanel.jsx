import React, { useState, useEffect, memo, useMemo } from 'react';
import { Activity, Clock, Zap, TrendingUp, Filter, Target, AlertCircle, RotateCcw } from 'lucide-react';
import { formatAge } from '../../../utils/v3-trader';

/**
 * LiveIndicator - Pulsing live status indicator
 */
const LiveIndicator = memo(({ isLive }) => (
  <div className="flex items-center space-x-1.5">
    <div className="relative">
      <div className={`w-2 h-2 rounded-full ${isLive ? 'bg-cyan-400' : 'bg-gray-600'}`} />
      {isLive && (
        <div className="absolute inset-0 w-2 h-2 rounded-full bg-cyan-400 animate-ping opacity-75" />
      )}
    </div>
    <span className={`text-xs font-bold uppercase tracking-wider ${isLive ? 'text-cyan-400' : 'text-gray-600'}`}>
      {isLive ? 'LIVE' : 'OFFLINE'}
    </span>
  </div>
));

LiveIndicator.displayName = 'LiveIndicator';

/**
 * TradeRow - Memoized row component for recent trades table with smooth hover
 */
const TradeRow = memo(({ trade, isNew }) => (
  <tr className={`
    border-b border-gray-700/20
    transition-all duration-300 ease-out
    hover:bg-gradient-to-r hover:from-gray-800/60 hover:via-gray-800/40 hover:to-transparent
    ${isNew ? 'bg-cyan-900/20 animate-pulse-once' : ''}
  `}>
    <td className="px-3 py-2.5">
      <span className="font-mono text-gray-200 text-xs truncate block max-w-[140px]" title={trade.market_ticker}>
        {trade.market_ticker}
      </span>
    </td>
    <td className="px-3 py-2.5 text-center">
      <span className={`
        inline-flex items-center justify-center
        px-2.5 py-1 rounded-md text-xs font-bold uppercase
        transition-all duration-200
        ${trade.side === 'yes'
          ? 'bg-gradient-to-r from-green-900/40 to-green-800/20 text-green-400 border border-green-600/30 shadow-sm shadow-green-900/20'
          : 'bg-gradient-to-r from-red-900/40 to-red-800/20 text-red-400 border border-red-600/30 shadow-sm shadow-red-900/20'
        }
      `}>
        {trade.side}
      </span>
    </td>
    <td className="px-3 py-2.5 text-right">
      <span className="font-mono text-gray-200 text-sm">{trade.price_cents}<span className="text-gray-500">c</span></span>
    </td>
    <td className="px-3 py-2.5 text-right">
      <span className="font-mono text-gray-200 text-sm">{trade.count?.toLocaleString()}</span>
    </td>
    <td className="px-3 py-2.5 text-right">
      <span className="font-mono text-gray-500 text-xs">{formatAge(trade.age_seconds)}</span>
    </td>
  </tr>
));

TradeRow.displayName = 'TradeRow';

/**
 * StatBox - Enhanced stat display component with gradient and icon
 */
const StatBox = memo(({ label, value, valueClass = 'text-white', icon: Icon, accentColor = 'gray' }) => {
  const accentStyles = {
    cyan: 'border-cyan-500/30 bg-gradient-to-br from-cyan-950/30 via-gray-900/50 to-gray-900/30',
    gray: 'border-gray-700/50 bg-gradient-to-br from-gray-800/40 via-gray-900/50 to-gray-900/30',
    green: 'border-green-500/30 bg-gradient-to-br from-green-950/30 via-gray-900/50 to-gray-900/30',
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
        {typeof value === 'number' ? value.toLocaleString() : value}
      </div>
    </div>
  );
});

StatBox.displayName = 'StatBox';

/**
 * DecisionBox - Compact decision stat component with subtle glow
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
      {typeof value === 'number' ? value.toLocaleString() : value}
    </div>
  </div>
));

DecisionBox.displayName = 'DecisionBox';

/**
 * EmptyState - Beautiful empty state with animated icon
 */
const EmptyState = memo(() => (
  <div className="
    bg-gradient-to-b from-gray-800/20 to-gray-900/20
    rounded-xl p-8 border border-gray-700/30
    flex flex-col items-center justify-center
  ">
    <div className="relative mb-4">
      <div className="w-16 h-16 rounded-full bg-gradient-to-br from-gray-800/80 to-gray-900/80 border border-gray-700/50 flex items-center justify-center">
        <Activity className="w-7 h-7 text-gray-600 animate-pulse" />
      </div>
      <div className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-gray-800 border border-gray-700 flex items-center justify-center">
        <div className="w-1.5 h-1.5 rounded-full bg-gray-600" />
      </div>
    </div>
    <div className="text-gray-400 text-sm font-medium mb-1">Waiting for Trades</div>
    <div className="text-gray-600 text-xs text-center max-w-[200px]">
      Trades matching RLM criteria will appear here in real-time
    </div>
  </div>
));

EmptyState.displayName = 'EmptyState';

/**
 * TradeProcessingPanel - Displays trade processing stats and recent tracked trades
 * with world-class UX including animations, gradients, and live indicators
 */
const TradeProcessingPanel = ({ tradeProcessing }) => {
  const [lastSyncAge, setLastSyncAge] = useState(null);
  const [newTradeIds, setNewTradeIds] = useState(new Set());

  // Update last sync age every second
  useEffect(() => {
    const updateAge = () => {
      if (tradeProcessing?.last_updated) {
        const age = Math.floor(Date.now() / 1000 - tradeProcessing.last_updated);
        setLastSyncAge(age);
      }
    };

    updateAge();
    const interval = setInterval(updateAge, 1000);
    return () => clearInterval(interval);
  }, [tradeProcessing?.last_updated]);

  // Track new trades for pulse animation
  useEffect(() => {
    const trades = tradeProcessing?.recent_trades || [];
    if (trades.length > 0) {
      const latestTradeId = trades[0]?.trade_id;
      if (latestTradeId && !newTradeIds.has(latestTradeId)) {
        setNewTradeIds(prev => new Set([...prev, latestTradeId]));
        // Remove from new set after animation completes
        setTimeout(() => {
          setNewTradeIds(prev => {
            const next = new Set(prev);
            next.delete(latestTradeId);
            return next;
          });
        }, 1000);
      }
    }
  }, [tradeProcessing?.recent_trades]);

  const stats = useMemo(() => tradeProcessing?.stats || {
    trades_seen: 0,
    trades_filtered: 0,
    trades_tracked: 0,
    filter_rate_percent: 0
  }, [tradeProcessing?.stats]);

  const decisions = useMemo(() => tradeProcessing?.decisions || {
    detected: 0,
    executed: 0,
    rate_limited: 0,
    skipped: 0,
    reentries: 0
  }, [tradeProcessing?.decisions]);

  const recentTrades = useMemo(() =>
    (tradeProcessing?.recent_trades || []).slice(0, 100),
    [tradeProcessing?.recent_trades]
  );

  const isLive = lastSyncAge !== null && lastSyncAge < 10;

  return (
    <div className="
      bg-gradient-to-br from-gray-900/70 via-gray-900/50 to-gray-950/70
      backdrop-blur-md rounded-2xl
      border border-gray-800/80
      shadow-xl shadow-black/20
      p-5
    ">
      {/* Header */}
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-lg bg-gradient-to-br from-cyan-900/30 to-cyan-950/20 border border-cyan-800/30">
            <Activity className="w-4 h-4 text-cyan-400" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-gray-200 uppercase tracking-wider">Trade Processing</h3>
            <div className="flex items-center space-x-2 mt-0.5">
              <Clock className="w-3 h-3 text-gray-600" />
              <span className="font-mono text-[10px] text-gray-500">
                {lastSyncAge !== null ? `${lastSyncAge}s ago` : 'Connecting...'}
              </span>
            </div>
          </div>
        </div>
        <LiveIndicator isLive={isLive} />
      </div>

      {/* Stats Grid - Primary metrics */}
      <div className="grid grid-cols-4 gap-3 mb-4">
        <StatBox
          label="Trades Seen"
          value={stats.trades_seen}
          icon={TrendingUp}
          valueClass="text-gray-100"
        />
        <StatBox
          label="Filtered"
          value={stats.trades_filtered}
          icon={Filter}
          valueClass="text-gray-400"
        />
        <StatBox
          label="Tracked"
          value={stats.trades_tracked}
          icon={Target}
          valueClass="text-cyan-400"
          accentColor="cyan"
        />
        <StatBox
          label="Filter Rate"
          value={`${stats.filter_rate_percent?.toFixed(1) || 0}%`}
          icon={Zap}
          valueClass="text-gray-400"
        />
      </div>

      {/* Decisions Grid - Secondary metrics */}
      <div className="grid grid-cols-5 gap-2 mb-5">
        <DecisionBox
          label="Detected"
          value={decisions.detected}
          valueClass="text-gray-200"
          icon={Target}
        />
        <DecisionBox
          label="Executed"
          value={decisions.executed}
          valueClass="text-green-400"
          icon={Zap}
        />
        <DecisionBox
          label="Rate Lim."
          value={decisions.rate_limited}
          valueClass="text-red-400"
          icon={AlertCircle}
        />
        <DecisionBox
          label="Skipped"
          value={decisions.skipped}
          valueClass="text-yellow-400"
          icon={Filter}
        />
        <DecisionBox
          label="Reentries"
          value={decisions.reentries}
          valueClass="text-purple-400"
          icon={RotateCcw}
        />
      </div>

      {/* Recent Trades Section */}
      <div className="relative">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs text-gray-500 uppercase tracking-wider font-medium">Recent Tracked Trades</span>
          {recentTrades.length > 0 && (
            <span className="text-[10px] text-gray-600 font-mono">{recentTrades.length} trades</span>
          )}
        </div>

        {recentTrades.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="
            bg-gradient-to-b from-gray-800/30 to-gray-900/30
            rounded-xl border border-gray-700/40
            max-h-[280px] overflow-y-auto
          ">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-gray-900/60 border-b border-gray-700/40">
                  <th className="px-3 py-2.5 text-left text-[10px] text-gray-500 uppercase font-semibold tracking-wider">Market</th>
                  <th className="px-3 py-2.5 text-center text-[10px] text-gray-500 uppercase font-semibold tracking-wider">Side</th>
                  <th className="px-3 py-2.5 text-right text-[10px] text-gray-500 uppercase font-semibold tracking-wider">Price</th>
                  <th className="px-3 py-2.5 text-right text-[10px] text-gray-500 uppercase font-semibold tracking-wider">Count</th>
                  <th className="px-3 py-2.5 text-right text-[10px] text-gray-500 uppercase font-semibold tracking-wider">Age</th>
                </tr>
              </thead>
              <tbody>
                {recentTrades.map((trade) => (
                  <TradeRow
                    key={trade.trade_id}
                    trade={trade}
                    isNew={newTradeIds.has(trade.trade_id)}
                  />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Custom animation styles */}
      <style>{`
        @keyframes pulse-once {
          0% { background-color: rgba(6, 182, 212, 0.15); }
          100% { background-color: transparent; }
        }
        .animate-pulse-once {
          animation: pulse-once 1s ease-out forwards;
        }
      `}</style>
    </div>
  );
};

export default memo(TradeProcessingPanel);
