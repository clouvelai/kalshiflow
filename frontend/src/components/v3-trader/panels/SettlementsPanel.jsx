import React, { useState, memo, useMemo } from 'react';
import {
  ChevronRight,
  ChevronDown,
  CheckCircle,
  DollarSign,
  TrendingUp,
  Percent,
  Calculator,
  XCircle,
  Receipt,
  Filter
} from 'lucide-react';
import { formatSettlementCurrency, formatTime, formatCents, getPnLColor, formatAge } from '../../../utils/v3-trader';

/**
 * Strategy configuration for display and styling
 */
const STRATEGY_CONFIG = {
  deep_agent: {
    label: 'DEEP AGENT',
    bgClass: 'bg-emerald-900/20',
    borderClass: 'border-emerald-700/30',
    textClass: 'text-emerald-400',
    rowBgClass: 'hover:bg-emerald-900/10',
  },
};

/**
 * Get strategy display config, with fallback for unknown strategies
 */
const getStrategyConfig = (strategyId) => {
  if (!strategyId) {
    return {
      label: '-',
      bgClass: 'bg-gray-800/20',
      borderClass: 'border-gray-700/20',
      textClass: 'text-gray-600',
      rowBgClass: '',
    };
  }
  return STRATEGY_CONFIG[strategyId] || {
    label: strategyId.toUpperCase(),
    bgClass: 'bg-purple-900/20',
    borderClass: 'border-purple-700/30',
    textClass: 'text-purple-400',
    rowBgClass: 'hover:bg-purple-900/10',
  };
};

/**
 * StrategyBadge - Shows strategy identifier with color coding
 */
const StrategyBadge = memo(({ strategyId }) => {
  const config = getStrategyConfig(strategyId);
  return (
    <span className={`
      inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-bold
      ${config.bgClass} ${config.textClass} border ${config.borderClass}
    `}>
      {config.label}
    </span>
  );
});

StrategyBadge.displayName = 'StrategyBadge';

/**
 * StatBox - Enhanced stat display component with gradient and icon
 * Matches TradeProcessingPanel styling
 */
const StatBox = memo(({ label, value, subtitle, valueClass = 'text-white', icon: Icon, accentColor = 'gray' }) => {
  const accentStyles = {
    cyan: 'border-cyan-500/30 bg-gradient-to-br from-cyan-950/30 via-gray-900/50 to-gray-900/30',
    gray: 'border-gray-700/50 bg-gradient-to-br from-gray-800/40 via-gray-900/50 to-gray-900/30',
    green: 'border-green-500/30 bg-gradient-to-br from-green-950/30 via-gray-900/50 to-gray-900/30',
    red: 'border-red-500/30 bg-gradient-to-br from-red-950/30 via-gray-900/50 to-gray-900/30',
    emerald: 'border-emerald-500/30 bg-gradient-to-br from-emerald-950/30 via-gray-900/50 to-gray-900/30',
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
 * Matches TradeProcessingPanel styling
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
 * ResultBadge - Shows market result (YES/NO/PENDING)
 */
const ResultBadge = memo(({ result }) => {
  // Empty string or falsy result means result is pending (not yet fetched from REST)
  if (!result || result === '' || result.toLowerCase() === 'unknown') {
    return (
      <span className="
        inline-flex items-center px-2 py-0.5 rounded text-xs font-bold
        bg-amber-900/30 text-amber-400 border border-amber-600/30
        animate-pulse
      ">
        PENDING
      </span>
    );
  }

  const isYes = result.toLowerCase() === 'yes';
  return (
    <span className={`
      inline-flex items-center px-2 py-0.5 rounded text-xs font-bold
      ${isYes
        ? 'bg-green-900/40 text-green-400 border border-green-600/30'
        : 'bg-red-900/40 text-red-400 border border-red-600/30'
      }
    `}>
      {result.toUpperCase()}
    </span>
  );
});

ResultBadge.displayName = 'ResultBadge';

/**
 * SettlementRow - Memoized row component for settlements table
 */
const SettlementRow = memo(({ settlement }) => {
  const qty = Math.abs(settlement.position || 0);
  const totalCost = settlement.total_cost || 0;
  const netPnl = settlement.net_pnl ?? settlement.realized_pnl ?? 0;
  const entryPrice = settlement.entry_price || 0;
  const tradeRoi = settlement.trade_roi || 0;
  const strategyConfig = getStrategyConfig(settlement.strategy_id);

  return (
    <tr className={`
      border-b border-gray-700/20
      transition-all duration-300 ease-out
      ${strategyConfig.rowBgClass || 'hover:bg-gradient-to-r hover:from-gray-800/60 hover:via-gray-800/40 hover:to-transparent'}
    `}>
      <td className="px-3 py-2.5">
        <span className="font-mono text-gray-200 text-xs truncate block max-w-[120px]" title={settlement.ticker}>
          {settlement.ticker}
        </span>
      </td>
      <td className="px-2 py-2.5 text-center">
        <StrategyBadge strategyId={settlement.strategy_id} />
      </td>
      <td className="px-3 py-2.5 text-center">
        <ResultBadge result={settlement.market_result} />
      </td>
      <td className="px-3 py-2.5 text-center">
        <span className={`
          inline-flex items-center justify-center
          px-2 py-0.5 rounded text-xs font-bold uppercase
          ${settlement.side === 'yes'
            ? 'bg-green-900/30 text-green-400 border border-green-600/20'
            : 'bg-red-900/30 text-red-400 border border-red-600/20'
          }
        `}>
          {settlement.side}
        </span>
      </td>
      <td className="px-2 py-2.5 text-right">
        <span className="font-mono text-gray-300 text-sm">{entryPrice}<span className="text-gray-500">c</span></span>
      </td>
      <td className="px-2 py-2.5 text-right">
        <span className="font-mono text-gray-300 text-sm">{qty}</span>
      </td>
      <td className="px-2 py-2.5 text-right">
        <span className="font-mono text-gray-400 text-sm">{formatCents(totalCost)}</span>
      </td>
      <td className={`px-2 py-2.5 text-right font-mono font-bold text-sm ${getPnLColor(netPnl)}`}>
        {formatSettlementCurrency(netPnl)}
      </td>
      <td className={`px-2 py-2.5 text-right font-mono text-sm ${tradeRoi >= 0 ? 'text-green-400' : 'text-red-400'}`}>
        {tradeRoi >= 0 ? '+' : ''}{tradeRoi}%
      </td>
      <td className="px-2 py-2.5 text-right">
        <span className="font-mono text-gray-500 text-xs">{formatAge(Math.floor(Date.now() / 1000 - settlement.closed_at))}</span>
      </td>
    </tr>
  );
});

SettlementRow.displayName = 'SettlementRow';

/**
 * EmptyState - Beautiful empty state
 */
const EmptyState = memo(() => (
  <div className="
    bg-gradient-to-b from-gray-800/20 to-gray-900/20
    rounded-xl p-8 border border-gray-700/30
    flex flex-col items-center justify-center
  ">
    <div className="relative mb-4">
      <div className="w-16 h-16 rounded-full bg-gradient-to-br from-gray-800/80 to-gray-900/80 border border-gray-700/50 flex items-center justify-center">
        <CheckCircle className="w-7 h-7 text-gray-600" />
      </div>
    </div>
    <div className="text-gray-400 text-sm font-medium mb-1">No Settlements Yet</div>
    <div className="text-gray-600 text-xs text-center max-w-[200px]">
      Closed positions will appear here with full P&L analysis
    </div>
  </div>
));

EmptyState.displayName = 'EmptyState';

/**
 * SettlementsPanel - Professional trading performance dashboard
 * Shows settlement history with stats summary and detailed table
 */
const SettlementsPanel = ({ settlements }) => {
  const [isExpanded, setIsExpanded] = useState(true);

  const settlementsData = settlements || [];
  const hasSettlements = settlementsData.length > 0;

  // Compute all stats with useMemo
  const stats = useMemo(() => {
    if (!hasSettlements) {
      return {
        totalPnL: 0,
        totalCost: 0,
        totalFees: 0,
        wins: 0,
        losses: 0,
        winRate: 0,
        edge: 0,
        profitFactor: 0,
        avgWin: 0,
        avgLoss: 0,
        grossWins: 0,
        grossLosses: 0,
      };
    }

    const totalPnL = settlementsData.reduce((sum, s) => sum + (s.net_pnl || s.realized_pnl || 0), 0);
    const totalCost = settlementsData.reduce((sum, s) => sum + (s.total_cost || 0), 0);
    const totalFees = settlementsData.reduce((sum, s) => sum + (s.fees || 0), 0);

    const winners = settlementsData.filter(s => (s.net_pnl || s.realized_pnl || 0) > 0);
    const losers = settlementsData.filter(s => (s.net_pnl || s.realized_pnl || 0) < 0);

    const wins = winners.length;
    const losses = losers.length;
    const winRate = settlementsData.length > 0 ? (wins / settlementsData.length) * 100 : 0;
    const edge = totalCost > 0 ? (totalPnL / totalCost) * 100 : 0;

    const grossWins = winners.reduce((sum, s) => sum + (s.net_pnl || s.realized_pnl || 0), 0);
    const grossLosses = Math.abs(losers.reduce((sum, s) => sum + (s.net_pnl || s.realized_pnl || 0), 0));
    const profitFactor = grossLosses > 0 ? grossWins / grossLosses : (grossWins > 0 ? Infinity : 0);

    const avgWin = wins > 0 ? grossWins / wins : 0;
    const avgLoss = losses > 0 ? grossLosses / losses : 0;

    return {
      totalPnL,
      totalCost,
      totalFees,
      wins,
      losses,
      winRate,
      edge,
      profitFactor,
      avgWin,
      avgLoss,
      grossWins,
      grossLosses,
    };
  }, [settlementsData, hasSettlements]);

  // Format helpers
  const formatCurrency = (cents) => {
    const dollars = cents / 100;
    const prefix = dollars >= 0 ? '+' : '';
    return `${prefix}$${Math.abs(dollars).toFixed(2)}`;
  };

  const formatProfitFactor = (pf) => {
    if (pf === Infinity) return '∞';
    if (pf === 0) return '0.00';
    return pf.toFixed(2);
  };

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
          <div className="p-2 rounded-lg bg-gradient-to-br from-emerald-900/30 to-emerald-950/20 border border-emerald-800/30">
            <CheckCircle className="w-4 h-4 text-emerald-400" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-gray-200 uppercase tracking-wider">Settlements</h3>
            <div className="flex items-center space-x-2 mt-0.5">
              <span className="font-mono text-[10px] text-gray-500">
                {settlementsData.length} trades
              </span>
            </div>
          </div>
        </div>
        <div className="flex items-center space-x-3">
          <div className={`px-3 py-1 rounded-lg text-sm font-bold font-mono ${getPnLColor(stats.totalPnL)}`}>
            {formatCurrency(stats.totalPnL)}
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
          {hasSettlements ? (
            <>
              {/* Stats Grid - Primary metrics */}
              <div className="grid grid-cols-4 gap-3 mb-4">
                <StatBox
                  label="Net P&L"
                  value={formatCurrency(stats.totalPnL)}
                  icon={DollarSign}
                  accentColor={stats.totalPnL >= 0 ? 'green' : 'red'}
                  valueClass={stats.totalPnL >= 0 ? 'text-green-400' : 'text-red-400'}
                />
                <StatBox
                  label="Win Rate"
                  value={`${stats.winRate.toFixed(1)}%`}
                  subtitle={`${stats.wins}/${settlementsData.length}`}
                  icon={TrendingUp}
                  accentColor={stats.winRate >= 50 ? 'green' : 'red'}
                  valueClass={stats.winRate >= 50 ? 'text-green-400' : 'text-red-400'}
                />
                <StatBox
                  label="Edge (ROI)"
                  value={`${stats.edge >= 0 ? '+' : ''}${stats.edge.toFixed(1)}%`}
                  icon={Percent}
                  accentColor={stats.edge >= 0 ? 'green' : 'red'}
                  valueClass={stats.edge >= 0 ? 'text-green-400' : 'text-red-400'}
                />
                <StatBox
                  label="Profit Factor"
                  value={formatProfitFactor(stats.profitFactor)}
                  icon={Calculator}
                  accentColor={stats.profitFactor >= 1 ? 'green' : 'red'}
                  valueClass={stats.profitFactor >= 1 ? 'text-green-400' : 'text-red-400'}
                />
              </div>

              {/* Secondary Stats Grid */}
              <div className="grid grid-cols-5 gap-2 mb-5">
                <DecisionBox
                  label="Wins"
                  value={stats.wins}
                  valueClass="text-green-400"
                  icon={CheckCircle}
                />
                <DecisionBox
                  label="Losses"
                  value={stats.losses}
                  valueClass="text-red-400"
                  icon={XCircle}
                />
                <DecisionBox
                  label="Avg Win"
                  value={`$${(stats.avgWin / 100).toFixed(2)}`}
                  valueClass="text-green-400"
                  icon={TrendingUp}
                />
                <DecisionBox
                  label="Avg Loss"
                  value={`$${(stats.avgLoss / 100).toFixed(2)}`}
                  valueClass="text-red-400"
                  icon={TrendingUp}
                />
                <DecisionBox
                  label="Fees"
                  value={`$${(stats.totalFees / 100).toFixed(2)}`}
                  valueClass="text-gray-400"
                  icon={Receipt}
                />
              </div>

              {/* Settlements Table */}
              <div className="relative">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-xs text-gray-500 uppercase tracking-wider font-medium">Trade History</span>
                  <span className="text-[10px] text-gray-600 font-mono">{settlementsData.length} settlements</span>
                </div>

                <div className="
                  bg-gradient-to-b from-gray-800/30 to-gray-900/30
                  rounded-xl border border-gray-700/40
                  overflow-hidden
                  max-h-[320px] overflow-y-auto
                ">
                  <table className="w-full text-sm">
                    <thead className="sticky top-0 z-10">
                      <tr className="bg-gray-900/80 border-b border-gray-700/40">
                        <th className="px-3 py-2.5 text-left text-[10px] text-gray-500 uppercase font-semibold tracking-wider">Ticker</th>
                        <th className="px-2 py-2.5 text-center text-[10px] text-gray-500 uppercase font-semibold tracking-wider">Strategy</th>
                        <th className="px-3 py-2.5 text-center text-[10px] text-gray-500 uppercase font-semibold tracking-wider">Result</th>
                        <th className="px-3 py-2.5 text-center text-[10px] text-gray-500 uppercase font-semibold tracking-wider">Side</th>
                        <th className="px-2 py-2.5 text-right text-[10px] text-gray-500 uppercase font-semibold tracking-wider">Entry</th>
                        <th className="px-2 py-2.5 text-right text-[10px] text-gray-500 uppercase font-semibold tracking-wider">Qty</th>
                        <th className="px-2 py-2.5 text-right text-[10px] text-gray-500 uppercase font-semibold tracking-wider">Cost</th>
                        <th className="px-2 py-2.5 text-right text-[10px] text-gray-500 uppercase font-semibold tracking-wider">P&L</th>
                        <th className="px-2 py-2.5 text-right text-[10px] text-gray-500 uppercase font-semibold tracking-wider">ROI</th>
                        <th className="px-2 py-2.5 text-right text-[10px] text-gray-500 uppercase font-semibold tracking-wider">Age</th>
                      </tr>
                    </thead>
                    <tbody>
                      {settlementsData.map((s, idx) => (
                        <SettlementRow
                          key={`${s.ticker}-${s.closed_at}-${idx}`}
                          settlement={s}
                        />
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          ) : (
            <EmptyState />
          )}
        </>
      )}
    </div>
  );
};

export default memo(SettlementsPanel);
