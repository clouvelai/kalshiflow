import React, { memo } from 'react';
import { DollarSign, TrendingUp, Layers, Zap } from 'lucide-react';

/**
 * MetricCard - Single metric display.
 */
const MetricCard = memo(({ label, value, icon: Icon, valueClass = 'text-white', accentColor = 'gray', testId }) => {
  const accentStyles = {
    cyan: 'border-cyan-500/15 bg-gradient-to-br from-cyan-950/20 to-gray-900/30',
    green: 'border-green-500/15 bg-gradient-to-br from-green-950/20 to-gray-900/30',
    red: 'border-red-500/15 bg-gradient-to-br from-red-950/20 to-gray-900/30',
    violet: 'border-violet-500/15 bg-gradient-to-br from-violet-950/20 to-gray-900/30',
    amber: 'border-amber-500/15 bg-gradient-to-br from-amber-950/20 to-gray-900/30',
    gray: 'border-gray-700/30 bg-gradient-to-br from-gray-800/30 to-gray-900/30',
  };

  return (
    <div
      data-testid={testId}
      className={`rounded-xl p-4 border backdrop-blur-sm transition-colors duration-200 hover:brightness-110 ${accentStyles[accentColor] || accentStyles.gray}`}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-[10px] text-gray-500 uppercase tracking-wider font-medium">{label}</span>
        {Icon && <Icon className="w-3.5 h-3.5 text-gray-600" />}
      </div>
      <div data-testid={`${testId}-value`} className={`text-2xl font-mono font-bold tracking-tight tabular-nums ${valueClass}`}>
        {value}
      </div>
    </div>
  );
});
MetricCard.displayName = 'MetricCard';

/**
 * ArbMetricsBar - Balance, P&L, Events, Trades.
 */
const ArbMetricsBar = ({ tradingState, arbTradeCount, events }) => {
  const balance = tradingState?.balance ?? 0;
  const pnl = tradingState?.pnl;
  const tradeCount = arbTradeCount ?? 0;
  const activeCount = events?.size ?? 0;

  const formatDollars = (cents) => {
    const dollars = (cents || 0) / 100;
    return `$${dollars.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  };

  const pnlCents = pnl?.realized_pnl_cents ?? pnl?.total_pnl_cents ?? 0;
  const pnlDollars = pnlCents / 100;
  const pnlPrefix = pnlDollars >= 0 ? '+' : '';
  const pnlDisplay = `${pnlPrefix}$${Math.abs(pnlDollars).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

  return (
    <div id="arb-metrics-bar" data-testid="arb-metrics-bar" className="grid grid-cols-4 gap-4">
      <MetricCard label="Balance" value={formatDollars(balance)} icon={DollarSign} accentColor="cyan" valueClass="text-cyan-400" testId="metric-balance" />
      <MetricCard label="P&L" value={pnlDisplay} icon={TrendingUp} accentColor={pnlCents >= 0 ? 'green' : 'red'} valueClass={pnlCents >= 0 ? 'text-green-400' : 'text-red-400'} testId="metric-pnl" />
      <MetricCard label="Active Events" value={activeCount} icon={Layers} accentColor="violet" valueClass="text-violet-400" testId="metric-active-events" />
      <MetricCard label="Arb Trades" value={tradeCount} icon={Zap} accentColor="amber" valueClass="text-amber-400" testId="metric-arb-trades" />
    </div>
  );
};

export default memo(ArbMetricsBar);
