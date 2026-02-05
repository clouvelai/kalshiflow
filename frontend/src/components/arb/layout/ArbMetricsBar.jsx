import React, { memo } from 'react';
import { DollarSign, TrendingUp, Layers, Zap } from 'lucide-react';

/**
 * MetricCard - Single metric display
 */
const MetricCard = memo(({ label, value, icon: Icon, valueClass = 'text-white', accentColor = 'gray' }) => {
  const accentStyles = {
    cyan: 'border-cyan-500/30 bg-gradient-to-br from-cyan-950/30 via-gray-900/50 to-gray-900/30',
    green: 'border-green-500/30 bg-gradient-to-br from-green-950/30 via-gray-900/50 to-gray-900/30',
    red: 'border-red-500/30 bg-gradient-to-br from-red-950/30 via-gray-900/50 to-gray-900/30',
    violet: 'border-violet-500/30 bg-gradient-to-br from-violet-950/30 via-gray-900/50 to-gray-900/30',
    amber: 'border-amber-500/30 bg-gradient-to-br from-amber-950/30 via-gray-900/50 to-gray-900/30',
    gray: 'border-gray-700/50 bg-gradient-to-br from-gray-800/40 via-gray-900/50 to-gray-900/30',
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
    </div>
  );
});

MetricCard.displayName = 'MetricCard';

/**
 * ArbMetricsBar - Row of 4 metric cards
 *
 * Balance ($), P&L ($), Active Events/Pairs (#), Arb Trades Today (#)
 */
const ArbMetricsBar = ({ tradingState, arbTradeCount, events }) => {
  const balance = tradingState?.balance ?? 0;
  const pnl = tradingState?.pnl;
  const tradeCount = arbTradeCount ?? 0;

  const activeCount = events?.size ?? 0;
  const activeLabel = 'Active Events';

  const formatDollars = (cents) => {
    const dollars = (cents || 0) / 100;
    return `$${dollars.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  };

  const pnlCents = pnl?.realized_pnl_cents ?? pnl?.total_pnl_cents ?? 0;
  const pnlDollars = pnlCents / 100;
  const pnlPrefix = pnlDollars >= 0 ? '+' : '';
  const pnlDisplay = `${pnlPrefix}$${Math.abs(pnlDollars).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

  return (
    <div className="grid grid-cols-4 gap-4">
      <MetricCard
        label="Balance"
        value={formatDollars(balance)}
        icon={DollarSign}
        accentColor="cyan"
        valueClass="text-cyan-400"
      />
      <MetricCard
        label="P&L"
        value={pnlDisplay}
        icon={TrendingUp}
        accentColor={pnlCents >= 0 ? 'green' : 'red'}
        valueClass={pnlCents >= 0 ? 'text-green-400' : 'text-red-400'}
      />
      <MetricCard
        label={activeLabel}
        value={activeCount}
        icon={Layers}
        accentColor="violet"
        valueClass="text-violet-400"
      />
      <MetricCard
        label="Arb Trades"
        value={tradeCount}
        icon={Zap}
        accentColor="amber"
        valueClass="text-amber-400"
      />
    </div>
  );
};

export default memo(ArbMetricsBar);
