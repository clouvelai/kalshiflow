import React, { memo, useRef, useEffect, useState } from 'react';
import {
  DollarSign,
  Briefcase,
  Database,
  TrendingUp,
  TrendingDown,
  CheckCircle,
  Activity,
  Shield,
  ShoppingCart,
  FileText,
  Clock,
  ArrowUpCircle,
  ArrowDownCircle,
  Hash
} from 'lucide-react';

// ============================================================================
// HOOKS
// ============================================================================

/**
 * useAnimatedValue - Hook for tracking value changes and providing animation classes
 *
 * Returns animation class and key when value changes:
 * - Green glow + slight scale-up for increases
 * - Red glow + slight scale-down for decreases
 * - Uses key prop to re-trigger CSS animation
 *
 * @param {number} value - Current value to track
 * @param {boolean} enabled - Whether animations are enabled (default true)
 * @returns {{ animClass: string, animKey: number }} Animation class and unique key
 */
const useAnimatedValue = (value, enabled = true) => {
  const prevValueRef = useRef(value);
  const [animState, setAnimState] = useState({ class: '', key: 0 });

  useEffect(() => {
    if (!enabled) return;

    const prevValue = prevValueRef.current;

    // Only animate if we have a valid previous value and value actually changed
    if (prevValue !== null && prevValue !== undefined && value !== prevValue) {
      const isIncrease = value > prevValue;
      const animClass = isIncrease ? 'animate-value-increase' : 'animate-value-decrease';

      // Update animation state with new key to force re-render
      setAnimState(prev => ({
        class: animClass,
        key: prev.key + 1
      }));

      // Clear animation class after animation completes
      const timer = setTimeout(() => {
        setAnimState(prev => ({ ...prev, class: '' }));
      }, 500);

      prevValueRef.current = value;
      return () => clearTimeout(timer);
    }

    prevValueRef.current = value;
  }, [value, enabled]);

  return { animClass: animState.class, animKey: animState.key };
};

// ============================================================================
// FORMATTING HELPERS
// ============================================================================

const formatCurrency = (cents) => {
  const dollars = cents / 100;
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  }).format(dollars);
};

const formatPnLCurrency = (cents) => {
  const dollars = cents / 100;
  const prefix = cents >= 0 ? '+' : '';
  return prefix + new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  }).format(dollars);
};

const formatTime = (timestamp) => {
  if (!timestamp) return 'N/A';
  const date = new Date(timestamp * 1000);
  return date.toLocaleTimeString('en-US', {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  });
};

// ============================================================================
// ACCENT COLOR SYSTEM
// ============================================================================

const accentStyles = {
  green: {
    border: 'border-green-500/30',
    bg: 'bg-gradient-to-br from-green-950/30 via-gray-900/50 to-gray-900/30',
    hover: 'hover:border-green-500/50 hover:shadow-green-900/20',
    icon: 'text-green-400'
  },
  blue: {
    border: 'border-blue-500/30',
    bg: 'bg-gradient-to-br from-blue-950/30 via-gray-900/50 to-gray-900/30',
    hover: 'hover:border-blue-500/50 hover:shadow-blue-900/20',
    icon: 'text-blue-400'
  },
  white: {
    border: 'border-gray-500/30',
    bg: 'bg-gradient-to-br from-gray-800/40 via-gray-900/50 to-gray-900/30',
    hover: 'hover:border-gray-400/50 hover:shadow-gray-900/20',
    icon: 'text-gray-400'
  },
  red: {
    border: 'border-red-500/30',
    bg: 'bg-gradient-to-br from-red-950/30 via-gray-900/50 to-gray-900/30',
    hover: 'hover:border-red-500/50 hover:shadow-red-900/20',
    icon: 'text-red-400'
  },
  amber: {
    border: 'border-amber-500/30',
    bg: 'bg-gradient-to-br from-amber-950/30 via-gray-900/50 to-gray-900/30',
    hover: 'hover:border-amber-500/50 hover:shadow-amber-900/20',
    icon: 'text-amber-400'
  },
  cyan: {
    border: 'border-cyan-500/30',
    bg: 'bg-gradient-to-br from-cyan-950/30 via-gray-900/50 to-gray-900/30',
    hover: 'hover:border-cyan-500/50 hover:shadow-cyan-900/20',
    icon: 'text-cyan-400'
  },
  purple: {
    border: 'border-purple-500/30',
    bg: 'bg-gradient-to-br from-purple-950/30 via-gray-900/50 to-gray-900/30',
    hover: 'hover:border-purple-500/50 hover:shadow-purple-900/20',
    icon: 'text-purple-400'
  },
  orange: {
    border: 'border-orange-500/30',
    bg: 'bg-gradient-to-br from-orange-950/30 via-gray-900/50 to-gray-900/30',
    hover: 'hover:border-orange-500/50 hover:shadow-orange-900/20',
    icon: 'text-orange-400'
  },
  gray: {
    border: 'border-gray-700/50',
    bg: 'bg-gradient-to-br from-gray-800/40 via-gray-900/50 to-gray-900/30',
    hover: 'hover:border-gray-600/50 hover:shadow-gray-900/20',
    icon: 'text-gray-400'
  },
  yellow: {
    border: 'border-yellow-500/30',
    bg: 'bg-gradient-to-br from-yellow-950/30 via-gray-900/50 to-gray-900/30',
    hover: 'hover:border-yellow-500/50 hover:shadow-yellow-900/20',
    icon: 'text-yellow-400'
  },
  indigo: {
    border: 'border-indigo-500/30',
    bg: 'bg-gradient-to-br from-indigo-950/30 via-gray-900/50 to-gray-900/30',
    hover: 'hover:border-indigo-500/50 hover:shadow-indigo-900/20',
    icon: 'text-indigo-400'
  }
};

// ============================================================================
// MEMOIZED SUB-COMPONENTS
// ============================================================================

/**
 * ValueCard - Main value display card with animations (Row 1)
 * Features: gradient background, backdrop blur, hover effects, value-change animations
 */
const ValueCard = memo(({
  label,
  value,
  formattedValue,
  icon: Icon,
  accentColor = 'gray',
  valueClass = 'text-white',
  subtitle = null,
  subtitleClass = 'text-gray-400'
}) => {
  const { animClass, animKey } = useAnimatedValue(value);
  const accent = accentStyles[accentColor] || accentStyles.gray;

  return (
    <div
      key={animKey}
      className={`
        rounded-xl p-4 border backdrop-blur-sm
        transition-all duration-300 ease-out
        hover:scale-[1.02] hover:shadow-lg
        ${accent.border} ${accent.bg} ${accent.hover}
        ${animClass}
      `}
    >
      <div className="flex items-center space-x-2 mb-2">
        {Icon && <Icon className={`w-4 h-4 ${accent.icon}`} />}
        <span className="text-xs text-gray-500 uppercase tracking-wider font-medium">{label}</span>
        {subtitle && (
          <span className={`text-xs font-mono ml-auto ${subtitleClass}`}>
            {subtitle}
          </span>
        )}
      </div>
      <div className={`text-2xl font-mono font-bold tracking-tight ${valueClass}`}>
        {formattedValue}
      </div>
    </div>
  );
});

ValueCard.displayName = 'ValueCard';

/**
 * FlowCard - Session cash flow card (Row 2)
 * Compact design with animations for tracking session activity
 */
const FlowCard = memo(({
  label,
  value,
  formattedValue,
  icon: Icon,
  accentColor = 'gray',
  valueClass = 'text-white',
  title = ''
}) => {
  const { animClass, animKey } = useAnimatedValue(value);
  const accent = accentStyles[accentColor] || accentStyles.gray;

  return (
    <div
      key={animKey}
      className={`
        rounded-lg p-3 border backdrop-blur-sm
        transition-all duration-300 ease-out
        hover:scale-[1.02] hover:shadow-md
        ${accent.border} ${accent.bg} ${accent.hover}
        ${animClass}
      `}
    >
      <div className="flex items-center space-x-2 mb-1">
        {Icon && <Icon className={`w-3 h-3 ${accent.icon}`} />}
        <span
          className="text-xs text-gray-500 uppercase tracking-wider font-medium"
          title={title}
        >
          {label}
        </span>
      </div>
      <div className={`text-lg font-mono font-bold ${valueClass}`}>
        {formattedValue}
      </div>
    </div>
  );
});

FlowCard.displayName = 'FlowCard';

/**
 * MetricRow - Horizontal metric display (Row 3)
 * Inline layout with icon, label, and value
 */
const MetricRow = memo(({
  label,
  value,
  icon: Icon,
  accentColor = 'gray',
  children = null
}) => {
  const { animClass, animKey } = useAnimatedValue(value);
  const accent = accentStyles[accentColor] || accentStyles.gray;

  return (
    <div
      key={animKey}
      className={`
        rounded-lg p-3 border backdrop-blur-sm
        transition-all duration-300 ease-out
        hover:scale-[1.01] hover:shadow-md
        ${accent.border} ${accent.bg} ${accent.hover}
        ${animClass}
      `}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          {Icon && <Icon className={`w-4 h-4 ${accent.icon}`} />}
          <span className="text-xs text-gray-500 uppercase tracking-wider font-medium">{label}</span>
        </div>
        {children ? children : (
          <span className="text-lg font-mono font-bold text-white">{value}</span>
        )}
      </div>
    </div>
  );
});

MetricRow.displayName = 'MetricRow';

/**
 * EmptyState - Displayed when no trading state is available
 */
const EmptyState = memo(() => (
  <div className="
    bg-gradient-to-br from-gray-900/70 via-gray-900/50 to-gray-950/70
    backdrop-blur-md rounded-2xl
    border border-gray-800/80
    shadow-xl shadow-black/20
    p-5 mb-4
  ">
    <div className="flex items-center justify-between">
      <div className="flex items-center space-x-3">
        <div className="p-2 rounded-lg bg-gradient-to-br from-gray-800/50 to-gray-900/50 border border-gray-700/30">
          <Activity className="w-4 h-4 text-gray-500" />
        </div>
        <h3 className="text-sm font-bold text-gray-400 uppercase tracking-wider">Trading Session</h3>
      </div>
      <span className="text-xs text-gray-500 font-mono">Waiting for data...</span>
    </div>
  </div>
));

EmptyState.displayName = 'EmptyState';

// ============================================================================
// MAIN COMPONENT
// ============================================================================

/**
 * TradingSessionPanel - Unified trading session display
 * Shows balance, portfolio, P&L from Kalshi sync with elegant animations
 *
 * Features:
 * - Gradient backgrounds matching TradeProcessingPanel design
 * - Value-change animations (green glow for increases, red for decreases)
 * - Hover micro-interactions
 * - Memoized sub-components for performance
 *
 * @param {Object} props
 * @param {Object} props.tradingState - Trading state from WebSocket
 * @param {number} props.lastUpdateTime - Last update timestamp
 */
const TradingSessionPanel = ({ tradingState, lastUpdateTime }) => {
  // Extract values with defaults
  const hasState = tradingState?.has_state ?? false;
  const balance = tradingState?.balance ?? 0;
  const portfolioValue = tradingState?.portfolio_value ?? 0;
  const totalValue = balance + portfolioValue;
  const positionCount = tradingState?.position_count ?? 0;
  const orderCount = tradingState?.order_count ?? 0;
  const pnl = tradingState?.pnl ?? null;
  const orderGroup = tradingState?.order_group ?? null;
  const syncTimestamp = tradingState?.sync_timestamp ?? null;

  // Early return for no state
  if (!hasState) {
    return <EmptyState />;
  }

  // Determine P&L accent color
  const pnlAccent = pnl && pnl.session_pnl >= 0 ? 'green' : 'red';
  const pnlValueClass = pnl && pnl.session_pnl >= 0 ? 'text-green-400' : 'text-red-400';

  return (
    <div className="
      bg-gradient-to-br from-gray-900/70 via-gray-900/50 to-gray-950/70
      backdrop-blur-md rounded-2xl
      border border-gray-800/80
      shadow-xl shadow-black/20
      p-5 mb-4
    ">
      {/* Header */}
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-lg bg-gradient-to-br from-cyan-900/30 to-cyan-950/20 border border-cyan-800/30">
            <TrendingUp className="w-4 h-4 text-cyan-400" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-gray-200 uppercase tracking-wider">Trading Session</h3>
            <div className="flex items-center space-x-2 mt-0.5">
              <Clock className="w-3 h-3 text-gray-600" />
              <span className="font-mono text-[10px] text-gray-500">
                {pnl?.session_start_time ? `Started ${formatTime(pnl.session_start_time)}` : 'Active'}
              </span>
            </div>
          </div>
        </div>
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <Activity className="w-3.5 h-3.5 text-blue-400" />
            <span className="text-xs text-gray-400 font-mono">
              <span className="text-gray-500">Sync:</span> {formatTime(syncTimestamp)}
            </span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="relative">
              <div className="w-2 h-2 rounded-full bg-green-500" />
              <div className="absolute inset-0 w-2 h-2 rounded-full bg-green-500 animate-ping opacity-75" />
            </div>
            <span className="text-xs text-gray-400 font-mono">
              <span className="text-gray-500">Update:</span> {formatTime(lastUpdateTime)}
            </span>
          </div>
        </div>
      </div>

      {/* Main Value Cards - Row 1 */}
      <div className="grid grid-cols-4 gap-4 mb-4">
        <ValueCard
          label="Cash Available"
          value={balance}
          formattedValue={formatCurrency(balance)}
          icon={DollarSign}
          accentColor="green"
          valueClass="text-green-400"
        />

        <ValueCard
          label="In Positions"
          value={portfolioValue}
          formattedValue={formatCurrency(portfolioValue)}
          icon={Briefcase}
          accentColor="blue"
          valueClass="text-blue-400"
        />

        <ValueCard
          label="Total Value"
          value={totalValue}
          formattedValue={formatCurrency(totalValue)}
          icon={Database}
          accentColor="white"
          valueClass="text-white"
        />

        <ValueCard
          label="Session P&L"
          value={pnl?.session_pnl ?? 0}
          formattedValue={pnl ? formatPnLCurrency(pnl.session_pnl ?? 0) : '--'}
          icon={pnl && pnl.session_pnl >= 0 ? TrendingUp : TrendingDown}
          accentColor={pnlAccent}
          valueClass={pnl ? pnlValueClass : 'text-gray-500'}
          subtitle={pnl?.session_pnl_percent !== undefined
            ? `(${pnl.session_pnl_percent >= 0 ? '+' : ''}${pnl.session_pnl_percent.toFixed(1)}%)`
            : null
          }
          subtitleClass={pnl?.session_pnl_percent >= 0 ? 'text-green-400' : 'text-red-400'}
        />
      </div>

      {/* Session Cash Flow - Row 2 */}
      {pnl && (
        <div className="grid grid-cols-4 gap-4 mb-4">
          <FlowCard
            label="Cash Invested"
            value={pnl.session_cash_invested ?? 0}
            formattedValue={formatCurrency(pnl.session_cash_invested ?? 0)}
            icon={ArrowUpCircle}
            accentColor="amber"
            valueClass="text-amber-400"
            title="Cash spent on orders this session"
          />

          <FlowCard
            label="Cash Received"
            value={pnl.session_cash_received ?? 0}
            formattedValue={formatCurrency(pnl.session_cash_received ?? 0)}
            icon={ArrowDownCircle}
            accentColor="cyan"
            valueClass="text-cyan-400"
            title="Cash received from settlements this session"
          />

          <FlowCard
            label="Orders Placed"
            value={pnl.session_orders_count ?? 0}
            formattedValue={pnl.session_orders_count ?? 0}
            icon={Hash}
            accentColor="purple"
            valueClass="text-purple-400"
            title="Orders placed this session"
          />

          <FlowCard
            label="Settled"
            value={pnl.session_settlements_count ?? 0}
            formattedValue={pnl.session_settlements_count ?? 0}
            icon={CheckCircle}
            accentColor="orange"
            valueClass="text-orange-400"
            title="Positions settled this session"
          />
        </div>
      )}

      {/* Positions, Orders, Order Group - Row 3 */}
      <div className="grid grid-cols-3 gap-4">
        <MetricRow
          label="Positions"
          value={positionCount}
          icon={ShoppingCart}
          accentColor="purple"
        />

        <MetricRow
          label="Orders"
          value={orderCount}
          icon={FileText}
          accentColor="yellow"
        />

        <MetricRow
          label="Order Group"
          value={orderGroup?.order_count ?? 0}
          icon={Shield}
          accentColor="indigo"
        >
          {orderGroup && orderGroup.id ? (
            <div className="flex items-center space-x-2">
              <span className="text-xs font-mono text-gray-400">
                {orderGroup.id.substring(0, 8)}
              </span>
              <span className={`px-1.5 py-0.5 text-xs font-bold rounded ${
                orderGroup.status === 'active'
                  ? 'bg-green-900/50 text-green-400 border border-green-700/30'
                  : orderGroup.status === 'inactive'
                  ? 'bg-gray-900/50 text-gray-400 border border-gray-700/30'
                  : 'bg-yellow-900/50 text-yellow-400 border border-yellow-700/30'
              }`}>
                {(orderGroup.status || 'N/A').toUpperCase()}
              </span>
              <span className="text-sm font-mono font-bold text-white">
                {orderGroup.order_count || 0}
              </span>
            </div>
          ) : (
            <span className="text-sm font-mono text-gray-500">--</span>
          )}
        </MetricRow>
      </div>

      {/* Animation Keyframes */}
      <style>{`
        @keyframes value-increase {
          0% {
            transform: scale(1);
            box-shadow: none;
          }
          50% {
            transform: scale(1.02);
            box-shadow: 0 0 12px rgba(34, 197, 94, 0.4);
          }
          100% {
            transform: scale(1);
            box-shadow: none;
          }
        }

        @keyframes value-decrease {
          0% {
            transform: scale(1);
            box-shadow: none;
          }
          50% {
            transform: scale(0.98);
            box-shadow: 0 0 12px rgba(239, 68, 68, 0.4);
          }
          100% {
            transform: scale(1);
            box-shadow: none;
          }
        }

        .animate-value-increase {
          animation: value-increase 450ms ease-out;
        }

        .animate-value-decrease {
          animation: value-decrease 450ms ease-out;
        }
      `}</style>
    </div>
  );
};

export default memo(TradingSessionPanel);
