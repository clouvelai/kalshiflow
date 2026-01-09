import React, { useState, useEffect, memo, useMemo } from 'react';
import {
  ChevronRight,
  ChevronDown,
  Zap,
  Target,
  TrendingDown,
  AlertCircle,
  Clock,
  Activity,
  Layers,
  Settings,
  History,
  CheckCircle,
  XCircle,
  Filter,
  Brain,
  Database,
  FileSearch,
  TrendingUp
} from 'lucide-react';
import { formatAge } from '../../../utils/v3-trader';

/**
 * Strategy display configuration
 *
 * MULTI-STRATEGY SUPPORT: To add a new strategy (e.g., "s013"):
 * 1. Add an entry here with unique colors (see SettlementsPanel.jsx for color options)
 * 2. The backend already broadcasts all strategies - no backend changes needed
 * 3. See "MULTI-STRATEGY REFACTOR" comment below for rendering changes
 */
const STRATEGY_CONFIG = {
  rlm_no: {
    label: 'RLM NO',
    description: 'Reverse Line Movement',
    accentColor: 'violet',
    bgClass: 'bg-violet-900/20',
    borderClass: 'border-violet-700/30',
    textClass: 'text-violet-400',
  },
  odmr: {
    label: 'ODMR',
    description: 'Dip Buyer / Mean Reversion',
    accentColor: 'amber',
    bgClass: 'bg-amber-900/20',
    borderClass: 'border-amber-700/30',
    textClass: 'text-amber-400',
  },
  agentic_research: {
    label: 'AI Research',
    description: 'Agentic Event Research',
    accentColor: 'emerald',
    bgClass: 'bg-emerald-900/20',
    borderClass: 'border-emerald-700/30',
    textClass: 'text-emerald-400',
  },
};

const getStrategyConfig = (strategyId) => {
  return STRATEGY_CONFIG[strategyId] || {
    label: strategyId?.toUpperCase() || 'UNKNOWN',
    description: 'Strategy',
    accentColor: 'gray',
    bgClass: 'bg-gray-800/20',
    borderClass: 'border-gray-700/30',
    textClass: 'text-gray-400',
  };
};

/**
 * HealthBadge - Shows strategy health status
 */
const HealthBadge = memo(({ healthy, running }) => {
  if (!running) {
    return (
      <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold bg-gray-800/50 text-gray-500 border border-gray-700/30">
        STOPPED
      </span>
    );
  }

  return (
    <span className={`
      inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold
      ${healthy
        ? 'bg-green-900/40 text-green-400 border border-green-600/30'
        : 'bg-red-900/40 text-red-400 border border-red-600/30'
      }
    `}>
      {healthy ? 'HEALTHY' : 'UNHEALTHY'}
    </span>
  );
});

HealthBadge.displayName = 'HealthBadge';

/**
 * StrategyTabs - Tab bar for switching between strategies
 * Only shown when 2+ strategies are available
 */
const StrategyTabs = memo(({ strategies, activeId, onSelect }) => {
  if (strategies.length <= 1) return null;

  return (
    <div className="flex space-x-1 mb-4 bg-gray-800/30 rounded-lg p-1">
      {strategies.map(([id, data]) => {
        const config = getStrategyConfig(id);
        const isActive = id === activeId;
        const isRunning = data?.status?.running;

        return (
          <button
            key={id}
            onClick={() => onSelect(id)}
            className={`
              px-4 py-2 rounded-md text-xs font-medium transition-all
              ${isActive
                ? `${config.bgClass} ${config.textClass} ${config.borderClass} border`
                : 'text-gray-400 hover:text-gray-300 hover:bg-gray-800/50'
              }
            `}
          >
            {config.label}
            {isRunning && (
              <span className="ml-2 w-1.5 h-1.5 rounded-full bg-green-500 inline-block" />
            )}
          </button>
        );
      })}
    </div>
  );
});

StrategyTabs.displayName = 'StrategyTabs';

/**
 * StatBox - Primary metrics display
 */
const StatBox = memo(({ label, value, subtitle, valueClass = 'text-white', icon: Icon, accentColor = 'gray' }) => {
  const accentStyles = {
    violet: 'border-violet-500/30 bg-gradient-to-br from-violet-950/30 via-gray-900/50 to-gray-900/30',
    green: 'border-green-500/30 bg-gradient-to-br from-green-950/30 via-gray-900/50 to-gray-900/30',
    red: 'border-red-500/30 bg-gradient-to-br from-red-950/30 via-gray-900/50 to-gray-900/30',
    yellow: 'border-yellow-500/30 bg-gradient-to-br from-yellow-950/30 via-gray-900/50 to-gray-900/30',
    cyan: 'border-cyan-500/30 bg-gradient-to-br from-cyan-950/30 via-gray-900/50 to-gray-900/30',
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
      {subtitle && (
        <div className="text-xs text-gray-500 mt-1 font-mono">{subtitle}</div>
      )}
    </div>
  );
});

StatBox.displayName = 'StatBox';

/**
 * SkipBox - Mini stat box for skip breakdown
 */
const SkipBox = memo(({ label, value, accentColor = 'yellow' }) => {
  const hasValue = value > 0;

  const colorStyles = {
    yellow: {
      active: 'bg-yellow-900/20 border-yellow-700/30',
      text: 'text-yellow-400',
    },
    emerald: {
      active: 'bg-emerald-900/20 border-emerald-700/30',
      text: 'text-emerald-400',
    },
    violet: {
      active: 'bg-violet-900/20 border-violet-700/30',
      text: 'text-violet-400',
    },
  };

  const colors = colorStyles[accentColor] || colorStyles.yellow;

  return (
    <div className={`
      rounded-lg p-2 border text-center
      transition-all duration-200 ease-out
      ${hasValue
        ? colors.active
        : 'bg-gray-800/30 border-gray-700/30'
      }
    `}>
      <div className="text-[9px] text-gray-500 uppercase tracking-wider font-medium mb-0.5 truncate">
        {label}
      </div>
      <div className={`text-sm font-mono font-bold ${hasValue ? colors.text : 'text-gray-600'}`}>
        {value}
      </div>
    </div>
  );
});

SkipBox.displayName = 'SkipBox';

/**
 * SkipBreakdownSection - Renders strategy-specific skip breakdown
 */
const SkipBreakdownSection = memo(({ strategyId, skipBreakdown }) => {
  // Agentic Research uses different skip categories
  if (strategyId === 'agentic_research') {
    return (
      <div className="mb-4">
        <div className="flex items-center space-x-2 mb-2">
          <AlertCircle className="w-3 h-3 text-gray-600" />
          <span className="text-[10px] text-gray-500 uppercase tracking-wider font-medium">Skip Breakdown</span>
        </div>
        <div className="grid grid-cols-4 gap-2">
          <SkipBox label="Threshold" value={skipBreakdown.threshold || 0} accentColor="emerald" />
          <SkipBox label="Position Lim" value={skipBreakdown.position_limit || 0} accentColor="emerald" />
          <SkipBox label="Event Lim" value={skipBreakdown.event_limit || 0} accentColor="emerald" />
          <SkipBox label="Hold Rec" value={skipBreakdown.hold_recommendation || 0} accentColor="emerald" />
        </div>
      </div>
    );
  }

  // Default: RLM/ODMR skip breakdown
  return (
    <div className="mb-4">
      <div className="flex items-center space-x-2 mb-2">
        <AlertCircle className="w-3 h-3 text-gray-600" />
        <span className="text-[10px] text-gray-500 uppercase tracking-wider font-medium">Skip Breakdown</span>
      </div>
      <div className="grid grid-cols-6 gap-2">
        <SkipBox label="Spread Vol" value={skipBreakdown.spread_volatility || 0} />
        <SkipBox label="Spread Wide" value={skipBreakdown.spread_wide || 0} />
        <SkipBox label="Stale OB" value={skipBreakdown.stale_orderbook || 0} />
        <SkipBox label="Rate Lim" value={skipBreakdown.rate_limited || 0} />
        <SkipBox label="Pos Max" value={skipBreakdown.position_maxed || 0} />
        <SkipBox label="No OB Data" value={skipBreakdown.no_ob_data || 0} />
      </div>
    </div>
  );
});

SkipBreakdownSection.displayName = 'SkipBreakdownSection';

/**
 * AgenticMetricsSection - Extended metrics for Agentic Research strategy
 */
const AgenticMetricsSection = memo(({ strategyData }) => {
  const agenticMetrics = strategyData?.agentic_metrics || {};

  // Only show for agentic_research strategy
  if (!agenticMetrics || Object.keys(agenticMetrics).length === 0) {
    return null;
  }

  const eventsResearched = agenticMetrics.events_researched || 0;
  const marketsResearched = agenticMetrics.markets_researched || 0;
  const cacheHitRate = (agenticMetrics.cache_hit_rate || 0) * 100;
  const calibrationSamples = agenticMetrics.calibration_samples || 0;
  const calibrationError = agenticMetrics.calibration_avg_error_cents || 0;

  return (
    <div className="mb-4">
      <div className="flex items-center space-x-2 mb-2">
        <Brain className="w-3 h-3 text-emerald-500" />
        <span className="text-[10px] text-gray-500 uppercase tracking-wider font-medium">Research Metrics</span>
      </div>
      <div className="grid grid-cols-4 gap-2">
        {/* Events Researched */}
        <div className="rounded-lg p-2 border text-center bg-emerald-900/20 border-emerald-700/30">
          <div className="text-[9px] text-gray-500 uppercase tracking-wider font-medium mb-0.5">
            Events
          </div>
          <div className="text-sm font-mono font-bold text-emerald-400">
            {eventsResearched}
          </div>
        </div>

        {/* Markets Researched */}
        <div className="rounded-lg p-2 border text-center bg-emerald-900/20 border-emerald-700/30">
          <div className="text-[9px] text-gray-500 uppercase tracking-wider font-medium mb-0.5">
            Markets
          </div>
          <div className="text-sm font-mono font-bold text-emerald-400">
            {marketsResearched}
          </div>
        </div>

        {/* Cache Hit Rate */}
        <div className="rounded-lg p-2 border text-center bg-cyan-900/20 border-cyan-700/30">
          <div className="text-[9px] text-gray-500 uppercase tracking-wider font-medium mb-0.5">
            Cache Hit
          </div>
          <div className="text-sm font-mono font-bold text-cyan-400">
            {cacheHitRate.toFixed(0)}%
          </div>
        </div>

        {/* LLM Calibration */}
        <div className="rounded-lg p-2 border text-center bg-violet-900/20 border-violet-700/30">
          <div className="text-[9px] text-gray-500 uppercase tracking-wider font-medium mb-0.5">
            Calib Err
          </div>
          <div className="text-sm font-mono font-bold text-violet-400">
            {calibrationSamples > 0 ? `${calibrationError}c` : '-'}
          </div>
        </div>
      </div>
    </div>
  );
});

AgenticMetricsSection.displayName = 'AgenticMetricsSection';

/**
 * RateLimiterBar - Progress bar showing rate limiter utilization
 */
const RateLimiterBar = memo(({ tokens, capacity, utilization }) => {
  const barWidth = Math.max(0, Math.min(100, 100 - utilization));

  return (
    <div className="flex items-center space-x-3">
      <div className="flex-1">
        <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-violet-600 to-violet-400 transition-all duration-500 rounded-full"
            style={{ width: `${barWidth}%` }}
          />
        </div>
      </div>
      <div className="text-xs font-mono text-gray-400 w-16 text-right">
        {tokens}/{capacity}
      </div>
    </div>
  );
});

RateLimiterBar.displayName = 'RateLimiterBar';

/**
 * ConfigSection - Collapsible configuration display
 */
const ConfigSection = memo(({ config }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!config || Object.keys(config).length === 0) {
    return null;
  }

  return (
    <div className="border border-gray-700/30 rounded-lg overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between px-3 py-2 bg-gray-800/30 hover:bg-gray-800/50 transition-colors"
      >
        <div className="flex items-center space-x-2">
          <Settings className="w-3.5 h-3.5 text-gray-500" />
          <span className="text-xs text-gray-400 uppercase tracking-wider font-medium">Config</span>
        </div>
        {isExpanded ? (
          <ChevronDown className="w-3.5 h-3.5 text-gray-500" />
        ) : (
          <ChevronRight className="w-3.5 h-3.5 text-gray-500" />
        )}
      </button>

      {isExpanded && (
        <div className="px-3 py-2 bg-gray-900/30 grid grid-cols-3 gap-2 text-xs">
          {Object.entries(config).map(([key, value]) => (
            <div key={key} className="flex justify-between">
              <span className="text-gray-500">{key.replace(/_/g, ' ')}:</span>
              <span className="text-gray-300 font-mono">
                {typeof value === 'boolean' ? (value ? 'ON' : 'OFF') : String(value)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
});

ConfigSection.displayName = 'ConfigSection';

/**
 * Format P&L with color coding - defined outside component for performance
 */
const formatPnl = (pnlCents) => {
  if (pnlCents === null || pnlCents === undefined) return null;
  const prefix = pnlCents > 0 ? '+' : '';
  return {
    text: `${prefix}${pnlCents}c`,
    colorClass: pnlCents > 0 ? 'text-green-400' : pnlCents < 0 ? 'text-red-400' : 'text-gray-400'
  };
};

/**
 * DecisionRow - Row in the recent decisions table
 * Renders strategy-specific columns based on strategyId
 */
const DecisionRow = memo(({ decision, strategyId }) => {
  const isExecuted = decision.action === 'executed' || decision.action === 'reentry' ||
                     decision.action === 'entry_placed' || decision.action?.startsWith('exit_');
  const isSkipped = decision.action?.startsWith('skipped');

  // Render strategy-specific columns
  const renderStrategyColumns = () => {
    if (strategyId === 'agentic_research') {
      const aiProb = decision.ai_probability;
      const edge = decision.edge;
      return (
        <>
          <td className="px-2 py-2 text-right">
            <span className="text-xs text-emerald-400">
              {aiProb != null ? `${(aiProb * 100).toFixed(0)}%` : '-'}
            </span>
          </td>
          <td className="px-2 py-2 text-right">
            <span className={`text-xs ${edge != null && edge > 0 ? 'text-green-400' : edge != null && edge < 0 ? 'text-red-400' : 'text-gray-400'}`}>
              {edge != null ? `${(edge * 100).toFixed(1)}%` : '-'}
            </span>
          </td>
        </>
      );
    }
    if (strategyId === 'odmr') {
      const pnl = formatPnl(decision.pnl_cents);
      return (
        <>
          <td className="px-2 py-2 text-right">
            <span className="text-xs text-gray-400">
              {decision.dip_depth != null ? `${decision.dip_depth}c` : '—'}
            </span>
          </td>
          <td className="px-2 py-2 text-right">
            <span className={`text-xs ${pnl?.colorClass || 'text-gray-400'}`}>
              {pnl?.text || '—'}
            </span>
          </td>
        </>
      );
    }
    // Default: RLM_NO columns
    return (
      <>
        <td className="px-2 py-2 text-right">
          <span className="text-xs text-gray-400">
            {decision.yes_ratio != null ? `${(decision.yes_ratio * 100).toFixed(0)}%` : '—'}
          </span>
        </td>
        <td className="px-2 py-2 text-right">
          <span className="text-xs text-gray-400">
            {decision.price_drop != null ? `${decision.price_drop}c` : '—'}
          </span>
        </td>
      </>
    );
  };

  return (
    <tr className="border-b border-gray-700/20 hover:bg-gray-800/30 transition-colors">
      <td className="px-2 py-2">
        <span className="font-mono text-gray-300 text-xs truncate block max-w-[100px]" title={decision.market_ticker}>
          {decision.market_ticker || '—'}
        </span>
      </td>
      <td className="px-2 py-2 text-center">
        <span className={`
          inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-bold
          ${isExecuted
            ? 'bg-green-900/30 text-green-400 border border-green-600/20'
            : isSkipped
              ? 'bg-yellow-900/30 text-yellow-400 border border-yellow-600/20'
              : 'bg-red-900/30 text-red-400 border border-red-600/20'
          }
        `}>
          {decision.action?.replace('skipped_', '').replace('exit_', '').toUpperCase() || 'UNKNOWN'}
        </span>
      </td>
      {renderStrategyColumns()}
      <td className="px-2 py-2 text-right">
        <span className="font-mono text-gray-500 text-[10px]">
          {formatAge(decision.age_seconds)}
        </span>
      </td>
    </tr>
  );
});

DecisionRow.displayName = 'DecisionRow';

/**
 * RecentDecisionsSection - Collapsible table of recent decisions
 * Renders strategy-specific column headers based on strategyId
 */
const RecentDecisionsSection = memo(({ decisions, strategyId }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!decisions || decisions.length === 0) {
    return null;
  }

  // Render strategy-specific column headers
  const renderColumnHeaders = () => {
    if (strategyId === 'agentic_research') {
      return (
        <>
          <th className="px-2 py-1.5 text-right text-[9px] text-gray-500 uppercase font-semibold">AI Prob</th>
          <th className="px-2 py-1.5 text-right text-[9px] text-gray-500 uppercase font-semibold">Edge</th>
        </>
      );
    }
    if (strategyId === 'odmr') {
      return (
        <>
          <th className="px-2 py-1.5 text-right text-[9px] text-gray-500 uppercase font-semibold">Dip</th>
          <th className="px-2 py-1.5 text-right text-[9px] text-gray-500 uppercase font-semibold">P&L</th>
        </>
      );
    }
    // Default: RLM_NO column headers
    return (
      <>
        <th className="px-2 py-1.5 text-right text-[9px] text-gray-500 uppercase font-semibold">YES%</th>
        <th className="px-2 py-1.5 text-right text-[9px] text-gray-500 uppercase font-semibold">Drop</th>
      </>
    );
  };

  return (
    <div className="border border-gray-700/30 rounded-lg overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between px-3 py-2 bg-gray-800/30 hover:bg-gray-800/50 transition-colors"
      >
        <div className="flex items-center space-x-2">
          <History className="w-3.5 h-3.5 text-gray-500" />
          <span className="text-xs text-gray-400 uppercase tracking-wider font-medium">Recent Decisions</span>
          <span className="text-[10px] text-gray-600 font-mono">({decisions.length})</span>
        </div>
        {isExpanded ? (
          <ChevronDown className="w-3.5 h-3.5 text-gray-500" />
        ) : (
          <ChevronRight className="w-3.5 h-3.5 text-gray-500" />
        )}
      </button>

      {isExpanded && (
        <div className="bg-gray-900/30 max-h-[200px] overflow-y-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-gray-900/80">
              <tr className="border-b border-gray-700/40">
                <th className="px-2 py-1.5 text-left text-[9px] text-gray-500 uppercase font-semibold">Market</th>
                <th className="px-2 py-1.5 text-center text-[9px] text-gray-500 uppercase font-semibold">Action</th>
                {renderColumnHeaders()}
                <th className="px-2 py-1.5 text-right text-[9px] text-gray-500 uppercase font-semibold">Age</th>
              </tr>
            </thead>
            <tbody>
              {decisions.map((d, idx) => (
                <DecisionRow key={d.signal_id || idx} decision={d} strategyId={strategyId} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
});

RecentDecisionsSection.displayName = 'RecentDecisionsSection';

/**
 * EmptyState - Shown when no strategies are running
 */
const EmptyState = memo(() => (
  <div className="
    bg-gradient-to-b from-gray-800/20 to-gray-900/20
    rounded-xl p-8 border border-gray-700/30
    flex flex-col items-center justify-center
  ">
    <div className="relative mb-4">
      <div className="w-16 h-16 rounded-full bg-gradient-to-br from-gray-800/80 to-gray-900/80 border border-gray-700/50 flex items-center justify-center">
        <Zap className="w-7 h-7 text-gray-600" />
      </div>
    </div>
    <div className="text-gray-400 text-sm font-medium mb-1">No Strategies Running</div>
    <div className="text-gray-600 text-xs text-center max-w-[200px]">
      Trading strategies will appear here when active
    </div>
  </div>
));

EmptyState.displayName = 'EmptyState';

/**
 * formatUptime - Format seconds into human-readable duration
 */
const formatUptime = (seconds) => {
  if (!seconds || seconds < 0) return '0s';

  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);

  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  return `${minutes}m`;
};

/**
 * TradingStrategiesPanel - Professional trading strategies dashboard
 * Shows strategy status, performance metrics, skip breakdown, and decision history
 */
const TradingStrategiesPanel = ({ strategyStatus }) => {
  const [isExpanded, setIsExpanded] = useState(true);
  const [activeStrategy, setActiveStrategy] = useState('rlm_no');

  const coordinator = strategyStatus?.coordinator || {};
  const strategies = strategyStatus?.strategies || {};
  const recentDecisions = strategyStatus?.recent_decisions || [];

  const hasStrategies = Object.keys(strategies).length > 0;
  const strategyEntries = Object.entries(strategies);

  // Auto-select first strategy if none selected or current selection is invalid
  useEffect(() => {
    if (strategyEntries.length > 0) {
      const validIds = strategyEntries.map(([id]) => id);
      if (!activeStrategy || !validIds.includes(activeStrategy)) {
        setActiveStrategy(strategyEntries[0][0]);
      }
    }
  }, [strategyEntries, activeStrategy]);

  // Get active strategy data (tab-based selection)
  const strategyId = activeStrategy || strategyEntries[0]?.[0] || '';
  const strategyData = strategies[strategyId] || null;
  const strategyConfig = getStrategyConfig(strategyId);

  // Extract strategy data
  const status = strategyData?.status || {};
  const config = strategyData?.config || {};
  const skipBreakdown = strategyData?.skip_breakdown || {};

  // Calculate derived metrics
  const performance = strategyData?.performance || {};
  const skipRate = useMemo(() => {
    const total = strategyData?.performance?.signals_detected || 0;
    const skipped = strategyData?.performance?.signals_skipped || 0;
    if (total === 0) return 0;
    return (skipped / total * 100).toFixed(1);
  }, [strategyData?.performance?.signals_detected, strategyData?.performance?.signals_skipped]);

  // Filter decisions by active strategy tab
  const filteredDecisions = useMemo(() => {
    if (!activeStrategy) return recentDecisions;
    return recentDecisions.filter(d =>
      !d.strategy_id || d.strategy_id === activeStrategy
    );
  }, [recentDecisions, activeStrategy]);

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
          <div className={`p-2 rounded-lg bg-gradient-to-br ${strategyConfig.bgClass} border ${strategyConfig.borderClass}`}>
            <Zap className={`w-4 h-4 ${strategyConfig.textClass}`} />
          </div>
          <div>
            <h3 className="text-sm font-bold text-gray-200 uppercase tracking-wider">Trading Strategies</h3>
            <div className="flex items-center space-x-2 mt-0.5">
              {hasStrategies && strategyData ? (
                <>
                  <span className={`font-mono text-[10px] ${strategyConfig.textClass}`}>
                    {strategyConfig.label}
                  </span>
                  <span className="text-gray-600">|</span>
                  <span className="font-mono text-[10px] text-gray-500">
                    {status.running ? `${formatUptime(status.uptime_seconds)} uptime` : 'stopped'}
                  </span>
                </>
              ) : (
                <span className="font-mono text-[10px] text-gray-500">
                  No active strategies
                </span>
              )}
            </div>
          </div>
        </div>
        <div className="flex items-center space-x-3">
          {hasStrategies && (
            <HealthBadge healthy={status.healthy} running={status.running} />
          )}
          {isExpanded ? (
            <ChevronDown className="w-4 h-4 text-gray-400" />
          ) : (
            <ChevronRight className="w-4 h-4 text-gray-400" />
          )}
        </div>
      </div>

      {isExpanded && (
        <>
          {/* Strategy Tabs - only shown when 2+ strategies */}
          <StrategyTabs
            strategies={strategyEntries}
            activeId={activeStrategy}
            onSelect={setActiveStrategy}
          />

          {hasStrategies && strategyData ? (
            <>
              {/* Primary Metrics Grid */}
              <div className="grid grid-cols-4 gap-3 mb-4">
                <StatBox
                  label={strategyId === 'agentic_research' ? 'Order Rate' : 'Execution Rate'}
                  value={`${performance.execution_rate || 0}%`}
                  subtitle={strategyId === 'agentic_research'
                    ? `${performance.signals_executed || 0}/${performance.signals_detected || 0} orders`
                    : `${performance.signals_executed || 0}/${performance.signals_detected || 0}`}
                  icon={Target}
                  accentColor={performance.execution_rate >= 50 ? 'green' : 'yellow'}
                  valueClass={performance.execution_rate >= 50 ? 'text-green-400' : 'text-yellow-400'}
                />
                <StatBox
                  label="Skip Rate"
                  value={`${skipRate}%`}
                  subtitle={`${performance.signals_skipped || 0} skipped`}
                  icon={Filter}
                  accentColor={parseFloat(skipRate) > 50 ? 'yellow' : 'gray'}
                  valueClass={parseFloat(skipRate) > 50 ? 'text-yellow-400' : 'text-gray-400'}
                />
                <StatBox
                  label="Signals Today"
                  value={performance.signals_detected || 0}
                  subtitle={`${performance.reentries || 0} reentries`}
                  icon={Activity}
                  accentColor="violet"
                  valueClass="text-violet-400"
                />
                <StatBox
                  label="Rate Limit"
                  value={
                    <RateLimiterBar
                      tokens={coordinator.rate_limiter?.tokens_available || 0}
                      capacity={coordinator.rate_limiter?.capacity || 20}
                      utilization={coordinator.rate_limiter?.utilization_percent || 0}
                    />
                  }
                  subtitle={`${coordinator.rate_limiter?.refill_rate_per_minute || 0}/min refill`}
                  icon={Layers}
                  accentColor="cyan"
                />
              </div>

              {/* Skip Breakdown Grid - Strategy-specific */}
              <SkipBreakdownSection
                strategyId={strategyId}
                skipBreakdown={skipBreakdown}
              />

              {/* Agentic Research Extended Metrics */}
              {strategyId === 'agentic_research' && (
                <AgenticMetricsSection strategyData={strategyData} />
              )}

              {/* Collapsible Sections */}
              <div className="space-y-3">
                <ConfigSection config={config} />
                <RecentDecisionsSection decisions={filteredDecisions} strategyId={strategyId} />
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

export default memo(TradingStrategiesPanel);
