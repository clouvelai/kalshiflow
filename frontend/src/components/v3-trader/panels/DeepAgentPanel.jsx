import React, { useState, memo, useMemo } from 'react';
import useRelativeTime from '../../../hooks/v3-trader/useRelativeTime';
import {
  ChevronRight,
  ChevronDown,
  Brain,
  Zap,
  CheckCircle,
  XCircle,
  TrendingUp,
  TrendingDown,
  Clock,
  Activity,
  FileText,
  RefreshCw,
  DollarSign,
  Target,
  BookOpen,
  AlertCircle,
  Newspaper,
} from 'lucide-react';
import renderThinkingMarkdown from '../../../utils/renderThinkingMarkdown';

/**
 * Thinking Stream - Real-time agent reasoning display
 */
const ThinkingStream = memo(({ thinking, isLearning }) => {
  if (!thinking?.text && !isLearning) {
    return (
      <div className="p-4 bg-gray-800/30 rounded-lg border border-gray-700/20 text-center">
        <div className="text-gray-500 text-sm flex items-center justify-center gap-2">
          <Brain className="w-4 h-4" />
          <span>Agent is observing...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 bg-violet-900/20 rounded-lg border border-violet-700/30 max-h-[400px] overflow-y-auto">
      <div className="flex items-start gap-3">
        <div className="p-2 rounded-lg bg-violet-900/40 flex-shrink-0">
          <Brain className="w-4 h-4 text-violet-400 animate-pulse" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-xs text-violet-400 font-medium mb-2 uppercase tracking-wider flex items-center justify-between">
            <span>Thinking</span>
            {thinking?.cycle > 0 && (
              <span className="text-violet-500 font-mono">Cycle {thinking.cycle}</span>
            )}
          </div>
          <div className="break-words">
            {renderThinkingMarkdown(thinking?.text)}
            <span className="animate-pulse text-violet-400">█</span>
          </div>
        </div>
      </div>
    </div>
  );
});

ThinkingStream.displayName = 'ThinkingStream';

/**
 * Tool Call Row - Shows a single tool invocation
 */
const ToolCallRow = memo(({ toolCall }) => {
  const getToolIcon = (tool) => {
    switch (tool) {
      case 'get_markets':
        return <Target className="w-3 h-3 text-cyan-400" />;
      case 'query_gdelt_news':
        return <Newspaper className="w-3 h-3 text-blue-400" />;
      case 'query_gdelt_events':
        return <Activity className="w-3 h-3 text-red-400" />;
      case 'search_gdelt_articles':
        return <Newspaper className="w-3 h-3 text-blue-300" />;
      case 'get_gdelt_volume_timeline':
        return <TrendingUp className="w-3 h-3 text-purple-400" />;
      case 'trade':
        return <DollarSign className="w-3 h-3 text-green-400" />;
      case 'get_session_state':
        return <Activity className="w-3 h-3 text-blue-400" />;
      case 'read_memory':
        return <BookOpen className="w-3 h-3 text-violet-400" />;
      case 'write_memory':
        return <FileText className="w-3 h-3 text-violet-400" />;
      default:
        return <Zap className="w-3 h-3 text-gray-400" />;
    }
  };

  return (
    <div className="flex items-center justify-between py-2 border-b border-gray-700/20 last:border-b-0">
      <div className="flex items-center gap-2 flex-1 min-w-0">
        <div className="p-1 rounded bg-gray-800/50">
          {getToolIcon(toolCall.tool)}
        </div>
        <span className="font-mono text-xs text-gray-300 truncate">
          {toolCall.tool}
        </span>
        <span className="text-[10px] text-gray-500 truncate max-w-[200px]">
          {(() => { const s = JSON.stringify(toolCall.input); return s.length > 50 ? s.slice(0, 50) + '...' : s; })()}
        </span>
      </div>
      <span className="font-mono text-[10px] text-gray-500">
        {toolCall.timestamp}
      </span>
    </div>
  );
});

ToolCallRow.displayName = 'ToolCallRow';

/**
 * Trade Row - Shows a single trade
 */
const TradeRow = memo(({ trade }) => {
  const sideColor = trade.side === 'yes' ? 'text-green-400' : 'text-red-400';
  const sideBg = trade.side === 'yes' ? 'bg-green-900/30 border-green-600/20' : 'bg-red-900/30 border-red-600/20';
  const isSell = trade.action === 'sell';

  return (
    <div className="p-3 bg-gray-800/30 rounded-lg border border-gray-700/20 mb-2">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold border ${
            isSell ? 'bg-amber-900/30 border-amber-600/20 text-amber-400' : 'bg-blue-900/30 border-blue-600/20 text-blue-400'
          }`}>
            {isSell ? 'SELL' : 'BUY'}
          </span>
          <span className={`px-2 py-0.5 rounded text-[10px] font-bold border ${sideBg} ${sideColor}`}>
            {trade.side.toUpperCase()}
          </span>
          <span className="font-mono text-xs text-gray-300">{trade.ticker}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-400">{trade.contracts} contracts</span>
          <span className="font-mono text-xs text-emerald-400">{trade.priceCents}c</span>
        </div>
      </div>
      <div className="text-[10px] text-gray-500 truncate" title={trade.reasoning}>
        {trade.reasoning}
      </div>
    </div>
  );
});

TradeRow.displayName = 'TradeRow';

/**
 * Settlement Row - Shows a trade settlement with P&L
 */
const SettlementRow = memo(({ settlement }) => {
  const isWin = settlement.result === 'win';
  const isLoss = settlement.result === 'loss';

  const resultStyle = isWin
    ? 'bg-green-900/30 text-green-400 border-green-600/20'
    : isLoss
      ? 'bg-red-900/30 text-red-400 border-red-600/20'
      : 'bg-gray-800/30 text-gray-400 border-gray-600/20';

  const ResultIcon = isWin ? TrendingUp : isLoss ? TrendingDown : Activity;

  return (
    <div className="flex items-center justify-between py-2 border-b border-gray-700/20 last:border-b-0">
      <div className="flex items-center gap-2">
        <span className={`flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold border ${resultStyle}`}>
          <ResultIcon className="w-3 h-3" />
          {(settlement.result || 'unknown').toUpperCase()}
        </span>
        <span className="font-mono text-xs text-gray-300">{settlement.ticker}</span>
      </div>
      <div className="flex items-center gap-3">
        {(settlement.entryPrice != null && settlement.exitPrice != null) && (
          <span className="text-[10px] text-gray-500">
            {settlement.entryPrice}c → {settlement.exitPrice}c
          </span>
        )}
        <span className={`font-mono text-xs font-bold ${isWin ? 'text-green-400' : isLoss ? 'text-red-400' : 'text-gray-400'}`}>
          ${((settlement.pnlCents || 0) / 100).toFixed(2)}
        </span>
      </div>
    </div>
  );
});

SettlementRow.displayName = 'SettlementRow';

/**
 * Learning Row - Shows an extracted learning
 */
const LearningRow = memo(({ learning }) => (
  <div className="flex items-start gap-2 py-2 border-b border-gray-700/20 last:border-b-0">
    <FileText className="w-3.5 h-3.5 text-violet-400 mt-0.5 flex-shrink-0" />
    <div className="flex-1 min-w-0">
      <div className="text-xs text-gray-300 break-words">
        {learning.content}
      </div>
      <div className="text-[10px] text-gray-500 mt-0.5">
        {learning.timestamp}
      </div>
    </div>
  </div>
));

LearningRow.displayName = 'LearningRow';

/**
 * Stats Card - Compact stat display
 */
const StatsCard = memo(({ label, value, color = 'text-gray-300', bgColor = 'bg-gray-800/30', borderColor = 'border-gray-700/20', testId }) => (
  <div className={`rounded-lg p-2 border text-center ${bgColor} ${borderColor}`} {...(testId ? { 'data-testid': testId } : {})}>
    <div className="text-[9px] text-gray-500 uppercase tracking-wider font-medium mb-0.5">{label}</div>
    <div className={`text-sm font-mono font-bold ${color}`}>{value}</div>
  </div>
));

StatsCard.displayName = 'StatsCard';

/**
 * CostPanel - Collapsible LLM cost tracking display
 */
const CostPanel = memo(({ costData }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!costData) return null;

  const sessionTotal = costData.sessionCost?.total_cost_usd || 0;
  const cacheSavings = costData.sessionCost?.cache_savings_usd || 0;
  const cycleCount = costData.cycle || 0;
  const avgPerCycle = cycleCount > 0 ? sessionTotal / cycleCount : 0;

  const formatUsd = (v) => {
    if (v >= 1) return `$${v.toFixed(2)}`;
    if (v >= 0.01) return `$${v.toFixed(3)}`;
    return `$${v.toFixed(4)}`;
  };

  const formatTokens = (n) => {
    if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
    if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
    return String(n);
  };

  return (
    <div className="rounded-xl border border-amber-800/30 overflow-hidden mb-4">
      {/* Collapsed bar - always visible */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between px-3 py-2 bg-gradient-to-r from-amber-900/20 via-amber-900/10 to-transparent hover:from-amber-900/30 transition-all duration-200"
      >
        <div className="flex items-center gap-2">
          <DollarSign className="w-3.5 h-3.5 text-amber-400" />
          <span className="text-[11px] text-amber-300 uppercase tracking-wider font-semibold">
            LLM Cost
          </span>
        </div>
        <div className="flex items-center gap-3">
          <span className="font-mono text-xs text-amber-400 font-bold">
            {formatUsd(sessionTotal)}
          </span>
          {cacheSavings > 0 && (
            <span className="text-[10px] text-emerald-400">
              saved {formatUsd(cacheSavings)}
            </span>
          )}
          {avgPerCycle > 0 && (
            <span className="text-[10px] text-gray-500 font-mono">
              ~{formatUsd(avgPerCycle)}/cycle
            </span>
          )}
          {isExpanded ? (
            <ChevronDown className="w-3.5 h-3.5 text-amber-400/60" />
          ) : (
            <ChevronRight className="w-3.5 h-3.5 text-amber-400/60" />
          )}
        </div>
      </button>

      {/* Expanded detail */}
      {isExpanded && (
        <div className="px-3 pb-3 pt-2 bg-gray-900/40 border-t border-amber-800/20">
          {/* Model */}
          <div className="flex items-center justify-between mb-2">
            <span className="text-[10px] text-gray-500">Model</span>
            <span className="text-[10px] text-gray-300 font-mono">{costData.model || 'unknown'}</span>
          </div>

          {/* Token cost grid */}
          <div className="grid grid-cols-2 gap-2 mb-2">
            <div className="bg-gray-800/40 rounded p-1.5">
              <div className="text-[9px] text-gray-500 uppercase">Input</div>
              <div className="text-xs font-mono text-gray-300">{formatTokens(costData.sessionTokens?.input || 0)}</div>
              <div className="text-[10px] font-mono text-amber-400">{formatUsd(costData.sessionCost?.input_cost_usd || 0)}</div>
            </div>
            <div className="bg-gray-800/40 rounded p-1.5">
              <div className="text-[9px] text-gray-500 uppercase">Output</div>
              <div className="text-xs font-mono text-gray-300">{formatTokens(costData.sessionTokens?.output || 0)}</div>
              <div className="text-[10px] font-mono text-amber-400">{formatUsd(costData.sessionCost?.output_cost_usd || 0)}</div>
            </div>
            <div className="bg-gray-800/40 rounded p-1.5">
              <div className="text-[9px] text-gray-500 uppercase">Cache Read</div>
              <div className="text-xs font-mono text-gray-300">{formatTokens(costData.sessionTokens?.cache_read || 0)}</div>
              <div className="text-[10px] font-mono text-emerald-400">{formatUsd(costData.sessionCost?.cache_read_cost_usd || 0)}</div>
            </div>
            <div className="bg-gray-800/40 rounded p-1.5">
              <div className="text-[9px] text-gray-500 uppercase">Cache Write</div>
              <div className="text-xs font-mono text-gray-300">{formatTokens(costData.sessionTokens?.cache_created || 0)}</div>
              <div className="text-[10px] font-mono text-amber-400">{formatUsd(costData.sessionCost?.cache_write_cost_usd || 0)}</div>
            </div>
          </div>

          {/* Cache savings highlight */}
          {cacheSavings > 0 && (
            <div className="flex items-center justify-between px-2 py-1.5 bg-emerald-900/20 rounded border border-emerald-800/30 mb-2">
              <span className="text-[10px] text-emerald-400">Cache Savings</span>
              <span className="text-xs font-mono font-bold text-emerald-400">{formatUsd(cacheSavings)}</span>
            </div>
          )}

          {/* Last cycle info */}
          {costData.lastCycleCost && (
            <div className="flex items-center justify-between text-[10px] text-gray-500">
              <span>Last cycle: {formatUsd(costData.lastCycleCost.total_cost_usd || 0)}</span>
              <span>{costData.lastCycleTokens?.api_calls || 0} API calls</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
});

CostPanel.displayName = 'CostPanel';

/**
 * DeepAgentPanel - Main panel for the self-improving deep agent
 *
 * Props:
 * - statsOnly: When true, renders only header and stat summary (for Trader tab)
 * - agentState: Current agent state object
 * - thinking: Current thinking stream
 * - toolCalls: Array of tool call records
 * - trades: Array of trade records
 * - settlements: Array of settlement records
 * - learnings: Array of learning records
 * - isRunning: Whether agent is currently running
 * - isLearning: Whether agent is in learning mode
 * - costData: LLM cost tracking data
 *
 * When statsOnly={true}, displays:
 * - Agent status header with RUNNING/STOPPED badge
 * - Session stats (P&L, Trades, Win Rate, Cycles)
 * - Extraction pipeline stats
 *
 * When statsOnly={false} (default), displays full view with all sections.
 */
const DeepAgentPanel = ({
  agentState,
  thinking,
  toolCalls = [],
  trades = [],
  settlements = [],
  learnings = [],
  isRunning = false,
  isLearning = false,
  statsOnly = false,
  costData = null,
}) => {
  const [isExpanded, setIsExpanded] = useState(true);
  const [showTools, setShowTools] = useState(false);

  // Live-updating timestamps: forces re-render every 30s
  useRelativeTime(30000);

  // Calculate session P&L from settlements
  const sessionPnL = useMemo(() => {
    return settlements.reduce((sum, s) => sum + (s.pnlCents || 0), 0);
  }, [settlements]);

  // Calculate win rate
  const winRate = useMemo(() => {
    if (settlements.length === 0) return 0;
    const wins = settlements.filter(s => s.result === 'win').length;
    return (wins / settlements.length * 100).toFixed(0);
  }, [settlements]);

  const statusColor = isRunning ? 'text-emerald-400' : 'text-gray-500';
  const statusBg = isRunning ? 'bg-emerald-900/40 border-emerald-600/30' : 'bg-gray-800/50 border-gray-700/30';

  return (
    <div data-testid="deep-agent-panel" className="
      bg-gradient-to-br from-gray-900/70 via-gray-900/50 to-gray-950/70
      backdrop-blur-md rounded-2xl
      border border-violet-800/30
      shadow-xl shadow-black/20
      p-5 mb-6
    ">
      {/* Header */}
      <div
        className="flex items-center justify-between mb-4 cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center space-x-3">
          <div className="p-2 rounded-lg bg-gradient-to-br from-violet-900/40 to-violet-950/40 border border-violet-700/30">
            <Brain className={`w-5 h-5 ${isLearning ? 'text-violet-400 animate-pulse' : 'text-violet-500'}`} />
          </div>
          <div>
            <h3 className="text-sm font-bold text-gray-200 uppercase tracking-wider">
              Deep Agent
            </h3>
            <div className="flex items-center space-x-2 mt-0.5">
              <span className={`text-[10px] font-medium ${statusColor}`}>
                {isRunning ? 'LEARNING' : 'STOPPED'}
              </span>
              {agentState?.cycleCount > 0 && (
                <>
                  <span className="text-gray-600">|</span>
                  <span className="text-[10px] text-gray-500">
                    Cycle <span className="text-violet-400 font-mono">{agentState.cycleCount}</span>
                  </span>
                </>
              )}
            </div>
          </div>
        </div>
        <div className="flex items-center space-x-3">
          <div data-testid="deep-agent-status" className={`
            inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold border
            ${statusBg} ${statusColor}
          `}>
            {isRunning ? (
              <>
                <RefreshCw className="w-3 h-3 mr-1 animate-spin" />
                RUNNING
              </>
            ) : (
              <>
                <XCircle className="w-3 h-3 mr-1" />
                STOPPED
              </>
            )}
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
          {/* Session Stats */}
          <div className="grid grid-cols-4 gap-2 mb-4">
            <StatsCard
              label="Session P&L"
              value={`$${(sessionPnL / 100).toFixed(2)}`}
              color={sessionPnL >= 0 ? 'text-green-400' : 'text-red-400'}
              bgColor={sessionPnL >= 0 ? 'bg-green-900/20' : 'bg-red-900/20'}
              borderColor={sessionPnL >= 0 ? 'border-green-700/30' : 'border-red-700/30'}
              testId="deep-agent-pnl"
            />
            <StatsCard
              label="Trades"
              value={trades.length}
              color="text-cyan-400"
              bgColor="bg-cyan-900/20"
              borderColor="border-cyan-700/30"
              testId="deep-agent-trades"
            />
            <StatsCard
              label="Win Rate"
              value={`${winRate}%`}
              color={parseInt(winRate) >= 50 ? 'text-emerald-400' : 'text-amber-400'}
              bgColor={parseInt(winRate) >= 50 ? 'bg-emerald-900/20' : 'bg-amber-900/20'}
              borderColor={parseInt(winRate) >= 50 ? 'border-emerald-700/30' : 'border-amber-700/30'}
              testId="deep-agent-win-rate"
            />
            <StatsCard
              label="Cycles"
              value={agentState?.cycleCount || 0}
              color="text-violet-400"
              bgColor="bg-violet-900/20"
              borderColor="border-violet-700/30"
              testId="deep-agent-cycles"
            />
          </div>

          {/* Extraction Pipeline Stats (Sources → Extractions → Signals → GDELT) */}
          <div className={statsOnly ? "grid grid-cols-6 gap-2" : "grid grid-cols-5 gap-2 mb-4"}>
            <StatsCard
              label="Sources"
              value={agentState?.redditPostsProcessed || agentState?.postsProcessed || 0}
              color="text-orange-400"
              bgColor="bg-orange-900/20"
              borderColor="border-orange-700/30"
            />
            <StatsCard
              label="Extractions"
              value={agentState?.extractionsTotal || 0}
              color="text-violet-400"
              bgColor="bg-violet-900/20"
              borderColor="border-violet-700/30"
            />
            <StatsCard
              label="Signals"
              value={agentState?.extractionsMarketSignals || 0}
              color="text-cyan-400"
              bgColor="bg-cyan-900/20"
              borderColor="border-cyan-700/30"
            />
            <StatsCard
              label="Events"
              value={agentState?.eventsTracked || 0}
              color="text-blue-400"
              bgColor="bg-blue-900/20"
              borderColor="border-blue-700/30"
            />
            <StatsCard
              label="GDELT"
              value={agentState?.gdeltQueries || 0}
              color="text-blue-400"
              bgColor="bg-blue-900/20"
              borderColor="border-blue-800/30"
            />
            {statsOnly && (
              <StatsCard
                label="LLM Cost"
                value={costData?.sessionCost?.total_cost_usd != null
                  ? `$${costData.sessionCost.total_cost_usd < 1
                    ? costData.sessionCost.total_cost_usd.toFixed(3)
                    : costData.sessionCost.total_cost_usd.toFixed(2)}`
                  : '$0'}
                color="text-amber-400"
                bgColor="bg-amber-900/20"
                borderColor="border-amber-700/30"
              />
            )}
          </div>

          {/* Stats-only mode stops here - skip remaining sections */}
          {statsOnly ? null : (
          <>
          {/* LLM Cost Panel */}
          <CostPanel costData={costData} />

          {/* Thinking Stream */}
          <div className="mb-4">
            <div className="flex items-center space-x-2 mb-2">
              <Brain className="w-3 h-3 text-gray-500" />
              <span className="text-[10px] text-gray-500 uppercase tracking-wider font-medium">Thinking</span>
            </div>
            <ThinkingStream thinking={thinking} isLearning={isLearning} />
          </div>

          {/* Recent Activity */}
          <div className="mb-4">
            <button
              onClick={() => setShowTools(!showTools)}
              className="w-full flex items-center justify-between px-3 py-2 bg-gray-800/30 hover:bg-gray-800/50 rounded-lg transition-colors"
            >
              <div className="flex items-center gap-2">
                <Activity className="w-3.5 h-3.5 text-gray-500" />
                <span className="text-[10px] text-gray-400 uppercase tracking-wider font-medium">
                  Recent Activity
                </span>
                <span className="px-1.5 py-0.5 bg-gray-700/50 text-gray-400 text-[9px] rounded">
                  {toolCalls.length}
                </span>
              </div>
              {showTools ? (
                <ChevronDown className="w-3.5 h-3.5 text-gray-500" />
              ) : (
                <ChevronRight className="w-3.5 h-3.5 text-gray-500" />
              )}
            </button>
            {showTools && toolCalls.length > 0 && (
              <div className="mt-2 px-3 py-2 bg-gray-900/30 rounded-lg max-h-[200px] overflow-y-auto">
                {toolCalls.slice(0, 10).map((tc) => (
                  <ToolCallRow key={tc.id} toolCall={tc} />
                ))}
              </div>
            )}
          </div>

          {/* Trades */}
          {trades.length > 0 && (
            <div className="mb-4">
              <div className="flex items-center space-x-2 mb-2">
                <DollarSign className="w-3 h-3 text-gray-500" />
                <span className="text-[10px] text-gray-500 uppercase tracking-wider font-medium">
                  Recent Trades
                </span>
              </div>
              <div className="max-h-[200px] overflow-y-auto">
                {trades.slice(0, 5).map((trade) => (
                  <TradeRow key={trade.id} trade={trade} />
                ))}
              </div>
            </div>
          )}

          {/* Settlements */}
          {settlements.length > 0 && (
            <div className="mb-4">
              <div className="flex items-center space-x-2 mb-2">
                <CheckCircle className="w-3 h-3 text-gray-500" />
                <span className="text-[10px] text-gray-500 uppercase tracking-wider font-medium">
                  Settlements
                </span>
              </div>
              <div className="bg-gray-900/30 rounded-lg px-3 py-2 max-h-[150px] overflow-y-auto">
                {settlements.slice(0, 5).map((settlement) => (
                  <SettlementRow key={settlement.id} settlement={settlement} />
                ))}
              </div>
            </div>
          )}

          {/* Learnings */}
          {learnings.length > 0 && (
            <div className="mb-4">
              <div className="flex items-center space-x-2 mb-2">
                <BookOpen className="w-3 h-3 text-violet-400" />
                <span className="text-[10px] text-gray-500 uppercase tracking-wider font-medium">
                  Learnings (Live)
                </span>
              </div>
              <div className="bg-violet-900/20 rounded-lg px-3 py-2 border border-violet-700/30 max-h-[150px] overflow-y-auto">
                {learnings.slice(0, 5).map((learning) => (
                  <LearningRow key={learning.id} learning={learning} />
                ))}
              </div>
            </div>
          )}

          </>
          )}
        </>
      )}
    </div>
  );
};

export { CostPanel };
export default memo(DeepAgentPanel);
