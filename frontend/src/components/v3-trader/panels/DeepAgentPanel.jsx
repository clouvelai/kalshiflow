import React, { useState, memo, useMemo } from 'react';
import {
  ChevronRight,
  ChevronDown,
  Brain,
  Zap,
  CheckCircle,
  XCircle,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Clock,
  Activity,
  FileText,
  MessageSquare,
  RefreshCw,
  Loader2,
  DollarSign,
  Target,
  BookOpen,
  AlertCircle,
  ArrowRight,
  Sparkles,
  User,
  BarChart3,
} from 'lucide-react';

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
    <div className="p-4 bg-violet-900/20 rounded-lg border border-violet-700/30">
      <div className="flex items-start gap-3">
        <div className="p-2 rounded-lg bg-violet-900/40 flex-shrink-0">
          <Brain className="w-4 h-4 text-violet-400 animate-pulse" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-xs text-violet-400 font-medium mb-1 uppercase tracking-wider">
            Thinking
          </div>
          <div className="text-sm text-gray-300 whitespace-pre-wrap break-words">
            {thinking?.text}
            <span className="animate-pulse">█</span>
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
      case 'get_price_impacts':
        return <Sparkles className="w-3 h-3 text-orange-400" />;
      case 'search_news':
        return <MessageSquare className="w-3 h-3 text-amber-400" />;
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
          {JSON.stringify(toolCall.input).slice(0, 50)}...
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

  return (
    <div className="p-3 bg-gray-800/30 rounded-lg border border-gray-700/20 mb-2">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
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
          {settlement.result.toUpperCase()}
        </span>
        <span className="font-mono text-xs text-gray-300">{settlement.ticker}</span>
      </div>
      <div className="flex items-center gap-3">
        <span className="text-[10px] text-gray-500">
          {settlement.entryPrice}c → {settlement.exitPrice}c
        </span>
        <span className={`font-mono text-xs font-bold ${isWin ? 'text-green-400' : isLoss ? 'text-red-400' : 'text-gray-400'}`}>
          ${(settlement.pnlCents / 100).toFixed(2)}
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
 * Reddit Signal Row - Shows a Reddit-derived trading signal
 */
const RedditSignalRow = memo(({ signal }) => {
  const directionStyle = signal.direction === 'yes'
    ? 'bg-green-900/30 text-green-400 border-green-600/20'
    : signal.direction === 'no'
      ? 'bg-red-900/30 text-red-400 border-red-600/20'
      : 'bg-gray-800/30 text-gray-400 border-gray-600/20';

  return (
    <div className="p-2 bg-gray-800/30 rounded border border-gray-700/20 mb-2">
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-orange-400">r/{signal.subreddit}</span>
          <span className={`px-1.5 py-0.5 rounded text-[9px] font-bold border ${directionStyle}`}>
            {signal.direction.toUpperCase()}
          </span>
          <span className="text-[10px] text-gray-500">{signal.strength}</span>
        </div>
        <span className="text-[10px] text-amber-400">{(signal.confidence * 100).toFixed(0)}%</span>
      </div>
      <div className="text-[10px] text-gray-400 truncate" title={signal.title}>
        {signal.title}
      </div>
      <div className="text-[10px] text-gray-500 mt-1">{signal.reason}</div>
    </div>
  );
});

RedditSignalRow.displayName = 'RedditSignalRow';

/**
 * Price Impact Row - Shows the entity → sentiment → price impact transformation
 *
 * This is the key visualization showing how Reddit entity sentiment gets
 * transformed into market-specific price impact signals.
 */
const PriceImpactRow = memo(({ impact }) => {
  const sentimentIsPositive = impact.sentimentScore > 0;
  const impactIsPositive = impact.priceImpactScore > 0;
  const wasInverted = sentimentIsPositive !== impactIsPositive;

  // Color schemes based on impact direction
  const impactBg = impactIsPositive
    ? 'bg-gradient-to-r from-emerald-950/40 via-emerald-900/30 to-transparent'
    : 'bg-gradient-to-r from-rose-950/40 via-rose-900/30 to-transparent';

  const sideBadge = impact.suggestedSide?.toUpperCase() === 'YES'
    ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30 shadow-emerald-500/20'
    : 'bg-rose-500/20 text-rose-400 border-rose-500/30 shadow-rose-500/20';

  const sentimentColor = sentimentIsPositive ? 'text-emerald-400' : 'text-rose-400';
  const impactColor = impactIsPositive ? 'text-emerald-400' : 'text-rose-400';

  // Market type badge colors
  const marketTypeBadge = {
    OUT: 'bg-amber-900/30 text-amber-400 border-amber-600/30',
    CONFIRM: 'bg-blue-900/30 text-blue-400 border-blue-600/30',
    WIN: 'bg-emerald-900/30 text-emerald-400 border-emerald-600/30',
    NOMINEE: 'bg-violet-900/30 text-violet-400 border-violet-600/30',
    PRESIDENT: 'bg-cyan-900/30 text-cyan-400 border-cyan-600/30',
    UNKNOWN: 'bg-gray-800/30 text-gray-400 border-gray-600/30',
  }[impact.marketType] || 'bg-gray-800/30 text-gray-400 border-gray-600/30';

  // Format timestamp
  const formattedTime = typeof impact.timestamp === 'number'
    ? new Date(impact.timestamp * 1000).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
    : impact.timestamp;

  return (
    <div className={`
      relative overflow-hidden
      rounded-xl border border-gray-700/40
      ${impactBg}
      backdrop-blur-sm
      p-3 mb-2
      transition-all duration-300
      hover:border-gray-600/50 hover:shadow-lg
      group
    `}>
      {/* Subtle glow effect */}
      <div className={`
        absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500
        ${impactIsPositive ? 'bg-emerald-500/5' : 'bg-rose-500/5'}
      `} />

      {/* Header: Entity + Market Type + Suggested Side */}
      <div className="relative flex items-center justify-between mb-2">
        <div className="flex items-center gap-2 min-w-0 flex-1">
          <div className="p-1.5 rounded-lg bg-gray-800/50 border border-gray-700/30">
            <User className="w-3.5 h-3.5 text-gray-400" />
          </div>
          <div className="min-w-0 flex-1">
            <div className="text-sm font-medium text-gray-200 truncate">
              {impact.entityName}
            </div>
            {impact.subreddit && (
              <div className="text-[10px] text-orange-400/80">
                r/{impact.subreddit}
              </div>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2 flex-shrink-0">
          <span className={`px-2 py-0.5 rounded text-[9px] font-bold border ${marketTypeBadge}`}>
            {impact.marketType}
          </span>
          <span className={`
            px-2 py-1 rounded-lg text-[10px] font-bold border shadow-sm
            ${sideBadge}
          `}>
            {impact.suggestedSide}
          </span>
        </div>
      </div>

      {/* Transformation Pipeline: Sentiment → Impact */}
      <div className="relative flex items-center gap-2 mb-2 py-2 px-2 bg-gray-900/40 rounded-lg border border-gray-800/50">
        {/* Sentiment Score */}
        <div className="flex-1 text-center">
          <div className="text-[9px] text-gray-500 uppercase tracking-wider mb-1">Sentiment</div>
          <div className={`text-lg font-mono font-bold ${sentimentColor}`}>
            {impact.sentimentScore > 0 ? '+' : ''}{impact.sentimentScore}
          </div>
        </div>

        {/* Arrow with inversion indicator */}
        <div className="flex flex-col items-center px-2">
          <ArrowRight className={`w-4 h-4 ${wasInverted ? 'text-amber-400' : 'text-gray-500'}`} />
          {wasInverted && (
            <span className="text-[8px] text-amber-400 font-medium mt-0.5">INVERTED</span>
          )}
        </div>

        {/* Price Impact Score */}
        <div className="flex-1 text-center">
          <div className="text-[9px] text-gray-500 uppercase tracking-wider mb-1">Impact</div>
          <div className={`text-lg font-mono font-bold ${impactColor}`}>
            {impact.priceImpactScore > 0 ? '+' : ''}{impact.priceImpactScore}
          </div>
        </div>

        {/* Confidence */}
        <div className="flex-1 text-center border-l border-gray-700/50 pl-2">
          <div className="text-[9px] text-gray-500 uppercase tracking-wider mb-1">Confidence</div>
          <div className="text-lg font-mono font-bold text-cyan-400">
            {(impact.confidence * 100).toFixed(0)}%
          </div>
        </div>
      </div>

      {/* Market Ticker + Transformation Logic */}
      <div className="relative flex items-center justify-between">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-3 h-3 text-gray-500" />
          <span className="font-mono text-[11px] text-gray-400">{impact.marketTicker}</span>
        </div>
        <span className="text-[9px] text-gray-500 italic truncate max-w-[200px]" title={impact.transformationLogic}>
          {impact.transformationLogic || (wasInverted ? 'Sentiment inverted for market type' : 'Direct correlation')}
        </span>
      </div>

      {/* Timestamp */}
      {formattedTime && (
        <div className="absolute top-2 right-2 text-[9px] text-gray-600">
          {formattedTime}
        </div>
      )}
    </div>
  );
});

PriceImpactRow.displayName = 'PriceImpactRow';

/**
 * Stats Card - Compact stat display
 */
const StatsCard = memo(({ label, value, color = 'text-gray-300', bgColor = 'bg-gray-800/30', borderColor = 'border-gray-700/20' }) => (
  <div className={`rounded-lg p-2 border text-center ${bgColor} ${borderColor}`}>
    <div className="text-[9px] text-gray-500 uppercase tracking-wider font-medium mb-0.5">{label}</div>
    <div className={`text-sm font-mono font-bold ${color}`}>{value}</div>
  </div>
));

StatsCard.displayName = 'StatsCard';

/**
 * DeepAgentPanel - Main panel for the self-improving deep agent
 *
 * Displays:
 * - Agent status and session metrics
 * - Real-time thinking stream
 * - Tool call activity
 * - Price impacts (entity → market transformation pipeline)
 * - Trade executions and settlements
 * - Learnings (live updates to memory)
 * - Reddit signals (if enabled)
 */
const DeepAgentPanel = ({
  agentState,
  thinking,
  toolCalls = [],
  trades = [],
  settlements = [],
  learnings = [],
  redditSignals = [],
  priceImpacts = [],
  isRunning = false,
  isLearning = false,
}) => {
  const [isExpanded, setIsExpanded] = useState(true);
  const [showTools, setShowTools] = useState(false);
  const [showReddit, setShowReddit] = useState(false);
  const [showPriceImpacts, setShowPriceImpacts] = useState(true);

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
    <div className="
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
          <div className={`
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
            />
            <StatsCard
              label="Trades"
              value={trades.length}
              color="text-cyan-400"
              bgColor="bg-cyan-900/20"
              borderColor="border-cyan-700/30"
            />
            <StatsCard
              label="Win Rate"
              value={`${winRate}%`}
              color={parseInt(winRate) >= 50 ? 'text-emerald-400' : 'text-amber-400'}
              bgColor={parseInt(winRate) >= 50 ? 'bg-emerald-900/20' : 'bg-amber-900/20'}
              borderColor={parseInt(winRate) >= 50 ? 'border-emerald-700/30' : 'border-amber-700/30'}
            />
            <StatsCard
              label="Cycles"
              value={agentState?.cycleCount || 0}
              color="text-violet-400"
              bgColor="bg-violet-900/20"
              borderColor="border-violet-700/30"
            />
          </div>

          {/* Thinking Stream */}
          <div className="mb-4">
            <div className="flex items-center space-x-2 mb-2">
              <Brain className="w-3 h-3 text-gray-500" />
              <span className="text-[10px] text-gray-500 uppercase tracking-wider font-medium">Thinking</span>
            </div>
            <ThinkingStream thinking={thinking} isLearning={isLearning} />
          </div>

          {/* Price Impacts - Entity → Market Transformation Pipeline */}
          {priceImpacts.length > 0 && (
            <div className="mb-4">
              <button
                onClick={() => setShowPriceImpacts(!showPriceImpacts)}
                className="w-full flex items-center justify-between px-3 py-2.5 bg-gradient-to-r from-orange-900/20 via-amber-900/15 to-transparent hover:from-orange-900/30 rounded-lg transition-all duration-200 border border-orange-800/30"
              >
                <div className="flex items-center gap-2">
                  <div className="p-1 rounded-md bg-orange-900/40">
                    <Sparkles className="w-3.5 h-3.5 text-orange-400" />
                  </div>
                  <span className="text-[11px] text-orange-300 uppercase tracking-wider font-semibold">
                    Price Impacts
                  </span>
                  <span className="px-2 py-0.5 bg-orange-900/40 text-orange-400 text-[10px] font-bold rounded-full border border-orange-700/30">
                    {priceImpacts.length}
                  </span>
                  <span className="text-[9px] text-gray-500 ml-1">
                    Entity → Market
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  {priceImpacts.length > 0 && (
                    <span className={`text-[10px] font-mono ${
                      priceImpacts[0]?.priceImpactScore > 0 ? 'text-emerald-400' : 'text-rose-400'
                    }`}>
                      Latest: {priceImpacts[0]?.priceImpactScore > 0 ? '+' : ''}{priceImpacts[0]?.priceImpactScore}
                    </span>
                  )}
                  {showPriceImpacts ? (
                    <ChevronDown className="w-4 h-4 text-orange-400/60" />
                  ) : (
                    <ChevronRight className="w-4 h-4 text-orange-400/60" />
                  )}
                </div>
              </button>
              {showPriceImpacts && (
                <div className="mt-3 max-h-[400px] overflow-y-auto pr-1 space-y-0">
                  {priceImpacts.slice(0, 10).map((impact) => (
                    <PriceImpactRow key={impact.id} impact={impact} />
                  ))}
                </div>
              )}
            </div>
          )}

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

          {/* Reddit Signals */}
          {redditSignals.length > 0 && (
            <div>
              <button
                onClick={() => setShowReddit(!showReddit)}
                className="w-full flex items-center justify-between px-3 py-2 bg-gray-800/30 hover:bg-gray-800/50 rounded-lg transition-colors"
              >
                <div className="flex items-center gap-2">
                  <MessageSquare className="w-3.5 h-3.5 text-orange-400" />
                  <span className="text-[10px] text-gray-400 uppercase tracking-wider font-medium">
                    Reddit Signals
                  </span>
                  <span className="px-1.5 py-0.5 bg-orange-900/30 text-orange-400 text-[9px] rounded">
                    {redditSignals.length}
                  </span>
                </div>
                {showReddit ? (
                  <ChevronDown className="w-3.5 h-3.5 text-gray-500" />
                ) : (
                  <ChevronRight className="w-3.5 h-3.5 text-gray-500" />
                )}
              </button>
              {showReddit && (
                <div className="mt-2 max-h-[200px] overflow-y-auto">
                  {redditSignals.slice(0, 5).map((signal) => (
                    <RedditSignalRow key={signal.id} signal={signal} />
                  ))}
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default memo(DeepAgentPanel);
