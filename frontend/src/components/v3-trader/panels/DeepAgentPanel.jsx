import React, { useState, memo, useMemo } from 'react';
import { formatRelativeTimestamp, formatLatency } from '../../../utils/v3-trader/formatters';
import useRelativeTime from '../../../hooks/v3-trader/useRelativeTime';
import useSignalFreshness from '../../../hooks/v3-trader/useSignalFreshness';
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
        {(settlement.entryPrice != null && settlement.exitPrice != null) && (
          <span className="text-[10px] text-gray-500">
            {settlement.entryPrice}c → {settlement.exitPrice}c
          </span>
        )}
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
 *
 * Displays:
 * - Dual timestamp bar (Posted / Detected / Lag)
 * - Contextual subtitles for sentiment, impact, and confidence
 * - Source context (Reddit post title and entity context snippet)
 * - Agent status and market type badges
 * - Sentiment → Impact transformation pipeline
 */
const PriceImpactRow = memo(({ impact, isNew = false, freshnessTier = 'normal' }) => {
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

  // Market type badge colors and labels
  const marketTypeConfig = {
    OUT: { style: 'bg-amber-900/30 text-amber-400 border-amber-600/30', label: 'OUT' },
    CONFIRM: { style: 'bg-blue-900/30 text-blue-400 border-blue-600/30', label: 'CONFIRM' },
    WIN: { style: 'bg-emerald-900/30 text-emerald-400 border-emerald-600/30', label: 'WIN' },
    NOMINEE: { style: 'bg-violet-900/30 text-violet-400 border-violet-600/30', label: 'NOMINEE' },
    PRESIDENT: { style: 'bg-cyan-900/30 text-cyan-400 border-cyan-600/30', label: 'PRES' },
    MENTION: { style: 'bg-orange-900/30 text-orange-400 border-orange-600/30', label: 'MENTION' },
    UNKNOWN: { style: 'bg-gray-800/30 text-gray-500 border-gray-600/30', label: 'SIGNAL' },
  };
  const marketType = marketTypeConfig[impact.marketType] || marketTypeConfig.UNKNOWN;

  // Agent status badge configuration
  const agentStatusConfig = {
    pending: { style: 'bg-gray-800/40 text-gray-400 border-gray-600/30', label: 'Pending', icon: Clock },
    viewed: { style: 'bg-blue-900/30 text-blue-400 border-blue-600/30', label: 'Viewed', icon: Activity },
    traded: { style: 'bg-emerald-900/30 text-emerald-400 border-emerald-600/30', label: 'Traded', icon: CheckCircle },
    observed: { style: 'bg-amber-900/30 text-amber-400 border-amber-600/30', label: 'Watching', icon: AlertCircle },
    rejected: { style: 'bg-red-900/30 text-red-400 border-red-600/30', label: 'Rejected', icon: XCircle },
  };
  const agentStatus = agentStatusConfig[impact.agentStatus] || agentStatusConfig.pending;
  const StatusIcon = agentStatus.icon;

  // Source type badge configuration
  const sourceTypeConfig = {
    reddit_text: { style: 'bg-orange-900/30 text-orange-400 border-orange-600/30', label: 'Text', icon: FileText },
    video_transcript: { style: 'bg-rose-900/30 text-rose-400 border-rose-600/30', label: 'Video', icon: Activity },
    article_extract: { style: 'bg-blue-900/30 text-blue-400 border-blue-600/30', label: 'Article', icon: BookOpen },
  };
  const sourceType = sourceTypeConfig[impact.sourceType] || sourceTypeConfig.reddit_text;
  const SourceIcon = sourceType.icon;

  // Live-updating timestamps
  const postedAgo = formatRelativeTimestamp(impact.sourceCreatedAt);
  const detectedAgo = formatRelativeTimestamp(impact.timestamp);
  const lag = formatLatency(impact.sourceCreatedAt, impact.timestamp);

  // Contextual subtitles
  const sentimentSubtitle = impact.contextSnippet
    ? (impact.contextSnippet.length > 40 ? impact.contextSnippet.slice(0, 40) + '...' : impact.contextSnippet)
    : `${impact.sourceType === 'reddit_text' ? 'Reddit' : 'Source'} discussion: ${sentimentIsPositive ? 'positive' : 'negative'} tone`;

  const impactSubtitle = wasInverted
    ? `${impact.marketType || 'OUT'}: negative news helps`
    : `${impact.marketType === 'WIN' ? 'WIN' : 'Direct'}: sentiment aligns`;

  const confidenceSubtitle = impact.confidence >= 0.9
    ? 'High: strong match'
    : impact.confidence >= 0.7
      ? 'Medium: likely match'
      : 'Low: uncertain match';

  // Build animation classes
  const animationClasses = [
    isNew ? 'animate-signal-slide-in' : '',
    freshnessTier === 'fresh' ? 'animate-signal-fresh' : '',
    freshnessTier === 'recent' ? 'animate-signal-recent' : '',
  ].filter(Boolean).join(' ');

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
      ${animationClasses}
    `}>
      {/* Subtle glow effect */}
      <div className={`
        absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500
        ${impactIsPositive ? 'bg-emerald-500/5' : 'bg-rose-500/5'}
      `} />

      {/* Header: Entity + Badges Row */}
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
              <span className="text-[10px] text-orange-400/80">r/{impact.subreddit}</span>
            )}
          </div>
        </div>

        {/* Right side badges: Agent Status + Market Type + Side */}
        <div className="flex items-center gap-1.5 flex-shrink-0">
          {/* Agent Status Badge */}
          <span className={`
            px-1.5 py-0.5 rounded text-[9px] font-medium border flex items-center gap-1
            ${agentStatus.style}
          `}>
            <StatusIcon className="w-2.5 h-2.5" />
            {agentStatus.label}
          </span>
          {/* Market Type Badge */}
          <span className={`px-2 py-0.5 rounded text-[9px] font-bold border ${marketType.style}`}>
            {marketType.label}
          </span>
          {/* Suggested Side Badge */}
          {impact.suggestedSide && (
            <span className={`
              px-2 py-1 rounded-lg text-[10px] font-bold border shadow-sm
              ${sideBadge}
            `}>
              {impact.suggestedSide}
            </span>
          )}
        </div>
      </div>

      {/* Dual Timestamp Bar: Posted | Detected | Lag */}
      {(postedAgo || detectedAgo) && (
        <div className="relative flex items-center gap-3 mb-2 px-2 py-1.5 bg-gray-900/50 rounded-lg border border-gray-800/30 text-[10px]">
          {postedAgo && (
            <span className="flex items-center gap-1">
              <Clock className="w-3 h-3 text-gray-500" />
              <span className="text-gray-500">Posted</span>
              <span className="text-orange-400 font-medium">{postedAgo}</span>
            </span>
          )}
          {postedAgo && detectedAgo && (
            <span className="text-gray-700">|</span>
          )}
          {detectedAgo && (
            <span className="flex items-center gap-1">
              <Zap className="w-3 h-3 text-gray-500" />
              <span className="text-gray-500">Detected</span>
              <span className="text-cyan-400 font-medium">{detectedAgo}</span>
            </span>
          )}
          {lag && (
            <>
              <span className="text-gray-700">|</span>
              <span className="flex items-center gap-1">
                <Activity className="w-3 h-3 text-gray-500" />
                <span className="text-gray-500">{lag} lag</span>
              </span>
            </>
          )}
        </div>
      )}

      {/* Source Context: Reddit Post Title + Context Snippet - THE WHY */}
      <div className="relative mb-2 p-2.5 bg-gray-900/60 rounded-lg border border-gray-800/40">
        <div className="flex items-start gap-2">
          <div className="flex-shrink-0 mt-0.5">
            <SourceIcon className={`w-3 h-3 ${sourceType.style.includes('orange') ? 'text-orange-400/70' : sourceType.style.includes('rose') ? 'text-rose-400/70' : 'text-blue-400/70'}`} />
          </div>
          <div className="flex-1 min-w-0">
            {impact.sourceTitle ? (
              <>
                <div className="text-xs font-medium text-gray-200 line-clamp-2 leading-relaxed" title={impact.sourceTitle}>
                  "{impact.sourceTitle}"
                </div>
                {impact.contextSnippet && (
                  <div className="text-[10px] text-gray-400 mt-1 line-clamp-3" title={impact.contextSnippet}>
                    {impact.contextSnippet}
                  </div>
                )}
              </>
            ) : (
              <div className="text-[10px] text-gray-500 italic">
                No source context available
              </div>
            )}
            {/* Source type label */}
            <div className="mt-1">
              <span className={`px-1.5 py-0.5 rounded text-[8px] font-medium border ${sourceType.style}`}>
                {sourceType.label}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Transformation Pipeline: Sentiment → Impact */}
      <div className="relative flex items-center gap-2 mb-2 py-2 px-2 bg-gray-900/40 rounded-lg border border-gray-800/50">
        {/* Entity Sentiment Score */}
        <div className="flex-1 text-center">
          <div className="text-[9px] text-gray-500 uppercase tracking-wider mb-1">Entity Sentiment</div>
          <div className={`text-lg font-mono font-bold ${sentimentColor}`}>
            {impact.sentimentScore > 0 ? '+' : ''}{impact.sentimentScore}
          </div>
          <div className="text-[9px] text-gray-500 mt-0.5">
            {sentimentSubtitle}
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
          <div className="text-[9px] text-gray-500 uppercase tracking-wider mb-1">Price Impact</div>
          <div className={`text-lg font-mono font-bold ${impactColor}`}>
            {impact.priceImpactScore > 0 ? '+' : ''}{impact.priceImpactScore}
          </div>
          <div className="text-[9px] text-gray-500 mt-0.5" title={impact.transformationLogic}>
            {impactSubtitle}
          </div>
        </div>

        {/* Confidence */}
        <div className="flex-1 text-center border-l border-gray-700/50 pl-2">
          <div className="text-[9px] text-gray-500 uppercase tracking-wider mb-1">Confidence</div>
          <div className="text-lg font-mono font-bold text-cyan-400">
            {(impact.confidence * 100).toFixed(0)}%
          </div>
          <div className="text-[9px] text-gray-500 mt-0.5">
            {confidenceSubtitle}
          </div>
        </div>
      </div>

      {/* Market Ticker + Transformation Logic */}
      <div className="relative flex items-center justify-between border-t border-gray-800/30 pt-1">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-3 h-3 text-gray-500" />
          <span className="font-mono text-[11px] text-gray-400">{impact.marketTicker}</span>
        </div>
        <span className="text-[10px] text-gray-400 truncate max-w-[280px]" title={impact.transformationLogic}>
          {impact.transformationLogic || (wasInverted ? 'Sentiment inverted for market type' : 'Direct sentiment correlation')}
        </span>
      </div>
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
 * Props:
 * - statsOnly: When true, renders only header and 8-stat summary (for Trader tab)
 * - agentState: Current agent state object
 * - thinking: Current thinking stream
 * - toolCalls: Array of tool call records
 * - trades: Array of trade records
 * - settlements: Array of settlement records
 * - learnings: Array of learning records
 * - redditSignals: Array of reddit signal records
 * - priceImpacts: Array of price impact records
 * - isRunning: Whether agent is currently running
 * - isLearning: Whether agent is in learning mode
 *
 * When statsOnly={true}, displays:
 * - Agent status header with RUNNING/STOPPED badge
 * - Session stats (P&L, Trades, Win Rate, Cycles)
 * - Entity pipeline stats (Reddit, Entities, Signals, Index)
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
  redditSignals = [],
  priceImpacts = [],
  isRunning = false,
  isLearning = false,
  statsOnly = false,
}) => {
  const [isExpanded, setIsExpanded] = useState(true);
  const [showTools, setShowTools] = useState(false);
  const [showReddit, setShowReddit] = useState(false);
  const [showPriceImpacts, setShowPriceImpacts] = useState(true);

  // Live-updating timestamps: forces re-render every 30s
  useRelativeTime(30000);
  // Track new signal arrivals + freshness tiers
  const { newSignalIds, getFreshnessTier } = useSignalFreshness(priceImpacts);

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

          {/* Entity Pipeline Stats (Reddit → Entities → Signals) */}
          <div className={statsOnly ? "grid grid-cols-4 gap-2" : "grid grid-cols-4 gap-2 mb-4"}>
            <StatsCard
              label="Reddit"
              value={agentState?.redditPostsProcessed || redditSignals.length || 0}
              color="text-orange-400"
              bgColor="bg-orange-900/20"
              borderColor="border-orange-700/30"
            />
            <StatsCard
              label="Entities"
              value={agentState?.entitiesExtracted || 0}
              color="text-amber-400"
              bgColor="bg-amber-900/20"
              borderColor="border-amber-700/30"
            />
            <StatsCard
              label="Signals"
              value={agentState?.signalsGenerated || priceImpacts.length || 0}
              color="text-rose-400"
              bgColor="bg-rose-900/20"
              borderColor="border-rose-700/30"
            />
            <StatsCard
              label="Index"
              value={agentState?.indexSize || 0}
              color="text-blue-400"
              bgColor="bg-blue-900/20"
              borderColor="border-blue-700/30"
            />
          </div>

          {/* Stats-only mode stops here - skip remaining sections */}
          {statsOnly ? null : (
          <>
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
                    <PriceImpactRow
                      key={impact.id}
                      impact={impact}
                      isNew={newSignalIds.has(impact.id)}
                      freshnessTier={getFreshnessTier(impact)}
                    />
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
        </>
      )}
    </div>
  );
};

export default memo(DeepAgentPanel);
